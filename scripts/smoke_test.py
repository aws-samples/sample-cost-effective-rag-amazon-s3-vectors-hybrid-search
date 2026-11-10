#!/usr/bin/env python3
"""End-to-end smoke test for the deployed s3-vectors-hybrid-rag stacks.

Used by the "Ingest documents and test retrieval" epic. Auto-discovers the
deployed resources from CloudFormation outputs, then:
  1. Uploads a small test document with distinctive facts (tenant: smoketest)
  2. Waits for the ingestion pipeline to index it
  3. Verifies /search in hybrid, dense, and sparse modes
  4. Verifies /query returns a grounded answer with citations
  5. Verifies tenant isolation (another tenant sees nothing)
  6. With --cleanup: deletes the test document and verifies its passages are
     no longer searchable (the ObjectRemoved notification removes its vectors
     and BM25 postings)

Run from the project root with deployed-account credentials in the shell:
    python scripts/smoke_test.py [--prefix hybrid-rag] [--region us-east-1] [--cleanup]

Exit code 0 = all checks passed. Costs: a few embedding calls + one Claude
call (well under a cent).

HTTP 403 responses are retried with backoff for up to 3 minutes per request,
because API Gateway can take one to two minutes (about 60-95 seconds in
testing) to propagate a newly created API key and its usage-plan association
right after `cdk deploy`. That retry time doesn't count against the ingestion
and cleanup polling windows.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import boto3

TENANT = "smoketest"
DOC_KEY = f"tenants/{TENANT}/nimbuscache-notes.md"
DOC_BODY = """# NimbusCache Pro - internal notes

NimbusCache Pro is a fictional caching appliance used only for smoke testing.

## Pricing
NimbusCache Pro costs exactly 19 dollars per month on the standard plan.

## Performance
Benchmarks show NimbusCache Pro reduces median lookup latency by 42 percent
compared to the previous ZephyrStore appliance.

## Support
Support tickets are answered within one business day.
"""

QUESTION = "How much does NimbusCache Pro cost per month?"
EXPECT_IN_ANSWER = "19"
KEYWORD_QUERY = "NimbusCache latency benchmark"

# A new API key / usage-plan association can take one to two minutes to
# propagate after `cdk deploy` (about 94 s observed in testing); until then API
# Gateway returns 403 {"message": "Forbidden"}.
FORBIDDEN_RETRY_SECONDS = 180
FORBIDDEN_BACKOFF_INITIAL_SECONDS = 5
FORBIDDEN_BACKOFF_MAX_SECONDS = 15

# Outer /search polling windows (ingestion and --cleanup). Time http_post spends
# waiting out 403s is added to these, so key propagation can't shorten them.
POLL_BUDGET_SECONDS = 120
POLL_INTERVAL_SECONDS = 6


class ApiKeyForbiddenError(RuntimeError):
    """API Gateway kept returning 403 after FORBIDDEN_RETRY_SECONDS of retries."""


def stack_outputs(cfn, stack_name: str) -> dict:
    stacks = cfn.describe_stacks(StackName=stack_name)["Stacks"]
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


def http_post(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int = 65,
    *,
    opener=None,
    sleep=None,
    clock=None,
    on_forbidden_wait=None,
) -> tuple[int, dict]:
    """POST JSON and return (status, body).

    Retries HTTP 403 with backoff for up to FORBIDDEN_RETRY_SECONDS (API key
    propagation), then raises ApiKeyForbiddenError. Other 4xx/5xx codes are
    returned without retrying. ``opener``, ``sleep``, and ``clock`` default to
    urllib.request.urlopen, time.sleep, and time.monotonic (injectable for tests).
    If any 403s were retried, ``on_forbidden_wait`` (when given) is called with
    the seconds spent on them before the non-403 response is returned.
    """
    # Endpoint URLs come from CloudFormation outputs; enforce HTTPS anyway so
    # urlopen can never be pointed at file:// or other unexpected schemes.
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS URL: {url}")
    opener = opener or urllib.request.urlopen
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": api_key},
        method="POST",
    )
    start = clock()
    delay = FORBIDDEN_BACKOFF_INITIAL_SECONDS
    warned = False
    while True:
        attempt_start = clock()
        try:
            with opener(request, timeout=timeout) as response:  # nosec B310 - HTTPS enforced above
                result = response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 403:
                if not warned:
                    print("  [wait] API Gateway returned 403 (API key still propagating); retrying...")
                    warned = True
                remaining = FORBIDDEN_RETRY_SECONDS - (clock() - start)
                if remaining <= 0:
                    raise ApiKeyForbiddenError(
                        f"{url} still returned HTTP 403 after {FORBIDDEN_RETRY_SECONDS}s of retries. "
                        "Verify that the x-api-key header carries the value of the ApiKeyId output "
                        "(aws apigateway get-api-key --api-key <ApiKeyId> --include-value "
                        "--query value --output text)."
                    ) from err
                sleep(min(delay, remaining))
                delay = min(delay * 2, FORBIDDEN_BACKOFF_MAX_SECONDS)
                continue
            try:
                body = json.loads(err.read().decode("utf-8"))
            except Exception:  # noqa: BLE001
                body = {"error": str(err)}
            result = err.code, body
        if warned and on_forbidden_wait is not None:
            # Everything before this (non-403) attempt was spent on 403 retries.
            on_forbidden_wait(attempt_start - start)
        return result


def poll_search(
    url: str,
    api_key: str,
    payload: dict,
    done,
    budget_seconds: float = POLL_BUDGET_SECONDS,
    interval_seconds: float = POLL_INTERVAL_SECONDS,
    *,
    opener=None,
    sleep=None,
    clock=None,
) -> tuple[int, dict]:
    """POST ``payload`` until ``done(status, body)`` or ``budget_seconds`` of polling elapse.

    Returns the last (status, body). Time http_post spends waiting out 403s
    (API key propagation) extends the deadline, so it doesn't eat into the
    polling budget. ApiKeyForbiddenError from http_post propagates unchanged.
    """
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    forbidden_seconds = 0.0

    def add_forbidden_wait(seconds: float) -> None:
        nonlocal forbidden_seconds
        forbidden_seconds += seconds

    deadline = clock() + budget_seconds
    while True:
        status, body = http_post(
            url,
            api_key,
            payload,
            opener=opener,
            sleep=sleep,
            clock=clock,
            on_forbidden_wait=add_forbidden_wait,
        )
        if done(status, body):
            return status, body
        sleep(interval_seconds)
        if clock() >= deadline + forbidden_seconds:
            return status, body


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def resolve_region(cli_region: str | None) -> str:
    """Pick the Region the way the CDK app does, so both target the same stacks.

    Order: --region, then AWS_REGION, then AWS_DEFAULT_REGION or the AWS CLI
    profile's region (boto3 reads both; it doesn't read AWS_REGION), then
    CDK_DEFAULT_REGION, then us-east-1 (the app's fallback in infra/app.py).
    """
    return (
        cli_region
        or os.environ.get("AWS_REGION")
        or boto3.Session().region_name
        or os.environ.get("CDK_DEFAULT_REGION")
        or "us-east-1"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="hybrid-rag", help="stack name prefix")
    parser.add_argument(
        "--region",
        default=None,
        help="AWS Region (default: AWS_REGION, then AWS_DEFAULT_REGION or the AWS CLI "
        "profile region, then CDK_DEFAULT_REGION, then us-east-1)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="delete the test document afterwards and verify its passages are no longer searchable",
    )
    args = parser.parse_args()

    region = resolve_region(args.region)
    session = boto3.Session(region_name=region)
    print(f"== Using region {region} ==")
    cfn = session.client("cloudformation")
    s3 = session.client("s3")
    apigw = session.client("apigateway")

    print("== Discovering deployed resources ==")
    compute = stack_outputs(cfn, f"{args.prefix}-compute")
    api = stack_outputs(cfn, f"{args.prefix}-api")
    bucket = compute["DocsBucketName"]
    endpoint = api["ApiEndpoint"].rstrip("/")
    api_key = apigw.get_api_key(apiKey=api["ApiKeyId"], includeValue=True)["value"]
    print(f"  docs bucket : {bucket}")
    print(f"  api endpoint: {endpoint}")

    print("\n== 1. Upload test document ==")
    s3.put_object(Bucket=bucket, Key=DOC_KEY, Body=DOC_BODY.encode("utf-8"))
    print(f"  uploaded s3://{bucket}/{DOC_KEY}")

    print("\n== 2. Wait for ingestion (poll /search) ==")
    status, body = poll_search(
        f"{endpoint}/search",
        api_key,
        {"query": KEYWORD_QUERY, "tenant_id": TENANT, "top_k": 3, "mode": "hybrid"},
        lambda status, body: status == 200 and bool(body.get("results")),
    )
    results = body.get("results", []) if status == 200 else []
    ok_ingest = check(f"document indexed within {POLL_BUDGET_SECONDS}s", bool(results))
    if not ok_ingest:
        log_group = compute.get(
            "IngestionLogGroupName", "<IngestionLogGroupName output of the compute stack>"
        )
        print(f"  hint: aws logs tail {log_group} --since 10m")
        return 1

    all_ok = True
    print("\n== 3. Search modes ==")
    top_text = (results[0].get("text") or "").lower()
    all_ok &= check("hybrid result contains document text", "nimbuscache" in top_text)

    for mode in ("dense", "sparse"):
        status, body = http_post(
            f"{endpoint}/search",
            api_key,
            {"query": KEYWORD_QUERY, "tenant_id": TENANT, "top_k": 3, "mode": mode},
        )
        mode_results = body.get("results", []) if status == 200 else []
        all_ok &= check(f"{mode} mode returns results", status == 200 and bool(mode_results))

    print("\n== 4. Grounded generation ==")
    status, body = http_post(
        f"{endpoint}/query",
        api_key,
        {"question": QUESTION, "tenant_id": TENANT, "top_k": 3},
    )
    answer = body.get("answer", "")
    citations = body.get("citations", [])
    all_ok &= check("query returns HTTP 200", status == 200, f"status={status}")
    all_ok &= check(
        f"answer contains '{EXPECT_IN_ANSWER}'", EXPECT_IN_ANSWER in answer, f"answer={answer[:120]!r}"
    )
    all_ok &= check("answer includes citations", bool(citations), f"count={len(citations)}")
    if status != 200:
        print(f"  response body: {json.dumps(body)[:300]}")
        print("  hint: a 500 with AccessDeniedException in the query function's logs")
        print(f"  ({compute.get('QueryLogGroupName', '<QueryLogGroupName output of the compute stack>')})")
        print("  usually means the Anthropic use-case form or the AWS Marketplace subscription")
        print("  for the generation model is missing (see the Confirm Amazon Bedrock model")
        print("  access story).")

    print("\n== 5. Tenant isolation ==")
    status, body = http_post(
        f"{endpoint}/search",
        api_key,
        {"query": KEYWORD_QUERY, "tenant_id": "someone-else", "top_k": 3, "mode": "hybrid"},
    )
    other = body.get("results", []) if status == 200 else None
    all_ok &= check("other tenant sees zero results", other == [])

    if args.cleanup:
        print("\n== 6. Cleanup (poll /search until the document is gone) ==")
        s3.delete_object(Bucket=bucket, Key=DOC_KEY)
        print(f"  deleted s3://{bucket}/{DOC_KEY}; the ObjectRemoved notification removes its vectors and postings")
        status, body = poll_search(
            f"{endpoint}/search",
            api_key,
            {"query": KEYWORD_QUERY, "tenant_id": TENANT, "top_k": 3, "mode": "hybrid"},
            lambda status, body: status == 200 and body.get("results") == [],
        )
        remaining = body.get("results") if status == 200 else None
        all_ok &= check("deleted document no longer searchable", remaining == [])

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ApiKeyForbiddenError as err:
        print(f"  [FAIL] {err}")
        sys.exit(1)
