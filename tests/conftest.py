"""Shared test setup: environment variables and lightweight AWS fakes.

``pythonpath = src infra`` in pytest.ini makes both the Lambda packages and
the CDK stacks importable without installation.
"""
import io
import json
import os
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

# Set before any src module import reads them.
os.environ.setdefault("VECTOR_BUCKET_NAME", "test-vector-bucket")
os.environ.setdefault("VECTOR_INDEX_NAME", "chunks")
os.environ.setdefault("BM25_TABLE_NAME", "test-bm25-table")
os.environ.setdefault("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "8")
os.environ.setdefault("GENERATION_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
os.environ.setdefault("DEFAULT_TENANT", "default")
os.environ.setdefault("MAX_CONTEXT_TOKENS", "6000")


def _eq_condition(cond):
    """Extract (attribute_name, value) from a boto3 Key(...).eq(...) condition."""
    expression = cond.get_expression()
    attr, value = expression["values"]
    return attr.name, value


class FakeBatchWriter:
    def __init__(self, table):
        self._table = table

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def put_item(self, Item):
        self._table.put_item(Item=Item)

    def delete_item(self, Key):
        self._table.delete_item(Key=Key)


class FakeTable:
    """In-memory stand-in for a boto3 DynamoDB Table (subset used by the code)."""

    def __init__(self):
        self.items: dict[tuple, dict] = {}

    # -- writes ------------------------------------------------------------
    def put_item(self, Item):
        self.items[(Item["PK"], Item["SK"])] = dict(Item)

    def delete_item(self, Key):
        self.items.pop((Key["PK"], Key["SK"]), None)

    def update_item(self, Key, UpdateExpression, ExpressionAttributeValues):
        assert UpdateExpression.startswith("ADD "), "fake supports ADD expressions only"
        item = self.items.setdefault((Key["PK"], Key["SK"]), dict(Key))
        for clause in UpdateExpression[4:].split(","):
            attr, placeholder = clause.strip().split(" ")
            item[attr] = item.get(attr, 0) + ExpressionAttributeValues[placeholder]

    def batch_writer(self):
        return FakeBatchWriter(self)

    # -- reads ---------------------------------------------------------------
    def get_item(self, Key):
        item = self.items.get((Key["PK"], Key["SK"]))
        return {"Item": dict(item)} if item else {}

    def query(self, **kwargs):
        name, value = _eq_condition(kwargs["KeyConditionExpression"])
        if kwargs.get("IndexName") == "DocIndex":
            assert name == "gsi1pk"
            matches = [i for i in self.items.values() if i.get("gsi1pk") == value]
        else:
            assert name == "PK"
            matches = [i for i in self.items.values() if i["PK"] == value]
        matches = sorted(matches, key=lambda i: i["SK"])
        return {"Items": [dict(i) for i in matches]}  # single page, no pagination


class FakeBedrockBody:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data


class FakeBedrockRuntime:
    """Fake bedrock-runtime client: deterministic embeddings + canned converse."""

    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        self.invoke_calls: list[dict] = []
        self.converse_calls: list[dict] = []
        self.converse_reply = "The answer is 42. [1]"

    def invoke_model(self, **kwargs):
        self.invoke_calls.append(kwargs)
        body = json.loads(kwargs["body"])
        seed = float(len(body["inputText"]) % 97) / 97.0
        dims = int(body.get("dimensions", self.dimensions))
        payload = {
            "embedding": [round(seed + i * 0.001, 6) for i in range(dims)],
            "inputTextTokenCount": max(1, len(body["inputText"]) // 4),
        }
        return {"body": FakeBedrockBody(payload)}

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": self.converse_reply}]}},
            "usage": {"inputTokens": 100, "outputTokens": 20},
            "stopReason": "end_turn",
        }


class FakeS3:
    """In-memory S3 client (subset): objects keyed by (bucket, key)."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, dict]] = []

    def _missing(self, code: str, operation: str):
        return ClientError({"Error": {"Code": code, "Message": "Not Found"}}, operation)

    def put_object(self, Bucket, Key, Body=b""):
        self.calls.append(("put_object", {"Bucket": Bucket, "Key": Key}))
        self.objects[(Bucket, Key)] = Body.encode("utf-8") if isinstance(Body, str) else bytes(Body)
        return {}

    def delete_object(self, Bucket, Key):
        self.calls.append(("delete_object", {"Bucket": Bucket, "Key": Key}))
        self.objects.pop((Bucket, Key), None)
        return {}

    def get_object(self, Bucket, Key):
        self.calls.append(("get_object", {"Bucket": Bucket, "Key": Key}))
        if (Bucket, Key) not in self.objects:
            raise self._missing("NoSuchKey", "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        self.calls.append(("head_object", {"Bucket": Bucket, "Key": Key}))
        if (Bucket, Key) not in self.objects:
            raise self._missing("404", "HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


class FakeS3Vectors:
    """Records s3vectors calls; set ``fail_on[method] = exc`` to make one raise."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.query_reply: dict = {"vectors": []}
        self.get_reply: dict = {"vectors": []}
        self.fail_on: dict[str, Exception] = {}

    def _record(self, name: str, kwargs: dict) -> None:
        self.calls.append((name, kwargs))
        if name in self.fail_on:
            raise self.fail_on[name]

    def calls_to(self, name: str) -> list[dict]:
        return [kwargs for called, kwargs in self.calls if called == name]

    def put_vectors(self, **kwargs):
        self._record("put_vectors", kwargs)
        return {}

    def delete_vectors(self, **kwargs):
        self._record("delete_vectors", kwargs)
        return {}

    def query_vectors(self, **kwargs):
        self._record("query_vectors", kwargs)
        return self.query_reply

    def get_vectors(self, **kwargs):
        self._record("get_vectors", kwargs)
        return self.get_reply


@pytest.fixture
def fake_table():
    return FakeTable()


@pytest.fixture
def fake_bedrock(monkeypatch):
    from common import aws_clients

    fake = FakeBedrockRuntime()
    monkeypatch.setattr(aws_clients, "bedrock_runtime", lambda: fake)
    return fake


@pytest.fixture
def fake_s3():
    return FakeS3()


@pytest.fixture
def fake_s3vectors():
    return FakeS3Vectors()


@pytest.fixture
def ingestion_env(monkeypatch, fake_table, fake_bedrock, fake_s3, fake_s3vectors):
    """Route every AWS client the ingestion pipeline uses to in-memory fakes."""
    from common import aws_clients

    monkeypatch.setattr(aws_clients, "s3", lambda: fake_s3)
    monkeypatch.setattr(aws_clients, "s3vectors", lambda: fake_s3vectors)
    monkeypatch.setattr(aws_clients, "dynamodb_table", lambda name: fake_table)
    return SimpleNamespace(
        table=fake_table, bedrock=fake_bedrock, s3=fake_s3, s3vectors=fake_s3vectors
    )
