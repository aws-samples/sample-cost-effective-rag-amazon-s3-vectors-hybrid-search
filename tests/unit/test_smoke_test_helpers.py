"""Region resolution and HTTP retry in scripts/smoke_test.py (no AWS calls)."""
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "smoke_test.py"


@pytest.fixture
def smoke(monkeypatch, tmp_path):
    # Isolate from the developer's AWS config and environment.
    config_file = tmp_path / "config"
    config_file.write_text("[profile regional]\nregion = eu-west-1\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE", "CDK_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)

    spec = importlib.util.spec_from_file_location("smoke_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_region_wins(smoke, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert smoke.resolve_region("ap-southeast-2") == "ap-southeast-2"


def test_aws_region_env_next(smoke, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    assert smoke.resolve_region(None) == "us-west-2"


def test_profile_region(smoke, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "regional")
    monkeypatch.setenv("CDK_DEFAULT_REGION", "ca-central-1")
    assert smoke.resolve_region(None) == "eu-west-1"


def test_cdk_default_region_when_nothing_else(smoke, monkeypatch):
    monkeypatch.setenv("CDK_DEFAULT_REGION", "ca-central-1")
    assert smoke.resolve_region(None) == "ca-central-1"


def test_us_east_1_fallback(smoke):
    assert smoke.resolve_region(None) == "us-east-1"


# --- http_post retry (fake opener and clock; no network, no real sleeping) ---

_URL = "https://example.invalid/search"


class _FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status = status
        self._data = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int, msg: str, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(_URL, code, msg, hdrs={}, fp=io.BytesIO(body))


def _forbidden() -> urllib.error.HTTPError:
    return _http_error(403, "Forbidden", b'{"message": "Forbidden"}')


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _scripted_opener(outcomes):
    """Return an opener that yields each outcome in turn (the last one repeats)."""
    calls = {"count": 0}

    def opener(request, timeout):
        index = min(calls["count"], len(outcomes) - 1)
        calls["count"] += 1
        outcome = outcomes[index]
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return opener, calls


def test_http_post_retries_403_then_succeeds(smoke, capsys):
    clock = _FakeClock()
    opener, calls = _scripted_opener([_forbidden, _FakeResponse(200, {"results": []})])
    result = smoke.http_post(_URL, "key", {"q": 1}, opener=opener, sleep=clock.sleep, clock=clock)
    assert result == (200, {"results": []})
    assert calls["count"] == 2
    assert clock.sleeps == [5]
    assert capsys.readouterr().out.count("[wait] API Gateway returned 403") == 1


def test_http_post_403_gives_up_at_time_budget(smoke):
    clock = _FakeClock()
    opener, calls = _scripted_opener([_forbidden])
    with pytest.raises(smoke.ApiKeyForbiddenError) as excinfo:
        smoke.http_post(_URL, "key", {"q": 1}, opener=opener, sleep=clock.sleep, clock=clock)
    message = str(excinfo.value)
    assert "ApiKeyId" in message and "x-api-key" in message
    assert clock.now <= smoke.FORBIDDEN_RETRY_SECONDS
    assert sum(clock.sleeps) == smoke.FORBIDDEN_RETRY_SECONDS
    assert max(clock.sleeps) <= smoke.FORBIDDEN_BACKOFF_MAX_SECONDS
    assert calls["count"] == len(clock.sleeps) + 1


def test_http_post_reports_forbidden_wait(smoke):
    clock = _FakeClock()
    opener, _ = _scripted_opener([_forbidden, _forbidden, _FakeResponse(200, {"results": []})])
    waits: list[float] = []
    smoke.http_post(
        _URL, "key", {"q": 1}, opener=opener, sleep=clock.sleep, clock=clock, on_forbidden_wait=waits.append
    )
    assert waits == [15]  # 5 s + 10 s of backoff before the first non-403 attempt

    waits.clear()
    opener, _ = _scripted_opener([_FakeResponse(200, {"results": []})])
    smoke.http_post(
        _URL, "key", {"q": 1}, opener=opener, sleep=clock.sleep, clock=clock, on_forbidden_wait=waits.append
    )
    assert waits == []  # no 403s, nothing reported


# --- poll_search: 403 stalls must not shorten the outer polling window ---

_FORBIDDEN_UNTIL = 100.0  # the API key returns 403 until t=100 s, then works


def _ingestion_opener(clock, indexed_at: float | None):
    """403 before _FORBIDDEN_UNTIL, then empty results until ``indexed_at`` (None = never)."""
    calls: list[tuple[float, int]] = []

    def opener(request, timeout):
        if clock.now < _FORBIDDEN_UNTIL:
            calls.append((clock.now, 403))
            raise _forbidden()
        indexed = indexed_at is not None and clock.now >= indexed_at
        calls.append((clock.now, 200))
        return _FakeResponse(200, {"results": [{"text": "NimbusCache"}] if indexed else []})

    return opener, calls


def _indexed(status, body):
    return status == 200 and bool(body.get("results"))


def test_poll_403_stall_keeps_full_ingestion_budget(smoke):
    clock = _FakeClock()
    opener, calls = _ingestion_opener(clock, indexed_at=None)
    status, body = smoke.poll_search(
        _URL, "key", {"q": 1}, _indexed, opener=opener, sleep=clock.sleep, clock=clock
    )
    assert (status, body) == (200, {"results": []})
    first_ok = next(t for t, code in calls if code == 200)
    assert first_ok >= _FORBIDDEN_UNTIL  # the stall happened inside the poll
    # After the key starts working, polling still runs for the whole budget.
    assert clock.now - first_ok >= smoke.POLL_BUDGET_SECONDS
    last_ok = max(t for t, code in calls if code == 200)
    assert last_ok - first_ok >= smoke.POLL_BUDGET_SECONDS - smoke.POLL_INTERVAL_SECONDS


def test_poll_finds_document_indexed_late_in_budget_after_403_stall(smoke):
    clock = _FakeClock()
    # Indexed 110 s after the key starts working: inside the 120 s budget, but
    # past the original 120 s deadline once the ~105 s 403 stall is counted.
    indexed_at = 105.0 + 110.0
    opener, calls = _ingestion_opener(clock, indexed_at=indexed_at)
    status, body = smoke.poll_search(
        _URL, "key", {"q": 1}, _indexed, opener=opener, sleep=clock.sleep, clock=clock
    )
    assert _indexed(status, body)
    assert calls[-1][0] >= indexed_at


def test_poll_without_403_keeps_original_window(smoke):
    clock = _FakeClock()
    opener, calls = _scripted_opener([_FakeResponse(200, {"results": []})])
    smoke.poll_search(_URL, "key", {"q": 1}, _indexed, opener=opener, sleep=clock.sleep, clock=clock)
    assert clock.now == pytest.approx(smoke.POLL_BUDGET_SECONDS)
    assert set(clock.sleeps) == {smoke.POLL_INTERVAL_SECONDS}


def test_poll_never_clearing_403_raises_forbidden_error(smoke):
    clock = _FakeClock()
    opener, _ = _scripted_opener([_forbidden])  # every request returns 403
    with pytest.raises(smoke.ApiKeyForbiddenError):
        smoke.poll_search(_URL, "key", {"q": 1}, _indexed, opener=opener, sleep=clock.sleep, clock=clock)
    # http_post gave up after its own 403 budget, and poll_search let the error
    # through instead of returning the last 403 body.
    assert clock.now == pytest.approx(smoke.FORBIDDEN_RETRY_SECONDS)


def _stall_opener(clock, stall_start: float, stall_end: float):
    """Empty 200 results, except 403 from ``stall_start`` until ``stall_end``."""
    calls: list[tuple[float, int]] = []

    def opener(request, timeout):
        if stall_start <= clock.now < stall_end:
            calls.append((clock.now, 403))
            raise _forbidden()
        calls.append((clock.now, 200))
        return _FakeResponse(200, {"results": []})

    return opener, calls


def test_poll_403_stall_on_later_iteration_keeps_full_budget(smoke):
    clock = _FakeClock()
    # The key works on the first polls, then returns 403 from t=30 s to t=130 s.
    opener, calls = _stall_opener(clock, stall_start=30.0, stall_end=130.0)
    status, body = smoke.poll_search(
        _URL, "key", {"q": 1}, _indexed, opener=opener, sleep=clock.sleep, clock=clock
    )
    assert (status, body) == (200, {"results": []})
    assert calls[0] == (0.0, 200)  # the stall did not start on the first poll
    first_403 = next(t for t, code in calls if code == 403)
    resumed = next(t for t, code in calls if code == 200 and t > first_403)
    stall = resumed - first_403
    assert stall > smoke.POLL_BUDGET_SECONDS - first_403  # it outlasted the original window
    # Polling time outside the stall still adds up to the whole budget: the
    # deadline is extended by the stall, not cut short and not reset.
    assert smoke.POLL_BUDGET_SECONDS <= clock.now - stall < smoke.POLL_BUDGET_SECONDS + smoke.POLL_INTERVAL_SECONDS


def test_http_post_does_not_retry_400(smoke):
    clock = _FakeClock()
    opener, calls = _scripted_opener([lambda: _http_error(400, "Bad Request", b'{"error": "bad"}')])
    result = smoke.http_post(_URL, "key", {"q": 1}, opener=opener, sleep=clock.sleep, clock=clock)
    assert result == (400, {"error": "bad"})
    assert calls["count"] == 1
    assert clock.sleeps == []
