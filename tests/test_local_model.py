"""The local model path, end to end against a stand-in Ollama server.

This file exists because of one specific complaint: cheap turns were costing
money that they should not have. Every other test here mocks the provider, which
proves the wiring but not the thing that actually matters — that a request marked
cheap reaches Ollama, comes back, and never touches a paid API.

So this starts a real HTTP server speaking Ollama's ``/api/tags`` and
``/api/chat`` on a loopback port, points itsbob at it, and asserts on where the
traffic went. No network, no Ollama install, no key.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from itsbob.agent.brain import TieredBrain
from itsbob.llm.base import LLMRequest, LLMResponse, Usage, user
from itsbob.llm.local import OllamaProvider, default_ollama_config, is_ollama_running
from itsbob.router.tiers import Tier


class _Handler(BaseHTTPRequestHandler):
    """Ollama's two endpoints, plus a record of what was asked."""

    def log_message(self, *args):  # noqa: A003 - silence the test output
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's name
        if self.path.startswith("/api/tags"):
            self._json({"models": [{"name": "qwen2.5:1.5b"}]})
        else:
            self._json({}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        self.server.seen.append(request)
        if self.server.fail_with is not None:
            self._json({"error": "boom"}, self.server.fail_with)
            return
        if self.server.delay:
            time.sleep(self.server.delay)
        reply = self.server.reply
        self._json(
            {
                "model": request.get("model"),
                "message": {"role": "assistant", "content": reply},
                "done": True,
                "prompt_eval_count": 11,
                "eval_count": 5,
            }
        )


@pytest.fixture
def ollama(monkeypatch):
    """A stand-in Ollama on a loopback port, with itsbob pointed at it."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.seen = []
    server.reply = "ready"
    server.delay = 0.0
    server.fail_with = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("ITSBOB_OLLAMA_URL", url)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class NeverCalled:
    """A tier router that fails the test if anything reaches it."""

    def __init__(self):
        self.calls = 0

    def complete(self, request, *, purpose=""):
        self.calls += 1
        return LLMResponse(
            text='{"final": "from the cloud"}', model="paid-model", provider="google",
            usage=Usage(1000, 1000),
        )

    def describe(self):
        return [{"provider": "google", "configured": True, "models": ["paid-model"]}]


def _brain(**kwargs):
    cloud = NeverCalled()
    return TieredBrain({Tier.A: cloud, Tier.S: cloud}, local=OllamaProvider(**kwargs)), cloud


def test_the_liveness_probe_finds_a_running_server(ollama):
    assert is_ollama_running() is True
    from itsbob.llm.local import list_ollama_models

    assert list_ollama_models() == ["qwen2.5:1.5b"]


def test_a_cheap_turn_is_answered_locally_and_never_reaches_a_paid_model(ollama):
    brain, cloud = _brain()
    result = brain.complete(Tier.C, LLMRequest(messages=[user("hello")]))
    assert result.response.provider == "ollama"
    assert result.response.text == "ready"
    assert cloud.calls == 0
    assert brain.local_answers == 1 and brain.local_share == 1.0
    # And it is recorded as usage, so the ledger reflects where work went.
    assert any(r.provider == "ollama" for r in brain.tracker.records)


def test_bookkeeping_runs_locally_even_at_an_expensive_tier(ollama):
    """Classification, extraction and the feasibility check are marked local_ok."""
    brain, cloud = _brain()
    request = LLMRequest(messages=[user("classify this")], metadata={"local_ok": True})
    assert brain.complete(Tier.S, request).response.provider == "ollama"
    assert cloud.calls == 0


def test_expensive_work_still_goes_to_the_cloud(ollama):
    """The local model is preferred for cheap work, not substituted for judgement."""
    brain, cloud = _brain()
    result = brain.complete(Tier.S, LLMRequest(messages=[user("hard question")]))
    assert result.response.provider == "google"
    assert cloud.calls == 1
    assert brain.local_calls == 0


def test_the_gatekeeper_classifies_on_the_local_model(ollama):
    from itsbob.router.gatekeeper import Gatekeeper
    from itsbob.router.ingestion import compress

    ollama.reply = json.dumps({"tag": "TRIVIAL", "fingerprint": "say hello briefly"})
    decision = Gatekeeper(local_provider=OllamaProvider()).classify(compress("hello there"))
    assert decision.source == "local"
    assert decision.tier is Tier.C
    assert ollama.seen[-1]["format"] == "json"


def test_the_gatekeeper_keeps_its_tight_budget_while_answers_get_a_long_one(ollama):
    """One provider serves both, because they need very different timeouts."""
    from itsbob.router.gatekeeper import Gatekeeper
    from itsbob.router.ingestion import compress

    # The provider's own ceiling is generous — a 1.5B model answering from a
    # cold cache genuinely takes ten to twenty seconds.
    assert default_ollama_config({}).timeout >= 30.0

    # But classification, which is on the critical path of every turn, asks for
    # a much tighter one.
    gate = Gatekeeper(local_provider=OllamaProvider())
    assert gate._request(compress("hello")).metadata["timeout"] <= 10.0

    # And the per-request budget is honoured, not ignored: a reply slower than
    # its ceiling fails so the caller can fall back instead of waiting.
    from itsbob.llm.base import ProviderUnavailable

    ollama.delay = 1.0
    with pytest.raises((ProviderUnavailable, TimeoutError, OSError)):
        OllamaProvider().complete(
            LLMRequest(messages=[user("x")], metadata={"timeout": 0.5}),
            model="qwen2.5:1.5b",
        )


def test_a_dead_local_model_falls_through_to_the_cloud_rather_than_failing(ollama):
    brain, cloud = _brain()
    ollama.fail_with = 500
    result = brain.complete(Tier.C, LLMRequest(messages=[user("hello")]))
    assert result.response.provider == "google"
    assert brain.local_failures >= 1
    assert brain.last_local_error and "ollama" in brain.last_local_error.lower()
    assert cloud.calls == 1


def test_an_empty_local_reply_is_a_failure_not_an_answer(ollama):
    """A blank answer looks like success and is worse than an error."""
    brain, cloud = _brain()
    ollama.reply = "   "
    result = brain.complete(Tier.C, LLMRequest(messages=[user("hello")]))
    assert result.response.provider == "google"
    assert brain.local_answers == 0 and brain.local_failures == 1


def test_a_model_that_is_not_pulled_says_how_to_pull_it(ollama):
    from itsbob.llm.base import ProviderUnavailable

    ollama.fail_with = 404
    with pytest.raises(ProviderUnavailable, match="ollama pull"):
        OllamaProvider().complete(LLMRequest(messages=[user("x")]), model="missing:1b")


def test_doctor_makes_a_real_call_rather_than_trusting_the_probe(ollama):
    """'Reachable' and 'answering' are different claims; only one saves money."""
    from itsbob.cli import _probe_local

    ok, detail = _probe_local(default_ollama_config())
    assert ok and "ready" in detail

    ollama.reply = ""
    ok, detail = _probe_local(default_ollama_config())
    assert not ok and "empty" in detail


def test_a_short_back_and_forth_costs_nothing(ollama):
    """The case the whole local path exists for: quick chat, no bill."""
    brain, cloud = _brain()
    for message in ("hi", "thanks", "what's the time", "ok", "cheers"):
        brain.complete(Tier.C, LLMRequest(messages=[user(message)]))
    assert cloud.calls == 0
    assert brain.local_answers == 5
    assert len(ollama.seen) == 5
