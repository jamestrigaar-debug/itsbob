# itsbob — Complexity-Based Hierarchical Router

**Classify First, Execute Cheapest, Fallback Gracefully.**

`itsbob` decides, for every incoming piece of state, the *cheapest tier of
intelligence* that can safely handle it — a deterministic script, a small
local model, a cheap cloud API, an expensive one, or (if nothing can parse
the state) a halt-and-ask-the-human alert — and never lets anything but a
pre-registered script name touch actuation.

```
Tier D   Direct Script      no LLM, <50ms        trivial if/else
Tier C   Local Back Brain   local LLM, ~600ms     summarize/paraphrase/format
Tier B   Standard Cloud     cheap API, ~1.2s      tactical, multi-step reasoning
Tier A   Premium Cloud      expensive API         high-stakes, nuanced judgment
Tier S   Critical Fallback  halt, ask the user     nothing else could parse the state
```

A browser **GUI** ships with it — paste a game state, click Route, watch the
tier badge, the Gatekeeper's reasoning, the cache hit/miss, and the scripts
that ran. See [The GUI](#the-gui).

This repository also carries `itsbob`'s original foundation project — a
tick-based character simulation with an energy-metered decision loop — which
the router is layered on top of and can run independently. See
[The character simulation](#the-character-simulation-the-original-foundation).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,gui]"

itsbob doctor                    # what's configured: cloud keys + local Back Brain
itsbob gui                       # opens the browser GUI at http://127.0.0.1:8765
```

Runs with **zero configuration** — no API keys, no local model download
required. Missing pieces degrade gracefully exactly as the spec describes:
no local model → the Gatekeeper's classifier falls back to a fast rule-based
one; no cloud keys → Tier B/A calls fail over to the safe local default; the
whole pipeline is exercised end to end either way.

## The router pipeline

```
raw state ──► Ingestion & Compression (compress, truncate to last 20 events)
                 │
                 ▼
        ┌── Gatekeeper.classify() ──┐
        │  local model, or if       │──► GateDecision{tier, fingerprint, reasoning}
        │  unreachable: heuristic   │
        └────────────────────────────┘
                 │
                 ▼
        semantic cache check (fingerprint-keyed, 5 min TTL)
                 │  hit ──────────────────────────────► replay cached actions
                 ▼  miss
        ┌─────────────────────────────────────────────┐
        │ D → execute named script directly            │
        │ C → local model generation head (paraphrase)  │
        │ B → cheap cloud API, cost-aware prompt         │
        │ A → premium cloud API, cost-aware prompt        │
        │ S → halt, alert the user (needs_user=true)       │
        └─────────────────────────────────────────────┘
                 │
        cloud call times out / returns nothing usable?
                 ▼
        downgrade to local model's safe pick, then to
        MAINTAIN_FORMATION (hardcoded safe default), then Tier S
                 │
                 ▼
        Script name validated against the registry and executed —
        never free-form code, only a name the registry already knows.
```

### Where each piece of the spec lives

| Spec section | Module | What it does |
|---|---|---|
| §1 Local Back Brain (1.5B-3B, Ollama/llama.cpp) | `itsbob/llm/local.py` — `OllamaProvider` | Talks to `ollama serve`'s native `/api/chat`. `is_ollama_running()` is the liveness probe `doctor` and the GUI use. |
| §2 Complexity Classification Taxonomy (S/A/B/C/D) | `itsbob/router/tiers.py` — `Tier`, `GateDecision` | The tier enum and the tag→tier map (`SCRIPT`/`LOCAL_SUM`/`CLOUD_B`/`CLOUD_A`). |
| §3 Ingestion & Compression (truncate to last 20 events) | `itsbob/router/ingestion.py` — `compress()`, `GameState` | Accepts raw dict/JSON, keeps only the most recent `event_window` events. |
| §3 The Gatekeeper classifier prompt | `itsbob/router/gatekeeper.py` — `Gatekeeper` | The exact system prompt from the spec, JSON tag + fingerprint out. Falls back to a rule-based classifier (reasoning-depth / data-volume / action-risk heuristics) when no local model answers. |
| §3 Execution Handshake (SCRIPT→run, LOCAL_SUM→generate, CLOUD_B/A→API) | `itsbob/router/pipeline.py` — `ComplexityRouter._dispatch()` | Routes a `GateDecision` to the matching tier handler. |
| §4 Cost-aware system prompt ("respond in 30 words, strict JSON array") | `itsbob/router/pipeline.py` — `CLOUD_SYSTEM_PREFIX` | Prepended to every Tier B/A call. |
| §4 Semantic caching (fingerprint hash, ~5 min) | `itsbob/router/cache.py` — `SemanticCache` | TTL cache keyed on a normalized fingerprint hash; `itsbob route` and the GUI show hit/miss + hit rate. |
| §4 Async batch processing | *not implemented* | The pipeline is synchronous/per-call today; batching non-urgent Tier B work is listed under [Not yet built](#not-yet-built). |
| §5 Phase 0 — hard-coded classifier, no free text | `Gatekeeper.classify()` | Always asks for a tag + fingerprint only (`json_mode=True`, 60 max tokens); never lets the local model narrate. |
| §5 Phase 1 — Cold Cloud Router, 3 tactical scripts, <1.8s budget | `itsbob/router/pipeline.py` — `END_TO_END_LATENCY_BUDGET_MS`, `RouteResult.within_budget` | Every route reports total latency and whether it stayed under budget. |
| §5 Phase 2 — Timeout monitor → downgrade to local → Tier S | `ComplexityRouter._escalate_to_local()`, `_tier_s()` | A Tier B/A call that fails, times out, or names no known script downgrades to the local model's safe pick, then to `MAINTAIN_FORMATION`, then to Tier S. |
| §5 Phase 3 — predictive pre-fetching (background loop, pre-cache) | *not implemented* | Listed under [Not yet built](#not-yet-built) — the `SemanticCache` it would pre-populate already exists. |
| §6 Golden Rule (cloud/local only ever *name* an action) | `itsbob/router/scripts.py` — `ScriptRegistry` | `ComplexityRouter` only ever calls `registry.execute(name, ...)` with a name the model returned; unregistered names are dropped before execution, never run. |

## Setup

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui]"      # drop ",gui" if you only want the CLI/library
```

### The local Back Brain (Tier C) — optional but recommended

```bash
# install Ollama: https://ollama.com/download
ollama serve                      # runs on 127.0.0.1:11434
ollama pull qwen2.5:1.5b          # ~1GB, the default model this repo asks for
# or: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

`itsbob doctor` and the GUI's status strip both report whether Ollama is
reachable. If it isn't (not installed, not running, or a different port),
**nothing breaks** — `Gatekeeper` transparently falls back to a rule-based
classifier so Tier C/B/A routing still works, just less accurately than a
tuned local model would. Pin a different model or point at Ollama on another host/port:

```bash
export ITSBOB_OLLAMA_MODEL=phi3.5:3.8b-mini-instruct-q4_K_M
export ITSBOB_OLLAMA_URL=http://127.0.0.1:11434    # default; change if Ollama runs elsewhere
```

### Cloud providers (Tier B / Tier A) — optional

All three speak the OpenAI-compatible `/chat/completions` format:

```bash
export GROQ_API_KEY=...          # https://console.groq.com — fast, generous free tier
export GOOGLE_API_KEY=...        # https://aistudio.google.com — Gemini Flash
export OPENROUTER_API_KEY=...    # https://openrouter.ai — dozens of `:free` models
```

**Using Google?** Getting a key, picking a specific Gemini model (Flash vs.
Pro, pinning a model id, finding current model names), rate limits, and
Google-specific troubleshooting are all in
**[docs/GOOGLE_SETUP.md](docs/GOOGLE_SETUP.md)** — the two-line env var above
is enough to get going, that doc is for everything past that.

Or copy `.env.example` to `.env` and fill it in — every CLI command and the
GUI load it automatically:

```bash
cp .env.example .env
```

### Google-tiered mode — Google answers every cloud tier itself

By default (`mode="priority"`) Tier B and Tier A share the same pool of
cloud providers, tried in `ITSBOB_PROVIDER_ORDER`. If you'd rather **Google
alone** answer both cloud tiers — at two different Gemini models, cheap for
Tier B and stronger for Tier A — with Groq/OpenRouter only as a backup that's
tried *after* Google rather than before it, use **google-tiered** mode:

```bash
export ITSBOB_ROUTER_MODE=google-tiered
itsbob doctor            # now shows Tier B and Tier A as two separate Gemini chains
itsbob route '{"facts": {"stamina": 15}}'
itsbob gui                # the status strip shows the same split
```

or per-call, without changing the env:

```bash
itsbob route '{"facts": {"stamina": 15}}' --mode google-tiered
```

The ladder, in cheapest-to-strongest order:

| Tier | Default model | Falls back to |
|---|---|---|
| B (standard cloud) | `gemini-3.1-flash-lite` | `gemini-3.5-flash-lite` → `gemini-3.6-flash` |
| A (premium cloud) | `gemini-3.6-flash` | `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` |

Only `GOOGLE_API_KEY` is required. If Groq/OpenRouter keys are *also* set,
they're appended behind Google on both tiers as low-tier backups — reached
only once every Gemini model tried has failed or hit quota, never tried
first. Set them or leave them unset independently; nothing here requires
them. To go fully Google-only with no Groq/OpenRouter involved at all:

```python
from itsbob.router import build_google_tiered_router
from itsbob.config import Settings

router = build_google_tiered_router(Settings.from_env(), low_tier_backup=False)
```

Pin different Gemini model ids (if your account's model access differs)
directly on the call:

```python
router = build_google_tiered_router(
    Settings.from_env(),
    tier_b_model="gemini-3.1-flash-lite",
    tier_a_model="gemini-3.6-flash",
    fallback_model="gemini-3.5-flash-lite",
)
```

See [docs/GOOGLE_SETUP.md](docs/GOOGLE_SETUP.md) for getting the key itself
and finding which Gemini model ids your account currently has access to
(the exact names above may not match what's live for you).

### Wiring in a different Tier A provider entirely

`google-tiered` mode is the built-in way to split Tier B/A between two
models. If you want Tier A answered by a non-Google, non-free model
(GPT-4o-class, say) instead, build `ComplexityRouter` directly:

```python
from itsbob.router import ComplexityRouter, Gatekeeper, default_registry, SemanticCache
from itsbob.factory import build_router
from itsbob.config import Settings, ProviderConfig

settings = Settings.from_env()
cheap = build_router(settings)                       # Groq / Gemini Flash / OpenRouter free
expensive = build_router(replace_with_gpt4o_settings)  # your own Settings pointing at an
                                                         # OpenAI-compatible GPT-4o-class endpoint
router = ComplexityRouter(
    registry=default_registry(),
    gatekeeper=Gatekeeper(registry=default_registry()),
    cloud_router=cheap,
    premium_router=expensive,
    cache=SemanticCache(),
)
```

### Optional tuning env vars

| Variable | Effect |
|---|---|
| `ITSBOB_ROUTER_MODE` | `priority` (default) or `google-tiered` — see [Google-tiered mode](#google-tiered-mode--google-answers-every-cloud-tier-itself) |
| `ITSBOB_PROVIDER_ORDER` | comma-separated cloud try-order, e.g. `groq,google,openrouter` (ignored in `google-tiered` mode) |
| `ITSBOB_GROQ_MODEL` / `ITSBOB_GOOGLE_MODEL` / `ITSBOB_OPENROUTER_MODEL` | pin a cloud model id (old default becomes a fallback, never dropped) |
| `ITSBOB_MAX_ATTEMPTS` | total cloud provider attempts per call (default 4) |
| `ITSBOB_ALLOW_OFFLINE` | set `false` to make missing cloud keys a hard error instead of the offline `EchoProvider` |
| `ITSBOB_OLLAMA_MODEL` | pin the local Back Brain's model id (default `qwen2.5:1.5b`; old default becomes a fallback) |
| `ITSBOB_OLLAMA_URL` | Ollama server URL (default `http://127.0.0.1:11434`) |
| `ITSBOB_MEMORY_DB` | SQLite path for the character simulation's long-term memory |
| `ITSBOB_SEED` | RNG seed, for reproducible simulation runs |

## Launching it

```bash
# What's configured — cloud keys AND the local Back Brain
itsbob doctor
itsbob doctor --probe                          # + one real request per cloud provider

# Classify only — see the tier + fingerprint, nothing executed
itsbob classify '{"facts": {"stamina": 15, "minute": 60}}'

# Full pipeline: classify, cache-check, route, execute
itsbob route '{"facts": {"stamina": 15, "minute": 60}}'
itsbob route @path/to/state.json --goal "win the league"
itsbob route '{"facts": {"stamina": 15}}' --mode google-tiered   # Google alone, two Gemini models — see below

# The GUI
itsbob gui                                     # http://127.0.0.1:8765, opens a browser tab
itsbob gui --port 9000 --no-browser
```

## The GUI

```bash
pip install -e ".[gui]"     # one extra dependency: Flask
itsbob gui                  # opens http://127.0.0.1:8765 in your browser automatically
```

Two panels, side by side:

**Left — chat.** Type a message and hit Send (or Enter). Paste JSON
(`{"facts": {"stamina": 15, "minute": 60}}`) if you want to test a specific
state, or just type plain text — anything that isn't valid JSON is
automatically wrapped into `{"facts": {"message": "<your text>"}}` so you
can talk to it conversationally without hand-writing a state object every
time. Click one of the `example N` links under the input to load a ready
state. itsbob's reply appears as a chat bubble; a Tier S response (nothing
could parse the state) renders as a highlighted red bubble — "Unrecognized
state. Manual override required." — exactly where the spec says the system
should halt and ask you. Check **classify only** above the input to see how
something would be tagged without executing anything.

**Right — live processing.** Every message you send appends a trace card
here, newest on top, so you can *watch it think* rather than read logs:

- the tier badge (color-coded D/C/B/A/S) and, if it downgraded, what tier it
  escalated from
- the Gatekeeper's reasoning, its fingerprint, and its own latency
- **which model actually got called** — provider + model name, or "—
  (script/local/none)" when nothing needed to be
- whether it was served from the semantic cache, and the running cache hit
  rate
- which scripts executed
- total latency against the 1.8s budget, flagged if it went over
- a collapsed **raw JSON** section per card with everything `itsbob route`
  itself would print, for anyone who wants the full detail

The **status strip** at the top shows the active router mode
(`priority`/`google-tiered`), the local Back Brain's reachability, and every
configured cloud provider — split into Tier B/Tier A when in google-tiered
mode — refreshed on load.

It binds to `127.0.0.1` only (not exposed to your network) and has no
authentication — it's a local development tool, not a deployed service.
Nothing about it requires the character simulation to be running. The old
single-shot `/api/route` and `/api/classify` endpoints are still there
unchanged, if you're scripting against the GUI's backend directly (`curl`,
a notebook, etc.) rather than using the page.

## Using the router as a library

```python
from itsbob.router import build_complexity_router
from itsbob.config import Settings

router = build_complexity_router(Settings.from_env(), goal="win the league")

result = router.route({
    "facts": {"score": "1-0", "minute": 78, "opponent_formation": "4-4-2", "morale": "low"},
    "events": ["68' Yellow card", "74' Corner won"],
})
print(result.tier, result.actions, result.note)
```

`ComplexityRouter`, `Gatekeeper`, `ScriptRegistry`, and `SemanticCache` are
each independently constructible (see `itsbob/router/pipeline.py`) if you
want to swap in your own script macros, a different local model, or a
custom cache backend.

### Registering your own scripts (the actuation layer)

```python
from itsbob.router import ScriptRegistry, ScriptResult

registry = ScriptRegistry()
registry.register(
    "HIGH_PRESS",
    lambda state, params: ScriptResult(ok=True, action="HIGH_PRESS", detail="pressed high"),
    description="Aggressive high-press tactical macro.",
    trigger=None,  # or a callable(GameState) -> bool for a Tier-D auto-trigger
)
```

Only names registered here can ever execute — see the Golden Rule row in
the spec-mapping table above.

## Not yet built

Per the spec's own phasing, these are real gaps, not oversights:

- **Phase 0's LoRA fine-tuning** — training a tiny LoRA on historical
  game logs to make the local classifier hyper-accurate. The heuristic
  fallback and the raw Ollama classifier prompt exist; the training loop
  doesn't.
- **§4 Asynchronous batch processing** — buffering non-urgent Tier B work
  and firing it every 10 minutes in one batch call. `ComplexityRouter.route()`
  is synchronous, one state in, one result out.
- **Phase 3 predictive pre-fetching** — a continuous background loop
  simulating the next few minutes and pre-warming the semantic cache before
  a high-risk event happens. `SemanticCache` is ready to be pre-populated;
  nothing populates it proactively yet.
- **A real screen-scraper** — nothing in this repo reads a game window. You
  provide `raw_state` as JSON (from wherever your scraper lives); `compress()`
  is the ingestion boundary it should feed into.
- **Tier A on a genuinely separate, non-Google premium provider** (a real
  GPT-4o-class endpoint, say) — `google-tiered` mode covers Tier A on a
  stronger *Gemini* model out of the box; a different vendor entirely still
  needs the manual `ComplexityRouter(...)` wiring in
  [Wiring in a different Tier A provider entirely](#wiring-in-a-different-tier-a-provider-entirely).

## Troubleshooting

**A `--probe` run with some `!!` lines is normal, not broken.** `itsbob
doctor --probe` tries every model on every configured provider and reports
each attempt — a `!!` next to a specific model just means *that model* is
gone or blocked; the router already walks past it to the next one
automatically during a real call. What matters is that **at least one
provider shows `ok`** on at least one model. If Groq or Google (or Ollama)
answers, the router works end to end — you do not need OpenRouter, or every
model listed, to be green.

| Symptom | Cause | Fix |
|---|---|---|
| `itsbob: command not found` | venv not active, or install didn't register the entry point | `source .venv/bin/activate`; re-run `pip install -e ".[dev,gui]"` |
| Every OpenRouter model — including one you just pinned yourself — 404s with `"This model is unavailable"` or `"No endpoints available matching your guardrail restrictions and data policy"` | **Not a code or key problem.** OpenRouter's own account-level Privacy setting blocks all `:free` models from routing until you opt in. | Go to [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy) and enable the free-model / prompt-training data policy toggle (OpenRouter requires this before any `:free` model will serve a request to a new key). Also clear any "Allowed Providers" / "Ignored Providers" restriction under [openrouter.ai/settings/preferences](https://openrouter.ai/settings/preferences) — an over-restricted provider list produces the same 404. Re-run `itsbob doctor --probe` after saving. |
| `ITSBOB_GROQ_MODEL=llama-3.1-8b-instruct` (or another hand-typed id) still 404s | a typo, or that exact id doesn't exist on Groq (`instruct` vs. the real `instant`) | `itsbob doctor --probe` first to see which ids actually answer for your key, then pin one of *those* — don't guess a model id from memory |
| `itsbob route '...'` crashes with a `JSONDecodeError` traceback | you passed the literal `'...'` from a README example — it's a placeholder, not real input | pass actual JSON: `itsbob route '{"facts": {"stamina": 15}}'`. Newer versions print a friendly error instead of a traceback here; update if you still see the raw traceback |
| `ollama serve` prints `address already in use` | Ollama is already running (often auto-started as a background service on install) | that's fine, nothing to fix — `itsbob doctor` will still show it reachable; you don't need to run `ollama serve` yourself in that case |
| Flask prints `Tip: install python-dotenv to use them` when you run `itsbob gui` | Flask's own unrelated `.env` auto-loading feature noticing a `.env` file exists | harmless — itsbob loads `.env` itself (`config.load_dotenv`) before Flask ever starts; ignore the tip or `pip install python-dotenv` to silence it |
| Browser console/log shows `GET /favicon.ico 404` | no favicon route | harmless, cosmetic only |
| `itsbob gui` prints an error about Flask | the `gui` extra wasn't installed | `pip install -e ".[gui]"` |
| `itsbob route ...` always returns Tier S | both the local Back Brain and every cloud provider are unavailable | `itsbob doctor`; you need at least one of Ollama running or a cloud key set, or Tier D/C-only states, to avoid the halt |
| `itsbob doctor` shows `ollama -- not reachable` | Ollama isn't installed or `ollama serve` isn't running | `ollama serve` in another terminal, and `ollama pull qwen2.5:1.5b`; re-run `itsbob doctor` |
| Every `classify` result has `"source": "heuristic"` | same as above — the Gatekeeper is using its rule-based fallback, not a model | start Ollama; the fallback is intentional degraded behavior, not a bug |
| Tier B/A results always say `"downgraded to hardcoded safe default"` | no cloud provider configured (falls to the offline echo, which can't produce a valid actions array), or the model named scripts that aren't registered | `itsbob doctor` to check cloud keys; check `ScriptRegistry.names()` matches what your cloud prompt is allowed to say |
| `itsbob doctor` shows only `echo` as `ok` under cloud providers | no API keys in the environment or `.env` | export a key or add it to `.env`; re-run `itsbob doctor` |
| `RateLimited` / lots of `429` in `--probe` output | free-tier quota hit | wait, reorder providers with `ITSBOB_PROVIDER_ORDER`, or add another key |
| `BadRequest` naming a model | the hardcoded default model was retired/renamed by the vendor | pin a live one: `ITSBOB_GROQ_MODEL=...` etc., or call `itsbob.llm.catalog.discover_openrouter_free_models()` |
| GUI's status strip shows a provider as `no key` even though you set it | shell env var set after the GUI process started, or `.env` not in the working directory you launched `itsbob gui` from | restart `itsbob gui` from the repo root after exporting/creating `.env` |
| GUI's Route call is slow / times out | a cloud provider is hanging near its own timeout before failing over | check `itsbob doctor --probe` for which provider is slow; lower `ITSBOB_MAX_ATTEMPTS` so it fails over faster |
| `itsbob run` (the character simulation) — see its own troubleshooting | this is the older, separate subsystem the router is built on top of | see [Troubleshooting the character simulation](#troubleshooting-the-character-simulation) below |

## The character simulation (the original foundation)

Before the router existed, this repo already had a tick-based character
simulation whose "character" decides, every tick, whether a choice is worth
spending scarce energy to think about with an LLM, or cheap enough to decide
on instinct alone. The router (`itsbob/router/`) is layered on top of the
same `itsbob/llm/` stack this simulation uses, but the two run
independently — the simulation doesn't call the router, and the router
doesn't need the simulation running.

```
itsbob run --ticks 20 --policy hybrid          # hybrid is the default
itsbob run --ticks 20 --policy heuristic       # never calls an LLM
itsbob run --ticks 20 --policy llm             # always deliberates when affordable
itsbob run --ticks 20 --offline                # force the EchoProvider, ignore any keys
itsbob run --ticks 20 --json                   # machine-readable output
itsbob run --ticks 20 --db data/memory.sqlite  # persist long-term memory

itsbob ask "What's the fastest way to calm down a frustrated star player?"
itsbob memory data/memory.sqlite --query boats --limit 10
```

Every tick, `Simulation`:

1. advances the `World` and lets `Character.needs` drift upward (rest,
   sustenance, social, curiosity, purpose all creep toward "unmet"),
2. recalls relevant memories from the two-tier `MemoryBank`,
3. asks a `DecisionPolicy` to pick one `Action`:
   - **`HeuristicPolicy`** — free, deterministic-ish scoring by need-relief
     vs. energy price. Always available, always the fallback.
   - **`LLMPolicy`** — spends `deliberation_cost` energy, calls the
     `LLMRouter` for a JSON decision, falls back to the heuristic if
     unaffordable, unavailable, or the model hallucinates an action name.
   - **`HybridPolicy`** — deliberates only when need pressure is high,
     curiosity/energy allow it, or a trait-weighted random roll says so.
4. runs the chosen `Action`, pays its energy cost, updates mood/needs/memory.

### Troubleshooting the character simulation

| Symptom | Cause | Fix |
|---|---|---|
| Every decision comes from `instinct`/`fallback`, never `deliberation` | no cloud provider configured, or energy too low to deliberate | `itsbob doctor`; check `ITSBOB_ENERGY_*` if keys *are* set |
| A run raises `RuntimeError: no LLM providers configured` | no keys **and** `ITSBOB_ALLOW_OFFLINE=false` | unset `ITSBOB_ALLOW_OFFLINE`, pass `--offline` deliberately, or add a key |
| Long-term memory is empty after a run | no `--db` was passed, so it used the in-memory (`:memory:`) SQLite store, gone on exit | re-run with `itsbob run --db data/memory.sqlite ...` |
| `itsbob memory <db>` errors that the file doesn't exist | wrong path, or the earlier run never persisted (see above) | check the path you passed to `--db` on the run that created it |
| `pytest` finds nothing / warns "No files were found in testpaths" | there is no `tests/` directory in this snapshot despite `pyproject.toml` pointing at one | add tests under `tests/` as you build on this foundation |

## Repository map

```
docs/GOOGLE_SETUP.md  getting a key, picking a specific Gemini model, rate limits
pyproject.toml     deps: openai>=1.40 (base); pytest (dev); flask (gui); langchain (optional)
src/itsbob/
  __init__.py       public API (build_complexity_router, ComplexityRouter, build_simulation, ...)
  cli.py            itsbob run | doctor | ask | memory | classify | route | gui
  config.py         Settings, ProviderConfig, MemorySettings, EnergySettings, load_dotenv
  factory.py        build_router / build_character / build_simulation (the cloud LLMRouter + simulation)

  router/           the complexity-tier router (this spec)
    tiers.py           Tier enum (S/A/B/C/D), GateDecision
    ingestion.py       compress() — truncate/normalize raw state
    gatekeeper.py       Gatekeeper — classify(), local model + heuristic fallback
    cache.py            SemanticCache — fingerprint-keyed, TTL
    scripts.py          ScriptRegistry, default_registry() — the Golden Rule's name→macro map
    pipeline.py          ComplexityRouter — dispatch, cost-aware prompts, timeout escalation
    google_tiered.py      build_google_tiered_router() — Google-only, two Gemini models, one per tier

  gui/               browser GUI (needs the `gui` extra)
    app.py             Flask app: /, /api/status, /api/route, /api/classify

  llm/               provider-agnostic LLM access, shared by the router and the simulation
    base.py             Message/LLMRequest/LLMResponse/Provider contract
    local.py             OllamaProvider — the local Back Brain
    providers.py         OpenRouter, Groq, Google (OpenAI-compatible) + EchoProvider
    catalog.py            default model IDs per provider, env overrides
    router.py             LLMRouter — failover, rate limiting, circuit breaker, usage tracking

  character/         the original foundation's simulation pieces
    state.py, energy.py, actions.py, decisions.py
  engine/             World, EventBus, Simulation, TickReport
  memory/             ShortTermMemory, LongTermMemory (SQLite), MemoryBank facade
```
