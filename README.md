# itsbob (foundation build)

A tick-based character simulation whose "character" has to decide, every
tick, whether a choice is worth spending scarce energy to think about with an
LLM, or cheap enough to decide on instinct alone.

**This is not the final "itsbob" complexity-router.** It is a working
foundation — a real energy-metered decision loop with a resilient
multi-provider LLM router already built — that a hierarchical
Classify → Route → Map-to-Script system (Tiers S/A/B/C/D, local
Back-Brain classifier, cheap-cloud workhorse, deterministic script layer)
can be built on top of. See [Relationship to the itsbob router
design](#relationship-to-the-itsbob-router-design) below for how the pieces
already map onto that design.

> **Note on this README:** an earlier draft of this file described a
> different, unbuilt chatbot concept ("1tsb0b", leetspeak, a 426-line tone
> pool, credits/standing economy). None of that exists in `src/`. This
> version documents the code that is actually in this repository.

## What's actually here

```
src/itsbob/
  config.py          env-driven Settings: providers, memory, energy
  factory.py          build_router() / build_character() / build_simulation()
  cli.py               itsbob run | doctor | ask | memory
  character/
    state.py           Character, Traits, Needs
    energy.py           EnergyLedger, TokenCostModel — the metered budget
    actions.py          Action, ActionRegistry — what the character can do
    decisions.py         HeuristicPolicy (free) / LLMPolicy (costs energy) /
                          HybridPolicy (decides which to use, per tick)
  engine/
    world.py, events.py, simulation.py  — the tick loop and its output
  memory/
    short_term.py, long_term.py, bank.py — two-tier recall memory (SQLite-backed)
  llm/
    base.py             Message/LLMRequest/LLMResponse/Provider contract
    providers.py         OpenRouter, Groq, Google (all OpenAI-compatible) + EchoProvider
    catalog.py           default model IDs per provider, env overrides
    router.py            failover, rate limiting, circuit breaker, usage tracking
```

There is no `tests/` directory in this snapshot and no `.env.example` file —
both are called out under [Known gaps](#known-gaps-vs-the-old-readme) so you
don't go looking for them.

## How the loop works

Every tick, `Simulation`:

1. advances the `World` and lets `Character.needs` drift upward (hunger,
   fatigue, curiosity, social, purpose all creep toward "unmet"),
2. recalls relevant memories from the two-tier `MemoryBank`,
3. asks a `DecisionPolicy` to pick one `Action`:
   - **`HeuristicPolicy`** — free, deterministic-ish scoring of each action
     by how much need-pressure it relieves vs. its energy price. Always
     available, always the fallback.
   - **`LLMPolicy`** — spends `deliberation_cost` energy up front, calls the
     `LLMRouter` for a JSON `{"action": ..., "rationale": ..., "confidence":
     ...}` decision, and falls back to the heuristic if the model is
     unaffordable, unavailable, or hallucinates an action name that doesn't
     exist.
   - **`HybridPolicy`** — the interesting one: deliberates only when need
     pressure is high, curiosity/energy allow it, or a random roll driven by
     the `curiosity` trait says so. This is the closest existing analogue to
     the S/A/B/C/D tier gate described in the itsbob router design.
4. runs the chosen `Action`, pays its energy cost, and lets the outcome
   update mood, needs, and memory.

## Setup

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`itsbob` runs with **zero configuration and zero API keys** — with no
provider configured it falls back to the built-in offline `EchoProvider`
(deterministic, hash-derived replies), so the whole tick loop, energy
economy, and JSON decision-parsing path are exercised without spending
anything or touching the network.

### Adding real providers (optional)

All three supported providers speak the OpenAI-compatible `/chat/completions`
wire format, so one HTTP client (`openai` SDK) covers all of them. Set
whichever keys you have; unset ones are simply skipped:

```bash
export GROQ_API_KEY=...          # https://console.groq.com — fast, generous free tier
export GOOGLE_API_KEY=...        # https://aistudio.google.com — Gemini Flash
export OPENROUTER_API_KEY=...    # https://openrouter.ai — dozens of `:free` models
```

There's no `.env.example` in this snapshot; a `.env` file (dotenv-style,
`KEY=value` per line, `#` comments allowed) in the repo root is loaded
automatically by every CLI command — create one yourself if you'd rather not
export shell variables:

```
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...
```

Optional tuning env vars (see `config.py` / `llm/catalog.py` for the full
list):

| Variable | Effect |
|---|---|
| `ITSBOB_PROVIDER_ORDER` | comma-separated try-order, e.g. `groq,google,openrouter` |
| `ITSBOB_GROQ_MODEL` / `ITSBOB_GOOGLE_MODEL` / `ITSBOB_OPENROUTER_MODEL` | pin a model id (old default becomes a fallback, never dropped) |
| `ITSBOB_MAX_ATTEMPTS` | total provider attempts per logical call (default 4) |
| `ITSBOB_ALLOW_OFFLINE` | set `false` to make missing keys a hard error instead of falling back to `EchoProvider` |
| `ITSBOB_MEMORY_DB` | path to a SQLite file for long-term memory (default `:memory:`) |
| `ITSBOB_ENERGY_CAPACITY` / `ITSBOB_ENERGY_START` / `ITSBOB_ENERGY_REGEN` | tune the energy budget |
| `ITSBOB_SEED` | RNG seed, for reproducible runs |

## Launching it

```bash
# Check what's configured, without spending quota
itsbob doctor

# Check what's configured AND send one real request per provider
itsbob doctor --probe

# Run the simulation
itsbob run --ticks 20 --policy hybrid          # hybrid is the default
itsbob run --ticks 20 --policy heuristic       # never calls an LLM
itsbob run --ticks 20 --policy llm             # always deliberates when affordable
itsbob run --ticks 20 --offline                # force the EchoProvider, ignore any keys
itsbob run --ticks 20 --json                   # machine-readable output
itsbob run --ticks 20 --db data/memory.sqlite  # persist long-term memory

# One-shot call through the router (bypasses the simulation entirely)
itsbob ask "What's the fastest way to calm down a frustrated star player?"
itsbob ask "..." --provider groq               # restrict to one provider

# Inspect a saved run's long-term memory
itsbob memory data/memory.sqlite --query boats --limit 10
```

`data/` is not created automatically unless you pass `--db` — the default
memory store is in-process SQLite (`:memory:`) and disappears when the
process exits. Create a `data/` directory (or let `--db data/whatever.sqlite`
create the file) if you want memory to persist between runs.

## Using the pieces standalone

The two heaviest subsystems are independently usable, which is exactly what
the router-design foundation needs:

```python
from itsbob import build_router, Settings
from itsbob.llm.base import LLMRequest, user

router = build_router(Settings.from_env())
response = router.complete(
    LLMRequest(messages=[user("Classify this game state")], max_tokens=200),
    purpose="classification",
)
print(response.provider, response.model, response.text)
```

```python
from itsbob.memory.bank import MemoryBank
from itsbob.config import MemorySettings

bank = MemoryBank(MemorySettings(database="data/memory.sqlite"))
```

## Relationship to the itsbob router design

The complexity-tier router design (S/A/B/C/D, local Back-Brain classifier,
cheap-cloud workhorse, deterministic script mapping) is **not implemented
here** — but the scaffolding it needs already exists and maps cleanly:

| Router design concept | Existing foundation |
|---|---|
| Tier D — deterministic script, no LLM | `HeuristicPolicy` (already free, always available) and `ActionRegistry` (`character/actions.py`) as the pre-hardened "script" map |
| Tier C — local Back-Brain classifier/summarizer | not present; would slot in as a new `Provider` (e.g. an Ollama/llama.cpp-backed one) added to `llm/providers.py` and given priority in `build_router` |
| Tier B/A — cheap/premium cloud APIs | `LLMRouter` + `providers.py` already implement OpenRouter/Groq/Google failover, rate limiting, and a circuit breaker — this *is* the workhorse layer |
| Tier S — critical fallback / user alert | `LLMPolicy._fall_back` degrades to the heuristic today; there's no user-facing halt-and-ask yet — that's new work |
| "Classify first" gate | `HybridPolicy.should_deliberate` is the existing analogue — it already decides, per tick, whether the situation is worth paying to think about |
| Cost-aware prompting, energy budget | `EnergyLedger` / `TokenCostModel` (`character/energy.py`) already meter every LLM call against a budget and refuse deliberation when unaffordable |
| Semantic caching / state fingerprint | not present — would sit in front of `LLMRouter.complete()`, keyed on a hash of the compressed game state |
| "Never let the local LLM write scripts, only name them" | already the shape of `LLMPolicy.decide()`: the model's JSON reply is validated against the real `ActionRegistry` and any hallucinated/unknown action name falls back rather than executing |

Building the full router on this foundation is mostly: (1) add a local
model `Provider`, (2) give it a strict classifier-only prompt and a
tier-tag output contract, (3) route tier tags to `HeuristicPolicy` /
`LLMPolicy` / a new premium-tier policy instead of `HybridPolicy`'s current
probabilistic gate, and (4) add the semantic cache in front of `LLMRouter`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `itsbob: command not found` | venv not active, or install didn't register the entry point | `source .venv/bin/activate`; re-run `pip install -e ".[dev]"` |
| Every decision comes from `instinct`/`fallback`, never `deliberation` | no provider configured, or energy too low to deliberate | `itsbob doctor`; check `ITSBOB_ENERGY_*` if keys *are* set |
| `itsbob doctor` shows only `echo` as `ok` | no API keys in the environment or `.env` | export a key or add it to `.env`; re-run `itsbob doctor` |
| `itsbob doctor --probe` prints `ProviderNotConfigured` for a provider that has a key | the `openai` package isn't installed | `pip install openai` (it's a base dependency of `pyproject.toml`, so a normal `pip install -e .` should have it — check `pip show openai`) |
| `RateLimited` / lots of `429` in probe output | free-tier quota hit | wait, or reorder providers with `ITSBOB_PROVIDER_ORDER`, or add another key |
| `BadRequest` naming a model | the hardcoded default model was retired/renamed by the vendor | pin a live one: `ITSBOB_GROQ_MODEL=...` (etc.), or call `itsbob.llm.catalog.discover_openrouter_free_models()` to list what OpenRouter currently serves free |
| A run raises `RuntimeError: no LLM providers configured` | no keys **and** `ITSBOB_ALLOW_OFFLINE=false` | unset `ITSBOB_ALLOW_OFFLINE`, or pass `--offline` deliberately, or add a key |
| Long-term memory is empty after a run | no `--db` was passed, so it used the in-memory (`:memory:`) SQLite store, gone on exit | re-run with `itsbob run --db data/memory.sqlite ...` |
| `itsbob memory <db>` errors that the file doesn't exist | wrong path, or the earlier run never persisted (see above) | check the path you passed to `--db` on the run that created it |
| `pytest` finds nothing / warns "No files were found in testpaths" | there is no `tests/` directory in this snapshot despite `pyproject.toml` pointing at one | expected for now — add tests under `tests/` as you build on this foundation |

## Known gaps vs. the old README

The previous `README.md` in this repo described a materially different,
unbuilt system (leetspeak chatbot, tone-pool selection, a credits/standing
economy, a `server.py`/`ui.py` web front end, 212 tests). None of that code
exists here. Specifically absent from this snapshot, in case you go looking:

- `tests/` directory (pyproject.toml references it; it isn't present)
- `.env.example`
- `.gitignore` (a `data/` directory you create for `--db` output is **not**
  currently ignored — add one before committing real conversation/memory
  data)
- any HTTP server, web UI, or tunnel support
- the local Back-Brain classifier / complexity-tier router itself (see the
  mapping table above)

## Repository map (accurate)

```
pyproject.toml     deps: openai>=1.40 (base), pytest (dev), langchain (optional extra)
src/itsbob/
  __init__.py       public API surface (build_simulation, LLMRouter, MemoryBank, ...)
  cli.py            itsbob run | doctor | ask | memory
  config.py         Settings, ProviderConfig, MemorySettings, EnergySettings, load_dotenv
  factory.py        build_router / build_character / build_simulation
  character/        Character, Traits, Needs, EnergyLedger, Action(Registry), DecisionPolicy family
  engine/            World, EventBus, Simulation, TickReport
  llm/               Provider contract, OpenRouter/Groq/Google/Echo providers, catalog, failover router
  memory/            ShortTermMemory, LongTermMemory (SQLite), MemoryBank facade
```
