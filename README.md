# itsbob

A tick-based character simulation where the character can call LLMs — and has to
pay for it.

Bob has two memory banks, a finite pool of energy, and access to several free
LLM providers. Every tick he decides what to do. Deciding *well* means thinking
with a model, and thinking with a model costs energy priced on the tokens the
call actually burned. Deciding *cheaply* means falling back on instinct. That
trade-off, made every tick under a real budget, is the whole game.

This is the framework, not the game: the loop runs, everything is wired
together, and the interesting knobs are exposed and documented.

```
                    ┌─────────────────────────────────────────┐
   world ──perceive─▶│              MemoryBank                 │
                    │  short term (bounded, decaying)         │
                    │        │ consolidate ▼                  │
                    │  long term (SQLite, importance-ranked)  │
                    └────────┬────────────────────────────────┘
                             │ recall
                             ▼
   needs ──▶  DecisionPolicy ──▶ Decision ──▶ Action ──▶ ActionResult
   traits         │  instinct (free)                        │
                  │  deliberation ──▶ LLMRouter ──▶ tokens ──┤
                  └───────────────────────────────┐         │
                                                  ▼         ▼
                                            EnergyLedger ◀───┘
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The only runtime dependency is `openai` — all three providers speak the OpenAI
wire format, so no per-vendor SDK is needed. LangChain is available as an
optional extra (`pip install -e ".[langchain]"`) if you want to wrap the router
in a chain, but nothing here requires it.

## Keys

```bash
cp .env.example .env   # then fill in whichever keys you have
```

`.env` is gitignored. Every provider is optional: with no keys at all the
simulation runs on a deterministic offline provider and still exercises every
real code path, so nothing about the framework requires a network.

```bash
itsbob doctor --probe        # which providers actually answer right now
```

## Run

```bash
itsbob run --ticks 25 --seed 7            # narrate a run
itsbob run --ticks 25 --offline           # no network, deterministic
itsbob run --ticks 50 --db runs/bob.db    # persist long-term memory
itsbob run --ticks 25 --json              # machine-readable
itsbob ask "what is a memory bank?"       # one-shot through the router
itsbob memory runs/bob.db --query oracle  # inspect what stuck
```

```
t  1  [  instinct   ] consult_oracle  energy 100.0→ 96.8 | Bob consults the oracle — Systematically inspect and catalog…
t  2  [  instinct   ] eat             energy  96.8→100.0 | Bob eats, without ceremony.
t  3  [deliberation ] observe         energy 100.0→ 90.8 | Bob looks around: there are footprints that do not match anyone here
t  4  [  instinct   ] rest            energy  90.8→100.0 | Bob sits down and does nothing on purpose.
t  5  [  fallback   ] socialize       energy 100.0→ 90.5 | Bob talks with the quiet neighbour.
```

In code:

```python
from itsbob import build_simulation

sim = build_simulation(seed=7, policy="hybrid")
for report in sim.stream(20):
    print(report.line())
sim.finish()
print(sim.summary())
```

## How it works

### The tick

`Simulation.step()` runs one cycle, and every other part of the framework exists
to serve it:

1. **world advances** — clock, phase, weather drift
2. **perceive** — an observation is written to memory
3. **recall** — memory is queried using the most pressing need as the query
4. **decide** — a policy picks one available action
5. **act** — the action runs and reports consequences
6. **pay & record** — energy is debited, needs and mood shift, memories are written
7. **consolidate** — the working set decays; what survives moves to long-term
8. **regenerate** — a trickle of energy comes back

### Energy

One currency, `EnergyLedger`, with a full audit trail (`spent_by_reason()`
breaks a run down by where it went). Actions have fixed prices; LLM calls do
not. `TokenCostModel` bills a call *after* it returns, from real usage, with
completion tokens weighted double — so a rambling answer genuinely costs more
than a terse one.

Two consequences worth knowing:

- `affordable_max_tokens()` shrinks the output budget to what the ledger can
  cover, so a tired character asks a shorter question rather than being locked
  out of thinking entirely.
- Below `exhaustion_threshold`, deliberation is off the table and recovery
  actions get a large scoring bonus. Exhaustion changes what Bob *can* consider,
  not just what he prefers.

### Memory

Two stores behind one `MemoryBank` facade.

**Short term** is a bounded deque with per-tick salience decay. Capacity is the
point: every arrival forces a keep-or-drop decision. A record leaves by being
pushed off the end or by fading below the salience floor — either way it is
handed to consolidation, never silently dropped.

**Long term** is SQLite. Retrieval blends three signals — lexical relevance,
importance, and recency decay — the generative-agents recipe. SQL narrows the
candidates, Python scores them, which means swapping the bag-of-words scorer for
a real embedder is a `relevance_fn` argument and nothing else.

**Consolidation** is selective, and that is the interesting part. A memory is
promoted if it was important when formed, *or* if it was recalled more than once
(rehearsal is evidence of usefulness), *or* if it is a reflection or a fact. The
rest is forgotten on purpose.

**Reflection** periodically asks an LLM to distill recent memories into durable
insights, stored as high-importance records that then shape later recalls. It
degrades to a no-op without a router — reflection is a luxury, never a
dependency.

### Decisions

Three policies, differing only in what deliberation costs:

| policy | cost | behavior |
|---|---|---|
| `heuristic` | free | scores actions by need relief ÷ energy price, nudged by traits |
| `llm` | `deliberation_cost` + tokens | asks a model, validates the reply against the real option list |
| `hybrid` | *sometimes* | deliberates only when pressure is high and energy allows |

Traits amplify needs rather than acting as flat bonuses — a curious character
reaches for the oracle *when curious*, not unconditionally. Without this, one
trait quietly dominates every choice.

The LLM policy never trusts the model: a hallucinated action name, malformed
JSON, or a dead provider all fall back to instinct with the reason recorded in
the decision's rationale. A tick cannot fail because a model misbehaved.

### The LLM layer

`LLMRouter` sits over any number of providers and is useful on its own,
independent of the simulation.

- **Failover** in priority, round-robin, random, or least-used order.
- **Error semantics that match reality.** A bad model id (404) is a *model*
  problem, so the router tries that provider's next model. A 5xx or timeout is a
  *provider* problem, so it moves on — retrying a second model on the same dead
  host is wasted time. A 429 is either, depending on the vendor:
  `rate_limit_scope` marks Gemini as per-model (siblings still have budget) and
  Groq/OpenRouter as per-account.
- **Local rate budget** per provider, so free quota is spent deliberately rather
  than discovered through 429s.
- **Circuit breaker** with a half-open probe after cooldown.
- **Usage tracking** of every attempt, success or failure — this is what the
  energy economy bills against.

Free model ids churn constantly. Everything in `llm/catalog.py` is a default,
overridable per provider by env var, with fallbacks the router walks
automatically. `discover_openrouter_free_models()` asks OpenRouter what is
actually free today.

## Extending it

Each seam is a constructor argument, so nothing requires forking the loop.

```python
from itsbob import ActionResult, build_simulation, default_registry
from itsbob.memory import MemoryKind

registry = default_registry()

@registry.add("forage", "Search the yard for food and see what else turns up.",
              energy_cost=4, satisfies={"sustenance": 0.5, "curiosity": 0.35})
def forage(ctx):
    found = ctx.rng.random() < 0.5
    return ActionResult(
        narrative="Bob turns up something edible." if found else "Bob finds nothing.",
        needs_delta={"sustenance": -0.5 if found else 0.0, "curiosity": -0.2},
    ).remember(f"Foraged the yard: {'found food' if found else 'nothing'}",
               MemoryKind.ACTION, 0.4)

sim = build_simulation(registry=registry, policy="heuristic", seed=11)
```

Note the `satisfies` map is what the heuristic policy scores against, so it has
to be honest about what the action relieves — declare too little and the action
never gets picked, too much and it crowds everything else out.

- **New verbs** — register an `Action`; every policy picks it up automatically.
- **New policies** — anything with `decide(ctx) -> Decision` satisfies the protocol.
- **New providers** — add a `ProviderConfig`; unknown names get the generic
  OpenAI-compatible client, so a new vendor is usually config, not code.
- **Real embeddings** — pass `relevance_fn` to either memory store.
- **Observability** — subscribe to the `EventBus` (`decision`, `action`, `tick`,
  `action_error`); a listener that raises cannot take the run down.

## Tests

```bash
pytest
```

87 tests, no network, no API keys, fully deterministic — the offline provider is
a real fallback rather than a mock, so the suite exercises the same paths a live
run takes.

## Known constraints

- **Free tiers fail constantly**, and that is the router's entire reason to
  exist. In live testing Gemini returned both a 503 (model overloaded) and a 429
  (quota) within a handful of ticks; the run continued on fallbacks. Expect
  `doctor --probe` to show a partly-degraded set of providers as the normal
  state.
- **Reasoning models need output headroom.** Gemini's flash models spend part of
  `max_tokens` on hidden thinking and will return empty content with
  `finish_reason: length` if the budget is too tight. The provider surfaces that
  as retryable rather than as an empty answer; keep `max_tokens` ≥ ~256.
- **The economy constants are a starting point, not a balanced game.** They live
  in `EnergySettings` and the action definitions, and are meant to be tuned.
