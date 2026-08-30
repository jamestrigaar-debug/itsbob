# Models: the tier ladder, keys, and what to do when an id dies

## The ladder

Each *step* of each turn is classified, then answered by the cheapest tier that
can handle it.

| Tier | What it is for | Default model | Then |
|---|---|---|---|
| **D** | a registered routine — no model at all | — | — |
| **C** | greetings, recall, rephrasing, summarising text you supplied | Ollama if running, else `gemini-3.1-flash-lite` | `gemini-3.5-flash-lite` |
| **B** | tools, files, commands, multi-step work with clear steps | `gemini-3.5-flash` | `gemini-3.6-flash` → `gemini-3.1-flash-lite` |
| **A** | judgement, ambiguity, anything hard to undo | `gemini-pro-latest` | `gemini-3.6-flash` → `gemini-3.5-flash` |
| **S** | nothing could answer — stop and ask a person | — | — |

Embeddings: `gemini-embedding-2` at 768 dimensions, then
`gemini-embedding-001`, then an offline hashing projection.

Every id above was checked against the live models endpoint, not recalled. They
will still go stale — see [When a model id dies](#when-a-model-id-dies).

## Getting a key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Create a key (free tier, no card).
3. Put it in `.env` beside `pyproject.toml`:

```bash
GOOGLE_API_KEY=...
```

```bash
itsbob doctor            # should show all three tiers as ok
itsbob doctor --probe    # sends one real request per model
```

One key covers every tier. That is why Google is the default: it exposes models
at genuinely different price points under a single credential, which is what
the ladder needs.

### Backups

`GROQ_API_KEY` ([console.groq.com](https://console.groq.com)) and
`OPENROUTER_API_KEY` ([openrouter.ai](https://openrouter.ai)) are optional.
They sit *behind* Google on every tier — reached only once every Gemini model
on that tier has failed — because neither has a comparable cheap/premium split
to map onto the ladder.

### The local model

Free, private, and preferred for Tier C when it is running:

```bash
ollama serve
ollama pull qwen2.5:1.5b          # ~1GB
export ITSBOB_OLLAMA_MODEL=phi3.5:3.8b-mini-instruct-q4_K_M   # or pin another
```

`itsbob doctor` reports whether Ollama is reachable **and** which models are
actually pulled. Those are different failures with the same symptom: a
configured-but-never-pulled model 404s on every call, and the Gatekeeper
silently falls back to its rule-based classifier.

## Pinning a model

Per tier — an override is *promoted*, and the old default becomes its first
fallback, so a stale pin still degrades into something that works:

```bash
export ITSBOB_TIER_C_MODEL=gemini-3.5-flash-lite
export ITSBOB_TIER_B_MODEL=gemini-3.6-flash
export ITSBOB_TIER_A_MODEL=gemini-pro-latest
```

Or in code:

```python
from itsbob.agent.brain import TIER_MODELS, build_brain
from itsbob.router.tiers import Tier

TIER_MODELS[Tier.A] = ("gemini-pro-latest",)
brain = build_brain()
```

## When a key is rejected

Symptom: every model on one provider fails at once, and `itsbob doctor
--probe` marks them `[auth]`.

```
!! google  gemini-3.5-flash  [auth] google: Please pass a valid API key — check the key for this provider
```

That is the *key*, not the models. Two things to check:

**Is it the right kind of credential?** An AI Studio key starts with `AIza`
and is about 39 characters. A value starting `AQ.` or `ya29.` is an OAuth
token — a different thing entirely, and Google rejects it with exactly the
message above. `itsbob setup` warns about this before it spends a request.

**Is it still valid?** Keys can be revoked or scoped to a project without
Generative Language API access. Create a fresh one at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

While a rejected key is set, itsbob still tries that provider first on every
call — it costs one wasted attempt each time and pushes everything onto your
backup provider. Either fix it or remove it from `~/.itsbob/.env`.

## When a model id dies

Free and preview model ids get retired and renamed constantly. This is the
normal failure, not an unusual one — the whole `llm/` layer exists to absorb
it. Symptom: `BadRequest ... 404 ... is no longer available`.

```bash
itsbob models             # what each provider serves today, vs what itsbob asks for
itsbob models --provider groq --all
itsbob doctor --probe     # actually call each one
```

`itsbob models` reads the provider's own `/models` endpoint, so it is never
stale the way this page will eventually be. Anything itsbob is configured to
try but the provider no longer serves is flagged, with the env var to pin a
live one.

Or ask Google directly:

```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" \
  | python3 -c 'import json,sys; [print(m["name"].split("/")[-1]) for m in json.load(sys.stdin)["models"] if "generateContent" in m.get("supportedGenerationMethods",[])]'
```

Then pin one of *those*. Do not guess an id from memory — the previous defaults
in this repo (`gemini-2.0-flash`, `gemini-2.5-flash`) were plausible, current
at the time, and both 404 now.

## Reading `doctor --probe`

`!!` lines are normal. The probe tries **every** model on **every** tier and
reports each attempt; a `!!` means that specific model is gone, gated, or over
quota, and the router already walks past it during a real call. What matters is
one `ok` per tier.

## Rate limits

Gemini meters **per model**, so a 429 on one leaves its siblings usable — the
router knows this (`rate_limit_scope="model"` in `llm/catalog.py`) and moves to
the next model rather than writing off the provider. Groq and OpenRouter meter
per account, so a 429 there means moving on entirely.

Free-tier limits are per-minute and per-day. If you hit them constantly, the
tier ladder is the fix: most turns should be Tier C, and if `doctor` shows
otherwise your requests are being classified harder than they are.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404 ... no longer available` | id retired | probe, then pin a live one |
| `404 ... not available to new users` | gated to existing accounts | pin a different one |
| `503 ... high demand` | transient | it retries the next model automatically; nothing to do |
| `401` / `403` | bad or unset key | check `.env` is beside `pyproject.toml` and has no quotes |
| Everything works but is slow | reasoning models spend budget before the first token | pin a `-flash-lite` model on Tier B |
| Every OpenRouter `:free` model 404s | account privacy setting, not a code or key problem | enable the free-model data policy at [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy) |
| Ollama reachable, still using the heuristic | configured model never pulled | `itsbob doctor` shows pulled vs. wanted |
