# Setting up Google (Gemini) as a provider

`itsbob`'s `google` provider talks to Gemini through Google's
OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`)
— that's why no Google-specific SDK is needed, only the `openai` package
already in `pyproject.toml`'s base dependencies. This document covers getting
a key, picking which Gemini model itsbob actually calls, and troubleshooting
the Google-specific failure modes.

Google is normally used as **Tier B** (the cheap cloud workhorse — Gemini
Flash) in the [complexity router](../README.md#the-router-pipeline), and as
an ordinary failover provider in the [character
simulation](../README.md#the-character-simulation-the-original-foundation).

## 1. Get an API key

1. Go to **[aistudio.google.com](https://aistudio.google.com)** and sign in
   with a Google account.
2. Click **Get API key** (left sidebar) → **Create API key**.
3. Choose an existing Google Cloud project, or let AI Studio create one for
   you — for personal/free-tier use the auto-created project is fine.
4. Copy the key. It looks like `AIzaSy...`.

No billing setup is required to get a key or to use the free tier. If you
later want the paid tier's higher rate limits, billing is enabled from the
same Google Cloud project in the [Cloud Console](https://console.cloud.google.com/billing).

## 2. Give the key to itsbob

Either export it:

```bash
export GOOGLE_API_KEY=AIzaSy...
```

or put it in `.env` (copy `.env.example` first if you haven't):

```
GOOGLE_API_KEY=AIzaSy...
```

Verify it's picked up:

```bash
itsbob doctor            # shows "ok  google" once the key is set
itsbob doctor --probe    # sends one real request — confirms the key actually works
```

## 3. Picking a specific model

itsbob ships with a **default model and a fallback list** for Google (see
`src/itsbob/llm/catalog.py`):

```python
GOOGLE = ProviderConfig(
    ...
    default_model="gemini-3.6-flash",
    fallback_models=(
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-2.0-flash",
    ),
    ...
)
```

If the default model 404s (retired, renamed, or not available to your
account), the router automatically walks the fallback list — you don't have
to do anything. But if you want a **specific** model — a different Flash
tier, Gemini Pro for heavier reasoning, or whatever Google's current catalog
calls its latest release — override it with an env var:

```bash
export ITSBOB_GOOGLE_MODEL=gemini-2.5-pro
```

This is read by `default_provider_configs()` in `llm/catalog.py`: your
override becomes the new default, and itsbob's own hardcoded default slides
into the fallback list rather than disappearing — so a typo or a
model that later gets retired still degrades into something that works
instead of hard-failing.

### Where to find current model names

Google's model lineup and names change more often than this README does.
Two ways to check what's actually available to your key right now:

```bash
# List every model your key can see, with supported methods:
curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY"
```

or check the model list in [AI Studio](https://aistudio.google.com) itself
(the model picker in the chat UI shows the current names) or Google's
[Gemini API model docs](https://ai.google.dev/gemini-api/docs/models).

Match what you find there against the `generateContent`-capable, chat-style
models — that's what itsbob's `/chat/completions`-shaped requests need.
Flash-class models (`gemini-*-flash*`) are the free-tier/cheap workhorses
this router's Tier B is designed around; Pro-class models cost more and
reason better, closer to what Tier A is for.

### Picking a model per-request instead of globally

You don't have to change the env var to try a different model once — `itsbob
ask` and the router both accept an explicit model on the request:

```bash
itsbob ask "..." --provider google
```

(`--provider` restricts which provider answers; to pin the exact model for a
single call rather than globally, set `ITSBOB_GOOGLE_MODEL` just for that
invocation: `ITSBOB_GOOGLE_MODEL=gemini-2.5-pro itsbob ask "..." --provider google`.)

## 4. Rate limits and quota

Google's free tier meters **per model**, not per account — reflected in
itsbob's config as `rate_limit_scope="model"`. Practically: if
`gemini-3.6-flash` hits its quota, the router moves on to
`gemini-3.1-flash-lite` etc. automatically rather than treating the whole
provider as down (unlike Groq/OpenRouter, which meter per account — see
`config.py`'s `rate_limit_scope` docstring).

itsbob's own local rate limiter is set conservatively
(`requests_per_minute=15` in `catalog.py`) to spend quota deliberately rather
than discover the ceiling via 429s. If your key has a higher quota (e.g. a
paid tier), the local limiter is the thing capping you — there's no env var
for it today; edit `GOOGLE.requests_per_minute` in
`src/itsbob/llm/catalog.py` if you want to raise it.

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `itsbob doctor` shows `-- google` | `GOOGLE_API_KEY` not set, or set somewhere itsbob isn't reading from | check `.env` is in the directory you run `itsbob` from, or `echo $GOOGLE_API_KEY` |
| `itsbob doctor --probe` → `ProviderNotConfigured` for google despite a key | key has a leading/trailing space or quote left in from copy-paste | re-copy the key; `load_dotenv` strips surrounding quotes but not internal whitespace |
| `BadRequest: ... 404 ...` naming a model | the pinned/default model id was retired or renamed by Google | run the `curl` command above to see current names, then `export ITSBOB_GOOGLE_MODEL=<a live one>` |
| `RateLimited` / `429` on one model but other Google models still work | that model's per-model quota is spent; expected, not a bug | wait for the quota window to reset, or let the router's fallback list carry on (it already does this automatically) |
| `RateLimited` on every Google model | the whole key's quota is spent, or you're hitting a project-level cap | check quota in [Cloud Console](https://console.cloud.google.com/) → APIs & Services → Generative Language API, or enable billing for higher limits |
| `ProviderNotConfigured: 401/403` | key is invalid, revoked, or the Generative Language API isn't enabled on that project | regenerate the key in AI Studio; confirm the project has the Gemini API enabled |
| Response is empty with `finish_reason: length` | the model spent its whole `max_tokens` budget on hidden reasoning before emitting visible text (common on newer Gemini reasoning-capable models) | raise `--max-tokens` on `itsbob ask`, or `LLMRequest.max_tokens` if calling the router directly |
| Slow responses vs. Groq | expected — Gemini Flash is fast for a hosted API but not as fast as Groq's inference hardware | if raw speed matters more than Google specifically, reorder providers: `export ITSBOB_PROVIDER_ORDER=groq,google,openrouter` |

## 6. Using Google specifically, not just "whichever provider answers first"

By default the router tries providers in priority order and Google is just
one of several. To force Google specifically for one call:

```bash
itsbob ask "your prompt" --provider google
```

Programmatically:

```python
from itsbob.factory import build_router
from itsbob.config import Settings
from itsbob.llm.base import LLMRequest, user

router = build_router(Settings.from_env())
response = router.complete(
    LLMRequest(messages=[user("your prompt")], max_tokens=300),
    providers=["google"],   # restrict the failover set to just this provider
)
print(response.model, response.text)
```

Or to make Google the *first* choice (but still allow failover to the
others) rather than restricting to only it:

```bash
export ITSBOB_PROVIDER_ORDER=google,groq,openrouter
```

## 7. Google answering every complexity-router cloud tier

If you want Google to be the *only* cloud provider the
[complexity router](../README.md#the-router-pipeline) uses — at two
different Gemini models, a cheap one for Tier B and a stronger one for
Tier A, with Groq/OpenRouter only as a backup tried after Google rather
than before it — that's **google-tiered mode**:

```bash
export ITSBOB_ROUTER_MODE=google-tiered
itsbob doctor       # shows Tier B and Tier A as two separate Gemini chains
itsbob route '{"facts": {"opponent_formation": "4-3-3"}, "events": ["opponent tactic shift"]}'
```

Only `GOOGLE_API_KEY` is required for this mode. See [README:
Google-tiered mode](../README.md#google-tiered-mode--google-answers-every-cloud-tier-itself)
for the full model ladder, how to disable the Groq/OpenRouter backup
entirely, and how to pin different Gemini model ids.
