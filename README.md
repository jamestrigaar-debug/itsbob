# itsbob

A personal assistant that runs on your own laptop. It remembers things across
conversations, uses tools you can audit, and routes each step of its thinking
to the cheapest model that can handle it.

```bash
pip install -e ".[gui,speed]"
echo "GOOGLE_API_KEY=..." > .env

itsbob doctor          # what's configured and what actually answers
itsbob chat            # talk to it
itsbob serve           # let it work on its own
```

---

## What it does

**It remembers.** Not a chat transcript — durable facts, extracted after each
turn and recalled by meaning as well as by keyword. Tell it you keep your SSH
keys in `~/.ssh/work` today; ask "where are my keys again?" next month.

**It uses tools, and you can check.** Files, shell, Python, HTTP, and any API
you configure. Every call is gated by a policy and written to an append-only
log, so "it said it updated the file" and "it updated the file" are separable
claims.

**It picks the right model per step.** Deciding to call `read_file` and
deciding whether a half-finished migration is safe to resume are not the same
problem and should not cost the same. A classifier picks a tier for each step;
trivial work never reaches an expensive model.

**It works while you're not there.** Scheduled tasks run through the same
agent, and a cheap gate decides whether the result is worth interrupting you
for. An assistant that pings you about every successful backup gets muted, and
a muted assistant is worth nothing.

---

## The tier ladder

```
D   Direct routine     no model at all              a registered routine fires
C   Cheapest model     local, or cheapest cloud     chat, recall, rephrasing
B   Standard model     the everyday workhorse       tools, files, multi-step work
A   Premium model      judgement                    ambiguity, anything hard to undo
S   Halt               nothing could answer         a person has to decide
```

Defaults, all verified live against the models API rather than recalled:

| Tier | Model | Falls back to |
|---|---|---|
| C | `gemini-3.1-flash-lite` | `gemini-3.5-flash-lite` |
| B | `gemini-3.5-flash` | `gemini-3.6-flash` → `gemini-3.1-flash-lite` |
| A | `gemini-pro-latest` | `gemini-3.6-flash` → `gemini-3.5-flash` |
| embeddings | `gemini-embedding-2` @ 768d | `gemini-embedding-001` → offline hashing |

One `GOOGLE_API_KEY` covers all of it. `GROQ_API_KEY` and `OPENROUTER_API_KEY`
are optional backups, tried only after every Gemini model on that tier has
failed. With Ollama running, Tier C prefers it — free, private, and fast enough
for the work Tier C is given.

Escalation goes **up before down**. A Tier B call whose providers are all
failing tries A next: one premium call beats a wrong cheap answer to a question
that has already proved hard. Within a turn the tier can rise but never fall —
a turn that needed the premium model at step 1 has not become easy by step 4.

Pin any of them:

```bash
export ITSBOB_TIER_B_MODEL=gemini-3.6-flash
```

---

## Memory

Two retrieval signals, fused:

- **Lexical** — SQLite FTS5 with BM25. The only thing that reliably finds a
  proper noun, an error code, or a path.
- **Semantic** — cosine similarity over embeddings. Finds the memory that says
  the same thing in different words, which is most of what people actually ask
  for.

Neither is enough alone. BM25 misses `"what do I like to drink?"` →
`"prefers dark roast coffee"`; pure vector search misses `ERR_CONN_REFUSED`.

```bash
itsbob memory add "the fuse box is behind the coats" --tags house
itsbob memory search "where's the electrics"
itsbob memory stats          # is semantic recall actually live?
itsbob memory reindex        # after changing embedding model or dimensions
```

Every hit reports *why* it surfaced (`keyword #1`, `semantic #2 (cos 0.74)`),
which is the difference between a memory system you can debug and one you have
to trust.

Three things that took getting wrong first:

- **Relevance dominates.** Importance and recency are tiebreakers, weighted
  like tiebreakers (0.12 / 0.10 against a relevance budget of 1.0). Weighted
  higher, every query returns the most *important* memory rather than the most
  relevant one.
- **Scores, not ranks.** Rank-based fusion makes the gap between hit #1 and #2
  larger than the entire importance-plus-recency budget, so equally good
  matches get ordered by whatever the index returned first.
- **Cosine is normalized within the result set, not thresholded.** Gemini puts
  unrelated text at 0.45–0.60 and related text at 0.65–0.85. The signal is the
  spread; its absolute position moves with the model.

Every vector is tagged with the model that produced it, and recall only ever
compares within one tag. Vectors from different models are not comparable, and
comparing them anyway returns plausible nonsense rather than an error.

Memory lives in `~/.itsbob/memory.sqlite`. Set `ITSBOB_EMBED_OFFLINE=true` to
keep every memory on the machine — recall degrades to keyword plus a
dependency-free hashing embedder rather than failing.

---

## Tools and the safety envelope

Two locks, because they stop different things.

**The registry** is an allow-list. A model can only *name* a registered tool;
an unregistered name is a routing error, never a best-effort execution. That
stops a hallucinated action.

**The policy** decides whether *this* call, with *these* arguments, may run
right now. That stops a well-formed `run_shell("rm -rf ~/Documents")`, which
the allow-list has no opinion about.

```
readonly   observation only
guarded    (default) reads and workspace writes run; commands and network ask first
dry_run    mutating tools report what they would do and change nothing
trusted    everything runs unattended except deleting and reaching outside the workspace
```

```bash
itsbob chat --mode trusted            # this session
export ITSBOB_TOOL_MODE=guarded       # the default
export ITSBOB_AUTO_ALLOW=run_shell    # exempt specific tools
export ITSBOB_ALWAYS_CONFIRM=delete_file
```

**Confirmation fails closed.** A call needing a human with no handler attached
— the daemon, a web request, a piped command — is denied. A prompt nobody can
see is not consent. This is what makes the always-on mode safe by construction
rather than by convention; `itsbob serve` tells you on startup which tools it
therefore cannot use.

The four fences around `run_shell` / `run_python`:

1. The child starts in the workspace, and the file tools are jailed to the same
   root — checked on the *resolved* path, so `../`, an absolute path and an
   outward symlink all fail the same check.
2. It inherits only an allow-listed environment. **Every API key is withheld**,
   so a generated script cannot read a credential it was never handed.
3. A hard timeout kills the whole process group, so `sleep 999 & wait` cannot
   outlive it. A caller may shorten the timeout, never extend it.
4. Both are `EXECUTE` risk, so `guarded` mode puts a human in front of them.

There is also a deny-list (`rm -rf /`, `curl | sh`, fork bombs, `sudo`) that
refuses unconditionally, in any mode. **It is a guardrail, not a security
boundary** — it catches what a confused model emits, not what a determined
adversary writes. The boundaries are the four fences above.

What this is *not* is a container. A command that runs can still read your home
directory, because the OS says it may. Run it under a dedicated user account if
that matters — which, on a laptop that exists to be the assistant's, is the
natural setup anyway.

```bash
itsbob tools     # what exists, and the policy in force
itsbob audit     # every call, including the refused ones
```

---

## APIs

Adding an API is a config entry, not a code change. The model names the API and
the path; the catalog attaches the base URL and the credential. **The key never
enters the prompt, the model's output, or the audit log** — a model that cannot
see a secret cannot leak one.

`apis.json` in the working directory (or `ITSBOB_API_CONFIG`):

```json
{
  "weather": {
    "base_url": "https://api.openweathermap.org/data/2.5",
    "key_env": "OPENWEATHER_KEY",
    "auth": "query",
    "query_param": "appid",
    "description": "Current conditions and forecast by city."
  },
  "github": {
    "base_url": "https://api.github.com",
    "key_env": "GITHUB_TOKEN",
    "auth": "bearer",
    "description": "Repos, issues and pull requests."
  }
}
```

`auth` is `bearer` (default), `header`, `query`, or `none`. Or use env vars:

```bash
ITSBOB_API_WEATHER_BASE=https://api.openweathermap.org/data/2.5
ITSBOB_API_WEATHER_KEY_ENV=OPENWEATHER_KEY
ITSBOB_API_WEATHER_AUTH=query
ITSBOB_API_WEATHER_QUERY_PARAM=appid
```

Then the key itself goes in `.env`, which is gitignored. Narrow what it can
reach at all with `ITSBOB_ALLOWED_HOSTS=api.github.com,api.openweathermap.org`.

---

## Running on its own

```bash
itsbob task add standup "Summarise yesterday's git commits in ~/work" "weekdays at 08:30"
itsbob task add inbox   "Check ~/inbox for new CSVs and describe anything odd" "every 30m"
itsbob task list
itsbob task run inbox        # now, off-schedule
itsbob task runs inbox       # its history

itsbob serve                 # the loop
itsbob serve --once          # run what's due, then exit
```

Schedules are written in words, not cron: `every 15m`, `every 2 hours`,
`daily at 08:30`, `weekdays at 09:00`, `friday at 17:00`,
`at 2026-09-01T06:00`. An interval task fires immediately when created —
"every 15m" means you want it working now, and a task that does nothing for
fifteen minutes reads as broken.

Whether a result reaches you is a separate decision, made by a cheap model and
biased toward silence. It sees the previous run's output too, so "still fine"
three mornings running is not three notifications. Notifications go to the
desktop (`notify-send` / `osascript`), always to
`~/.itsbob/notifications.jsonl`, and to `ITSBOB_WEBHOOK_URL` if you set one.

Each run gets a fresh conversation but the shared long-term memory: a nightly
task shouldn't inherit yesterday's context, but noticing "third morning the
backup has failed" is exactly what makes this worth running. Failed runs are
written to memory for that reason; successful ones are not.

Five consecutive failures disable a task. At that point it is broken rather
than unlucky, and running it hourly forever helps nobody.

To keep it running across reboots, use whatever your OS already has — a
`systemd --user` unit, a launchd plist, or `tmux`. There is no bespoke
supervisor here and there shouldn't be.

---

## The browser interface

```bash
itsbob gui       # http://127.0.0.1:8765
```

Chat on the left; on the right, every step as it happens — the tier and why it
was chosen, each tool call with its arguments and result, what was recalled and
what was written back — plus memory, task and audit panels.

It binds to localhost with no authentication, and takes no confirm handler, so
a web request cannot approve a risky tool on behalf of whoever left the tab
open. Anything that can reach the port can still run allowed tools as you.

---

## As a library

```python
from itsbob import build_agent

bob = build_agent(home="~/bob", mode="guarded")
turn = bob.chat("summarise the CSVs in the workspace")
print(turn.final, turn.tools_used, turn.tier)
```

Every piece is independently constructible and independently useful:

```python
from itsbob import LongTermMemory, default_embedder, build_toolbox, build_brain

store = LongTermMemory("mem.sqlite", embedder=default_embedder())
box   = build_toolbox(memory=store, mode="trusted")
brain = build_brain()                       # just the tier ladder
```

Adding a tool is a registration:

```python
from itsbob.tools import Risk, Tool, ToolResult

def run(params, ctx):
    return ToolResult(ok=True, output=f"pinged {params['host']}")

box.registry.register(Tool(
    name="ping", description="Check whether a host is up.", run=run,
    risk=Risk.NETWORK,
    parameters={"type": "object", "properties": {"host": {"type": "string"}},
                "required": ["host"]},
))
```

---

## Further reading

- **[docs/MODELS.md](docs/MODELS.md)** — the tier ladder, getting a key,
  pinning a model, and what to do when an id gets retired (it will).
- **[docs/SECURITY.md](docs/SECURITY.md)** — what it can do, what stops it, and
  what the envelope explicitly does *not* cover.

---

## Setup

Python ≥ 3.10.

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[gui,speed,dev]"
cp .env.example .env        # then paste your keys in
itsbob doctor
```

`.env` is gitignored, so keys never reach the repository — which also means a
zip or clone won't carry them. Paste them in on the machine that runs it.

Runs with **zero configuration**: no keys falls back to the offline provider,
no embedding API falls back to keyword-only recall, no Ollama falls back to the
cheapest cloud model. Every path is exercised either way; things get less
capable, not broken.

**Optional: the local model.** Free, private, and preferred for Tier C.

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

**Optional: `numpy`** (`pip install -e ".[speed]"`). Without it, recall
compares the most recent 5,000 vectors in pure Python; with it, the whole table
in one matrix multiply. Matters past a few thousand memories.

Everything lives in `~/.itsbob` (`ITSBOB_HOME`):

```
memory.sqlite         what it remembers
tasks.sqlite          scheduled work and its history
workspace/            the only directory tools may touch
audit.jsonl           every tool call, including refusals
notifications.jsonl   everything it decided was worth saying
persona.md            optional: standing instructions, read on every turn
```

---

## Troubleshooting

Start with `itsbob doctor`, then `itsbob doctor --probe` to send one real
request per model.

| Symptom | Cause | Fix |
|---|---|---|
| `itsbob: command not found` | venv not active | `source .venv/bin/activate` |
| Every recall says `keyword #N`, never `semantic` | no embedding API reachable | `itsbob memory stats`; set `GOOGLE_API_KEY`, then `itsbob memory reindex` |
| Recall was good, then got worse | embedding model changed, so old vectors carry a different signature and are ignored | `itsbob memory reindex` |
| "needs confirmation and nothing is available to ask" | running unattended, by design | run `itsbob chat` interactively, add the tool to `ITSBOB_AUTO_ALLOW`, or use `--mode trusted` |
| A model id 404s | free-tier models get retired and renamed constantly | `itsbob doctor --probe` to see what answers, then pin it with `ITSBOB_TIER_B_MODEL=...` |
| A `--probe` run with some `!!` lines | normal | one `ok` per tier is all that's needed |
| It says it did something but nothing changed | it answered instead of calling a tool | `itsbob audit` — if no call is logged, it didn't happen. Report it; the prompt is meant to prevent this |
| The daemon runs but never notifies | the gate is working | it is biased toward silence. `~/.itsbob/notifications.jsonl` has everything it decided; `itsbob task runs <name>` has every result |
| A task keeps failing | five failures disable it | `itsbob task runs <name>` for the errors, fix the prompt, `itsbob task enable <name>` |
| OpenRouter 404s every `:free` model | account privacy setting, not a code or key problem | enable the free-model data policy at [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy) |
| Ollama reachable but the Gatekeeper still uses the heuristic | the configured model was never pulled | `itsbob doctor` shows pulled vs. wanted; `ollama pull <the missing one>` |
| Flask prints `Tip: install python-dotenv` | Flask noticing a `.env` exists | harmless — itsbob loads `.env` itself |

---

## Repository map

```
src/itsbob/
  agent/          the loop
    brain.py        the tier ladder and escalation between tiers
    loop.py         classify → recall → step → act → observe, and the turn guard
    context.py      prompt assembly (one system message; steps as ReAct turns)
    persona.py      the system prompt
    writer.py       post-turn extraction of durable facts
  memory/         hybrid recall
    long_term.py    SQLite + FTS5 + vectors, score fusion, migration
    base.py         MemoryRecord and scoring
    short_term.py   the simulation's decaying working set
    bank.py         two-tier facade over both
  tools/          capability and consent
    base.py         Tool, ToolRegistry — the allow-list
    policy.py       the gate: modes, deny-list, env scrubbing, host allow-list
    sandbox.py      run_shell / run_python, fenced four ways
    files.py        path-jailed filesystem tools
    http.py         http_request, call_api, the API catalog
    memory_tools.py remember / recall / forget / update
    audit.py        append-only JSONL, credentials redacted
  daemon/         the always-on half
    schedule.py     schedules in words
    tasks.py        SQLite task store and run history
    notify.py       the gate on interrupting, and where notices go
    service.py      the loop
  llm/            provider-agnostic model access
    router.py       failover, rate limiting, circuit breaker, usage tracking
    embeddings.py   the embedding chain and signature isolation
    providers.py    Groq / Google / OpenRouter (one OpenAI-compatible client)
    local.py        Ollama
    catalog.py      default model ids per provider
  router/         classification
    tiers.py        the S/A/B/C/D taxonomy
    gatekeeper.py   which tier, by model or by rule
    ingestion.py    anything in → one Snapshot shape
    pipeline.py     the original one-shot router (still importable)
  character/, engine/   the original tick simulation — `itsbob run`
  gui/app.py      the browser interface
  cli.py          every command
tests/            260 tests, none of which touch the network
```

## The original simulation

This repo grew out of a tick-based character simulation with an energy-metered
decision loop: every tick, a character decides whether a choice is worth
spending scarce energy to think about with an LLM. It still runs, shares the
`llm/` layer, and nothing in the assistant depends on it.

```bash
itsbob run --ticks 20 --policy hybrid
itsbob run --ticks 20 --offline --json
```

## Not built

- **A local embedding model.** Embeddings go to Google by default; the offline
  fallback is a hashing projection, not a learned one.
- **Multi-user anything.** One person, one laptop. No auth, no tenancy.
- **A supervisor.** Use systemd/launchd/tmux.
- **Streaming.** Turns complete before they return. The GUI shows steps after
  the fact, not token by token.
