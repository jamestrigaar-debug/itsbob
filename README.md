# itsbob

A personal assistant that runs on your own laptop. It remembers things across
conversations, uses tools you can audit, and routes each step of its thinking
to the cheapest model that can handle it.

```bash
git clone https://github.com/jamestrigaar-debug/itsbob && cd itsbob
./install.sh
```

That creates the virtualenv, installs everything, asks for a key, and checks it
against the real API before telling you it worked. Then:

```bash
itsbob chat            # talk to it
itsbob gui             # the browser interface
itsbob serve           # let it work on its own
itsbob doctor          # what's configured, and what actually answers
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

## What it costs, and how that is kept down

Every turn is billed, so five things work to keep the bill honest:

- **The local model gets first refusal on everything cheap.** With Ollama
  running, Tier C answers *and* every bookkeeping chore — classification,
  memory extraction, the feasibility check, condensing the briefing — run
  locally and cost nothing. `itsbob doctor` makes a real call and reports what
  came back, because "reachable" and "answering" are different claims and only
  the second one saves money.
- **A turn it cannot finish does not start.** One cheap call reads the request
  against the tools that actually exist. Discovering "there is no key for that"
  in one small call beats discovering it eight premium steps later. It is
  biased toward yes: a false refusal is worse than a wasted turn — which is
  also why it only runs on every request when Ollama makes it free. Without a
  local model it is reserved for requests long enough to imply a long turn,
  since a screen that rarely refuses anything must not cost a call per turn.
- **The prompt is on a diet.** The big one: after the first step of a turn,
  tool *descriptions* stop being re-sent. The description is how you *choose* a
  tool and the signature is how you *call* one — by step two the choosing is
  done, so everything stays listed and callable but only the tools in play keep
  their prose. At 37 tools that is ~1,900 tokens a step down to ~650, measured
  at **23% off a 16-step turn and 19% off a three-step one**. On top: cheap
  tiers get a short system prompt (754 characters against 3002), the API
  catalogue appears only where it can change a decision, older scratchpad steps
  collapse to one line, observation clipping tightens as a turn goes on, and
  tool output is condensed at the source rather than dumped raw.
- **A spend ceiling per turn and per day.** Hitting it does not kill the turn —
  it tells it to stop and answer with what it has.
- **Durable facts go in memory rather than into every prompt.**

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

### Whose memory is it

Every memory records its **subject**: `user`, `bob`, or `world`. This is not a
nicety. Asked for its own favourite films, it listed five and the extractor
wrote all five down as *the user's* favourites; recall then served them back as
facts about a person who had never mentioned any of them. A store that cannot
say whose opinion it is holding will, given enough turns, replace you with the
assistant. Recalled memories are shown grouped by subject, and a sentence that
gives itself away ("I liked …") overrides a wrong label — a model that writes
that and files it under `user` has contradicted itself in one line.

### Short term and long term

Memories also carry a **horizon**. Short-horizon rows are the working set —
what is being worked on today, a state the machine is in, a thread still open.
They are capped by count *and* by clock and pruned at the end of every turn, so
a busy hour cannot quietly become the corpus and a row from last Tuesday cannot
sit there forever. Long-horizon rows are what should still be true in a year.
`keep_memory` promotes one that turned out to matter after all.

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
— the daemon, a piped command — is denied. A prompt nobody can see is not
consent. This is what makes the always-on mode safe by construction rather than
by convention; `itsbob serve` tells you on startup which tools it therefore
cannot use. The browser interface *can* answer, and an unanswered card there is
denied after three minutes for the same reason.

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

### The ones that ship configured

Four services have their base URL, auth style and header name built in, so a
key in `.env` is the whole setup — nothing to write out, nothing to get subtly
wrong:

| Put this in `.env` | Gives you |
|---|---|
| `OPENWEATHER_API_KEY=…` | `weather` — conditions and today's outlook for your location |
| `NEWSAPI_KEY=…` | `news` — headlines, merged and deduplicated |
| `GNEWS_API_KEY=…` | a second news source, and the fallback when NewsAPI rate-limits |
| `FOOTBALL_DATA_KEY=…` | `football` via `call_api` — fixtures, standings, scorers |

With a weather key and either news key, `daily_briefing` becomes one tool: the
day's weather, then the day's significant news condensed into prose with its
sources listed. It is built to be a morning task:

```bash
itsbob task add briefing "Run daily_briefing and send me the result" "daily at 07:00"
itsbob task add pl "Use the football API (competitions/PL/matches) for today's \
  Premier League fixtures and kickoff times" "daily at 07:00"
```

The weather location defaults to Hull, UK and moves with
`ITSBOB_WEATHER_PLACE`, `ITSBOB_WEATHER_LAT` and `ITSBOB_WEATHER_LON`.

**It can look at its own screen.** `look_at_screen` captures and reads in a
single tool call — "what does that error say", "is the build finished yet",
"what's this chart showing". Doing it as two tools (screenshot, then vision on
the path) works and is still available, but costs two model calls where one
does; and the PNG is an implementation detail of the question, so it is
discarded afterwards unless you pass `keep`. `look_at_window` does the focused
window, `look_at_image` reads a picture already on disk. All three need
`GOOGLE_API_KEY`, and say so *before* taking a screenshot nobody could read.

**Web search needs no key at all.** `web_search` uses `ddgr` or `googler` if
either is installed (`sudo apt install ddgr`), and falls back to DuckDuckGo's
HTML endpoint otherwise. It is a separate tool rather than a `run_shell`
instruction on purpose: search is a read-only fetch, and routing it through the
broadest capability in the system would mean either approving `run_shell`
permanently or answering a prompt every time you want to look something up.

The **apis** panel in the browser shows which are live and which are missing a
key, with a *schedule…* button that starts a task against one — so you find out
a key is missing before you write the task, not at 07:00 tomorrow.

### Anything else

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
itsbob gui       # http://localhost:8765
```

The console is one page: the conversation on the left at full height, and on
the right six panels of evidence about it — **activity** (every step as it
happens), **memory**, **tasks**, **tokens**, **messages**, and **system**
(APIs, scripts, tools). Across the top, the four things asked at a glance:
whether it is thinking, whether the daemon is serving, whether Discord is
connected, and what today has cost.

Two rules it was rebuilt around. Every panel owns its endpoint, so one broken
subsystem costs one panel and says so in place — the old tasks panel read from
the shared status payload and went dark whenever anything unrelated in it was
slow. And nothing spins without saying why: every request is bounded, and a
timeout is a named error with a retry rather than a placeholder that never
resolves.

The **tokens** panel reports estimated money, not just counts. A million cheap
tokens and a million premium ones differ by roughly forty times in price, so a
bare token number invites the wrong conclusion. It splits spend by model and by
*what the call was for* — answering, classifying, extracting memories — and
shows what share ran locally for nothing. A model with no published price is
reported as unpriced rather than guessed at.

The previous interface is still at `/old` for one release.

Chat on the left. On the right, **every step as it happens** — streamed over
server-sent events, not delivered in one lump when the turn ends. You watch the
tier get chosen, each tool call go out, and each result come back, which is the
difference between an assistant you can supervise and one you have to trust.

Six panels: activity, memory (with *why* each hit surfaced), scheduled tasks,
scripts, APIs (which are live, which need a key), and the audit log.

### The messages window

`/messages` is a **separate page** for everything itsbob said without being
asked: task results, alerts, the morning briefing. Proactive notices and a
conversation are different kinds of thing — interleaving them gives the
conversation interruptions and the notices a context they do not have. It reads
the same `notifications.jsonl` the daemon already writes, updates live over its
own SSE stream, and tracks read/unread separately so marking one read never
rewrites a log another process is appending to. The header of the main page
carries the unread count and opens it in its own window.

### Discord

Set `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` and the channel becomes a
two-way workspace. Outbound: anything the notice gate passes is posted there,
including messages itsbob starts himself. Inbound: what you type in the channel
becomes an ordinary turn, queued alongside anything typed in the browser — one
agent, one queue, so nothing interleaves. Built on the REST API with `urllib`
rather than the gateway, so there is no websocket, no async runtime and no new
dependency; it polls every few seconds, which is how often a person looks at a
channel anyway. Long messages are split on paragraph breaks, rate limits are
waited out per Discord's own `retry_after`, and the bot never answers itself.

**Network access is a setup question.** Fetching a page, searching the web and
calling your own APIs each raise an approval prompt by default, and a web
search that needs a yes every time is not a web search. `itsbob setup` offers to
let network calls run without asking; shell commands, deletions and stopping
processes still ask, because those are the ones you cannot take back. Change it
any time with `ITSBOB_AUTO_ALLOW_RISKS=network` in `.env`, or
`itsbob setup --open-network`. It is a *risk level* rather than a list of tool
names on purpose: there are eight network tools and more arrive with every API
you add, so a per-tool list is out of date the moment it is written.

**You can approve tools from the page.** When the agent reaches something
`guarded` mode gates, a card appears showing the exact command and why it wants
to run it, with *Allow once* / *Deny* / *Always allow this tool*. The agent
waits on your answer. This is what makes `guarded` mode usable in a browser at
all — before, the interface passed no confirmation handler, so every command
was correctly but uselessly refused.

An unanswered card is **denied** after three minutes, counting down on the
card. A closed tab is not a yes.

One turn runs at a time (two would interleave into each other's conversation
history). It binds to localhost with no authentication: anything that can reach
the port can run allowed tools as you. `--public` binds to all interfaces and
warns you at the point of use.

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

Python ≥ 3.10. One command:

```bash
./install.sh
```

It finds a suitable Python, creates the virtualenv, installs with the browser
interface and the fast recall path, links `itsbob` onto your PATH if it can,
and runs `itsbob setup`. Re-running it is safe — it upgrades in place and never
touches your keys.

If anything in that chain does not suit you, the manual path still works:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
itsbob setup
```

`itsbob setup` writes your keys to `~/.itsbob/.env` at mode 600 and then
**makes a real API call to check each one** — "the variable is set" and "the key
works" are different claims, and only the second is worth being told. Because
they live in the home directory rather than the working directory, the daemon
and the GUI find them wherever they are started from.

It then offers each optional capability in turn — Discord, OpenWeather,
NewsAPI, GNews, football-data — saying what each one actually gives you before
asking for its key. They used to be *reported* at the end instead, which meant
finding out a capability existed just after deciding you had finished
configuring. Everything there is skippable, and none of it is verified with a
live call: they are not providers, and a wizard that spends five API calls
proving keys that are allowed to be absent is one people learn to skip.

Every one of them also has a flag, so an unattended install is one command:

```bash
itsbob setup --google-key … --openweather-api-key … --newsapi-key … \
             --discord-bot-token … --discord-channel-id …
```

`make help` lists the shortcuts for working *on* itsbob rather than with it.

### Everything you can put in `~/.itsbob/.env`

Only the first line is needed. Everything else adds a capability, and anything
absent is reported as absent rather than breaking something — `itsbob doctor`
lists what is on and what each missing piece would give you.

```bash
# Thinking. One key covers all four tiers.
GOOGLE_API_KEY=…
GROQ_API_KEY=…                  # backups, tried only after every Gemini model fails
OPENROUTER_API_KEY=…

# Services. Base URL and auth ship built in; the key is the whole setup.
OPENWEATHER_API_KEY=…           # the `weather` tool and half of `daily_briefing`
NEWSAPI_KEY=…                   # the `news` tool and the other half
GNEWS_API_KEY=…                 # second news source, and the rate-limit fallback
FOOTBALL_DATA_KEY=…             # fixtures, standings and scorers via call_api

# Discord: the channel becomes a two-way workspace.
DISCORD_BOT_TOKEN=…
DISCORD_CHANNEL_ID=…

# Where the weather is. Defaults to Hull, UK.
ITSBOB_WEATHER_PLACE="Hull, UK"
ITSBOB_WEATHER_LAT=53.7767
ITSBOB_WEATHER_LON=-0.3274

# Speaking first when idle. `off` disables it; hours between attempts; waking hours.
ITSBOB_INITIATIVE=on
ITSBOB_INITIATIVE_HOURS=3
ITSBOB_INITIATIVE_WAKING=8-22

# The local model. Keep the default unless you have pulled something else.
ITSBOB_OLLAMA_MODEL=qwen2.5:1.5b
ITSBOB_OLLAMA_URL=http://127.0.0.1:11434

# Safety. `guarded` (default) asks before anything outside the workspace.
ITSBOB_TOOL_MODE=guarded
ITSBOB_AUTO_ALLOW=                # tools to run without asking. Leave empty.
ITSBOB_ALLOWED_HOSTS=             # empty means any host; a list narrows it
ITSBOB_SCRIPTS_DIR=~/.itsbob/scripts
```

`ITSBOB_AUTO_ALLOW=run_shell` is worth naming, because it is the one people
reach for. It gives the model unattended shell on your machine for the life of
the process, and it is not needed for web search — `web_search` is its own
NETWORK-gated tool precisely so that looking something up does not require
opening that door.

### Running in the background

```bash
itsbob service install     # systemd --user unit, or a launchd plist
itsbob service status
itsbob service print       # see the unit without installing it
```

No supervisor ships with itsbob on purpose: your OS already has one that is
better tested and is what an administrator expects to find.

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
    persona.py      the system prompt, full and dieted
    writer.py       post-turn extraction, with attribution and horizon
    budget.py       the spend ceiling and the can-it-be-done check
    initiative.py   what it does when nobody has asked it anything
  memory/         hybrid recall
    long_term.py    SQLite + FTS5 + vectors, score fusion, migration
    base.py         MemoryRecord, Subject, Horizon, and scoring
    short_term.py   the simulation's decaying working set
    bank.py         two-tier facade over both
  tools/          capability and consent
    base.py         Tool, ToolRegistry — the allow-list
    policy.py       the gate: modes, deny-list, env scrubbing, host allow-list
    sandbox.py      run_shell / run_python, fenced four ways
    files.py        path-jailed filesystem tools
    http.py         http_request, call_api, the API catalog
    memory_tools.py remember / recall / forget / update / keep
    websearch.py    ddgr, googler, or DuckDuckGo — no key needed
    vision.py       describe_image / image_info
    audit.py        append-only JSONL, credentials redacted
  integrations/   the outside world
    apis.py         built-in specs: weather, news, gnews, football
    briefing.py     weather + news + the condensed daily report
    discord.py      the channel as a two-way workspace
  scripts/        what it can do to this machine — drop a file in to add one
    system_monitor.py, network_checker.py, process_manager.py,
    file_cleaner.py, screenshot.py, screen_reader.py, scheduler.py
  daemon/         the always-on half
    schedule.py     schedules in words
    tasks.py        SQLite task store and run history
    notify.py       the gate on interrupting, and where notices go
    service.py      the loop
  llm/            provider-agnostic model access
    router.py       failover, rate limiting, circuit breaker, usage tracking
    pricing.py      what a call cost, and which share of it was free
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
  gui/            the browser interface
    app.py          Flask routes and the SSE stream
    session.py      one running agent: event fan-out and the approval gate
    console.py      the console: chat, activity, memory, tasks, tokens, system
    page.py         the previous interface, served at /old for one release
    messages.py     the standalone /messages window and its log reader
    autonomous.py   continuous mode: scheduled work through the chat queue
  store.py        locked SQLite: one lock per file, WAL, busy timeout
  logfile.py      append-only JSONL with rotation
  setup_wizard.py `itsbob setup` — keys, directories, and a live check
  service.py      systemd/launchd unit generation
  cli.py          every command
install.sh        one-command install
tests/            508 tests, none of which touch the network
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
