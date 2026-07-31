# 1tsb0b

A roleplay chatbot who speaks l33t, has to earn the right to message you, and
never lets a language model write his dialogue.

He runs on your laptop, always on, reachable from a browser anywhere. Talking
*to* him is free. Messaging *you* first costs credits he earns by waiting and
behaving — and when he spends them, you judge whether it was worth it.

Two things matter. **The API makes decisions, not sentences** — it is called
under a rigid protocol and must answer with exactly one control line, and the
words come from a pool of 426 pre-written, tone-tagged lines. And **his
personality is learned, not configured** — from your feedback and his memory,
with no API calls at all.

```
   1TSB0B|P00L|8.4|0.3|0.0|7.5|1.0|7.0|good news
   ───┬── ─┬── ─────────┬───────────── ────┬────
   sentinel │      six tone axes        topic hint
          action
```

Anything else — a stray "Sure!", a code fence, `8` instead of `8.0` — is a
protocol violation. One terse correction, then a deterministic local decision
takes over. A model having a bad day can produce a malformed line; it cannot
produce a surprising personality or an unbounded bill.

## The loop

```
  message ──► read persona.md + short_term.md + long_term.md, recall memories
                 │
                 ▼
            STRICT PROTOCOL CALL ──► 1TSB0B|<ACTION>|<6 tone axes>|<topic>
                 │                        │
                 │                   violation? one correction, then local
                 ▼
            P00L ────► nearest line in tone-space (426 pre-written l33t lines)
            C0MP0S3 ─► write one in that exact tone, forced to l33t
            S1L3NC3 ─► say nothing (a real veto on spending credits)
```

## Start here

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env        # paste in whichever free API keys you have
itsbob init                 # creates data/ — persona, memories, pool
itsbob serve --open         # he's live; the browser opens with your token
```

No keys? Still runs. The local decision path covers everything and the pool
supplies the words, so he is never mute and never out of character.

## Commands

```bash
itsbob serve                       # always-on: web UI + the outreach loop
itsbob serve --tunnel              # ...and publish a public URL
itsbob chat                        # terminal, shows the tone of every reply
itsbob pool --tone 0,8,9.5,0,2,9   # what he'd say at that mood
itsbob pool --category sarcasm     # browse by declared category
itsbob credits                     # balance, standing, why he can't message you
itsbob traits                      # his learned personality and what it earns him
itsbob doctor --probe              # which providers answer right now
itsbob models                      # what each provider will actually try
itsbob memory --query boats        # what he remembers
```

## The six axes

Every pool line is tagged, in leet, with where it sits on:

```
[P0s1t1v1ty: 8.4] [N3g4t1v1ty: 0.3] [H0st1l1ty: 0.0]
[C0m3dy: 7.5] [S4rc4sm: 1.0] [H0n3sty: 7.0]
```

The **first tag is the declared category** — that is how the shipped pool
breaks down:

| category | lines |
|---|---|
| comedy | 95 |
| positivity | 89 |
| honesty | 71 |
| negativity | 64 |
| sarcasm | 55 |
| hostility | 52 |

(Category comes from tag *order*, not the largest number. Honesty is rated high
on nearly every line, so taking the max would file 100 of them under "honesty".)

Selection trades off three things, because nearest-tone alone is not enough:

- **tone** — distance in six-dimensional mood space, the dominant term;
- **relevance** — lexical overlap with what you actually said, so the reply is
  about something rather than merely correctly-moody;
- **recency** — a penalty on lines used lately, or the pool repeats
  itself in an afternoon.

Variety comes from sampling among lines *close to* the best, never from a flat
top-K — otherwise a badly-toned line gets picked exactly when the pool is
thinnest.

## L33t

`a→4 e→3 i→1 o→0`, encoded on the way out, decoded for matching so "h3ll0" and
"hello" are the same word to retrieval. Composed replies are re-encoded rather
than trusted to the model, because models drift out of leet constantly.

Real numbers survive: a lone `1` is "i", but `100,000` stays `100,000`.

## The economy

| | |
|---|---|
| **credits** | Spendable. Accrue with time, consumed by outreach. |
| **standing** | Reputation, 0–1. Scales earn rate, gates outreach, and **colours his voice** — a Bob who's been told off gets audibly sourer. |

| What you do | Credits | Standing |
|---|---|---|
| Approve a message | +6 | +0.10 |
| Reply to it | +3 | +0.05 |
| Turn it down | −8 | −0.20 |
| Ignore it for 6 hours | −3 | −0.08 |

Silence counts. Unanswered outreach is swept as *ignored* — the mechanism that
stops him learning that sending more messages is free.

Even with credits he won't message you if he's grounded (standing under 0.15),
within 45 minutes of his last, over the daily cap of 8, in quiet hours
(23:00–08:00), or already holding an unanswered message. And the protocol can
return `S1L3NC3`, which is a real veto — deciding not to speak costs nothing.

## Personality he earns

He starts with a disposition and it moves. Approve a hostile line and hostility
becomes native. Turn his sarcasm down enough times and it drains out. **None of
this costs an API call** — it is arithmetic over verdicts and memory, so it keeps
working when every free tier is rate limited.

```
   you approve  ──► disposition pulls toward the tone that worked
   you turn down ─► disposition pushes toward its opposite (harder than approval pulls)
   you reply     ─► a weaker pull — engaging is the outcome he wants
   you ignore    ─► a weaker push, and patience erodes
```

Five traits fall out of that, all derived rather than stored:

| trait | what it measures |
|---|---|
| **attunement** | share of judged messages that landed, smoothed while the sample is small |
| **rapport** | shared history — grows with what he actually *remembers* about you |
| **warmth** | how warm the things you approve of are |
| **volatility** | how much his temperament has been swinging |
| **patience** | tolerance for being ignored; rapport buys it |

**Rewards are literally more credits.** Attunement and rapport set an earn-rate
multiplier from x0.4 to x1.8, so judgement you keep approving compounds and
judgement you keep rejecting starves. Standing (fast, forgiving) and the trait
multiplier (slow, earned) move on different timescales rather than
double-counting one good day.

The disposition then biases every reply — protocol decisions *and* the local
floor — weakly at first and more strongly as rapport grows. It never replaces
the decision; the situation still leads.

```bash
itsbob traits
```

```
after you approve 8 hostile messages
  disposition  pos 1.2, neg 6.8, hos 7.1, com 1.4, sar 6.3, hon 8.8
  attunement 0.83   rapport 0.32
  credit rate  x1.46

after you then turn down 10 of them
  disposition  pos 9.1, neg 2.5, hos 1.7, com 9.1, sar 3.3, hon 1.8
  credit rate  x1.09
```

One thing to know: disapproval reflects *every* axis of the rejected tone, not
just the loud one. Turning down a hostile-but-honest line teaches him to be
less honest too. That is the simple, predictable rule; retune `_PULL`/`_PUSH` in
`traits.py` if you want something more surgical.

Delete `data/traits.json` and he reverts to his starting temperament.

## The files are the interface

| File | What it is |
|---|---|
| `data/persona.md` | Who he is. Read on every response — edit it and his next message changes. |
| `data/memory/short_term.md` | The last ~20 turns. Rewritten automatically. |
| `data/memory/long_term.md` | What he knows about you. **Source of truth** — delete a line and he forgets it. |
| `data/pool.txt` | The 426 tone-tagged lines. Add your own. |
| `data/traits.json` | His learned personality. Delete it to reset him. |
| `data/responses.json` | Intent-matched lines, used as the last-resort floor. |

All reloaded on change. No restart.

Behind the markdown, `memory/` keeps a SQLite store doing ranked retrieval so
recall works past what fits in a prompt. Handwritten long-term lines survive
every sync, and important disclosures are written immediately rather than
waiting for eviction.

## Reaching him from anywhere

`itsbob serve` binds to `127.0.0.1` and every API call needs a token (generated
into `data/token.txt`). To reach him off your laptop:

```bash
itsbob serve --tunnel            # cloudflared or ngrok, whichever is installed
```

It prints a public URL with the token in it. **Treat that link as a password.**
If no tunnel client is installed it says so and carries on locally.

## Examples

```bash
python examples/01_talk_to_bob.py          # replies, with the decision behind each
python examples/02_credits_and_feedback.py # earning, spending, discipline
python examples/03_router_only.py          # the failover router by itself
python examples/04_memory_only.py          # the memory bank by itself
python examples/05_the_protocol.py         # what's accepted, rejected, and why
python examples/06_personality.py          # feedback becoming personality, offline
```

## Repository map

```
src/itsbob/
  protocol.py      the strict control-line spec, validator, and fallback
  tone.py          the six axes: parsing, distance, blending
  pool.py          the tone-tagged pool and its selection
  leet.py          a→4 e→3 i→1 o→0, both directions
  bob.py           reply() and consider_outreach()
  credits.py       the economy: balance, standing, the outreach gate
  traits.py        learned personality, and the credit reward loop
  persona.py       reads/writes persona.md and the memory files
  conversation.py  the message log
  server.py        HTTP API + the always-on outreach loop
  ui.py            the browser page, showing each reply's tone
  tunnel.py        cloudflared / ngrok
  llm/             failover router over the free providers
  memory/          two-tier memory behind the markdown files
  templates/       the shipped pool
tests/             212 tests, no network required
```

## Tests

```bash
pytest
```

212 tests, no network, no API keys. The protocol tests pin every rejection case
individually — preamble, missing decimal, wrong sentinel, two lines, fenced
output — because "highly strict" is only true if it's enforced. One test asserts
the whole trait engine makes exactly zero API calls. Tests named as
regressions pin behaviour that was actually broken once.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every reply says `source: local` | no provider answering, or none complying | `itsbob doctor --probe` |
| Replies feel off-tone | the pool has nothing near that mood | add lines, or retune vectors in `data/pool.txt` |
| He never messages first | working as designed — he declines most ticks | `itsbob credits` names the gate that's blocking him |
| He's suddenly sour | low standing, or a learned disposition | `itsbob traits`; approve a few, or delete `data/traits.json` |
| He earns credits oddly fast/slow | the trait multiplier | `itsbob traits` shows it; it follows your approvals |
| Replies aren't in leet | a composed line drifted | it's re-encoded automatically; report it if you see plain text |
| Forgot the token | | `cat data/token.txt` |
| Want him to forget something | | delete the line from `data/memory/long_term.md` |

## A note on privacy

`data/` holds your access token, the whole conversation, and everything he has
concluded about you. It's gitignored. Behind a tunnel, the link is the only
thing standing between that and anyone who has it.
