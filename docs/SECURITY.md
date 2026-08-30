# What it can do, and what stops it

This is an agent with shell access on a laptop. That is the point, and it is
also the risk. This document is what the safety envelope actually is — and,
more importantly, what it is **not**, so you can decide what to run it as.

## Two locks, for two different problems

**The registry** is an allow-list. A model may only *name* a registered tool.
An unregistered name raises rather than being best-effort'd. This stops a
hallucinated action — `file_search`, `execute_bash`, tools that do not exist.

**The policy** decides whether *this* call, with *these* arguments, may run
right now. The registry has no opinion about `run_shell("rm -rf ~/Documents")`
— it is a perfectly well-formed call to a registered tool. The policy is what
stops it.

Neither is sufficient alone, which is why there are two.

## Modes

| Mode | Read | Write in workspace | Network | Run code | Delete |
|---|---|---|---|---|---|
| `readonly` | yes | no | no | no | no |
| `guarded` *(default)* | yes | yes | ask | ask | ask |
| `dry_run` | yes | rehearse | rehearse | rehearse | rehearse |
| `trusted` | yes | yes | yes | yes | ask |

Per-tool overrides: `ITSBOB_AUTO_ALLOW`, `ITSBOB_ALWAYS_CONFIRM` (which wins),
`ITSBOB_BLOCKED_TOOLS` (which wins over both).

## Confirmation fails closed

A call that needs a human, with no confirmation handler attached, is **denied**.
Not queued, not assumed-yes.

This applies to the daemon and to any piped or scripted invocation. `itsbob
chat` attaches a handler only when stdin is a terminal.

The browser interface *does* attach one, because a browser is somewhere a
person actually is: a tool needing approval raises a card showing the exact
arguments, and the agent blocks on the answer. The fail-closed property is kept
by the timeout — a card nobody answers within three minutes is **denied**, so a
closed tab, a backgrounded laptop, or a card scrolled off screen can never
become an approval. "Always allow this tool" lasts for that browser session
only and is never written to disk.

The reasoning: a prompt nobody can see is not consent. It also means the
always-on mode is safe *by construction* rather than by convention — you cannot
accidentally end up with an unattended agent running arbitrary commands,
because the code path that would allow it does not exist. `itsbob serve` prints
which tools it therefore cannot use, so the constraint is visible rather than
discovered later.

## The four fences around code execution

`run_shell` and `run_python` are the deliberate hole in the allow-list. They
are fenced on four sides:

1. **Working directory.** The child starts in the workspace, and the file tools
   are jailed to the same root. The check is on the *resolved* path, so `../`,
   an absolute path, and a symlink pointing outward all fail the same test —
   rather than three separate string checks that each miss a case.
2. **Environment.** The child inherits only an allow-list (`PATH`, `HOME`,
   `LANG`, `TZ`, `TERM`, `USER`, `SHELL`, `TMPDIR`). **Every API key is
   withheld.** A generated script cannot read a credential it was never handed.
3. **Time.** A hard timeout kills the whole process group — `start_new_session`
   means `sleep 999 & wait` cannot outlive the call. A caller may shorten the
   timeout, never extend it past the policy's.
4. **Consent.** Both are `EXECUTE` risk, which `guarded` mode puts a human in
   front of, and which fails closed unattended.

## The deny-list is a guardrail, not a boundary

Some shapes are refused unconditionally, in every mode, with no confirm option:
`rm -rf /` or `~`, fork bombs, `mkfs`, raw writes to block devices, `sudo`,
`curl | sh`, netcat reverse shells, clearing shell history, force pushes.

**This is not a security boundary.** It catches what a confused model actually
emits, which is worth doing. It does not stop anyone who can spell the same
thing another way — and there are a hundred other ways.

The boundaries are the four fences above. The deny-list is a seatbelt.

## What this is not

**It is not a container.** The workspace jail binds the *file tools*, not the
kernel. `read_file` refuses `/etc/passwd`; `run_shell("head /etc/passwd")` in
trusted mode succeeds, because at that point the OS is the only thing deciding
and it says yes. That is not a hole in the jail — it is what granting code
execution means, and it is why `guarded` mode puts a human in front of
`run_shell` and the daemon cannot use it at all without an explicit grant.

So a command that runs can read your home directory, your SSH keys, and your
browser profile.

If that matters — and for anything running unattended it should — run it as a
dedicated user:

```bash
sudo useradd -m bob
sudo -u bob -H bash -c 'cd ~ && python3 -m venv .venv && ...'
```

On a laptop that exists to be the assistant's, this is the natural setup
anyway. Then the blast radius is that user's home directory, which is also
where `~/.itsbob` lives.

**It is not multi-user.** No authentication anywhere. The browser interface
binds to `127.0.0.1`, and anything that can reach that port can run allowed
tools as you — including approving its own confirmation prompts. `itsbob gui
--public` binds to every interface and warns you at the point of use; only do
that on a network where you would hand someone your shell.

**It does not sandbox the network.** `ITSBOB_ALLOWED_HOSTS` restricts which
hosts the *tools* will reach. It does nothing about a script that opens its own
socket. Use a firewall if you need one.

## Credentials

API keys live in `.env`, which is gitignored, and are read into the parent
process only.

- They are **never** put in a prompt. `call_api` takes an API *name*; the
  catalog attaches the credential. A model that cannot see a secret cannot leak
  one.
- They are **never** written to the audit log. Any parameter whose name
  contains `authorization`, `api_key`, `token`, `secret`, `password` or
  `cookie` is redacted.
- They are **never** passed to a child process (fence 2 above).

The one thing to be careful about: `http_request` takes a `headers` argument,
so a model *could* be told a key by you and then put it in one. Don't paste
credentials into the chat — configure them as an API instead.

## The audit log

`~/.itsbob/audit.jsonl`, append-only, one JSON object per line, flushed on
write. Every call is recorded, **including refused ones** — what the agent
tried and was stopped from doing is the more interesting half.

```bash
itsbob audit                      # recent activity
grep '"denied"' ~/.itsbob/audit.jsonl | tail -20
```

If it claims to have done something, the log is how you check. No entry means
it did not happen.

## Reporting

If you find a way past the fences — a path that escapes the workspace, a way to
get a credential into a child process, a confirm gate that can be bypassed —
that is a real bug, not a hardening suggestion. The deny-list is explicitly not
in scope: it is not claimed to be complete.
