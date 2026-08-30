"""The outside world: services itsbob talks to, and the one it talks *through*.

Kept apart from :mod:`itsbob.tools` because these are different in kind. A tool
is a capability the agent has; these are *relationships with services* — each
one has a key, a base URL, a rate limit and its own way of failing, and each one
can be absent without anything else breaking.

Everything here follows the same three rules:

* **Configuration, not code.** A key in ``.env`` is the whole setup; the base
  URL, auth style and header name ship with the spec (:mod:`.apis`).
* **Absent is a normal state.** No key means the capability reports what is
  missing, never that something is broken. One dead source never costs another
  one its results.
* **``urllib``, not a client library.** Every service here is two or three HTTP
  calls. A dependency per service is a worse trade than the fifty lines.
"""

from __future__ import annotations

from .apis import BUILTIN_SPECS, builtin_status, register_builtins

__all__ = ["BUILTIN_SPECS", "builtin_status", "register_builtins"]
