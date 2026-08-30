"""Reaching external APIs — as configuration, not code.

Two tools, deliberately separate:

``http_request``
    An arbitrary URL. Gated as :attr:`~itsbob.tools.base.Risk.NETWORK`, and
    narrowable to a host allow-list.

``call_api``
    A *named* API from the catalog. The model names the API and the path; the
    catalog attaches the base URL and the credential. The key never appears in
    the prompt, in the model's output, or in the audit log — which is the whole
    point: a model that cannot see a secret cannot leak one, and adding an API
    stays a config entry instead of a code change.

Configure an API either in ``apis.json``::

    {"weather": {"base_url": "https://api.example.com/v1",
                 "key_env": "WEATHER_API_KEY",
                 "auth": "query", "query_param": "appid",
                 "description": "Current conditions and forecast."}}

or with environment variables, one per field::

    ITSBOB_API_WEATHER_BASE=https://api.example.com/v1
    ITSBOB_API_WEATHER_KEY_ENV=WEATHER_API_KEY
    ITSBOB_API_WEATHER_AUTH=query
    ITSBOB_API_WEATHER_QUERY_PARAM=appid

``auth`` is ``bearer`` (default), ``header``, ``query``, or ``none``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["ApiSpec", "ApiCatalog", "http_tools"]

MAX_RESPONSE_BYTES = 200_000
DEFAULT_TIMEOUT = 30.0


@dataclass
class ApiSpec:
    """One configured API. ``key_env`` names the variable; the value is never stored here."""

    name: str
    base_url: str
    key_env: str = ""
    auth: str = "bearer"  # bearer | header | query | none
    header_name: str = "Authorization"
    query_param: str = "api_key"
    description: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT

    def api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if env is None else env
        return (env.get(self.key_env, "").strip() or None) if self.key_env else None

    def is_configured(self, env: Mapping[str, str] | None = None) -> bool:
        return self.auth == "none" or self.api_key(env) is not None

    def build(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Full URL and headers, credential attached. Never logged, never returned to the model."""
        query = dict(params or {})
        headers = {"Accept": "application/json", **self.headers}
        key = self.api_key(env)

        if self.auth != "none" and not key:
            raise ToolError(
                f"API {self.name!r} needs {self.key_env}, which is not set. "
                f"Add it to .env and restart."
            )
        if self.auth == "bearer":
            headers[self.header_name] = f"Bearer {key}"
        elif self.auth == "header":
            headers[self.header_name] = key or ""
        elif self.auth == "query":
            query[self.query_param] = key

        url = f"{self.base_url.rstrip('/')}/{str(path).lstrip('/')}" if path else self.base_url
        if query:
            joiner = "&" if urllib.parse.urlparse(url).query else "?"
            url = f"{url}{joiner}{urllib.parse.urlencode(query, doseq=True)}"
        return url, headers

    def describe(self, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "auth": self.auth,
            "key_env": self.key_env,
            "configured": self.is_configured(env),
            "description": self.description,
        }


class ApiCatalog:
    """The set of APIs the agent may call by name."""

    def __init__(self, specs: Mapping[str, ApiSpec] | None = None) -> None:
        self._specs: dict[str, ApiSpec] = dict(specs or {})

    def register(self, spec: ApiSpec) -> ApiSpec:
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ApiSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def describe(self, env: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        return [self._specs[name].describe(env) for name in self.names()]

    def render_for_prompt(self, env: Mapping[str, str] | None = None) -> str:
        rows = []
        for name in self.names():
            spec = self._specs[name]
            state = "" if spec.is_configured(env) else "  [NOT CONFIGURED — do not call]"
            rows.append(f"- {name}: {spec.description or spec.base_url}{state}")
        return "\n".join(rows) or "- (no APIs configured)"

    def __len__(self) -> int:
        return len(self._specs)

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "ApiCatalog":
        p = Path(path).expanduser()
        if not p.is_file():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ToolError(f"{p} is not valid JSON: {exc}") from exc
        catalog = cls()
        for name, body in (raw or {}).items():
            if not isinstance(body, dict) or not body.get("base_url"):
                continue
            catalog.register(
                ApiSpec(
                    name=str(name),
                    base_url=str(body["base_url"]),
                    key_env=str(body.get("key_env", "")),
                    auth=str(body.get("auth", "bearer")).lower(),
                    header_name=str(body.get("header_name", "Authorization")),
                    query_param=str(body.get("query_param", "api_key")),
                    description=str(body.get("description", "")),
                    headers=dict(body.get("headers") or {}),
                    timeout=float(body.get("timeout", DEFAULT_TIMEOUT)),
                )
            )
        return catalog

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, path: str | Path | None = None) -> "ApiCatalog":
        """File first, then ``ITSBOB_API_<NAME>_*`` variables layered on top."""
        env = os.environ if env is None else env
        config = path or env.get("ITSBOB_API_CONFIG", "").strip() or "apis.json"
        catalog = cls.from_file(config)

        prefix = "ITSBOB_API_"
        names = {
            key[len(prefix) :].rsplit("_BASE", 1)[0].lower()
            for key in env
            if key.startswith(prefix) and key.endswith("_BASE")
        }
        for name in names:
            def _get(field: str, default: str = "", _upper: str = name.upper()) -> str:
                # _upper is bound at definition time: a closure over the loop
                # variable would read whatever the last iteration left behind
                # if this were ever called after the loop.
                return str(env.get(f"{prefix}{_upper}_{field}", default)).strip()

            catalog.register(
                ApiSpec(
                    name=name,
                    base_url=_get("BASE"),
                    key_env=_get("KEY_ENV"),
                    auth=(_get("AUTH") or "bearer").lower(),
                    header_name=_get("HEADER_NAME") or "Authorization",
                    query_param=_get("QUERY_PARAM") or "api_key",
                    description=_get("DESCRIPTION"),
                )
            )
        return catalog


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    body: Any = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, str, dict[str, str]]:
    data = None
    headers = dict(headers or {})
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        else:
            data = str(body).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1).decode("utf-8", errors="replace")
            return response.status, payload, dict(response.headers)
    except urllib.error.HTTPError as exc:
        # A 4xx/5xx body is usually the most useful thing in the whole
        # exchange ("invalid key", "rate limited, retry in 30s") — surface it
        # rather than only the status line.
        payload = exc.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
        return exc.code, payload, dict(exc.headers or {})
    except urllib.error.URLError as exc:
        raise ToolError(f"could not reach {urllib.parse.urlparse(url).netloc}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ToolError(f"request to {urllib.parse.urlparse(url).netloc} timed out after {timeout}s") from exc


def _summarize(status: int, payload: str, url: str) -> ToolResult:
    truncated = len(payload) > MAX_RESPONSE_BYTES
    if truncated:
        payload = payload[:MAX_RESPONSE_BYTES]
    parsed: Any = None
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        pass

    body = json.dumps(parsed, indent=2)[:MAX_RESPONSE_BYTES] if parsed is not None else payload
    if truncated:
        body += "\n… [response truncated]"
    ok = 200 <= status < 300
    safe_url = urllib.parse.urlparse(url)._replace(query="").geturl()
    return ToolResult(
        ok=ok,
        output=f"HTTP {status}\n{body}" if ok else f"HTTP {status} from {safe_url}\n{body}",
        error=None if ok else f"HTTP {status}",
        data={"status": status, "json": parsed, "url": safe_url},
    )


def _http_request(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = params["url"]
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ToolError(f"only http/https URLs are allowed, got {url!r}")
    status, payload, _ = _request(
        url,
        method=params.get("method", "GET"),
        headers=params.get("headers") or {},
        body=params.get("body"),
        timeout=float(params.get("timeout", DEFAULT_TIMEOUT)),
    )
    return _summarize(status, payload, url)


def _call_api(catalog: ApiCatalog):
    def run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = params["api"]
        spec = catalog.get(name)
        if spec is None:
            raise ToolError(
                f"no API named {name!r}. Configured: {', '.join(catalog.names()) or '(none)'}"
            )
        url, headers = spec.build(
            params.get("path", ""), params=params.get("params") or {}, env=ctx.env or os.environ
        )
        started = time.perf_counter()
        status, payload, _ = _request(
            url,
            method=params.get("method", "GET"),
            headers=headers,
            body=params.get("body"),
            timeout=spec.timeout,
        )
        result = _summarize(status, payload, url)
        result.data["api"] = name
        result.data["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result

    return run


def http_tools(catalog: ApiCatalog | None = None) -> list[Tool]:
    catalog = catalog if catalog is not None else ApiCatalog()
    tools = [
        Tool(
            name="http_request",
            description="Fetch an arbitrary http(s) URL. Returns the status and body.",
            run=_http_request,
            risk=Risk.NETWORK,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "description": "GET (default), POST, PUT, DELETE."},
                    "headers": {"type": "object"},
                    "body": {"type": "object", "description": "JSON body for POST/PUT."},
                    "timeout": {"type": "number"},
                },
                "required": ["url"],
            },
        )
    ]
    if len(catalog):
        tools.append(
            Tool(
                name="call_api",
                description=(
                    "Call a configured API by name. Credentials are attached automatically — "
                    "never put a key in the parameters. Available: "
                    + (", ".join(catalog.names()) or "(none)")
                ),
                run=_call_api(catalog),
                risk=Risk.NETWORK,
                parameters={
                    "type": "object",
                    "properties": {
                        "api": {"type": "string", "description": f"One of: {', '.join(catalog.names())}"},
                        "path": {"type": "string", "description": "Path appended to the API's base URL."},
                        "method": {"type": "string"},
                        "params": {"type": "object", "description": "Query string parameters."},
                        "body": {"type": "object"},
                    },
                    "required": ["api"],
                },
            )
        )
    return tools
