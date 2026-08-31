"""Driving a real browser, two ways, because neither works everywhere.

Some things are only reachable through a browser with a logged-in session: a
chat site with no API, a timeline that requires an account. Two mechanisms can
do it, and they fail in opposite directions, so both are here and the better one
is tried first.

**Playwright with a persistent profile** is the one to want. It talks to the
page through real selectors, runs headless, keeps its own cookie jar, and does
not touch the desktop. Costs a dependency and a browser download.

**xdotool and the clipboard** needs neither, and works with the browser already
open — including one already signed in. It is also genuinely intrusive: it
steals focus, types into whatever is focused, and overwrites the clipboard. If
somebody is using the machine it will type into their window. So it is the
fallback, it says so, and it refuses to run when it cannot find its own window
rather than typing into someone else's.

The extraction problem is the same for both and worth stating. ``ctrl+a ctrl+c``
captures the whole page — navigation, buttons, the lot — and the answer has to
be found inside it. The Playwright path avoids that entirely by reading the
message elements; the xdotool path brackets the reply with a marker it typed
itself, which is more reliable than hunting for the question text and far more
reliable than taking everything after the last occurrence of something.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "BrowserError",
    "PlaywrightSession",
    "XdotoolSession",
    "available",
    "chromium",
    "profile_dir",
]


class BrowserError(RuntimeError):
    """A browser action that could not be completed."""


def profile_dir(env: Any = None) -> Path:
    """Where the automated browser keeps its cookies.

    Its own profile, never the one a person uses: an automated session that
    shares a profile can log itself out of the real one, and a crash takes the
    person's tabs with it.
    """
    env = os.environ if env is None else env
    home = str(env.get("ITSBOB_HOME", "")).strip()
    root = Path(home).expanduser() if home else Path.home() / ".itsbob"
    return Path(env.get("ITSBOB_BROWSER_PROFILE", "").strip() or root / "browser-profile")


def _have(command: str) -> bool:
    return shutil.which(command) is not None


#: What a system Chromium can be called, most-standard first.
_CHROMIUM_NAMES = (
    "chromium",
    "chromium-browser",
    "google-chrome-stable",
    "google-chrome",
    "chrome",
)

#: Where a downloaded Chromium sits inside one of Playwright's version folders.
_PLAYWRIGHT_BINARIES = (
    "chrome-linux/chrome",
    "chrome-linux/headless_shell",
    "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    "chrome-win/chrome.exe",
)


def _playwright_root(env: Any) -> Path:
    """Where Playwright keeps the browsers it downloads."""
    configured = str(env.get("PLAYWRIGHT_BROWSERS_PATH", "")).strip()
    if configured and configured != "0":
        return Path(configured).expanduser()
    if os.name == "nt":
        local = str(env.get("LOCALAPPDATA", "")).strip()
        return Path(local or Path.home() / "AppData/Local") / "ms-playwright"
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/ms-playwright"
    return Path.home() / ".cache/ms-playwright"


def _playwright_chromium(env: Any) -> str:
    """The Chromium Playwright downloaded for itself, if it ever did."""
    root = _playwright_root(env)
    if not root.is_dir():
        return ""
    # Binary kind first, version second. A machine often holds both a full
    # Chromium and a headless shell, and the shell cannot open a visible window
    # — so a full browser in an older folder still beats a shell in a newer one.
    # Reverse-sorted folders so the newest build of a given kind wins.
    folders = sorted(root.glob("chromium*"), reverse=True)
    for relative in _PLAYWRIGHT_BINARIES:
        for folder in folders:
            candidate = folder / relative
            if candidate.exists():
                return str(candidate)
    return ""


def chromium(env: Any = None) -> dict[str, str]:
    """A Chromium binary this machine can actually launch, and where it came from.

    Having the ``playwright`` package is not the same as having a browser: the
    package is a driver, and on its own it launches nothing. Three places a
    browser can come from, in the order they should be preferred:

    ``configured``
        ``ITSBOB_CHROMIUM``, when somebody has pointed us at a specific build.
    ``playwright``
        The one ``playwright install chromium`` downloads. Best behaved, because
        it is the build Playwright was tested against.
    ``system``
        The Chromium already on the machine. Saves a 150MB download and is
        usually what somebody means by "we already have chromium installed".

    Returns ``{"path": ..., "source": ...}``, both empty when there is nothing
    to drive.
    """
    env = os.environ if env is None else env

    explicit = str(env.get("ITSBOB_CHROMIUM", "")).strip()
    if explicit:
        resolved = explicit if Path(explicit).exists() else (shutil.which(explicit) or "")
        # A path somebody set by hand and got wrong is worth reporting as set
        # and broken, rather than silently falling through to something else.
        return {"path": resolved, "source": "configured" if resolved else "configured-missing"}

    downloaded = _playwright_chromium(env)
    if downloaded:
        return {"path": downloaded, "source": "playwright"}

    for name in _CHROMIUM_NAMES:
        found = shutil.which(name)
        if found:
            return {"path": found, "source": "system"}

    return {"path": "", "source": ""}


def available(env: Any = None) -> dict[str, Any]:
    """Which mechanisms this machine can actually use, and why not otherwise."""
    env = os.environ if env is None else env
    try:
        import playwright  # noqa: F401, PLC0415

        package_ok = True
    except ImportError:
        package_ok = False

    browser = chromium(env)
    playwright_ok = package_ok and bool(browser["path"])
    if not package_ok:
        playwright_why = "not installed — pip install -e '.[browser]'"
    elif browser["source"] == "configured-missing":
        playwright_why = (
            f"ITSBOB_CHROMIUM points at {env.get('ITSBOB_CHROMIUM', '').strip()!r}, "
            "which is not there"
        )
    elif not browser["path"]:
        playwright_why = (
            "installed, but there is no chromium for it to drive — run "
            "'playwright install chromium', or set ITSBOB_CHROMIUM to one you have"
        )
    else:
        playwright_why = f"ready, driving the {browser['source']} chromium"

    display = bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    xdotool_ok = _have("xdotool") and (_have("xclip") or _have("xsel")) and display
    if not display:
        xdotool_why = "no display attached"
    elif not _have("xdotool"):
        xdotool_why = "xdotool not installed"
    elif not (_have("xclip") or _have("xsel")):
        xdotool_why = "neither xclip nor xsel installed"
    else:
        xdotool_why = "ready (intrusive: steals focus and the clipboard)"

    return {
        "playwright": {"ready": playwright_ok, "why": playwright_why},
        "xdotool": {"ready": xdotool_ok, "why": xdotool_why},
        "preferred": "playwright" if playwright_ok else ("xdotool" if xdotool_ok else None),
        "profile": str(profile_dir(env)),
        "chromium": browser,
    }


# -- the good path ---------------------------------------------------------


@dataclass
class PlaywrightSession:
    """A headless-capable browser with its own persistent profile."""

    headless: bool = True
    timeout_ms: int = 60_000
    profile: Path = field(default_factory=profile_dir)
    #: Chromium shipped with the machine, when Playwright's own is absent.
    executable: str = ""

    def _launch(self, playwright: Any) -> Any:
        self.profile.mkdir(parents=True, exist_ok=True)
        options: dict[str, Any] = {
            "user_data_dir": str(self.profile),
            "headless": self.headless,
            "args": ["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        }
        # Playwright finds its own download unaided; anything else has to be
        # handed to it, or it looks in the one place it has not been installed.
        executable = self.executable
        if not executable:
            found = chromium()
            executable = found["path"] if found["source"] != "playwright" else ""
        if executable:
            options["executable_path"] = executable
        return playwright.chromium.launch_persistent_context(**options)

    def fetch_text(self, url: str, *, wait_ms: int = 2500, scrolls: int = 0) -> str:
        """The readable text of a page, after JavaScript has run."""
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        with sync_playwright() as playwright:
            context = self._launch(playwright)
            try:
                page = context.new_page()
                page.set_default_timeout(self.timeout_ms)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(wait_ms)
                for _ in range(max(0, scrolls)):
                    page.keyboard.press("End")
                    page.wait_for_timeout(1200)
                # innerText rather than textContent: it respects layout, so
                # hidden nav and script bodies do not come along.
                return str(page.evaluate("() => document.body.innerText") or "")
            finally:
                context.close()

    def ask(
        self,
        url: str,
        prompt: str,
        *,
        input_selectors: Sequence[str],
        reply_selector: str,
        settle_ms: int = 4000,
        max_wait_ms: int = 180_000,
    ) -> str:
        """Type a prompt into a chat page and return the reply that appears.

        Waits for the reply to *stop growing* rather than for a fixed number of
        seconds. A streaming answer arrives a token at a time, and a fixed sleep
        either truncates a long one or wastes a minute on a short one.
        """
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        with sync_playwright() as playwright:
            context = self._launch(playwright)
            try:
                page = context.new_page()
                page.set_default_timeout(self.timeout_ms)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)

                box = None
                for selector in input_selectors:
                    try:
                        box = page.wait_for_selector(selector, timeout=8000)
                        if box:
                            break
                    except Exception:  # noqa: BLE001 - try the next selector
                        continue
                if box is None:
                    raise BrowserError(
                        "could not find the message box — the site's markup has "
                        f"changed, or it is asking for a login. Tried: {list(input_selectors)}"
                    )

                before = len(page.query_selector_all(reply_selector))
                box.click()
                box.fill(prompt)
                page.keyboard.press("Enter")

                deadline = time.time() + max_wait_ms / 1000
                previous, stable_since = "", 0.0
                while time.time() < deadline:
                    page.wait_for_timeout(1500)
                    blocks = page.query_selector_all(reply_selector)
                    if len(blocks) <= before:
                        continue
                    current = str(blocks[-1].inner_text() or "")
                    if current and current == previous:
                        # Unchanged for one full settle window means finished.
                        if stable_since and (time.time() - stable_since) * 1000 >= settle_ms:
                            return current
                        stable_since = stable_since or time.time()
                    else:
                        previous, stable_since = current, 0.0
                if previous:
                    return previous
                raise BrowserError(
                    f"no reply appeared within {max_wait_ms // 1000}s "
                    f"(selector {reply_selector!r})"
                )
            finally:
                context.close()


# -- the fallback ----------------------------------------------------------


@dataclass
class XdotoolSession:
    """Drives the browser already on screen. Intrusive, and says so."""

    window_class: str = "chromium"
    browser_cmd: str = "chromium-browser"
    clipboard: str = ""
    type_delay_ms: int = 12
    launch_wait: float = 6.0
    page_wait: float = 8.0

    def __post_init__(self) -> None:
        if not self.clipboard:
            self.clipboard = "xclip" if _have("xclip") else "xsel"

    # -- primitives --------------------------------------------------------

    def _run(self, command: Sequence[str], *, timeout: float = 15.0) -> str:
        try:
            done = subprocess.run(  # noqa: S603 - fixed binaries, list form, no shell
                list(command), capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BrowserError(f"{command[0]}: {exc}") from exc
        if done.returncode != 0 and not done.stdout:
            raise BrowserError(f"{command[0]} failed: {(done.stderr or '').strip()[:200]}")
        return done.stdout

    def window(self) -> str:
        """The browser window id, launching the browser if there is none.

        Refuses rather than guessing: typing a prompt into whatever happens to
        be focused is how this mechanism does real damage.
        """
        for attempt in range(4):
            try:
                found = self._run(
                    ["xdotool", "search", "--onlyvisible", "--class", self.window_class]
                ).strip()
            except BrowserError:
                found = ""
            if found:
                return found.splitlines()[-1]
            if attempt == 0:
                subprocess.Popen(  # noqa: S603 - configured command, no shell
                    self.browser_cmd.split(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(self.launch_wait)
            else:
                time.sleep(2.0)
        raise BrowserError(
            f"no visible {self.window_class!r} window, and launching "
            f"{self.browser_cmd!r} did not produce one — refusing to type into "
            "whatever else is focused"
        )

    def focus(self, window_id: str) -> None:
        self._run(["xdotool", "windowactivate", "--sync", window_id])
        self._run(["xdotool", "windowraise", window_id])
        time.sleep(0.4)

    def navigate(self, url: str) -> None:
        self._run(["xdotool", "key", "ctrl+l"])
        time.sleep(0.3)
        self._run(["xdotool", "type", "--delay", "40", url])
        self._run(["xdotool", "key", "Return"])
        time.sleep(self.page_wait)

    def type_text(self, text: str) -> None:
        # Newlines would submit the form early, so they are flattened. A chat
        # box takes one paragraph anyway.
        self._run(
            ["xdotool", "type", "--delay", str(self.type_delay_ms), " ".join(text.split())],
            timeout=180.0,
        )

    def read_clipboard(self) -> str:
        if self.clipboard == "xsel":
            return self._run(["xsel", "--clipboard", "--output"], timeout=10)
        return self._run(["xclip", "-selection", "clipboard", "-o"], timeout=10)

    def copy_page(self) -> str:
        self._run(["xdotool", "key", "ctrl+a"])
        time.sleep(0.3)
        self._run(["xdotool", "key", "ctrl+c"])
        time.sleep(0.6)
        self._run(["xdotool", "key", "ctrl+shift+Home"])  # collapse the selection
        return self.read_clipboard()

    # -- the operations ----------------------------------------------------

    def fetch_text(self, url: str, *, scrolls: int = 0, scroll_pause: float = 1.8) -> str:
        self.focus(self.window())
        self._run(["xdotool", "key", "ctrl+t"])
        time.sleep(0.5)
        self.navigate(url)
        for _ in range(max(0, scrolls)):
            self._run(["xdotool", "key", "Page_Down"])
            time.sleep(scroll_pause)
        text = self.copy_page()
        self._run(["xdotool", "key", "ctrl+w"])  # close the tab we opened
        return text

    def ask(self, url: str, prompt: str, *, settle: float = 25.0) -> str:
        """Send a prompt and return only what came back.

        The reply is bracketed by a marker typed as part of the prompt. Taking
        "everything after the question" breaks the moment the site echoes the
        question, wraps it, or renders it twice; a unique token does not.
        """
        marker = f"[[{uuid.uuid4().hex[:8]}]]"
        self.focus(self.window())
        self.navigate(url)
        self.type_text(f"{prompt}\n\nBegin your reply with {marker} on its own line.")
        self._run(["xdotool", "key", "Return"])
        time.sleep(settle)

        page = self.copy_page()
        # The last occurrence: the first is the prompt being echoed back.
        cut = page.rfind(marker)
        if cut == -1:
            raise BrowserError(
                "the reply marker never appeared — the page may still be "
                "generating, or the prompt did not reach the message box"
            )
        return page[cut + len(marker) :].strip()
