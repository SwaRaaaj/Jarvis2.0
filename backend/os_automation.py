import pyautogui
import os
import subprocess
import webbrowser
import time
import datetime
import urllib.parse
from typing import Dict, Any, Optional, Tuple

try:
    import win32gui
    import win32con
    import win32process
    import win32api
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = True

# Real, always-valid inbox/home URLs only. Chat platforms (Instagram, WhatsApp, Telegram) do NOT
# support deep-linking directly to an arbitrary contact's thread by username — any URL that pretends
# to do that is a guess that will 404 or land on the wrong page. The correct approach (used by
# OllamaEngine's ReAct loop) is: open the real inbox below, then visually/structurally locate the
# specific contact on screen and click it. Messenger's m.me/<username> is a real, documented
# exception, so it's handled separately in open_url rather than listed here.
SOCIAL_INBOX_URLS = {
    "instagram": "https://www.instagram.com/direct/inbox/",
    "whatsapp": "https://web.whatsapp.com/",
    "telegram": "https://web.telegram.org/a/",
    "messenger": "https://www.messenger.com/",
}

# Shared with OllamaEngine's deterministic "open/launch <app>" fast path, so a request for a known
# app never has to go through the LLM ReAct loop at all (small local models can and do hallucinate
# unrelated tool calls even for obvious single-tool requests — see launch_app's docstring).
# Chromium keeps its renderer accessibility tree switched off until something asks for it, and
# WM_GETOBJECT alone does not wake it (measured). Without the flag, UI Automation sees the browser
# frame — tabs, toolbar, Minimize — and *nothing at all* of the page, so clicking anything on a web
# page has to fall back to visual guessing, which lands ~80px off.
#
# With the flag, page content appears in the tree with exact bounding rectangles, which makes
# clicking pixel-perfect and free. Verified: a page's heading and links become real, addressable
# controls. Only applies to browsers JARVIS launches itself.
CHROMIUM_A11Y_FLAG = "--force-renderer-accessibility"

APP_ALIASES = {
    "vs code": "code",
    "vscode": "code",
    "code": "code",
    "visual studio code": "code",
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "calculator": "calc.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "terminal": "powershell.exe",
    "powershell": "powershell.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "chrome": "start chrome",
    "google chrome": "start chrome",
    "edge": "start msedge",
    "spotify": "start spotify",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
    "settings": "start ms-settings:",
    "word": "start winword",
    "excel": "start excel",
}

# Also shared with the deterministic fast path, for the extremely common "open <well-known site>"
# request — real, stable home-page URLs only, distinct from SOCIAL_INBOX_URLS which need the ReAct
# loop's on-screen contact lookup afterward. This exists because, without it, "open youtube" went
# through the full ReAct loop and the model invented an unrequested "now search for a video" step
# after the real task (just opening the site) was already done.
WEBSITE_ALIASES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "github": "https://github.com",
    "amazon": "https://www.amazon.com",
    "netflix": "https://www.netflix.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://www.twitter.com",
    "x": "https://www.x.com",
    "linkedin": "https://www.linkedin.com",
}


class OSAutomation:
    """Executes universal OS GUI actions, cursor tracking, element clicking, batch tab closing, typing, and desktop automation."""

    @staticmethod
    def get_time_and_date() -> Dict[str, Any]:
        """Returns current system time, day of week, and date."""
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        date_str = now.strftime("%A, %B %d, %Y")
        return {
            "status": "success",
            "time": time_str,
            "date": date_str,
            "formatted": f"The current time is {time_str} on {date_str}."
        }

    @staticmethod
    def get_cursor_position() -> Tuple[int, int]:
        """Gets current mouse cursor screen coordinates (x, y)."""
        return pyautogui.position()

    @staticmethod
    def switch_to_window(title_substring: str) -> Dict[str, Any]:
        """Brings an already-open window whose title contains title_substring to the foreground —
        a deterministic, reliable way to 'switch to X' rather than guessing via alt-tab or clicking."""
        if not HAS_WIN32:
            return {"status": "error", "message": "pywin32 is not available."}
        target = title_substring.lower().strip()
        if not target:
            return {"status": "error", "message": "title_substring is required."}
        matches = []

        def _enum_handler(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and target in title.lower():
                    matches.append((hwnd, title))

        try:
            win32gui.EnumWindows(_enum_handler, None)
            if not matches:
                return {"status": "error", "message": f"No open window found matching '{title_substring}'."}
            hwnd, title = matches[0]
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                # Windows blocks SetForegroundWindow from a process that doesn't already hold input
                # focus (this is exactly the situation a hands-free, voice-triggered background thread
                # is normally in — confirmed by live testing, not theoretical). Standard workaround:
                # temporarily attach our thread's input state to whichever thread currently owns
                # foreground focus, which grants the permission for the duration of the attachment.
                current_thread = win32api.GetCurrentThreadId()
                fg_hwnd = win32gui.GetForegroundWindow()
                fg_thread = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0
                attached = False
                try:
                    if fg_thread and fg_thread != current_thread:
                        win32process.AttachThreadInput(fg_thread, current_thread, True)
                        attached = True
                    win32gui.BringWindowToTop(hwnd)
                    win32gui.SetForegroundWindow(hwnd)
                finally:
                    if attached:
                        win32process.AttachThreadInput(fg_thread, current_thread, False)

            return {"status": "success", "action": "switch_to_window", "title": title}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def click_at_cursor() -> Dict[str, Any]:
        """Clicks at current mouse cursor position."""
        x, y = pyautogui.position()
        pyautogui.click(x, y)
        return {"status": "success", "action": "click_at_cursor", "x": x, "y": y}

    @staticmethod
    def close_active_tab() -> Dict[str, Any]:
        """Closes active browser tab using Ctrl+W."""
        try:
            pyautogui.hotkey('ctrl', 'w')
            return {"status": "success", "action": "close_active_tab"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def close_all_browser_tabs(max_tabs: int = 15) -> Dict[str, Any]:
        """Batch closes all open browser tabs at once using Ctrl+W loop."""
        try:
            for _ in range(max_tabs):
                pyautogui.hotkey('ctrl', 'w')
                time.sleep(0.08)
            return {"status": "success", "action": "close_all_browser_tabs", "closed_count": max_tabs}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def close_active_window() -> Dict[str, Any]:
        """Closes current foreground window using Alt+F4."""
        try:
            pyautogui.hotkey('alt', 'f4')
            return {"status": "success", "action": "close_active_window"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def focus_search_bar_and_type(text: str, press_enter: bool = True) -> Dict[str, Any]:
        """Focuses browser or app address/search bar using Ctrl+L and types target text."""
        try:
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.15)
            pyautogui.write(text, interval=0.01)
            if press_enter:
                pyautogui.press('enter')
            return {"status": "success", "action": "focus_search_bar_and_type", "text": text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def type_text(text: str, press_enter: bool = True) -> Dict[str, Any]:
        """Types text into current focused input box or active window."""
        try:
            pyautogui.write(text, interval=0.01)
            if press_enter:
                time.sleep(0.1)
                pyautogui.press('enter')
            return {"status": "success", "action": "type_text", "text": text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def click(x: int, y: int, button: str = "left", clicks: int = 1) -> Dict[str, Any]:
        """Clicks specified screen pixel coordinate."""
        try:
            screen_w, screen_h = pyautogui.size()
            x = max(0, min(x, screen_w - 1))
            y = max(0, min(y, screen_h - 1))
            pyautogui.click(x=x, y=y, clicks=clicks, button=button)
            return {"status": "success", "action": "click", "x": x, "y": y}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def key_combination(keys: str) -> Dict[str, Any]:
        """Executes key combination e.g. 'ctrl+c', 'ctrl+w', 'alt+tab', 'enter', 'escape', 'space'."""
        try:
            key_list = [k.strip().lower() for k in keys.split('+')]
            if len(key_list) == 1:
                pyautogui.press(key_list[0])
            else:
                pyautogui.hotkey(*key_list)
            return {"status": "success", "action": "key_combination", "keys": keys}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def scroll(direction: str = "down", amount: int = 5) -> Dict[str, Any]:
        """Scrolls mouse wheel up or down."""
        try:
            clicks = -amount if direction.lower() == "down" else amount
            pyautogui.scroll(clicks * 100)
            return {"status": "success", "action": "scroll", "direction": direction}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def open_url(url: str) -> Dict[str, Any]:
        """Opens URL in web browser."""
        try:
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
            webbrowser.open(url)
            return {"status": "success", "action": "open_url", "url": url}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def open_social_inbox(platform: str) -> Dict[str, Any]:
        """Opens the real inbox/home page of a supported chat platform (Instagram, WhatsApp, Telegram,
        Messenger). Never guesses a per-contact URL — the calling agent must then visually/structurally
        locate the specific contact on screen (see ScreenVision.find_visible_ui_elements / ask_vision)
        and click it, exactly like a human would."""
        try:
            key = platform.lower().strip()
            url = SOCIAL_INBOX_URLS.get(key)
            if not url:
                return {
                    "status": "error",
                    "message": f"Unsupported platform '{platform}'. Supported: {list(SOCIAL_INBOX_URLS.keys())}"
                }
            webbrowser.open(url)
            return {"status": "success", "action": "open_social_inbox", "platform": key, "url": url}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def move_mouse(x: int, y: int) -> Dict[str, Any]:
        """Moves the mouse cursor to a screen coordinate without clicking (e.g. to trigger hover UI)."""
        try:
            screen_w, screen_h = pyautogui.size()
            x = max(0, min(int(x), screen_w - 1))
            y = max(0, min(int(y), screen_h - 1))
            pyautogui.moveTo(x, y, duration=0.1)
            return {"status": "success", "action": "move_mouse", "x": x, "y": y}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def search_youtube(query: str) -> Dict[str, Any]:
        """Searches YouTube for video query."""
        try:
            encoded = urllib.parse.quote(query)
            webbrowser.open(f"https://www.youtube.com/results?search_query={encoded}")
            return {"status": "success", "action": "search_youtube", "query": query}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def search_google(query: str) -> Dict[str, Any]:
        """Searches Google for target query."""
        try:
            encoded = urllib.parse.quote(query)
            webbrowser.open(f"https://www.google.com/search?q={encoded}")
            return {"status": "success", "action": "search_google", "query": query}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def launch_app(app_name: str) -> Dict[str, Any]:
        """Launches an application by name, verifying it actually started rather than assuming success."""
        try:
            clean_name = app_name.lower().strip()
            target = APP_ALIASES.get(clean_name, clean_name)

            # Launch Chromium browsers with accessibility on, so page content is clickable by name
            # and coordinate rather than by visual guesswork. Harmless if the browser is already
            # running (the flag only applies to a cold start).
            if target in ("start chrome", "start msedge"):
                target = f"{target} {CHROMIUM_A11Y_FLAG}"

            if target.startswith("start "):
                # "start" is an async shell launcher with no reliable exit-code signal — this path is
                # only used for known-good aliases above, so we trust it.
                os.system(target)
                return {"status": "success", "action": "launch_app", "app": app_name}

            proc = subprocess.Popen(target, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.4)
            if proc.poll() is not None and proc.returncode != 0:
                stderr = proc.stderr.read().decode(errors="ignore").strip() if proc.stderr else ""
                return {"status": "error", "action": "launch_app", "app": app_name, "message": stderr[:200] or f"'{app_name}' could not be started."}
            return {"status": "success", "action": "launch_app", "app": app_name}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def run_shell_command(command: str) -> Dict[str, Any]:
        """Runs shell command safely."""
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
            return {
                "status": "success" if res.returncode == 0 else "warning",
                "stdout": res.stdout[:1500],
                "stderr": res.stderr[:500]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
