import mss
import mss.tools
from PIL import Image, ImageDraw
import io
import base64
import re
import requests
import pyautogui
from typing import Dict, Any, Tuple, Optional, List
import uiautomation as auto

OLLAMA_BASE_URL = "http://localhost:11434"
VISION_MODEL = "gemma3:4b"

# Bound worst-case UI Automation hangs against unresponsive/foreign providers (e.g. some
# Electron/Chromium windows) so a single perception call can never stall the ReAct loop for long.
auto.SetGlobalSearchTimeout(1.5)


class ScreenVision:
    """Captures continuous screen snapshots, extracts real browser tab names, cursor element targeting,
    UI Automation element trees, and gemma3:4b multimodal vision fallback for elements the accessibility
    tree can't describe (canvas/image-only controls, unread badges, icon-only buttons, etc.)."""

    def __init__(self):
        self.last_captured_image: Optional[Image.Image] = None
        self.last_captured_base64: Optional[str] = None

    def capture_screen_pil(self, monitor_index: int = 1) -> Optional[Image.Image]:
        """Captures primary monitor safely returning a PIL Image."""
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                target_mon = monitors[monitor_index] if len(monitors) > monitor_index else monitors[0]
                sct_img = sct.grab(target_mon)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                self.last_captured_image = img
                return img
        except Exception:
            return self.last_captured_image

    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Returns primary screen width and height."""
        try:
            with mss.mss() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                return mon["width"], mon["height"]
        except Exception:
            return 1920, 1080

    def capture_base64_jpeg(self, quality: int = 50, scale: float = 0.3) -> Optional[str]:
        """Captures desktop screen safely and returns base64 JPEG data URL."""
        img = self.capture_screen_pil()
        if img is None:
            return self.last_captured_base64
        try:
            if scale != 1.0:
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=quality)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            res = f"data:image/jpeg;base64,{img_str}"
            self.last_captured_base64 = res
            return res
        except Exception:
            return self.last_captured_base64

    def get_element_at_cursor(self) -> Dict[str, Any]:
        """Inspects and returns the exact UI element underneath current mouse cursor (x, y)."""
        cx, cy = pyautogui.position()
        try:
            control = auto.ControlFromPoint(cx, cy)
            name = control.Name if control else ""
            control_type = control.ControlTypeName if control else ""
            return {
                "name": name.strip() if name else "Target Item",
                "type": control_type,
                "x": cx,
                "y": cy
            }
        except Exception:
            return {"name": "Target Item", "type": "Control", "x": cx, "y": cy}

    def get_open_browser_tabs(self) -> List[str]:
        """Scans active foreground browser window using UI Automation to extract real open tab names."""
        tabs = []
        try:
            root = auto.GetForegroundControl()
            if not root:
                root = auto.GetRootControl()

            for control, depth in auto.WalkControl(root, maxDepth=6):
                try:
                    if control.ControlTypeName in ["TabItemControl", "HeaderItemControl"]:
                        name = control.Name
                        if name and name.strip() and len(name.strip()) > 1 and name.strip() not in tabs:
                            tabs.append(name.strip())
                except Exception:
                    pass
        except Exception as e:
            print(f"[Tab Extraction Error]: {e}")
        return tabs

    def find_visible_ui_elements(self, max_depth: int = 8, max_elements: int = 400) -> List[Dict[str, Any]]:
        """Walks the active window's UI Automation tree to extract visible, named, on-screen elements
        with their bounding boxes and center coordinates for instant (x, y) targeting."""
        elements = []
        try:
            root = auto.GetForegroundControl()
            if not root:
                root = auto.GetRootControl()

            for control, depth in auto.WalkControl(root, maxDepth=max_depth):
                if len(elements) >= max_elements:
                    break
                try:
                    name = control.Name
                    if not name or not name.strip():
                        continue
                    rect = control.BoundingRectangle
                    if rect.width() <= 0 or rect.height() <= 0:
                        continue
                    center_x = (rect.left + rect.right) // 2
                    center_y = (rect.top + rect.bottom) // 2
                    elements.append({
                        "name": name.strip(),
                        "type": control.ControlTypeName,
                        "x": center_x,
                        "y": center_y,
                        "left": rect.left,
                        "top": rect.top,
                        "width": rect.width(),
                        "height": rect.height()
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"[UIAutomation Walk Error]: {e}")
        return elements

    def find_nth_element(
        self,
        elements: Optional[List[Dict[str, Any]]] = None,
        name_contains: str = "",
        control_type: str = "",
        ordinal: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Resolves ordinal references like '3rd profile' or '2nd tab' by filtering elements
        (optionally by control type and/or a name substring) sorted in reading order (top-to-bottom,
        left-to-right) and picking the Nth (1-indexed) match."""
        pool = elements if elements is not None else self.find_visible_ui_elements()
        if control_type:
            pool = [e for e in pool if control_type.lower() in e["type"].lower()]
        if name_contains:
            pool = [e for e in pool if name_contains.lower() in e["name"].lower()]
        if not pool:
            return None
        pool = sorted(pool, key=lambda e: (e["top"], e["left"]))
        idx = max(0, min((ordinal - 1) if ordinal and ordinal >= 1 else 0, len(pool) - 1))
        return pool[idx]

    def find_element_coords(self, target_text: str) -> Optional[Tuple[int, int]]:
        """Locates exact screen center coordinates (x, y) of element matching target_text."""
        target = target_text.lower().strip()
        elements = self.find_visible_ui_elements(max_depth=5)

        for el in elements:
            if el["name"].lower() == target:
                return el["x"], el["y"]

        for el in elements:
            if target in el["name"].lower():
                return el["x"], el["y"]

        return None

    def see_element(self, target_text: str) -> Tuple[bool, str]:
        """Checks if element target_text is visible on screen right now."""
        coords = self.find_element_coords(target_text)
        if coords:
            return True, f"Yes, I see '{target_text}' on your screen at position ({coords[0]}, {coords[1]})."

        return False, f"I don't see '{target_text}' explicitly on screen right now."

    # ------------------------------------------------------------------
    # gemma3:4b multimodal vision fallback — used whenever the UI Automation
    # accessibility tree can't describe what's on screen (canvas-rendered web
    # UI, icon-only buttons, unread-message badges, chat avatars, etc.)
    # ------------------------------------------------------------------

    def ask_vision(
        self,
        question: str,
        model: str = VISION_MODEL,
        scale: float = 0.5,
        quality: int = 60,
        timeout: int = 45
    ) -> Optional[str]:
        """Sends the current screen to the local gemma3 vision model with a natural-language
        question and returns its text answer. Used for visual judgment calls the accessibility
        tree cannot make (e.g. 'which chat row shows unread messages?').

        The answer gets spoken aloud by TTS, not read on screen — gemma3 defaults to writing a
        structured markdown report (headers, bullet points) when asked "what's on screen", which
        reads out loud as unintelligible noise. The formatting instruction below is appended to every
        call so this is fixed at the source rather than depending on every caller to remember it."""
        img = self.capture_screen_pil()
        if img is None:
            return None
        try:
            if scale != 1.0:
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            spoken_question = (
                f"{question}\n\nAnswer in 1-3 short spoken sentences, like you're glancing over someone's "
                f"shoulder and telling them out loud. Plain conversational text only — no markdown, no "
                f"headers, no bullet points, no bold text, no numbered lists."
            )
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": spoken_question, "images": [b64]}],
                "stream": False,
                "options": {"temperature": 0.1}
            }
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
            if resp.status_code != 200:
                return None
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[Vision Error]: {e}")
            return None

    @staticmethod
    def _col_label(idx: int) -> str:
        """Spreadsheet-style column label: 0->A, 1->B, ..., 25->Z, 26->AA, ..."""
        label = ""
        idx += 1
        while idx > 0:
            idx, rem = divmod(idx - 1, 26)
            label = chr(65 + rem) + label
        return label

    def capture_grid_overlay_base64(
        self,
        target_width: int = 1024,
        cell_px: int = 80,
        quality: int = 70
    ) -> Tuple[Optional[str], Dict[str, Tuple[int, int]]]:
        """Captures the screen, draws a labeled reference grid (A1, B1, ... ) on top of it, and
        returns (base64 jpeg, {label: real_screen_pixel_center}). Small vision-language models are
        unreliable at regressing raw pixel coordinates directly, but are much more accurate at
        reading a printed grid-cell label off an image — this grid is the coordinate-grounding
        mechanism for locate_via_vision()."""
        img = self.capture_screen_pil()
        if img is None:
            return None, {}
        try:
            orig_w, orig_h = img.size
            scale = target_width / orig_w
            resized = img.resize((target_width, int(orig_h * scale)), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(resized)
            w, h = resized.size
            cols = max(1, w // cell_px)
            rows = max(1, h // cell_px)
            cell_w = w / cols
            cell_h = h / rows

            grid_meta: Dict[str, Tuple[int, int]] = {}
            for r in range(rows):
                for c in range(cols):
                    label = self._col_label(c) + str(r + 1)
                    x0, y0 = c * cell_w, r * cell_h
                    x1, y1 = x0 + cell_w, y0 + cell_h
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=1)
                    draw.text((x0 + 2, y0 + 2), label, fill=(255, 255, 0))
                    grid_meta[label] = (int((x0 + cell_w / 2) / scale), int((y0 + cell_h / 2) / scale))

            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return b64, grid_meta
        except Exception as e:
            print(f"[Grid Overlay Error]: {e}")
            return None, {}

    def locate_via_vision(self, description: str, model: str = VISION_MODEL, timeout: int = 45) -> Optional[Tuple[int, int]]:
        """Asks gemma3 to point at an on-screen target described in natural language (e.g. 'the 3rd
        profile avatar', 'the red record button') using the grid-overlay grounding trick, and returns
        the real screen pixel coordinates of the matching cell, or None if not found."""
        b64, grid_meta = self.capture_grid_overlay_base64()
        if not b64 or not grid_meta:
            return None
        prompt = (
            "This screenshot has a red reference grid drawn over it with yellow cell labels "
            "like A1, B3, C10 in the top-left corner of each cell.\n"
            f"Find this on screen: \"{description}\".\n"
            "Reply with ONLY the single grid cell label that contains it, nothing else (example: C4). "
            "If it is not visible anywhere on screen, reply with exactly: NONE"
        )
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False,
                "options": {"temperature": 0.0}
            }
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
            if resp.status_code != 200:
                return None
            answer = resp.json().get("message", {}).get("content", "").strip().upper()
            if "NONE" in answer and len(answer) < 8:
                return None
            match = re.search(r'\b([A-Z]{1,2}\d{1,3})\b', answer)
            if not match:
                return None
            return grid_meta.get(match.group(1))
        except Exception as e:
            print(f"[Vision Locate Error]: {e}")
            return None
