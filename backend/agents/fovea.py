"""FOVEA — precision targeting.

Named for the part of the retina with the sharpest vision: RETINA sees the whole field, FOVEA
resolves the fine detail at one point.

The problem it solves is measurable. Visual location works by drawing a labelled grid over the
screen and asking the vision model which cell holds the target. On a 1920x1080 display those cells
are **160 x 154 real pixels**, so a click derived from one lands up to **80 pixels** from where it
should. A typical button is about 30 pixels across. The estimate is in the right neighbourhood and
still misses the target completely — which is exactly what "it can't click on any part easily"
feels like from the outside.

FOVEA takes that coarse estimate and sharpens it, cheapest technique first:

    1. SNAP TO CONTROL      ControlFromPoint at the estimate, click the true centre    0 model calls
    2. PROBE THE AREA       sample a small lattice for a plausible control nearby      0 model calls
    3. ZOOM AND RE-ASK      crop the region, upscale, finer grid, ask again            1 vision call

Rung 1 is the important one and it is free. `ControlFromPoint` queries the UI Automation provider
at a single coordinate, and routinely returns a control that a whole-tree walk never surfaced — so
a rough visual guess becomes an exact control centre without spending anything.

Rung 3 only runs when the accessibility layer genuinely has nothing there (canvas, video, custom-
rendered UI), and even then it cuts the quantisation error by roughly an order of magnitude by
giving the model the same grid over a much smaller area.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import Agent, AgentContext, LLMClient, normalize, text_score

# A control this large is a container the point happens to fall inside, not the thing to click.
MAX_SNAP_WIDTH = 600
MAX_SNAP_HEIGHT = 420

# How far from the estimate to probe when the exact point yields nothing useful. Kept tight: the
# coarse estimate is already within about one grid cell, so a wider search would drift onto
# neighbouring controls.
# Measured: widening this HURT. A 120px lattice found more plausible-but-wrong controls and
# tripled confidently-wrong snaps without improving the hit rate. Geometry alone cannot recover a
# large estimate error against small controls - that is what the zoom pass is for.
PROBE_OFFSETS = (0, 24, 48, 72)

# Confidence floor for accepting a snapped control whose name should resemble the target.
NAME_MATCH_FLOOR = 0.42


@dataclass
class RefinedTarget:
    """A sharpened click point."""

    x: int
    y: int
    method: str = "coarse"          # snap | probe | zoom | coarse
    confidence: float = 0.5
    snapped_to: str = ""
    control_type: str = ""
    moved_px: float = 0.0
    reason: str = ""

    @property
    def coords(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "method": self.method,
            "confidence": round(self.confidence, 3), "snapped_to": self.snapped_to,
            "control_type": self.control_type, "moved_px": round(self.moved_px, 1),
            "reason": self.reason,
        }


class FoveaAgent(Agent):
    """Turns a rough visual estimate into an accurate click point."""

    name = "FOVEA"

    def __init__(self, vision: Any = None, llm: Optional[LLMClient] = None):
        super().__init__(llm)
        self.vision = vision
        self.refinements: int = 0
        self.snapped: int = 0
        self.probed: int = 0
        self.zoomed: int = 0
        self.unchanged: int = 0
        self.total_correction_px: float = 0.0

    # ------------------------------------------------------------------

    def refine(
        self,
        coarse: Tuple[int, int],
        description: str = "",
        ctx: Optional[AgentContext] = None,
        allow_zoom: bool = True,
    ) -> RefinedTarget:
        """Sharpens a coarse estimate into the most accurate click point available."""
        self.refinements += 1
        # A coarse estimate near a screen edge routinely lands outside it — clamp before probing,
        # or refinement fails exactly where it is needed most.
        width, height = self._screen_size()
        cx = max(0, min(int(coarse[0]), width - 1))
        cy = max(0, min(int(coarse[1]), height - 1))

        # --- rung 1: snap straight onto the control under the estimate --------------------
        snapped = self._snap(cx, cy, description)
        if snapped is not None:
            self.snapped += 1
            self.total_correction_px += snapped.moved_px
            self._emit(ctx, "ground", f"snapped to '{snapped.snapped_to}' ({snapped.moved_px:.0f}px)",
                       **snapped.as_dict())
            return snapped

        # --- rung 2: probe the neighbourhood ----------------------------------------------
        probed = self._probe(cx, cy, description)
        if probed is not None:
            self.probed += 1
            self.total_correction_px += probed.moved_px
            self._emit(ctx, "ground", f"found '{probed.snapped_to}' nearby ({probed.moved_px:.0f}px)",
                       **probed.as_dict())
            return probed

        # --- rung 3: zoom in and ask again -------------------------------------------------
        if allow_zoom and description and self.vision is not None \
                and hasattr(self.vision, "locate_in_region"):
            self._emit(ctx, "status", f"zooming in to place {description!r} precisely")
            try:
                fine = self.vision.locate_in_region(description, (cx, cy))
            except Exception:
                fine = None
            if fine:
                self.zoomed += 1
                moved = ((fine[0] - cx) ** 2 + (fine[1] - cy) ** 2) ** 0.5
                self.total_correction_px += moved
                # A zoomed estimate may now land on a real control - try snapping once more.
                resnapped = self._snap(int(fine[0]), int(fine[1]), description)
                if resnapped is not None:
                    resnapped.method = "zoom+snap"
                    resnapped.moved_px = moved
                    return resnapped
                return RefinedTarget(x=int(fine[0]), y=int(fine[1]), method="zoom",
                                     confidence=0.7, moved_px=moved,
                                     reason="located within a zoomed crop of the estimate")

        self.unchanged += 1
        return RefinedTarget(x=cx, y=cy, method="coarse", confidence=0.45,
                             reason="no control at the estimate and zoom did not improve it")

    # ------------------------------------------------------------------

    def _snap(self, x: int, y: int, description: str) -> Optional[RefinedTarget]:
        """Snaps to the control occupying this exact pixel, if it is a plausible click target."""
        element = self._element_at(x, y)
        if element is None:
            return None
        if not self._is_clickable_size(element):
            return None

        name = element.get("name") or ""
        confidence = 0.82

        # The estimate can be ~80px off, which frequently puts it inside a *different* control.
        # Snapping to that control's centre is confidently wrong and moves the click further from
        # the target, not closer. So when the order named something and the control here is named
        # something else, reject and let the probe go looking for the right one.
        if description and name:
            score = text_score(description, name)
            if score < NAME_MATCH_FLOOR:
                return None
            confidence = min(0.97, 0.82 + score * 0.15)
        elif description and not name:
            # Unnamed control (canvas, image, custom-drawn): nothing to verify against, so accept
            # it as a positional improvement but don't claim confidence we haven't earned.
            confidence = 0.6

        tx, ty = int(element["x"]), int(element["y"])
        moved = ((tx - x) ** 2 + (ty - y) ** 2) ** 0.5
        return RefinedTarget(
            x=tx, y=ty, method="snap", confidence=confidence,
            snapped_to=name or element.get("type", ""), control_type=element.get("type", ""),
            moved_px=moved, reason="snapped to the centre of the control at the estimate",
        )

    def _probe(self, x: int, y: int, description: str) -> Optional[RefinedTarget]:
        """Samples a lattice around the estimate for the *right* control, not merely a nearby one.

        Two things this has to get right, both learned from measurement:

        * Candidates are scored, not taken first-found. An early version accepted whatever control
          it stumbled on, which happily snapped "Minimize" onto a completely different control 40px
          away. When the order named something, a name match outweighs proximity by a wide margin.
        * Probes are clamped to the screen. A coarse estimate near a screen edge routinely lands at
          a negative coordinate, and probing off-screen finds nothing at all — which is exactly the
          case where refinement is most needed.
        """
        width, height = self._screen_size()
        scored: List[Tuple[float, RefinedTarget]] = []

        for dx in PROBE_OFFSETS:
            for dy in PROBE_OFFSETS:
                for sx, sy in {(dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)}:
                    if sx == 0 and sy == 0:
                        continue
                    px = max(0, min(x + sx, width - 1))
                    py = max(0, min(y + sy, height - 1))
                    candidate = self._snap(px, py, description)
                    if candidate is None:
                        continue
                    candidate.method = "probe"
                    candidate.moved_px = ((candidate.x - x) ** 2 + (candidate.y - y) ** 2) ** 0.5
                    candidate.reason = "nothing at the estimate; found a control alongside it"

                    name_match = text_score(description, candidate.snapped_to) if description else 0.0
                    # Name match dominates: being 60px away but correctly named beats being 10px
                    # away and wrong. Distance only breaks ties among equally plausible names.
                    score = name_match * 3.0 - (candidate.moved_px / 400.0)
                    scored.append((score, candidate))

        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        best_score, best = scored[0]
        # With a named target, refuse rather than snap onto something unrelated: a confident click
        # on the wrong control is worse than admitting the estimate could not be sharpened.
        if description and best_score <= 0 and len(description) > 2:
            return None
        return best

    def _screen_size(self) -> Tuple[int, int]:
        if self.vision is not None and hasattr(self.vision, "get_screen_dimensions"):
            try:
                return self.vision.get_screen_dimensions()
            except Exception:
                pass
        return 1920, 1080

    def _element_at(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        if self.vision is None or not hasattr(self.vision, "element_at_point"):
            return None
        try:
            return self.vision.element_at_point(x, y)
        except Exception:
            return None

    @staticmethod
    def _is_clickable_size(element: Dict[str, Any]) -> bool:
        """Rejects containers. A point inside a full-window pane tells you nothing about where to
        click, and snapping to its centre would move the click somewhere arbitrary."""
        width = int(element.get("width") or 0)
        height = int(element.get("height") or 0)
        if width <= 0 or height <= 0:
            return False
        return width <= MAX_SNAP_WIDTH and height <= MAX_SNAP_HEIGHT

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        resolved = self.snapped + self.probed + self.zoomed
        return {
            "refinements": self.refinements,
            "snapped": self.snapped,
            "probed": self.probed,
            "zoomed": self.zoomed,
            "left_coarse": self.unchanged,
            "improved_rate": round(resolved / self.refinements, 3) if self.refinements else 0.0,
            "avg_correction_px": round(self.total_correction_px / resolved, 1) if resolved else 0.0,
        }
