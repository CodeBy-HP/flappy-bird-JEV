"""
Jev Flappy Bird
---------------
Jev AI (TypeSafe) flies the bird autonomously.
Every ~200 ms a background thread asks Jev one Noul question:
  "Should the bird flap right now?"
The game loop runs at 60 fps and never blocks on the API.

Run:
    python game.py

Requires TYPESAFE_API_KEY in .env (project root).
Images (optional) in images/ :
    bird.png, pipe_top.png, pipe_bottom.png, background.png
    If missing, coloured rectangles are used as fallback.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pygame
from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient

# ── env ──────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ── constants ────────────────────────────────────────────────────────────────
W, H          = 480, 640
FPS           = 60
GRAVITY       = 0.45
JUMP_VEL      = -8.5
PIPE_SPEED    = 3.0
PIPE_GAP      = 155          # px between top and bottom pipe
PIPE_WIDTH    = 70
PIPE_INTERVAL = 90           # frames between pipe spawns
BIRD_X        = 90
BIRD_W, BIRD_H = 40, 30
FLOOR_Y       = H - 80       # ground level (y of floor top)

POLL_INTERVAL = 0.20         # seconds between Jev calls

# Colour palette — minimal, flat game look
C_SKY       = (106, 190, 240)
C_FLOOR     = (222, 184, 135)
C_FLOOR_TOP = (111, 196, 69)
C_PIPE      = (115, 190, 78)
C_PIPE_DARK = (87,  148, 58)
C_BIRD      = (255, 215, 0)
C_BIRD_EYE  = (30,  30,  30)
C_WHITE     = (255, 255, 255)
C_BLACK     = (0,   0,   0)
C_PANEL_BG  = (0,   0,   0,  160)   # RGBA for alpha surface
C_GREEN     = (60,  220, 100)
C_RED       = (230, 70,  70)
C_YELLOW    = (255, 215, 0)
C_GRAY      = (160, 160, 160)
C_DARK      = (20,  20,  30)

PIPE_COLORS = [
    ((115, 190, 78),  (87,  148, 58)),
    ((78,  160, 190), (58,  120, 148)),
    ((190, 115, 78),  (148, 87,  58)),
    ((160, 78,  190), (120, 58,  148)),
]


# ── asset loader (fallback to shapes if image missing) ───────────────────────
_IMG_DIR = Path(__file__).parent / "images"

def _load(name: str, size: tuple[int, int]) -> Optional[pygame.Surface]:
    p = _IMG_DIR / name
    if p.exists():
        try:
            img = pygame.image.load(str(p)).convert_alpha()
            return pygame.transform.smoothscale(img, size)
        except Exception:
            pass
    return None

def _load_assets() -> dict:
    return {
        "bird":       _load("bird.png",       (BIRD_W, BIRD_H)),
        "pipe_top":   _load("pipe_top.png",    (PIPE_WIDTH, 400)),
        "pipe_bot":   _load("pipe_bottom.png", (PIPE_WIDTH, 400)),
        "background": _load("background.png",  (W, H)),
    }


# ── dataclasses ──────────────────────────────────────────────────────────────
@dataclass
class Bird:
    y:   float = H / 2
    vel: float = 0.0
    angle: float = 0.0        # visual tilt

    def flap(self):
        self.vel = JUMP_VEL

    def update(self):
        self.vel  += GRAVITY
        self.y    += self.vel
        # tilt: point up on flap, droop on fall
        target = max(-30.0, min(90.0, self.vel * 6))
        self.angle += (target - self.angle) * 0.25

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(BIRD_X, int(self.y) - BIRD_H // 2, BIRD_W, BIRD_H)

    def hit_bounds(self) -> bool:
        return self.y - BIRD_H // 2 < 0 or self.y + BIRD_H // 2 > FLOOR_Y


@dataclass
class Pipe:
    x:        float
    gap_y:    int          # y-centre of gap
    color_idx: int = 0

    TOP_H  = 400           # max draw height for pipe surf
    BOT_H  = 400

    @property
    def top_rect(self) -> pygame.Rect:
        h = self.gap_y - PIPE_GAP // 2
        return pygame.Rect(int(self.x), 0, PIPE_WIDTH, h)

    @property
    def bot_rect(self) -> pygame.Rect:
        top = self.gap_y + PIPE_GAP // 2
        return pygame.Rect(int(self.x), top, PIPE_WIDTH, FLOOR_Y - top)

    def update(self, speed: float):
        self.x -= speed

    def off_screen(self) -> bool:
        return self.x + PIPE_WIDTH < 0

    def collides(self, bird: Bird) -> bool:
        br = bird.rect
        return br.colliderect(self.top_rect) or br.colliderect(self.bot_rect)


# ── shared state for poller ───────────────────────────────────────────────────
@dataclass
class PollData:
    """Written by JevPoller thread, read by main loop."""
    noul:       float = 0.3
    latency_ms: float = 0.0
    req_count:  int   = 0
    total_tokens: int = 0    # rough input-token accumulator
    last_flap_noul: float = 0.3   # noul at last flap decision

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, noul: float, latency_ms: float, tokens: int):
        with self._lock:
            self.noul        = noul
            self.latency_ms  = latency_ms
            self.req_count  += 1
            self.total_tokens += tokens

    def read(self) -> tuple[float, float, int, int]:
        with self._lock:
            return self.noul, self.latency_ms, self.req_count, self.total_tokens


# ── Jev poller thread ─────────────────────────────────────────────────────────
class JevPoller(threading.Thread):
    """
    Polls Jev every POLL_INTERVAL seconds.
    Reads a snapshot dict from main thread (written each frame).
    ponytail: single shared snapshot dict + lock — last write wins, fine at 5 Hz.
    """

    def __init__(self, poll_data: PollData):
        super().__init__(daemon=True)
        self._poll_data  = poll_data
        self._snapshot: dict = {}
        self._alive     = True
        self._snap_lock = threading.Lock()
        try:
            self._client = TypeSafeClient(
                api_key=os.environ["TYPESAFE_API_KEY"],
                model="jev-1.13.0",
            )
            self._api_ok = True
        except Exception as e:
            print(f"[JevPoller] SDK init failed: {e}", file=sys.stderr)
            self._api_ok = False

    def update_snapshot(self, snap: dict):
        with self._snap_lock:
            self._snapshot = snap

    def stop(self):
        self._alive = False

    def run(self):
        while self._alive:
            time.sleep(POLL_INTERVAL)
            if not self._api_ok:
                continue
            with self._snap_lock:
                snap = dict(self._snapshot)
            if not snap or snap.get("status") != "alive":
                continue
            try:
                t0 = time.perf_counter()
                resp = self._client.system_one(
                    state={
                        "bird_y":          snap["bird_y"],
                        "bird_vel":        snap["bird_vel"],
                        "pipe_dist_px":    snap["pipe_dist"],
                        "pipe_gap_center": snap["pipe_gap_center"],
                        "gap_height_px":   PIPE_GAP,
                        "screen_height":   H,
                        "floor_y":         FLOOR_Y,
                    },
                    questions={
                        "flap": Noul(
                            instructions=(
                                "The bird needs to pass through the pipe gap. "
                                "Given its current Y position, velocity, and distance "
                                "to the next pipe gap centre, the bird should flap "
                                "right now to stay on course and avoid crashing."
                            )
                        )
                    },
                )
                latency = (time.perf_counter() - t0) * 1000
                noul_val = resp.answers["flap"].noul
                # ponytail: token estimate = input JSON char count // 4, free output
                tokens = len(json.dumps(snap)) // 4
                self._poll_data.write(noul_val, latency, tokens)
            except Exception as e:
                print(f"[JevPoller] API error: {e}", file=sys.stderr)


# ── drawing helpers ───────────────────────────────────────────────────────────
def _draw_pipe(surf: pygame.Surface, pipe: Pipe, assets: dict):
    c_fill, c_dark = PIPE_COLORS[pipe.color_idx % len(PIPE_COLORS)]

    def _draw_one(rect: pygame.Rect, cap_bottom: bool):
        if rect.height <= 0:
            return
        pygame.draw.rect(surf, c_fill, rect)
        # right-edge shadow
        pygame.draw.rect(surf, c_dark,
                         (rect.right - 8, rect.y, 8, rect.height))
        # cap
        cap_h = 14
        cap_rect = pygame.Rect(rect.x - 4, 0, PIPE_WIDTH + 8, cap_h)
        if cap_bottom:
            cap_rect.top = rect.top
        else:
            cap_rect.bottom = rect.bottom
        pygame.draw.rect(surf, c_fill, cap_rect)
        pygame.draw.rect(surf, c_dark,
                         (cap_rect.right - 8, cap_rect.y, 8, cap_rect.height))

    _draw_one(pipe.top_rect, cap_bottom=True)
    _draw_one(pipe.bot_rect, cap_bottom=False)


def _draw_bird(surf: pygame.Surface, bird: Bird, assets: dict, frame: int):
    img = assets.get("bird")
    if img:
        rotated = pygame.transform.rotate(img, -bird.angle)
        r = rotated.get_rect(center=(BIRD_X + BIRD_W // 2,
                                      int(bird.y)))
        surf.blit(rotated, r)
        return

    # Fallback: draw a rounded bird shape
    cx = BIRD_X + BIRD_W // 2
    cy = int(bird.y)
    # wing flap animation
    wing_off = 4 if (frame // 6) % 2 == 0 else -2
    pygame.draw.ellipse(surf, C_BIRD,
                        (cx - BIRD_W // 2, cy - BIRD_H // 2,
                         BIRD_W, BIRD_H))
    # wing
    pygame.draw.ellipse(surf, (200, 165, 0),
                        (cx - 18, cy - 4 + wing_off, 16, 10))
    # eye
    pygame.draw.circle(surf, C_WHITE, (cx + 10, cy - 5), 6)
    pygame.draw.circle(surf, C_BIRD_EYE, (cx + 12, cy - 5), 3)
    # beak
    pygame.draw.polygon(surf, (255, 140, 0),
                        [(cx + 18, cy - 2), (cx + 26, cy + 1), (cx + 18, cy + 4)])


def _draw_background(surf: pygame.Surface, assets: dict, scroll: float):
    img = assets.get("background")
    if img:
        surf.blit(img, (0, 0))
        return
    surf.fill(C_SKY)
    # simple scrolling clouds
    for i in range(4):
        cx = int((i * 140 - scroll * 0.3) % (W + 60)) - 30
        pygame.draw.ellipse(surf, (240, 248, 255), (cx, 80 + i * 30, 90, 35))
        pygame.draw.ellipse(surf, (240, 248, 255), (cx + 20, 70 + i * 30, 60, 30))


def _draw_floor(surf: pygame.Surface, scroll: float):
    # grass stripe
    pygame.draw.rect(surf, C_FLOOR_TOP, (0, FLOOR_Y, W, 10))
    pygame.draw.rect(surf, C_FLOOR,     (0, FLOOR_Y + 10, W, H - FLOOR_Y - 10))
    # scrolling texture lines
    stripe_w = 30
    offset = int(scroll * 1.5) % stripe_w
    for x in range(-stripe_w + offset, W + stripe_w, stripe_w):
        pygame.draw.line(surf, (190, 155, 110),
                         (x, FLOOR_Y + 12), (x + 20, H), 2)


def _draw_debug_panel(surf: pygame.Surface, fonts: dict,
                      noul: float, latency: float,
                      req_count: int, total_tokens: int,
                      score: int, pipe_speed: float):
    """Semi-transparent HUD — top-right corner."""
    panel_w, panel_h = 200, 178
    px, py = W - panel_w - 8, 8

    # background
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((15, 15, 25, 175))
    pygame.draw.rect(panel, (80, 80, 120, 200),
                     (0, 0, panel_w, panel_h), 1)
    surf.blit(panel, (px, py))

    # header
    hdr = fonts["sm"].render("⬡ JEV DECISION", True, C_YELLOW)
    surf.blit(hdr, (px + 8, py + 6))
    pygame.draw.line(surf, (80, 80, 120),
                     (px + 4, py + 22), (px + panel_w - 4, py + 22), 1)

    # noul bar
    bar_x, bar_y = px + 8, py + 30
    bar_w = panel_w - 16
    bar_h = 16
    pygame.draw.rect(surf, (40, 40, 60), (bar_x, bar_y, bar_w, bar_h))
    fill_w = int(bar_w * noul)
    bar_color = C_GREEN if noul > 0.5 else C_RED
    if fill_w > 0:
        pygame.draw.rect(surf, bar_color, (bar_x, bar_y, fill_w, bar_h))
    pygame.draw.rect(surf, C_GRAY, (bar_x, bar_y, bar_w, bar_h), 1)
    # midline
    mid_x = bar_x + bar_w // 2
    pygame.draw.line(surf, (200, 200, 200),
                     (mid_x, bar_y), (mid_x, bar_y + bar_h), 1)

    action = "FLAP ▲" if noul > 0.5 else "FALL ▼"
    a_color = C_GREEN if noul > 0.5 else C_RED
    noul_lbl = fonts["sm"].render(f"noul: {noul:.3f}  {action}", True, a_color)
    surf.blit(noul_lbl, (bar_x, bar_y + bar_h + 3))

    # stats rows
    cost_usd = total_tokens * 0.042 / 1_000_000
    rows = [
        ("latency",   f"{latency:.0f} ms"),
        ("requests",  str(req_count)),
        ("tokens",    f"~{total_tokens:,}"),
        ("cost",      f"${cost_usd:.6f}"),
        ("score",     str(score)),
        ("pipe spd",  f"{pipe_speed:.1f} px/f"),
    ]
    ry = bar_y + bar_h + 20
    for label, val in rows:
        lbl_s = fonts["xs"].render(label, True, C_GRAY)
        val_s = fonts["xs"].render(val, True, C_WHITE)
        surf.blit(lbl_s, (px + 8, ry))
        surf.blit(val_s, (px + panel_w - val_s.get_width() - 8, ry))
        ry += 16


def _draw_score_big(surf: pygame.Surface, fonts: dict, score: int):
    shadow = fonts["lg"].render(str(score), True, C_BLACK)
    text   = fonts["lg"].render(str(score), True, C_WHITE)
    cx = W // 2 - text.get_width() // 2
    surf.blit(shadow, (cx + 2, 52))
    surf.blit(text,   (cx,     50))


def _draw_death_screen(surf: pygame.Surface, fonts: dict,
                       score: int, best: int, countdown: float):
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    surf.blit(overlay, (0, 0))

    panel_w, panel_h = 300, 160
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((15, 15, 25, 220))
    pygame.draw.rect(panel, C_RED, (0, 0, panel_w, panel_h), 2)
    surf.blit(panel, (W // 2 - panel_w // 2, H // 2 - panel_h // 2))

    go = fonts["md"].render("GAME OVER", True, C_RED)
    surf.blit(go, (W // 2 - go.get_width() // 2, H // 2 - 55))

    sc = fonts["sm"].render(f"Score: {score}   Best: {best}", True, C_WHITE)
    surf.blit(sc, (W // 2 - sc.get_width() // 2, H // 2 - 10))

    secs = max(0.0, countdown)
    rs = fonts["xs"].render(f"Restarting in {secs:.1f}s …", True, C_GRAY)
    surf.blit(rs, (W // 2 - rs.get_width() // 2, H // 2 + 35))


# ── game state reset ──────────────────────────────────────────────────────────
def _new_game() -> tuple[Bird, list[Pipe], int, float]:
    """Returns (bird, pipes, score, pipe_speed)."""
    import random
    bird  = Bird()
    gap_y = random.randint(160, FLOOR_Y - 160)
    pipes = [Pipe(x=W + 60, gap_y=gap_y, color_idx=0)]
    return bird, pipes, 0, PIPE_SPEED


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import random

    pygame.init()
    pygame.display.set_caption("Flappy Bird  ·  Powered by Jev AI")
    screen = pygame.display.set_mode((W, H))
    clock  = pygame.time.Clock()

    fonts = {
        "lg": pygame.font.SysFont("Arial", 52, bold=True),
        "md": pygame.font.SysFont("Arial", 32, bold=True),
        "sm": pygame.font.SysFont("Arial", 14, bold=True),
        "xs": pygame.font.SysFont("Arial", 12),
    }

    assets = _load_assets()

    poll_data  = PollData()
    poller     = JevPoller(poll_data)
    poller.start()

    bird, pipes, score, pipe_speed = _new_game()
    pipe_timer   = 0
    scroll       = 0.0
    status       = "alive"    # "alive" | "dead"
    death_timer  = 0.0
    DEATH_PAUSE  = 2.0
    best         = 0
    frame        = 0

    # edge-trigger: only flap on rising edge of noul > 0.5
    prev_noul_high = False

    while True:
        dt = clock.tick(FPS) / 1000.0  # seconds, capped by FPS

        # ── events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                poller.stop()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                poller.stop()
                pygame.quit()
                sys.exit()

        noul, latency, req_count, total_tokens = poll_data.read()

        # ── physics ──
        if status == "alive":
            # edge-trigger flap
            noul_high = noul > 0.5
            if noul_high and not prev_noul_high:
                bird.flap()
            prev_noul_high = noul_high

            bird.update()
            scroll += pipe_speed

            # pipes
            pipe_timer += 1
            if pipe_timer >= PIPE_INTERVAL:
                pipe_timer = 0
                gap_y = random.randint(160, FLOOR_Y - 160)
                ci    = random.randint(0, len(PIPE_COLORS) - 1)
                pipes.append(Pipe(x=float(W + 20), gap_y=gap_y, color_idx=ci))

            for p in pipes:
                p.update(pipe_speed)

            # score: passed a pipe
            for p in pipes:
                if not hasattr(p, "_scored"):
                    p._scored = False
                if not p._scored and p.x + PIPE_WIDTH < BIRD_X:
                    p._scored = True
                    score += 1
                    # speed ramp every 5 points
                    pipe_speed = PIPE_SPEED + (score // 5) * 0.5

            pipes = [p for p in pipes if not p.off_screen()]

            # collision
            dead = bird.hit_bounds()
            if not dead:
                for p in pipes:
                    if p.collides(bird):
                        dead = True
                        break

            if dead:
                status     = "dead"
                death_timer = DEATH_PAUSE
                best        = max(best, score)

            # update poller snapshot
            next_pipe = min(
                (p for p in pipes if p.x + PIPE_WIDTH > BIRD_X),
                key=lambda p: p.x,
                default=None,
            )
            if next_pipe:
                snap = {
                    "status":         "alive",
                    "bird_y":         int(bird.y),
                    "bird_vel":       round(bird.vel, 1),
                    "pipe_dist":      int(next_pipe.x - BIRD_X),
                    "pipe_gap_center": next_pipe.gap_y,
                }
            else:
                snap = {"status": "alive", "bird_y": int(bird.y),
                        "bird_vel": round(bird.vel, 1),
                        "pipe_dist": W, "pipe_gap_center": H // 2}
            poller.update_snapshot(snap)

        else:
            # dead — count down then restart
            death_timer -= dt
            poller.update_snapshot({"status": "dead"})
            if death_timer <= 0:
                bird, pipes, score, pipe_speed = _new_game()
                pipe_timer     = 0
                status         = "alive"
                prev_noul_high = False

        # ── render ──
        _draw_background(screen, assets, scroll)

        for p in pipes:
            _draw_pipe(screen, p, assets)

        _draw_floor(screen, scroll)
        _draw_bird(screen, bird, assets, frame)
        _draw_score_big(screen, fonts, score)
        _draw_debug_panel(screen, fonts,
                          noul, latency, req_count, total_tokens,
                          score, pipe_speed)

        if status == "dead":
            _draw_death_screen(screen, fonts, score, best, death_timer)

        pygame.display.flip()
        frame += 1


if __name__ == "__main__":
    main()
