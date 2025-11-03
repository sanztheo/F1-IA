from __future__ import annotations

import pygame
import numpy as np
from typing import Tuple, List


class Viewer:
    def __init__(self, size: Tuple[int, int] = (1200, 800), title: str = "F1-IA Evolution"):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.offset = np.array([size[0] // 2, size[1] // 2], dtype=float)
        self.scale = 0.05  # pixels per meter
        self.bg = (15, 17, 22)
        self.fg = (220, 220, 220)

    def world_to_screen(self, xy: np.ndarray) -> np.ndarray:
        if xy.size == 0:
            return xy
        pts = xy * self.scale
        pts[:, 1] *= -1
        pts += self.offset
        return pts

    def draw_polyline(self, xy: np.ndarray, color=(120, 120, 120), width: int = 2):
        if xy is None or len(xy) < 2:
            return
        pts = self.world_to_screen(xy.copy())
        pygame.draw.lines(self.screen, color, False, pts.astype(int).tolist(), width)

    def draw_cars(self, cars: List[Tuple[float, float, Tuple[int, int, int]]], radius: int = 4):
        # cars: list of (x, y, color)
        if not cars:
            return
        arr = np.array([[c[0], c[1]] for c in cars], dtype=float)
        colors = [c[2] for c in cars]
        pts = self.world_to_screen(arr)
        for p, col in zip(pts, colors):
            pygame.draw.circle(self.screen, col, p.astype(int), radius)

    def draw_text(self, text: str, topleft=(10, 10), color=(230, 230, 230)):
        surf = self.font.render(text, True, color)
        self.screen.blit(surf, topleft)

    def handle_events(self) -> Tuple[bool, float, float]:
        running = True
        d_off = np.array([0.0, 0.0])
        d_scale = 1.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # wheel up
                    self.scale *= 1.1
                elif event.button == 5:  # wheel down
                    self.scale /= 1.1
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:
                    rel = np.array(event.rel, dtype=float)
                    self.offset += rel
        return running, d_off[0], d_scale

    def tick(self, fps: int = 60) -> float:
        return self.clock.tick(fps) / 1000.0

    def clear(self):
        self.screen.fill(self.bg)

    def flip(self):
        pygame.display.flip()

