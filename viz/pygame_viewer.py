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
        self.screen_size = np.array([size[0], size[1]], dtype=float)
        self.offset = np.array([size[0] // 2, size[1] // 2], dtype=float)
        self.scale = 0.05  # pixels per meter
        self.follow = False
        self.bg = (15, 17, 22)
        self.fg = (220, 220, 220)

    def world_to_screen(self, xy: np.ndarray) -> np.ndarray:
        if xy.size == 0:
            return xy
        pts = xy * self.scale
        pts[:, 1] *= -1
        pts += self.offset
        return pts

    def screen_to_world(self, pt: Tuple[int, int]) -> np.ndarray:
        x, y = float(pt[0]), float(pt[1])
        wx = (x - self.offset[0]) / self.scale
        wy = - (y - self.offset[1]) / self.scale
        return np.array([wx, wy], dtype=float)

    def center_on(self, world_pos: Tuple[float, float]):
        wp = np.array(world_pos, dtype=float)
        # place world_pos au centre écran
        self.offset = self.screen_size/2.0 - np.array([wp[0], -wp[1]]) * self.scale

    def draw_polyline(self, xy: np.ndarray, color=(120, 120, 120), width: int = 2):
        if xy is None or len(xy) < 2:
            return
        pts = self.world_to_screen(xy.copy())
        pygame.draw.lines(self.screen, color, False, pts.astype(int).tolist(), width)

    def draw_polyline_fast(self, xy: np.ndarray, color=(120, 120, 120), width: int = 2, min_px: float = 2.0):
        if xy is None or len(xy) < 2:
            return
        pts = self.world_to_screen(xy.copy())
        # décimation en espace écran: garder un point si éloigné de min_px du précédent gardé
        out = []
        last = None
        for p in pts:
            if last is None:
                out.append(p)
                last = p
            else:
                if np.linalg.norm(p - last) >= min_px:
                    out.append(p)
                    last = p
        if len(out) >= 2:
            pygame.draw.lines(self.screen, color, False, np.array(out).astype(int).tolist(), width)

    def draw_cars(self, cars: List[Tuple[float, float, Tuple[int, int, int]]], radius: int = 4):
        # cars: list of (x, y, color)
        if not cars:
            return
        arr = np.array([[c[0], c[1]] for c in cars], dtype=float)
        colors = [c[2] for c in cars]
        pts = self.world_to_screen(arr)
        for p, col in zip(pts, colors):
            pygame.draw.circle(self.screen, col, p.astype(int), radius)

    def draw_car_rect(self, x: float, y: float, angle: float, length: float = 4.5, width: float = 2.0, color=(233, 30, 99)):
        # rectangle orienté
        c, s = np.cos(angle), np.sin(angle)
        # coins dans le repère voiture (avant en +x)
        corners = np.array([
            [ length/2,  width/2],
            [ length/2, -width/2],
            [-length/2, -width/2],
            [-length/2,  width/2],
        ])
        R = np.array([[c, -s], [s, c]])
        world = (corners @ R.T) + np.array([x, y])
        pts = self.world_to_screen(world.copy()).astype(int)
        # culling écran
        if ((pts[:,0] < -50).all() or (pts[:,0] > self.screen_size[0]+50).all() or
            (pts[:,1] < -50).all() or (pts[:,1] > self.screen_size[1]+50).all()):
            return
        pygame.draw.polygon(self.screen, color, pts.tolist(), width=0)
        # flèche avant
        nose = (np.array([length/2+0.7, 0.0]) @ R.T) + np.array([x, y])
        nose_pt = self.world_to_screen(nose.reshape(1, -1)).astype(int)[0]
        pygame.draw.circle(self.screen, (250, 250, 250), nose_pt, 2)

    def draw_rays(self, origin: Tuple[float, float], endpoints: np.ndarray, color=(255, 200, 0)):
        if endpoints is None or len(endpoints) == 0:
            return
        o = np.array([[origin[0], origin[1]]], dtype=float)
        A = self.world_to_screen(o.copy()).astype(int)[0]
        B = self.world_to_screen(endpoints.copy()).astype(int)
        for b in B:
            pygame.draw.line(self.screen, color, A.tolist(), b.tolist(), 1)

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
                mouse = pygame.mouse.get_pos()
                world_before = self.screen_to_world(mouse)
                if event.button == 4:  # wheel up
                    self.scale *= 1.15
                elif event.button == 5:  # wheel down
                    self.scale /= 1.15
                # maintenir le point sous le curseur fixe pendant le zoom
                screen = np.array(mouse, dtype=float)
                self.offset = screen - np.array([world_before[0], -world_before[1]]) * self.scale
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
