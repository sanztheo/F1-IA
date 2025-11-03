from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from scipy.spatial import cKDTree
from shapely.geometry import LineString, LinearRing


def _angle_of(vx: float, vy: float) -> float:
    return float(np.arctan2(vy, vx))


def _wrap_angle(a: float) -> float:
    return ((a + np.pi) % (2 * np.pi)) - np.pi


@dataclass
class CarParams:
    L: float = 3.6  # empattement (F1 ~3.6m)
    body_len: float = 5.6  # longueur (F1 ~5.6m)
    body_w: float = 2.0    # largeur (F1 max 2.0m)
    max_steer: float = np.deg2rad(30)
    max_accel: float = 9.0
    drag: float = 0.002
    max_speed: float = 90.0  # m/s ~ 324 km/h


class TrackEnv:
    def __init__(self, centerline: np.ndarray, half_width: float = 6.0, dt: float = 1/60.0):
        assert centerline.ndim == 2 and centerline.shape[1] == 2
        self.center = centerline.astype(float)
        self.half_w = float(half_width)
        self.dt = float(dt)
        # pré-calculs
        self.s = np.zeros(len(self.center))
        ds = np.linalg.norm(np.diff(self.center, axis=0), axis=1)
        self.s[1:] = np.cumsum(ds)
        self.length = float(self.s[-1])
        dx = np.gradient(self.center[:, 0])
        dy = np.gradient(self.center[:, 1])
        self.tang = np.stack([dx, dy], axis=1)
        n = np.stack([-dy, dx], axis=1)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        self.normals = n / n_norm
        self.tang_angles = np.array([_angle_of(t[0], t[1]) for t in self.tang])
        self.kdt = cKDTree(self.center)
        # bords de piste (robustes via offsets shapely, évite les "boîtes" aux virages serrés)
        try:
            # fermer la ligne si proche
            closed = np.linalg.norm(self.center[0] - self.center[-1]) < 1e-6
            geo = LinearRing(self.center) if closed else LineString(self.center)
            # Créer un "ruban" de largeur 2*half_w autour du centre
            poly = geo.buffer(self.half_w, join_style=1, cap_style=2, resolution=8)
            # Le contour contient deux anneaux (bord interne/externe). Choisir les deux plus longs
            boundary = poly.boundary
            lines = list(boundary.geoms) if hasattr(boundary, 'geoms') else [boundary]
            lines.sort(key=lambda g: g.length, reverse=True)
            if len(lines) >= 2:
                self.left_edge = np.asarray(lines[0].coords)
                self.right_edge = np.asarray(lines[1].coords)
            else:
                self.left_edge = self.center + self.normals * self.half_w
                self.right_edge = self.center - self.normals * self.half_w
        except Exception:
            self.left_edge = self.center + self.normals * self.half_w
            self.right_edge = self.center - self.normals * self.half_w
        # Portique départ/arrivée (ligne perpendiculaire au centre au début)
        self.gate_o = self.center[0]
        self.gate_n = self.normals[0]

        # état voiture
        self.params = CarParams()
        self.state: Dict[str, Any] = {}
        self._prev_pos = None
        self._s_travel = 0.0
        self._gate_prev = None

    def reset(self, s0: float = 0.0, random_start: bool = False) -> np.ndarray:
        # position initiale au début de la piste
        if random_start:
            s0 = float(np.random.uniform(0.0, self.length))
        idx = int(np.searchsorted(self.s, s0, side="left"))
        p = self.center[idx]
        th = self.tang_angles[idx]
        self.state = {
            "x": float(p[0]),
            "y": float(p[1]),
            "th": float(th),
            "v": 5.0,
            "progress": float(self.s[idx]),
            "lap": 0,
            "t_lap": 0.0,
        }
        self._prev_pos = np.array([self.state["x"], self.state["y"]], dtype=float)
        self._s_travel = 0.0
        self._gate_prev = self._gate_side(self._prev_pos)
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        # erreurs par rapport au centerline
        pos = np.array([self.state["x"], self.state["y"]])
        dist, idx = self.kdt.query(pos)
        n = self.normals[idx]
        lat_err = float(np.dot(pos - self.center[idx], n))
        head_err = _wrap_angle(self.state["th"] - self.tang_angles[idx])
        speed = float(self.state["v"]) / self.params.max_speed
        # 5 capteurs rudimentaires (raycasts approchés)
        rays = np.deg2rad(np.array([-60, -30, 0, 30, 60], dtype=float))
        dists = [self._raycast(pos, self.state["th"] + r, self.half_w*2.0) for r in rays]
        obs = np.array([lat_err / self.half_w, head_err / np.pi, speed] + [d/(self.half_w*2.0) for d in dists], dtype=float)
        return obs

    def ray_endpoints(self, num: int = 5, fov_deg: float = 120.0, max_r: float | None = None):
        """Retourne les points finaux des raycasts à partir de la position/heading courants.

        - num: nombre de rayons (5 par défaut)
        - fov_deg: champ de vision total en degrés (centré sur l'angle de la voiture)
        - max_r: portée max (défaut: 2*half_width)
        """
        max_r = max_r or (2.0 * self.half_w)
        pos = np.array([self.state["x"], self.state["y"]])
        th0 = self.state["th"]
        if num <= 1:
            angles = np.array([0.0])
        else:
            angles = np.linspace(-np.deg2rad(fov_deg)/2.0, np.deg2rad(fov_deg)/2.0, num)
        endpoints = []
        for a in angles:
            ang = th0 + a
            # marche jusqu'au mur (sortie de piste) ou portée max
            step = self.half_w / 20.0
            r = 0.0
            hit = pos.copy()
            while r < max_r:
                probe = pos + np.array([np.cos(ang), np.sin(ang)]) * r
                if not self._on_track_point(probe):
                    hit = probe
                    break
                hit = probe
                r += step
            endpoints.append(hit)
        return np.asarray(endpoints)

    def _raycast(self, pos: np.ndarray, ang: float, max_r: float) -> float:
        # avance petit à petit jusqu'à sortir de la piste
        step = self.half_w / 10.0
        r = 0.0
        while r < max_r:
            probe = pos + np.array([np.cos(ang), np.sin(ang)]) * r
            if not self._on_track_point(probe):
                return r
            r += step
        return max_r

    def _on_track_point(self, pos: np.ndarray) -> bool:
        _, idx = self.kdt.query(pos)
        n = self.normals[idx]
        lat = float(np.dot(pos - self.center[idx], n))
        return abs(lat) <= self.half_w

    def _car_corners(self, x: float, y: float, th: float) -> np.ndarray:
        L = self.params.body_len
        W = self.params.body_w
        corners = np.array([
            [ L/2,  W/2],
            [ L/2, -W/2],
            [-L/2, -W/2],
            [-L/2,  W/2],
        ])
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        world = (corners @ R.T) + np.array([x, y])
        return world

    def _on_track_car(self, x: float, y: float, th: float) -> bool:
        # Toute la voiture doit rester dans les bords (tolérance 0.05m)
        tol = 0.05
        pts = self._car_corners(x, y, th)
        for p in pts:
            if not self._on_track_point(p):
                return False
        # vérifier centre aussi
        return self._on_track_point(np.array([x, y]))

    # utilitaires
    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        return self.left_edge, self.right_edge

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # action: [steer(-1..1), throttle(0..1), brake(0..1)]
        steer = float(np.clip(action[0], -1, 1)) * self.params.max_steer
        throttle = float(np.clip(action[1], 0, 1))
        brake = float(np.clip(action[2], 0, 1))

        # physique simple (bicycle approx)
        v = self.state["v"]
        th = self.state["th"]
        ax = (11.0 * (throttle) - 12.0 * brake) - self.params.drag * v * v  # ~0-100 in ~2.6s, brake up to ~6g equivalent via limit below
        # lateral accel limit with aero: a_lat_max(v) ≈ (1.8 + k*v^2) * g, k~=0.00058 so that ~5.5g @ 80 m/s
        g = 9.80665
        a_lat_max = min(6.5 * g, (1.8 + 0.00058 * v * v) * g)
        # curvature commanded and clamped to a_lat limit
        curv_cmd = np.tan(steer) / self.params.L
        if v > 0.1:
            curv_max = a_lat_max / (v * v)
            curv_cmd = float(np.clip(curv_cmd, -curv_max, curv_max))
        # integrate
        v = float(np.clip(v + ax * self.dt, 0.0, self.params.max_speed))
        th = float(_wrap_angle(th + (v * curv_cmd) * self.dt))
        pos = np.array([self.state["x"], self.state["y"]]) + np.array([np.cos(th), np.sin(th)]) * v * self.dt

        self.state.update({"x": float(pos[0]), "y": float(pos[1]), "th": th, "v": v})

        # progrès et récompense
        dist, idx = self.kdt.query(pos)
        s_here = self.s[idx]
        prev_s = self.state.get("progress", 0.0)
        ds_prog = float(s_here - prev_s)
        if ds_prog < -self.length * 0.5:
            # wrap autour si on franchit l'origine
            ds_prog += self.length
        self.state["progress"] = s_here
        self.state["t_lap"] += self.dt
        # Distance parcourue le long de la tangente (plus fiable que s wrap)
        if self._prev_pos is None:
            self._prev_pos = pos.copy()
        delta = pos - self._prev_pos
        t_prev = np.array([np.cos(self.state["th"]), np.sin(self.state["th"])])
        ds_forward = max(0.0, float(np.dot(delta, t_prev)))
        self._s_travel += ds_forward
        self._prev_pos = pos.copy()
        lap_done = False
        # Détection passage portique (changement de signe par rapport à la normale de gate)
        gate_now = self._gate_side(pos)
        crossed_gate = (self._gate_prev is not None) and (gate_now * self._gate_prev < 0)
        self._gate_prev = gate_now
        if self._s_travel >= self.length * 0.8 and crossed_gate:
            self.state["lap"] = int(self.state.get("lap", 0)) + 1
            lap_done = True
            self._s_travel -= self.length
            self.state["t_lap"] = 0.0

        on = self._on_track_car(pos[0], pos[1], th)
        # erreurs courantes
        n = self.normals[idx]
        lat_err = float(np.dot(pos - self.center[idx], n))
        head_err = _wrap_angle(th - self.tang_angles[idx])
        # pénalités stabilité
        pen = 0.01 * abs(lat_err) + 0.005 * abs(head_err) + 0.0005 * v * v
        done = not on
        reward = (ds_prog - pen) if on else -1.0
        return self.get_obs(), reward, done, {"s": s_here, "on": on, "lap": int(self.state["lap"]), "t_lap": float(self.state["t_lap"]), "lap_done": lap_done}

    def _gate_side(self, p: np.ndarray) -> float:
        # signe de la projection du point par rapport à la normale du portique
        return float(np.dot(p - self.gate_o, self.gate_n))
