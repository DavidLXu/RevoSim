"""Deterministic press/slide/release paths and real ballistic cross-hand hops."""

import numpy as np
from revosim.resources import SIDES, FINGERS


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * u * (10 + u * (-15 + 6 * u))


class Demo:
    def __init__(self, world, quick=False):
        self.world = world
        self.segments = []
        self.last = None
        self.start_time = world.time
        fingers = ("index",) if quick else FINGERS
        # All ten optical pads and all ten MCP/PIP pairs are exercised explicitly.
        for region in ["tip", "mcp", "pip"]:
            for side in SIDES:
                for finger in fingers:
                    self.add_probe(side, finger, region, 0.65 if quick else 0.9)
        for side in SIDES:
            self.add_probe(side, "index", "palm", 2.0)
        self.add_hops()
        self.duration = sum(s["duration"] for s in self.segments)
        self.segment_index = -1
        first = self.segments[0]["from"]
        world.reset_object(first)
        world.drive_object(first)

    def add_probe(self, side, finger, region, duration):
        p, n = self.world.surface_point(side, finger, region)
        t = np.cross(n, [0, 1, 0])
        t /= np.linalg.norm(t)
        hover = p + n * 0.035
        if self.last is None:
            self.last = hover
        self.segments.append(
            dict(
                kind="travel",
                duration=0.35,
                **{"from": self.last.copy()},
                to=hover,
                phase=f"{side} {finger} / {region}: approach",
            )
        )
        self.segments.append(
            dict(
                kind="probe",
                duration=duration,
                point=p,
                normal=n,
                tangent=t,
                phase=f"{side} {finger} / {region}: press + slide",
            )
        )
        self.last = hover.copy()

    def add_hops(self):
        p0, n0 = self.world.surface_point("left", region="palm")
        p1, n1 = self.world.surface_point("right", region="palm")
        # Settle on the left palm, then apply launch impulses. Between launch and
        # landing, only gravity and contact act on the object.
        start = p0 + n0 * 0.012
        self.segments.append(
            dict(
                kind="travel",
                duration=0.8,
                **{"from": self.last},
                to=start,
                phase="left palm / prepare hop",
            )
        )
        for side, dest in [("right", p1 + n1 * 0.006), ("left", p0 + n0 * 0.006)]:
            self.segments.append(
                dict(
                    kind="hop",
                    duration=0.75,
                    flight=0.45,
                    to=dest,
                    phase=f"ballistic hop to {side} palm",
                )
            )
        self.segments.append(
            dict(
                kind="travel",
                duration=0.6,
                **{"from": p0 + n0 * 0.012},
                to=p0 + n0 * 0.06,
                phase="release / return to zero",
            )
        )
        self.segments.append(
            dict(
                kind="travel",
                duration=0.4,
                **{"from": p0 + n0 * 0.06},
                to=p0 + n0 * 0.06,
                phase="no contact",
            )
        )

    def update(self, t):
        elapsed = t
        index = 0
        for index, s in enumerate(self.segments):
            if elapsed < s["duration"]:
                break
            elapsed -= s["duration"]
        else:
            return None
        u = elapsed / s["duration"]
        new = index != self.segment_index
        self.segment_index = index
        if s["kind"] == "travel":
            pos = (1 - smooth(u)) * s["from"] + smooth(u) * s["to"]
            pos[2] += 0.045 * np.sin(np.pi * u) ** 2
            self.world.drive_object(pos)
        elif s["kind"] == "probe":
            # Hover -> sustained compliant contact + bidirectional tangential
            # travel -> release, with C2 transitions and no position teleportation.
            if u < 0.22:
                gap = 0.035 + (0.005 - 0.035) * smooth(u / 0.22)
                slide = -0.003 * smooth(u / 0.22)
            elif u < 0.78:
                gap = 0.005
                progress = (u - 0.22) / 0.56
                stroke = 2 * progress if progress < 0.5 else 2 * (1 - progress)
                slide = -0.003 + 0.006 * smooth(stroke)
            else:
                gap = 0.005 + (0.035 - 0.005) * smooth((u - 0.78) / 0.22)
                slide = -0.003 * (1 - smooth((u - 0.78) / 0.22))
            self.world.drive_object(
                s["point"] + s["normal"] * gap + s["tangent"] * slide
            )
        elif new:
            pos = self.world.object.data.root_pos_w[0].cpu().numpy()
            T = s["flight"]
            v = (s["to"] - pos) / T + np.array([0, 0, 9.81 * T / 2])
            self.world.release_object(v)
        return s["phase"]
