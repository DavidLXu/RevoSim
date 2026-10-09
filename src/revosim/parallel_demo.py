"""Seeded, simultaneous force-driven probes with region coverage and safe transfers."""

import numpy as np
from .demo import smooth
from .resources import SIDES, FINGERS


class ParallelDemo:
    def __init__(self, world, seed=42, round_seconds=1.65):
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.round_seconds = round_seconds
        jobs = []
        for side in SIDES:
            for region in ("tip", "mcp", "pip", "palm"):
                for finger in FINGERS if region != "palm" else ("index",):
                    point, normal = world.surface_point(side, finger, region)
                    tangent = np.cross(normal, [0, 1, 0])
                    tangent /= np.linalg.norm(tangent)
                    other = np.cross(normal, tangent)
                    angle = self.rng.uniform(-0.45, 0.45)
                    tangent = tangent * np.cos(angle) + other * np.sin(angle)
                    jobs.append(
                        dict(
                            side=side,
                            finger=finger,
                            region=region,
                            point=point,
                            normal=normal,
                            tangent=tangent,
                            stroke=float(self.rng.uniform(0.0018, 0.0032)),
                            gap=float(self.rng.uniform(0.0046, 0.0052)),
                        )
                    )
        pending = list(self.rng.permutation(len(jobs)))
        self.rounds = []
        while pending:
            selected = []
            for _ in range(3):

                def compatible(i):
                    a = jobs[i]
                    return all(
                        (a["side"], a["finger"]) != (jobs[j]["side"], jobs[j]["finger"])
                        and np.linalg.norm(a["point"] - jobs[j]["point"]) > 0.026
                        for j in selected
                    )

                candidates = [i for i in pending if compatible(i)]
                if not candidates:
                    candidates = [i for i in range(len(jobs)) if compatible(i)]
                i = candidates[int(self.rng.integers(len(candidates)))]
                selected.append(i)
                if i in pending:
                    pending.remove(i)
            self.rng.shuffle(selected)
            self.rounds.append([jobs[i] for i in selected])
        self.names = list(world.objects)
        self.duration = len(self.rounds) * round_seconds + 0.8
        self.coverage = {f"{j['side']}/{j['finger']}/{j['region']}": 0.0 for j in jobs}
        self.active = []
        for k, name in enumerate(self.names):
            j = self.rounds[0][k]
            pos = j["point"] + 0.035 * j["normal"]
            world.reset_object(pos, object_name=name)
            world.drive_object(pos, object_name=name)

    def update(self, t):
        if t >= self.duration:
            return None
        i = min(int(t / self.round_seconds), len(self.rounds) - 1)
        u = (t - i * self.round_seconds) / self.round_seconds
        labels = []
        self.active = []
        for k, name in enumerate(self.names):
            j = self.rounds[i][k]
            p, n, tangent = j["point"], j["normal"], j["tangent"]
            hover = p + 0.035 * n
            if t >= len(self.rounds) * self.round_seconds:
                pos = hover + n * 0.025 * smooth(
                    (t - len(self.rounds) * self.round_seconds) / 0.45
                )
            elif u < 0.30:
                previous = self.rounds[max(0, i - 1)][k]
                start = previous["point"] + 0.035 * previous["normal"]
                v = u / 0.30
                pos = start + smooth(v) * (hover - start)
                pos = pos + np.array([0, 0, 0.045 + 0.018 * k]) * np.sin(np.pi * v) ** 2
            else:
                v = (u - 0.30) / 0.70
                if v < 0.2:
                    blend = smooth(v / 0.2)
                elif v > 0.78:
                    blend = 1 - smooth((v - 0.78) / 0.22)
                else:
                    blend = 1.0
                gap = 0.035 + (j["gap"] - 0.035) * blend
                phase = np.clip((v - 0.20) / 0.58, 0, 1)
                stroke = j["stroke"] * np.sin(2 * np.pi * phase) * blend
                pos = p + n * gap + tangent * stroke
                if 0.2 <= v <= 0.78:
                    self.active.append(j)
            self.world.drive_object(pos, object_name=name)
            labels.append(f"{name}: {j['side'][0].upper()} {j['finger']} {j['region']}")
        return (
            " | ".join(labels)
            if t < len(self.rounds) * self.round_seconds
            else "Release / no contact"
        )

    def record(self, values):
        from .resources import pressure_regions

        for j in self.active:
            h, f = SIDES.index(j["side"]), FINGERS.index(j["finger"])
            if j["region"] == "tip":
                response = float(values["depth"][h, f].max())
            else:
                name = (
                    "palm" if j["region"] == "palm" else j["finger"] + "_" + j["region"]
                )
                response = float(values["pressure"][h, pressure_regions()[name]].max())
            key = f"{j['side']}/{j['finger']}/{j['region']}"
            self.coverage[key] = max(self.coverage[key], response)
