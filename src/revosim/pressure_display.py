"""A planar hand-shaped pressure display, independent of the RTX hand surface."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom
from revosim.resources import SIDES


class PressureDisplay:
    def __init__(self, world, font, small):
        self.font = font
        self.small = small
        self.hands = {}
        layout = world.sensor_layout()["hands"]
        stage = world.sim.stage
        cache = UsdGeom.XformCache()
        for side in SIDES:
            hand = world.hands[side]
            states = hand.data.body_link_state_w[0, :, :7].cpu().numpy()
            geometry = []
            taxels = []
            regions = []
            for prim in Usd.PrimRange(
                stage.GetPrimAtPath(f"/World/{side.capitalize()}")
            ):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                    continue
                link = prim
                while link.IsValid() and link.GetName() not in hand.body_names:
                    link = link.GetParent()
                if not link.IsValid():
                    continue
                i = hand.body_names.index(link.GetName())
                pose = states[i]
                rot = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                mesh = UsdGeom.Mesh(prim)
                vertices = np.asarray(mesh.GetPointsAttr().Get())
                mat = np.array(
                    cache.GetLocalToWorldTransform(prim)
                    * cache.GetLocalToWorldTransform(link).GetInverse()
                )
                vertices = (vertices @ mat[:3, :3] + mat[3, :3]) @ rot.T + pose[:3]
                idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                counts = mesh.GetFaceVertexCountsAttr().Get()
                start = 0
                for count in counts:
                    geometry.append(vertices[idx[start : start + count], :2])
                    start += count
            for region in layout[side]["pressure_regions"]:
                pose = states[hand.body_names.index(region["link"])]
                rot = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                points = np.asarray(region["points_local_m"]) @ rot.T + pose[:3]
                taxels.extend(points[:, :2])
                regions.append(region)
            vertices = np.concatenate(geometry)
            lo = vertices.min(0)
            hi = vertices.max(0)
            scale = min(580 / (hi[0] - lo[0]), 435 / (hi[1] - lo[1]))
            center = (lo + hi) / 2

            def project(points):
                return (points - center) * np.array([scale, -scale]) + [615, 275]

            mask = Image.new("L", (1250, 535))
            draw = ImageDraw.Draw(mask)
            for face in geometry:
                draw.polygon([tuple(p) for p in project(face)], fill=255)
            edge = mask.filter(ImageFilter.MaxFilter(3))
            background = Image.new("RGB", mask.size, (18, 25, 36))
            background.paste((72, 89, 111), (0, 0), edge)
            background.paste((27, 39, 54), (0, 0), mask)
            xy = project(np.asarray(taxels))
            draw = ImageDraw.Draw(background)
            for region in regions:
                name = region["region"]
                p = xy[region["start"] : region["stop"]]
                a = p.min(0) - [6, 7]
                b = p.max(0) + [6, 7]
                draw.rounded_rectangle(
                    (*a, *b), radius=4, outline=(79, 105, 130), width=1
                )
                if name == "palm":
                    label_x = b[0] + 15 if side == "left" else a[0] - 145
                    draw.text(
                        (label_x, (a[1] + b[1]) / 2 - 12),
                        "PALM / 117",
                        font=font,
                        fill="#afc1d3",
                    )
                else:
                    label = name.rsplit("_", 1)[1].upper()
                    draw.text((a[0], a[1] - 16), label, font=small, fill="#92a9bf")
            # Finger names sit above the actual finger axes, not above an unrelated grid.
            for finger in ["thumb", "index", "middle", "ring", "little"]:
                region = next(r for r in regions if r["region"] == finger + "_pip")
                p = xy[region["start"] : region["stop"]]
                at = p.mean(0)
                draw.text(
                    (at[0] - 15, p[:, 1].min() - 43), finger, font=small, fill="#c3d2df"
                )
            self.hands[side] = (background, xy)

    def draw(self, pressure):
        canvas = Image.new("RGB", (2560, 550), (12, 18, 27))
        draw = ImageDraw.Draw(canvas)
        for h, side in enumerate(SIDES):
            x = 15 + h * 1280
            im, xy = self.hands[side]
            im = im.copy()
            d = ImageDraw.Draw(im)
            d.rounded_rectangle(
                (0, 0, 1248, 533), radius=14, outline="#38506a", width=2
            )
            d.text(
                (22, 16),
                side.upper() + " / PRESSURE",
                font=self.font,
                fill="#72c8f4" if h == 0 else "#edc389",
            )
            d.text((22, 53), "247 original channels", font=self.small, fill="#b2c3d4")
            d.text((22, 78), "117 palm + 130 finger", font=self.small, fill="#b2c3d4")
            d.text((22, 110), "MCP: proximal", font=self.small, fill="#8da6bc")
            d.text((22, 134), "PIP: middle", font=self.small, fill="#8da6bc")
            d.text(
                (22, 176),
                "Max: %.3f" % float(pressure[h].max()),
                font=self.font,
                fill="#ffd489",
            )
            for (xx, yy), value in zip(xy, pressure[h]):
                u = float(np.clip(value, 0, 1))
                color = (
                    int(62 + 193 * u),
                    int(82 + 125 * np.sqrt(u)),
                    int(105 * (1 - u)),
                )
                r = 2.4 if u < 0.01 else 3.4
                d.ellipse((xx - r, yy - r, xx + r, yy + r), fill=color)
            d.text(
                (22, 474),
                "Open-hand reference projection",
                font=self.small,
                fill="#8da6bc",
            )
            d.text(
                (22, 500),
                "Normalized response | display 0–1",
                font=self.small,
                fill="#8da6bc",
            )
            canvas.paste(im, (x, 7))
        return canvas
