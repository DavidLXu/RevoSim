"""Left/right optical panels, a clean central simulation, and hand-shaped pressure."""

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import UsdGeom, UsdLux, Gf
import omni.replicator.core as rep
from revosim.resources import SIDES, FINGERS, sensor_paths
from revosim.pressure_display import PressureDisplay


def font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class Dashboard:
    def __init__(self, world, live=False):
        self.world = world
        stage = world.sim.stage
        self.font = font(20)
        self.small = font(15)
        self.tiny = font(12)
        self.title = font(29)
        cam = UsdGeom.Camera.Define(stage, "/World/OverviewCamera")
        cam.CreateFocalLengthAttr(42)
        cam.CreateHorizontalApertureAttr(36)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10))
        eye = (
            Gf.Vec3d(0, -0.26, 0.80)
            if world.hand_spacing <= 0.21
            else Gf.Vec3d(0, -0.29, 0.90)
        )
        target = Gf.Vec3d(0, -0.005, 0.18)
        UsdGeom.Xformable(cam).AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        world.sim.set_camera_view(tuple(eye), tuple(target))
        self.product = rep.create.render_product(str(cam.GetPath()), (1040, 780))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb.attach(self.product)
        for name, pos, power, w, h in [
            ("Key", (0, -0.2, 0.75), 2200, 0.5, 0.5),
            ("Rim", (0.35, 0.3, 0.55), 3000, 0.2, 0.5),
            ("Fill", (-0.4, 0, 0.55), 1300, 0.3, 0.6),
        ]:
            light = UsdLux.RectLight.Define(stage, "/World/" + name)
            light.CreateIntensityAttr(power)
            light.CreateWidthAttr(w)
            light.CreateHeightAttr(h)
            light.CreateNormalizeAttr(False)
            UsdGeom.Xformable(light).AddTransformOp().Set(
                Gf.Matrix4d()
                .SetLookAt(Gf.Vec3d(*pos), target, Gf.Vec3d(0, 1, 0))
                .GetInverse()
            )
        self.pressure = PressureDisplay(world, self.font, self.tiny)
        self.markers = {}
        for side in SIDES:
            d = world.calibration[side]
            self.markers[side] = []
            rects = json.loads(
                (
                    sensor_paths(side)["vtac"] / "camera_ray_rectangles_320x240.json"
                ).read_text()
            )["rectangles"]
            for finger in FINGERS:
                f = "pinky" if finger == "little" else finger
                pts = d[f + "_points_camera_m"][:, :2]
                R = d[f + "_camera_rotation_link"]
                valid = (
                    d["distortion_valid"]
                    & (d[f + "_method"] == "ray_hit")
                    & d[f + "_marker_ray_valid"]
                )
                rect = np.array(rects[f]["corners_camera_m"])[:, :2]
                self.markers[side].append((pts, R, valid, rect.min(0), rect.max(0)))
        self.provider = None
        if live:
            import omni.ui as ui

            self.window = ui.Window(
                "RevoSim — multimodal tactile dashboard", width=1600, height=1000
            )
            self.provider = ui.ByteImageProvider()
            with self.window.frame:
                ui.ImageWithProvider(self.provider)
        for _ in range(8):
            world.sim.render()

    def draw(self, values, phase):
        self.world.sim.render()
        self.world.sim.render()
        pixels = self.rgb.get_data()[..., :3]
        if pixels.shape != (780, 1040, 3):
            raise RuntimeError("Overview camera has no image")
        canvas = Image.new("RGB", (2560, 1600), (12, 18, 27))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (25, 20),
            "RevoSim / High-fidelity tactile hands",
            font=self.title,
            fill="white",
        )
        draw.text(
            (25, 62),
            f"{self.world.shape.upper()}  |  {phase}",
            font=self.font,
            fill="#aabfd4",
        )
        canvas.paste(Image.fromarray(pixels), (760, 170))
        draw.rounded_rectangle(
            (750, 112, 1808, 1027), radius=14, outline="#38506a", width=2
        )
        draw.text((785, 128), "LEFT HAND", font=self.font, fill="#72c8f4")
        draw.text((1620, 128), "RIGHT HAND", font=self.font, fill="#edc389")
        draw.text(
            (785, 955),
            f"t = {values['time']:.2f} s  |  PhysX 240 Hz  |  clean model surfaces",
            font=self.font,
            fill="#c4d2df",
        )
        draw.text(
            (785, 991),
            "Pressure channels appear in the two hand-shaped maps below.",
            font=self.small,
            fill="#8ea8bf",
        )
        for h, side in enumerate(SIDES):
            x0 = 14 if h == 0 else 1825
            color = "#72c8f4" if h == 0 else "#edc389"
            draw.text(
                (x0 + 8, 112),
                side.upper() + " / OPTICAL TACTILE + TIP WRENCH",
                font=self.font,
                fill=color,
            )
            for offset, label in [
                (10, "Raw RGB"),
                (177, "Marker x4"),
                (344, "Depth / mm"),
                (512, "Tip wrench N, N·mm"),
            ]:
                draw.text((x0 + offset, 148), label, font=self.small, fill="#d2dce5")
            for f, finger in enumerate(FINGERS):
                y = 179 + f * 169
                draw.rounded_rectangle(
                    (x0, y, x0 + 720, y + 158),
                    radius=8,
                    outline="#34465b",
                    fill=(20, 29, 41),
                )
                draw.text((x0 + 9, y + 5), finger.upper(), font=self.small, fill=color)
                yy = y + 30
                rgb = np.rint(
                    np.clip(values["rgb"][h, f].transpose(1, 2, 0), 0, 1) * 255
                ).astype(np.uint8)
                canvas.paste(Image.fromarray(rgb).resize((156, 117)), (x0 + 8, yy))
                u = np.clip(values["depth"][h, f] / 0.004, 0, 1)
                heat = np.stack(
                    [255 * u, 220 * np.sqrt(u), 30 + 140 * u], axis=-1
                ).astype(np.uint8)
                canvas.paste(Image.fromarray(heat).resize((156, 117)), (x0 + 342, yy))
                draw.text(
                    (x0 + 390, y + 5),
                    f"{values['depth'][h, f].max() * 1000:.2f} mm",
                    font=self.tiny,
                    fill="#c4d2df",
                )
                panel = Image.new("RGB", (156, 117), "#080f18")
                pd = ImageDraw.Draw(panel)
                pts, R, valid, lo, hi = self.markers[side][f]
                xy = (pts - lo) / (hi - lo) * [155, 116]
                delta = (values["marker"][h, f] @ R)[:, :2] / (hi - lo) * [155, 116] * 4
                for i, ((xx, zz), (dx, dz)) in enumerate(zip(xy, delta)):
                    if not valid[i]:
                        continue
                    pd.ellipse((xx - 1, zz - 1, xx + 1, zz + 1), fill="#74889b")
                    if np.hypot(dx, dz) > 0.15:
                        pd.line((xx, zz, xx + dx, zz + dz), fill="#ffd56c", width=1)
                canvas.paste(panel, (x0 + 175, yy))
                wr = values["wrench"][h, f].copy()
                wr[3:] *= 1000
                for j, (name, value) in enumerate(
                    zip(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], wr)
                ):
                    z = yy + j * 19
                    draw.text(
                        (x0 + 510, z),
                        f"{name} {value:+7.3f}",
                        font=self.small,
                        fill="#cfdae6",
                    )
                    base = x0 + 664
                    scale = 24 / (2 if j < 3 else 10)
                    draw.line((base, z + 2, base, z + 14), fill="#52657a")
                    dx = float(np.clip(value * scale, -38, 38))
                    draw.line((base, z + 8, base + dx, z + 8), fill="#68d4b5", width=4)
        canvas.paste(self.pressure.draw(values["pressure"]), (0, 1045))
        if self.provider:
            self.provider.set_data_array(
                np.asarray(canvas.convert("RGBA")).ravel(), [2560, 1600]
            )
        return np.asarray(canvas)

    def close(self):
        self.rgb.detach(self.product)
        self.product.destroy()
