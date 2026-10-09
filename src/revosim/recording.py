"""Lossless numeric observations plus a synchronized human-readable dashboard."""

import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio


class Recorder:
    def __init__(self, directory, fps=60, save_data=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            self.directory / "dashboard.mp4",
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=["-threads", "4", "-crf", "18", "-movflags", "+faststart"],
        )
        self.events = (self.directory / "frames.jsonl").open("w")
        self.count = 0
        self.data = None
        if save_data:
            import h5py

            self.data = h5py.File(self.directory / "observations.h5", "w")
        (self.directory / "metadata.json").write_text(
            json.dumps(
                {
                    "fps": fps,
                    "physics_hz": 240,
                    "sides": ["left", "right"],
                    "fingers": ["thumb", "index", "middle", "ring", "little"],
                    "rgb": "float32 [0,1], original simulated sensor RGB",
                    "depth": "metres",
                    "marker": "metres, DIP_rubber_link local xyz, 100 original marker indices; consult validity mask",
                    "wrench": "[Fx,Fy,Fz,Tx,Ty,Tz], N and N m, DIP_rubber_link origin and local frame; only DIP_rubber_link contacts, normal + friction",
                    "pressure": "247 normalized diffused responses per hand, can exceed 1; not Pa/N/ohms",
                    "scope": "two sensor-equipped hands; object_pose is the first object, object_poses contains all objects",
                },
                indent=2,
            )
        )

    def append(self, values, image, phase):
        self.writer.append_data(image)
        self.events.write(
            json.dumps({"frame": self.count, "time": values["time"], "phase": phase})
            + "\n"
        )
        if self.data is not None:
            for k, v in values.items():
                v = np.asarray(v)
                if k not in self.data:
                    self.data.create_dataset(
                        k,
                        shape=(0, *v.shape),
                        maxshape=(None, *v.shape),
                        dtype=v.dtype,
                        chunks=(1, *v.shape),
                        compression="gzip",
                        compression_opts=2,
                        shuffle=True,
                    )
                d = self.data[k]
                d.resize(self.count + 1, axis=0)
                d[self.count] = v
        self.count += 1

    def close(self):
        self.writer.close()
        self.events.close()
        if self.data is not None:
            self.data.close()
