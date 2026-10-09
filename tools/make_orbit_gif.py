#!/usr/bin/env python3
"""Export a looping README GIF from a complete orbit video."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--colors", type=int, default=256)
    p.add_argument("--dither", choices=("none", "bayer", "sierra2_4a"), default="bayer")
    p.add_argument(
        "--lossy",
        type=int,
        default=60,
        help="Optional gifsicle compression; 0 is lossless",
    )
    a = p.parse_args()
    if a.width <= 0 or a.fps <= 0 or not 4 <= a.colors <= 256:
        p.error("width/fps must be positive and colors between 4 and 256")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filters = (
        f"fps={a.fps},scale={a.width}:-1:flags=lanczos,split[video][palette];"
        f"[palette]palettegen=max_colors={a.colors}:stats_mode=single[p];"
        f"[video][p]paletteuse=new=1:dither={a.dither}:bayer_scale=5:diff_mode=rectangle"
    )
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(a.video),
            "-filter_complex_threads",
            "2",
            "-filter_complex",
            filters,
            "-threads",
            "4",
            "-loop",
            "0",
            str(a.output),
        ],
        check=True,
    )
    gifsicle = shutil.which("gifsicle")
    if gifsicle:
        with tempfile.TemporaryDirectory(
            prefix="revosim-gif-", dir=a.output.parent
        ) as tmp:
            optimized = Path(tmp) / "optimized.gif"
            subprocess.run(
                [
                    gifsicle,
                    "-O3",
                    f"--lossy={a.lossy}",
                    str(a.output),
                    "-o",
                    str(optimized),
                ],
                check=True,
            )
            optimized.replace(a.output)
    else:
        print(
            "Install gifsicle for a smaller GIF; the complete animation is still saved."
        )
    print(f"Saved {a.output}: {a.output.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
