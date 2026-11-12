"""Write the synthetic test clip for the browser test: pitch-blip chords under a drum + bass bed.

Run: .venv/bin/python -m tests.browser.make_clip <out_dir>
Writes song.wav (8 s) and, when ffmpeg is installed, song.m4a; prints the synth's span.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from tests.e2e_pipeline import REGION, build_song


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/browser")
    out.mkdir(parents=True, exist_ok=True)
    build_song(out)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "song.wav"), "-c:a", "aac", "-b:a", "192k",
                        str(out / "song.m4a")], check=True)
    print(f"{out / 'song.wav'} region {REGION[0]} {REGION[1]}")


if __name__ == "__main__":
    main()
