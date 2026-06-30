# VHS Glitch

[![Support me on Patreon](https://img.shields.io/badge/Patreon-Support%20my%20work-FF424D?style=flat&logo=patreon&logoColor=white)](https://www.patreon.com/AndersBjarby)

Beat-synced VHS tracking glitch effects for video. It simulates a badly tuned VHS player — horizontal band displacement, color channel separation, noise bands, head-switching artifacts and analog grain — all driven by audio onsets detected in three frequency bands (bass, mid, high).

## What it does

- Analyzes the video's audio with librosa, detecting onsets in separate bass / mid / high bands.
- Drives VHS-style distortion from those onsets: tracking displacement, chroma separation, noise, head-switch artifacts, wobble, washout, grain and scanlines.
- Ships presets (`subtle`, `normal`, `heavy`, `destroyed`) and an interactive tuning loop for dialing in the look.
- Renders the result back out with ffmpeg.

## Usage

Requires Python with `librosa`, `numpy`, `opencv-python` (cv2), and `ffmpeg` available on your PATH.

```bash
# Interactive mode
python vhs_glitch.py video.mp4

# One-shot render with a preset
python vhs_glitch.py video.mp4 --preset destroyed

# Custom frequency ranges: bass_lo bass_hi mid_lo mid_hi high_lo high_hi
python vhs_glitch.py video.mp4 20 200 200 2000 4000 8000
```

The output is written next to the input as `<name>_vhs.mp4`.

## Tech

Python, librosa, NumPy, OpenCV, ffmpeg.
