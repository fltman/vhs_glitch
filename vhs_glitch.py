#!/usr/bin/env python3
"""
Beat-synced VHS tracking glitch effects.
Simulates a badly tuned VHS player: horizontal band displacement,
color channel separation, noise bands, head-switching artifacts,
and analog grain — all driven by audio onsets.
"""

import librosa
import numpy as np
import cv2
import subprocess
import sys
import os


# ── audio analysis ──────────────────────────────────────────────

def analyze_audio(video_file,
                  bass_range=(20, 200),
                  mid_range=(200, 2000),
                  high_range=(4000, 8000)):
    """Detect onsets in three frequency bands."""
    print(f"🎵 Analyzing audio from {video_file}...")
    y, sr = librosa.load(video_file, sr=22050)
    D = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr)

    def _band_onsets(lo, hi, hop, **kw):
        mask = (freqs >= lo) & (freqs <= hi)
        Db = D.copy()
        Db[~mask, :] = 0
        yb = librosa.istft(Db)
        return librosa.onset.onset_detect(y=yb, sr=sr, units='time',
                                          hop_length=hop, backtrack=True, **kw)

    bass = _band_onsets(*bass_range, 512)
    mid  = _band_onsets(*mid_range,  256)
    high = _band_onsets(*high_range, 128,
                        pre_max=3, post_max=3,
                        pre_avg=3, post_avg=3,
                        delta=0.2, wait=10)

    # Also compute a continuous loudness envelope for subtle constant wobble
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    print(f"   Bass onsets: {len(bass)}  Mid: {len(mid)}  High: {len(high)}")
    return {'bass': bass, 'mid': mid, 'high': high,
            'rms': rms, 'rms_times': rms_times}


# ── envelope helpers ────────────────────────────────────────────

def compute_envelope(total_frames, fps, beat_times, duration, softness):
    """Smooth per-frame envelope 0-1 for a list of beat times."""
    env = np.zeros(total_frames, dtype=np.float32)
    fade = max(duration * softness, 0.001) if softness > 0 else 0
    for bt in beat_times:
        margin = duration + fade + 0.05
        sf = max(0, int((bt - margin) * fps))
        ef = min(total_frames, int((bt + duration + margin) * fps) + 1)
        for f in range(sf, ef):
            t = f / fps
            if softness > 0:
                ru = max(0, (t - bt) / fade)
                rd = max(0, (bt + duration - t) / fade)
                val = min(ru, 1.0, rd)
            else:
                val = 1.0 if bt <= t <= bt + duration else 0.0
            env[f] = min(1.0, env[f] + val)
    return env


def rms_at_frame(frame_idx, fps, rms, rms_times):
    """Interpolate RMS value for a given frame."""
    t = frame_idx / fps
    idx = np.searchsorted(rms_times, t)
    idx = min(idx, len(rms) - 1)
    return float(rms[idx])


# ── VHS glitch primitives ──────────────────────────────────────

def glitch_row_shift(frame, intensity, rng, num_bands=None, band_height_range=None):
    """Shift horizontal bands of rows left/right.
    
    This is the core "bad tracking" look: chunks of scan-lines
    get displaced horizontally by different amounts.
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    if num_bands is None:
        num_bands = rng.integers(3, 12)
    if band_height_range is None:
        band_height_range = (4, max(5, h // 8))

    max_shift = int(w * 0.35 * intensity)
    if max_shift < 1:
        return out

    for _ in range(num_bands):
        bh = rng.integers(band_height_range[0], band_height_range[1])
        y0 = rng.integers(0, max(1, h - bh))
        y1 = min(y0 + bh, h)
        shift = rng.integers(-max_shift, max_shift + 1)
        if shift == 0:
            continue
        band = out[y0:y1]
        out[y0:y1] = np.roll(band, shift, axis=1)
        # Fill the wrapped edge with a smeared column
        if shift > 0:
            out[y0:y1, :shift] = out[y0:y1, shift:shift+1]
        else:
            out[y0:y1, shift:] = out[y0:y1, shift-1:shift]

    return out


def glitch_color_separation(frame, intensity, rng):
    """Offset RGB channels horizontally — chromatic aberration / color bleed."""
    h, w = frame.shape[:2]
    max_offset = int(w * 0.04 * intensity)
    if max_offset < 1:
        return frame

    b, g, r = cv2.split(frame)
    r_shift = rng.integers(-max_offset, max_offset + 1)
    b_shift = rng.integers(-max_offset, max_offset + 1)
    r = np.roll(r, r_shift, axis=1)
    b = np.roll(b, b_shift, axis=1)
    return cv2.merge([b, g, r])


def glitch_noise_bands(frame, intensity, rng, num_bands=None):
    """Horizontal bands of static noise — like head-clog dropouts."""
    h, w = frame.shape[:2]
    out = frame.copy()

    if num_bands is None:
        num_bands = rng.integers(1, 6)

    for _ in range(num_bands):
        bh = rng.integers(1, max(2, int(6 * intensity) + 2))
        y0 = rng.integers(0, max(1, h - bh))
        y1 = min(y0 + bh, h)
        noise = rng.integers(0, 256, size=(y1 - y0, w, 3), dtype=np.uint8)
        alpha = 0.3 + 0.7 * intensity
        out[y0:y1] = cv2.addWeighted(out[y0:y1], 1 - alpha, noise, alpha, 0)

    return out


def glitch_head_switch(frame, intensity, rng):
    """Bottom-of-frame head switching noise — the classic VHS bar at the bottom."""
    h, w = frame.shape[:2]
    out = frame.copy()

    bar_h = int(h * (0.02 + 0.08 * intensity))
    y0 = h - bar_h

    # Shift the bottom bar significantly
    shift = rng.integers(int(-w * 0.3), int(w * 0.3) + 1)
    out[y0:] = np.roll(out[y0:], shift, axis=1)

    # Add some noise on top
    noise = rng.integers(0, 256, size=(bar_h, w, 3), dtype=np.uint8)
    out[y0:] = cv2.addWeighted(out[y0:], 0.5, noise, 0.5, 0)

    return out


def glitch_vertical_roll(frame, intensity, rng):
    """Vertical frame roll — when tracking is really off, the whole
    image slides up or down, wrapping around."""
    h = frame.shape[0]
    max_roll = int(h * 0.3 * intensity)
    if max_roll < 1:
        return frame
    shift = rng.integers(-max_roll, max_roll + 1)
    return np.roll(frame, shift, axis=0)


def apply_scanlines(frame, opacity=0.15):
    """Faint CRT/VHS scan-line overlay (every other row darkened)."""
    out = frame.astype(np.float32)
    out[::2] *= (1.0 - opacity)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_vhs_grain(frame, intensity, rng):
    """Analog-style luma noise — the general fuzz of VHS."""
    noise = rng.normal(0, 12 * intensity, frame.shape).astype(np.float32)
    out = frame.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_washout(frame, amount=0.15):
    """Desaturate + slightly shift hue for that aged-tape look."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1.0 - amount)          # lower saturation
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 - amount * 0.3) + amount * 20  # slight lift
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ── main render pipeline ───────────────────────────────────────

def render_vhs(input_file, output_file, beats, cfg):
    """Frame-by-frame VHS glitch render."""
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   Video: {width}x{height} @ {fps:.2f} fps, {total} frames")

    # Pre-compute envelopes for each band
    print("   Computing beat envelopes...")
    bass_env = compute_envelope(total, fps,
                                beats['bass'][:cfg['max_bass']],
                                cfg['bass_dur'], cfg['bass_soft'])
    mid_env  = compute_envelope(total, fps,
                                beats['mid'][:cfg['max_mid']],
                                cfg['mid_dur'], cfg['mid_soft'])
    high_env = compute_envelope(total, fps,
                                beats['high'][:cfg['max_high']],
                                cfg['high_dur'], cfg['high_soft'])

    rms      = beats['rms']
    rms_t    = beats['rms_times']
    rng      = np.random.default_rng(42)

    # ffmpeg output pipe
    cmd = [
        'ffmpeg',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-',
        '-i', input_file,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-c:a', 'copy', '-pix_fmt', 'yuv420p',
        '-y', output_file
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for f in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        b = bass_env[f]
        m = mid_env[f]
        h = high_env[f]
        r = rms_at_frame(f, fps, rms, rms_t)

        # ── constant base layer (always-on VHS look) ──
        if cfg['scanlines']:
            frame = apply_scanlines(frame, opacity=cfg['scanline_opacity'])
        if cfg['grain'] > 0:
            frame = apply_vhs_grain(frame, cfg['grain'], rng)
        if cfg['washout'] > 0:
            frame = apply_color_washout(frame, cfg['washout'])

        # ── subtle constant wobble driven by RMS loudness ──
        if cfg['wobble'] > 0:
            wobble_intensity = r * cfg['wobble']
            if wobble_intensity > 0.01:
                small_shift = int(wobble_intensity * width * 0.02)
                if small_shift > 0:
                    # Gentle horizontal shift of a few random rows
                    for _ in range(rng.integers(1, 4)):
                        bh = rng.integers(2, max(3, height // 20))
                        y0 = rng.integers(0, max(1, height - bh))
                        y1 = min(y0 + bh, height)
                        s = rng.integers(-small_shift, small_shift + 1)
                        frame[y0:y1] = np.roll(frame[y0:y1], s, axis=1)

        # ── bass → big tracking displacement ──
        if b > 0.01:
            intensity = b * cfg['track_intensity']
            frame = glitch_row_shift(frame, intensity, rng)
            # Heavy hits also roll the frame vertically
            if b > 0.5:
                frame = glitch_vertical_roll(frame, (b - 0.5) * 2 * cfg['roll_intensity'], rng)

        # ── mid → color channel separation ──
        if m > 0.01:
            frame = glitch_color_separation(frame, m * cfg['chroma_intensity'], rng)

        # ── high → noise bands + head-switch bar ──
        if h > 0.01:
            frame = glitch_noise_bands(frame, h * cfg['noise_intensity'], rng)
            if h > 0.3:
                frame = glitch_head_switch(frame, h * cfg['headswitch_intensity'], rng)

        proc.stdin.write(frame.tobytes())

        if f % 100 == 0:
            pct = 100 * f / total
            print(f"   Frame {f}/{total} ({pct:.0f}%)", end='\r')

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"   Frame {total}/{total} (100%)       ")
    return proc.returncode == 0


# ── interactive config ──────────────────────────────────────────

DEFAULT_CFG = {
    # bass → tracking glitch
    'track_intensity':     1.0,
    'roll_intensity':      0.5,
    'bass_dur':            0.15,
    'bass_soft':           0.4,
    'max_bass':            5000,
    # mid → color separation
    'chroma_intensity':    1.0,
    'mid_dur':             0.10,
    'mid_soft':            0.3,
    'max_mid':             5000,
    # high → noise / head-switch
    'noise_intensity':     1.0,
    'headswitch_intensity':1.0,
    'high_dur':            0.08,
    'high_soft':           0.2,
    'max_high':            5000,
    # constant base layer
    'scanlines':           True,
    'scanline_opacity':    0.12,
    'grain':               0.4,
    'washout':             0.15,
    'wobble':              0.5,
}

PRESETS = {
    'subtle': {
        'track_intensity': 0.4, 'roll_intensity': 0.1,
        'chroma_intensity': 0.4, 'noise_intensity': 0.3,
        'headswitch_intensity': 0.3,
        'grain': 0.2, 'washout': 0.08, 'wobble': 0.2,
        'scanline_opacity': 0.06,
    },
    'normal': {},  # uses defaults
    'heavy': {
        'track_intensity': 1.8, 'roll_intensity': 1.0,
        'chroma_intensity': 1.5, 'noise_intensity': 1.5,
        'headswitch_intensity': 1.5,
        'grain': 0.7, 'washout': 0.25, 'wobble': 0.8,
        'bass_dur': 0.20, 'bass_soft': 0.3,
    },
    'destroyed': {
        'track_intensity': 3.0, 'roll_intensity': 2.0,
        'chroma_intensity': 2.5, 'noise_intensity': 2.5,
        'headswitch_intensity': 2.0,
        'grain': 1.0, 'washout': 0.35, 'wobble': 1.5,
        'bass_dur': 0.25, 'bass_soft': 0.2,
        'high_dur': 0.15,
    },
}


def show_config(cfg, beats):
    """Display current config."""
    print(f"\n{'─'*55}")
    print(f"  TRACKING (bass hits: {len(beats['bass'])})")
    print(f"    intensity: {cfg['track_intensity']:.1f}   "
          f"roll: {cfg['roll_intensity']:.1f}   "
          f"dur: {cfg['bass_dur']:.2f}s   "
          f"soft: {cfg['bass_soft']:.1f}")
    print(f"  COLOR BLEED (mid hits: {len(beats['mid'])})")
    print(f"    intensity: {cfg['chroma_intensity']:.1f}   "
          f"dur: {cfg['mid_dur']:.2f}s   "
          f"soft: {cfg['mid_soft']:.1f}")
    print(f"  NOISE / HEAD-SWITCH (high hits: {len(beats['high'])})")
    print(f"    noise: {cfg['noise_intensity']:.1f}   "
          f"head-sw: {cfg['headswitch_intensity']:.1f}   "
          f"dur: {cfg['high_dur']:.2f}s   "
          f"soft: {cfg['high_soft']:.1f}")
    print(f"  BASE LAYER")
    print(f"    grain: {cfg['grain']:.2f}   "
          f"washout: {cfg['washout']:.2f}   "
          f"wobble: {cfg['wobble']:.2f}   "
          f"scanlines: {'on' if cfg['scanlines'] else 'off'} ({cfg['scanline_opacity']:.2f})")
    print(f"{'─'*55}")


def interactive_loop(video_file, beats):
    """Main interactive config + render loop."""
    base = os.path.splitext(video_file)[0]
    output_file = f"{base}_vhs.mp4"
    cfg = dict(DEFAULT_CFG)

    print("\n" + "=" * 55)
    print("  📼  VHS TRACKING GLITCH MACHINE  📼")
    print("=" * 55)

    while True:
        show_config(cfg, beats)

        print("\n  Commands:")
        print("    (r)ender          (p)reset [subtle/normal/heavy/destroyed]")
        print("    (t)racking cfg    (c)olor cfg     (n)oise cfg")
        print("    (b)ase layer cfg  (o)utput name   (q)uit")

        choice = input("\n  📼 > ").lower().strip()

        if choice == 'r':
            print(f"\n  🎬 Rendering → {output_file}\n")
            ok = render_vhs(video_file, output_file, beats, cfg)
            if ok:
                print(f"\n  ✓ Saved: {output_file}")
                again = input("  Happy? (y)es / (n)o, adjust: ").lower().strip()
                if again == 'y':
                    print("  ✓ Done!")
                    break
            else:
                print("  ❌ Render failed.")

        elif choice.startswith('p'):
            parts = choice.split()
            name = parts[1] if len(parts) > 1 else input("  Preset [subtle/normal/heavy/destroyed]: ").strip()
            if name in PRESETS:
                cfg = dict(DEFAULT_CFG)
                cfg.update(PRESETS[name])
                print(f"  ✓ Loaded preset: {name}")
            else:
                print(f"  Unknown preset: {name}")

        elif choice == 't':
            print(f"  Current tracking intensity: {cfg['track_intensity']:.1f}")
            v = input("  New tracking intensity (0-3): ")
            if v: cfg['track_intensity'] = float(v)
            v = input(f"  Roll intensity [{cfg['roll_intensity']:.1f}]: ")
            if v: cfg['roll_intensity'] = float(v)
            v = input(f"  Duration [{cfg['bass_dur']:.2f}]: ")
            if v: cfg['bass_dur'] = float(v)
            v = input(f"  Softness [{cfg['bass_soft']:.1f}]: ")
            if v: cfg['bass_soft'] = float(v)

        elif choice == 'c':
            v = input(f"  Chroma separation intensity [{cfg['chroma_intensity']:.1f}]: ")
            if v: cfg['chroma_intensity'] = float(v)
            v = input(f"  Duration [{cfg['mid_dur']:.2f}]: ")
            if v: cfg['mid_dur'] = float(v)

        elif choice == 'n':
            v = input(f"  Noise band intensity [{cfg['noise_intensity']:.1f}]: ")
            if v: cfg['noise_intensity'] = float(v)
            v = input(f"  Head-switch intensity [{cfg['headswitch_intensity']:.1f}]: ")
            if v: cfg['headswitch_intensity'] = float(v)
            v = input(f"  Duration [{cfg['high_dur']:.2f}]: ")
            if v: cfg['high_dur'] = float(v)

        elif choice == 'b':
            v = input(f"  Grain [{cfg['grain']:.2f}]: ")
            if v: cfg['grain'] = float(v)
            v = input(f"  Color washout [{cfg['washout']:.2f}]: ")
            if v: cfg['washout'] = float(v)
            v = input(f"  Wobble [{cfg['wobble']:.2f}]: ")
            if v: cfg['wobble'] = float(v)
            v = input(f"  Scanlines on/off [{cfg['scanlines']}]: ")
            if v: cfg['scanlines'] = v.lower() in ('on', 'true', '1', 'yes')
            if cfg['scanlines']:
                v = input(f"  Scanline opacity [{cfg['scanline_opacity']:.2f}]: ")
                if v: cfg['scanline_opacity'] = float(v)

        elif choice == 'o':
            v = input(f"  Output filename [{output_file}]: ")
            if v: output_file = v

        elif choice == 'q':
            print("  Bye!")
            sys.exit(0)


# ── entry point ─────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python vhs_glitch.py <video.mp4>")
        print("\nOptional frequency ranges:")
        print("  python vhs_glitch.py video.mp4 <bass_lo> <bass_hi> <mid_lo> <mid_hi> <hi_lo> <hi_hi>")
        print("\nPresets can be set as last arg:")
        print("  python vhs_glitch.py video.mp4 --preset destroyed")
        sys.exit(1)

    video_file = sys.argv[1]

    # Parse frequency ranges
    if len(sys.argv) >= 8 and not sys.argv[2].startswith('-'):
        bass_range = (int(sys.argv[2]), int(sys.argv[3]))
        mid_range  = (int(sys.argv[4]), int(sys.argv[5]))
        high_range = (int(sys.argv[6]), int(sys.argv[7]))
    else:
        bass_range = (20, 200)
        mid_range  = (200, 2000)
        high_range = (4000, 8000)

    beats = analyze_audio(video_file, bass_range, mid_range, high_range)

    # Quick non-interactive mode with --preset
    if '--preset' in sys.argv:
        idx = sys.argv.index('--preset')
        preset_name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'normal'
        cfg = dict(DEFAULT_CFG)
        if preset_name in PRESETS:
            cfg.update(PRESETS[preset_name])
        base = os.path.splitext(video_file)[0]
        output_file = f"{base}_vhs.mp4"
        print(f"\n  Rendering with preset '{preset_name}' → {output_file}\n")
        render_vhs(video_file, output_file, beats, cfg)
        print(f"\n  ✓ Done: {output_file}")
    else:
        interactive_loop(video_file, beats)


if __name__ == '__main__':
    main()
