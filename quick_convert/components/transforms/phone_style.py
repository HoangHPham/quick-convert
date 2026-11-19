"""Noise-free simulated VoIP phone channel."""

import hashlib
import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf


class PhoneStyleConverter:
    def __init__(self, severity: float | None = None, seed: int = 2026):
        if severity is not None and not 0 <= severity <= 1:
            raise ValueError("severity must be in [0, 1] or None")
        self.severity = severity
        self.seed = seed

    def __call__(self, x: np.ndarray, sr: int, *, key: str = "") -> np.ndarray:
        """Convert mono audio to 16 kHz mono phone-style audio."""
        from scipy.signal import butter, resample_poly, sosfilt

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1 or not len(x) or sr <= 0:
            raise ValueError("Expected nonempty mono waveform and positive sample rate")
        if sr != 16_000:
            d = gcd(sr, 16_000)
            x = resample_poly(x, 16_000 // d, sr // d).astype(np.float32)
        level = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        if level < 1e-10:
            return np.zeros_like(x)
        x = x * (0.07 / level)

        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        severity = float(rng.uniform()) if self.severity is None else self.severity
        hp = int(np.clip(95 + 100 * severity + rng.uniform(-15, 15), 70, 250))
        lp = int(np.clip(7000 - 3500 * severity + rng.uniform(-200, 200), 3100, 7300))
        bitrate = int(2 * round((32 - 16 * severity) / 2))
        ratio = 1.3 + 2.0 * severity

        y = sosfilt(butter(3, [hp, lp], btype="bandpass", fs=16_000, output="sos"), x)
        hop = 320
        frames = np.pad(y, (0, (-len(y)) % hop)).reshape(-1, hop)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-10)
        desired = np.clip(0.07 / (frame_rms + 0.01), 0.65, 2.5)
        smooth = np.empty_like(desired)
        last = 1.0
        for i, gain in enumerate(desired):
            last = 0.85 * last + 0.15 * gain
            smooth[i] = last
        gain = np.interp(np.arange(len(y)), np.arange(len(smooth)) * hop, smooth)
        y *= gain
        a = np.abs(y)
        y = np.sign(y) * np.where(a > 0.12, 0.12 + (a - 0.12) / ratio, a)
        y = np.clip(y, -0.98, 0.98).astype(np.float32)

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg (with libopus) is required")
        with tempfile.TemporaryDirectory() as tmp:
            src, opus, dec = (Path(tmp) / p for p in ("in.wav", "phone.ogg", "decoded.wav"))
            sf.write(src, y, 16_000, subtype="PCM_16")
            for command in (
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(src),
                    "-ac",
                    "1",
                    "-c:a",
                    "libopus",
                    "-application",
                    "voip",
                    "-frame_duration",
                    "20",
                    "-b:a",
                    f"{bitrate}k",
                    str(opus),
                ],
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(opus),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(dec),
                ],
            ):
                subprocess.run(command, check=True)
            decoded, _ = sf.read(dec, dtype="float32")
        return np.pad(decoded[: len(x)], (0, max(0, len(x) - len(decoded))))
