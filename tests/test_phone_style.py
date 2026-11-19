"""Minimal regression tests for the noise-free phone-style workflow."""

import shutil

import numpy as np
import pytest
import soundfile as sf

from quick_convert.components.transforms.phone_style import PhoneStyleConverter
from quick_convert.pipelines.phone_style import PhoneStylePipeline


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_file_and_folder_phone_conversion(tmp_path):
    src = tmp_path / "source"
    (src / "speaker").mkdir(parents=True)
    t = np.arange(8000) / 8000
    clean = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(src / "speaker" / "one.wav", clean, 8000)
    converter = PhoneStyleConverter(seed=2026)

    folder_out = PhoneStylePipeline(str(src), str(tmp_path / "folder"), converter).run()
    wav, sr = sf.read(folder_out / "speaker" / "one.flac", dtype="float32")
    assert sr == 16000 and len(wav) == 16000
    assert np.isfinite(wav).all()

    recorded, recorded_sr = sf.read(src / "speaker" / "one.wav", dtype="float32")
    again = converter(recorded, recorded_sr, key="speaker/one.wav")
    assert np.allclose(wav, again, atol=1 / 32768)

    file_out = PhoneStylePipeline(str(src / "speaker" / "one.wav"), str(tmp_path / "single"), converter).run()
    assert (file_out / "one.flac").is_file()
