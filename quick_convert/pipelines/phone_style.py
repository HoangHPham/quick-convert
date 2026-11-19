"""Single-file or recursive folder phone-style conversion."""

from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from quick_convert.components.transforms.phone_style import PhoneStyleConverter


class PhoneStylePipeline:
    def __init__(self, input_path: str, out_dir: str, converter: PhoneStyleConverter):
        self.input_path = Path(input_path).expanduser().resolve()
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.converter = converter
        self.config = None

    def write_config(self, config: str) -> None:
        self.config = config

    def run(self) -> Path:
        src = self.input_path
        out = self.out_dir
        if not src.exists():
            raise FileNotFoundError(src)
        if src.is_dir() and out.is_relative_to(src):
            raise ValueError("Output must be outside the input directory")
        extensions = {".wav", ".flac", ".ogg"}
        files = (
            [src]
            if src.is_file()
            else sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        )
        if not files or (src.is_file() and src.suffix.lower() not in extensions):
            raise ValueError(f"No supported audio files at {src}")
        destinations = [
            (out / (p.stem + ".flac") if src.is_file() else out / p.relative_to(src).with_suffix(".flac"))
            for p in files
        ]
        if len(destinations) != len(set(destinations)) or any(
            dest == p for p, dest in zip(files, destinations, strict=True)
        ):
            raise ValueError("Output paths collide or would overwrite source audio")
        out.mkdir(parents=True, exist_ok=True)
        if self.config is not None:
            (out / "resolved_config.yaml").write_text(self.config, encoding="utf-8")
        for audio, target in tqdm(zip(files, destinations, strict=True), total=len(files), desc="Phone style"):
            waveform, sr = sf.read(audio, dtype="float32", always_2d=True)
            key = audio.name if src.is_file() else audio.relative_to(src).as_posix()
            y = self.converter(waveform.mean(axis=1), sr, key=key)
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, y, 16_000, format="FLAC", subtype="PCM_16")
        return out
