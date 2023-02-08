from __future__ import annotations

from pathlib import Path, PurePath

from dynaconf import Dynaconf

settings_files: list[str] = ["settings.toml", ".secrets.toml"]
settings_files_absolute: list[PurePath] = [
    Path(__file__).resolve().parent / file for file in settings_files
]

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=settings_files_absolute,
)
