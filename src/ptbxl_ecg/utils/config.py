"""YAML configs with single/multiple inheritance (`inherit:`) and CLI overrides."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _load_raw(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    parents = cfg.pop("inherit", None)
    if parents is None:
        return cfg
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = deep_merge(merged, _load_raw((path.parent / parent).resolve()))
    return deep_merge(merged, cfg)


def parse_overrides(overrides: Iterable[str]) -> list[tuple[str, Any]]:
    parsed = []
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        parsed.append((key.strip(), yaml.safe_load(value)))
    return parsed


_PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


def _format_name(cfg: dict) -> None:
    """Expand `{dotted.key}` placeholders in cfg['name'] from the config itself."""
    name = cfg.get("name", "run")
    cfg["name"] = _PLACEHOLDER.sub(lambda m: str(get_by_path(cfg, m.group(1))), name)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    path = Path(path).resolve()
    cfg = _load_raw(path)
    for key, value in parse_overrides(overrides):
        set_by_path(cfg, key, value)
    _format_name(cfg)
    cfg["_config_path"] = str(path)
    cfg["_overrides"] = list(overrides)
    return cfg


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in cfg.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{full}."))
        else:
            flat[full] = value
    return flat


def dump_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
