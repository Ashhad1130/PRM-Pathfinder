"""Print one config value, so the shell driver can read a YAML file without parsing it.

    python scripts/lightning/_cfg_get.py configs/baseline.yaml run name

Missing keys print an empty line (exit 0) — the caller substitutes its own default.
"""

from __future__ import annotations

import sys

import yaml


def main() -> None:
    path, section, key = sys.argv[1], sys.argv[2], sys.argv[3]
    raw = yaml.safe_load(open(path, encoding="utf-8")) or {}
    value = (raw.get(section) or {}).get(key)
    if value is None:
        print("")
    elif isinstance(value, (list, tuple)):
        print(" ".join(str(v) for v in value))
    else:
        print(value)


if __name__ == "__main__":
    main()
