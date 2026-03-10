from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def get_git_commit(project_root: str | Path) -> str:
    """Return current git commit hash or `unknown` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(manifest: Dict[str, Any], manifests_dir: str | Path, run_id: str) -> Path:
    """Write JSON manifest to artifacts manifests directory."""
    out_dir = Path(manifests_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{run_id}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    return output_path
