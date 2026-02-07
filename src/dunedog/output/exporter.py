"""Export skeletons and synthesized stories to files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dunedog.models.skeleton import StorySkeleton


def to_json(skeletons: list[StorySkeleton], path: str | Path) -> None:
    """Export skeletons as JSON."""
    path = Path(path)
    data = [sk.to_dict() for sk in skeletons]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def to_csv(skeletons: list[StorySkeleton], path: str | Path) -> None:
    """Export skeleton summary as CSV."""
    path = Path(path)
    fieldnames = [
        "index", "tone", "coherence_score", "engine", "spread_type",
        "beat_count", "atom_count", "theme_tags", "violations",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, sk in enumerate(skeletons):
            writer.writerow({
                "index": i,
                "tone": sk.tone,
                "coherence_score": round(sk.coherence_score, 4),
                "engine": sk.stats.engine,
                "spread_type": sk.stats.spread_type,
                "beat_count": sk.stats.beat_count,
                "atom_count": len(sk.atoms),
                "theme_tags": "; ".join(sk.theme_tags),
                "violations": "; ".join(sk.stats.violations),
            })


def export_stories(stories: list[dict], path: str | Path) -> None:
    """Export synthesized stories as JSON."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(stories, f, indent=2)


def load_skeletons(path: str | Path) -> list[StorySkeleton]:
    """Load skeletons from a JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return [StorySkeleton.from_dict(d) for d in data]
