"""Persistent registry of validated tools.

Each registered tool is stored as a ``<name>.py`` module plus an entry in an
``index.json`` manifest holding its description and JSON input schema, so tools
survive across sessions and can be retrieved later.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field

_NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}")


class ToolRecord(BaseModel):
    name: str
    description: str = ""
    schema_: dict = Field(default_factory=dict, alias="schema")
    filename: str
    created_at: float = Field(default_factory=time.time)

    model_config = {"populate_by_name": True}


class ToolRegistry:
    """Stores tool modules and their metadata under a root directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    # ---------------- index persistence ----------------

    def _load_index(self) -> dict[str, dict]:
        if not self.index_path.is_file():
            return {}
        try:
            return json.loads(self.index_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_index(self, index: dict[str, dict]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2))

    # ---------------- operations ----------------

    def register(
        self, name: str, source: str, schema: dict | None = None, description: str = ""
    ) -> ToolRecord:
        if not _NAME_RE.fullmatch(name):
            raise ValueError(f"invalid tool name: {name!r}")
        filename = f"{name}.py"
        (self.root / filename).write_text(source)
        record = ToolRecord(
            name=name,
            description=description,
            schema=schema or {},
            filename=filename,
        )
        index = self._load_index()
        index[name] = record.model_dump(by_alias=True)
        self._save_index(index)
        return record

    def get(self, name: str) -> ToolRecord | None:
        entry = self._load_index().get(name)
        return ToolRecord.model_validate(entry) if entry else None

    def load_source(self, name: str) -> str:
        record = self.get(name)
        if record is None:
            raise KeyError(f"no such tool: {name!r}")
        return (self.root / record.filename).read_text()

    def list(self) -> list[ToolRecord]:
        return [ToolRecord.model_validate(e) for e in self._load_index().values()]

    def remove(self, name: str) -> bool:
        index = self._load_index()
        entry = index.pop(name, None)
        if entry is None:
            return False
        (self.root / entry["filename"]).unlink(missing_ok=True)
        self._save_index(index)
        return True
