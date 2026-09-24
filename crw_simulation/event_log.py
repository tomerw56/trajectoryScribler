"""Write compact structured diagnostic events for CRW operations."""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .common_geometry import HorizontalPolygonFace, Point


_DEFAULT_LOG_DIRECTORY = Path(__file__).resolve().parent.parent / "logs"
_log_directory = _DEFAULT_LOG_DIRECTORY
_log_path: Path | None = None
_write_lock = threading.Lock()


def configure_event_log(directory: str | Path) -> Path:
	"""Configure the CRW event directory and reset the active process log file."""
	resolved = Path(directory).expanduser().resolve()
	resolved.mkdir(parents=True, exist_ok=True)
	if not resolved.is_dir():
		raise NotADirectoryError(f"CRW event log path is not a directory: {resolved}")

	global _log_directory, _log_path
	with _write_lock:
		_log_directory = resolved
		_log_path = None
	return resolved


def _json_safe(value: Any) -> Any:
	if isinstance(value, Point):
		return {
			"coordinates": dict(value.coordinates),
			"point_type": value.point_type.name,
		}
	if isinstance(value, HorizontalPolygonFace):
		return {"points": [_json_safe(point) for point in value.points]}
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, Enum):
		return value.name
	if isinstance(value, Mapping):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
		return [_json_safe(item) for item in value]
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	return repr(value)[:500]


def _current_log_path() -> Path:
	global _log_path
	if _log_path is None:
		stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
		_log_path = _log_directory / f"crw-{stamp}-p{os.getpid()}.jsonl"
	return _log_path


def _append_json_line(event: dict[str, Any]) -> None:
	_log_directory.mkdir(parents=True, exist_ok=True)
	with _current_log_path().open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(_json_safe(event), separators=(",", ":")) + "\n")


def write_event(
	event: str,
	operation: str,
	parameters: dict[str, Any],
	*,
	details: dict[str, Any] | None = None,
	error: BaseException | None = None,
) -> None:
	payload: dict[str, Any] = {
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"process_id": os.getpid(),
		"event": event,
		"status": "failure" if error is not None else "success",
		"operation": operation,
		"parameters": parameters,
	}
	if details is not None:
		payload.update(details)
	if error is not None:
		payload["exception_type"] = type(error).__name__
		payload["exception_message"] = str(error)
	try:
		with _write_lock:
			_append_json_line(payload)
	except Exception as log_error:
		print(f"CRW event logging failed: {log_error}", file=sys.stderr)


@contextmanager
def log_construction(
	category: str,
	operation: str,
	parameters: dict[str, Any],
) -> Iterator[None]:
	try:
		yield
	except Exception as error:
		write_event(
			f"{category}.construction_failed",
			operation,
			parameters,
			error=error,
		)
		raise
	else:
		write_event(f"{category}.constructed", operation, parameters)


def _set_log_directory_for_testing(directory: Path | None) -> None:
	if directory is not None:
		configure_event_log(directory)
		return

	global _log_directory, _log_path
	with _write_lock:
		_log_directory = _DEFAULT_LOG_DIRECTORY
		_log_path = None