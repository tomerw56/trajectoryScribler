from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaybackState:
    time_sec: float = 0.0
    playing: bool = False

    def as_dict(self) -> dict[str, float | bool]:
        return {"time_sec": float(self.time_sec), "playing": bool(self.playing)}


def apply_playback_event(
    state: PlaybackState,
    event: str | None,
    *,
    step_sec: float,
    max_time_sec: float,
    slider_value: float | None = None,
) -> PlaybackState:
    """Pure playback state transition used by the Dash callback."""
    step = max(1e-9, float(step_sec))
    maximum = max(0.0, float(max_time_sec))
    time_sec = float(np.clip(state.time_sec, 0.0, maximum))
    playing = bool(state.playing)

    if event in (None, "scenario-select"):
        return PlaybackState(0.0, False)
    if event == "first-button":
        return PlaybackState(0.0, False)
    if event == "prev-button":
        return PlaybackState(max(0.0, time_sec - step), False)
    if event == "play-button":
        return PlaybackState(time_sec, not playing)
    if event == "next-button":
        return PlaybackState(min(maximum, time_sec + step), False)
    if event == "last-button":
        return PlaybackState(maximum, False)
    if event == "time-slider":
        requested = float(slider_value or 0.0)
        return PlaybackState(float(np.clip(requested, 0.0, maximum)), False)
    if event == "playback-interval" and playing:
        next_time = min(maximum, time_sec + step)
        return PlaybackState(next_time, next_time < maximum - 1e-12)
    return PlaybackState(time_sec, playing)
