# Drawing State

Drawing is an explicit edit transaction per target.

## State rules

- Default state: drawing is **disabled**.
- `Begin drawing / redraw`:
  - stops playback;
  - makes that target active;
  - deletes its previous scribble, generated trajectory, motion samples, and covariance;
  - arms left-drag drawing for that target only.
- `Generate selected trajectory` while drawing:
  - generates the trajectory;
  - on success, commits the scribble and automatically disables drawing.
- `Stop drawing` before successful generation:
  - deletes the draft scribble;
  - resets the target to no trajectory;
  - disables drawing.
- Playback/navigation, switching away from the drawing target, save/load, loading terrain, or starting a different redraw also cancels an ungenerated draft.
- Right-click inspection never draws.

This deliberately makes redraw destructive: once `Begin drawing / redraw` is pressed, the old trajectory is gone.

## Freehand drag invariant

`TerrainCanvas.set_draw_target()` is idempotent. A normal repaint/refresh may call it
many times while the left mouse button is held. If the requested draw target has not
changed, it must **not** reset the internal drag flag. This preserves continuous
freehand scribbling while keeping drawing disabled outside an explicit Begin Drawing
session.
