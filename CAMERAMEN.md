# Cameramen and view sectors

A cameraman is a static entity on the active terrain.

## State

Each cameraman owns:

- name
- color
- terrain position `[x, y, z]`
- view radius in meters
- start azimuth in degrees
- end azimuth in degrees

The altitude is not manually configurable. On placement, the application samples
the terrain at the clicked XY coordinate and stores that terrain altitude as Z.

## Placement

1. Open **Cameraman Manager**.
2. Add/select a cameraman.
3. Press **Place / move cameraman**.
4. Click once on the main terrain.
5. Placement mode automatically ends.

Starting target drawing cancels cameraman placement, and starting cameraman
placement cancels an uncommitted target drawing. Playback/navigation also leaves
placement mode.

## View sector

Azimuth convention:

- `0°`: +X
- `90°`: +Y
- angles increase counter-clockwise
- wrapped sectors are supported, e.g. `330° -> 30°` gives a 60° sector

The view sector is a 2D horizontal sector on the terrain display. Radius and both
angles update live from the manager. LOS/elevation filtering is intentionally not
part of this version.

## Persistence

Cameramen are stored in the `.trajectory` project. The project still references
a separate `.terrain` file. Project format version 4 adds the `cameramen` array;
versions 2 and 3 remain loadable and receive one default unplaced cameraman.
