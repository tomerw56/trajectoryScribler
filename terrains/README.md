# Demo terrains

These `.terrain` files are ready to load from the main application.

- `multi_hills.terrain` — several separated hills; useful for repeated climb/descent and velocity-component changes.
- `big_top.terrain` — one large broad summit with foothills; useful for cruise-altitude obstacle avoidance.
- `canyon.terrain` — elevated plateau cut by a winding canyon; useful for terrain-clearance trajectories.
- `ridge_pass.terrain` — a long ridge with a lower pass; useful for demonstrating path placement versus altitude behavior.

All presets use deterministic, explicitly generated height matrices. Loading a preset does not depend on the random terrain generator.
