from __future__ import annotations

import carb

_DEFAULT_WORLD_KWARGS = {
    "physics_dt": 1.0 / 30.0,
    "rendering_dt": 1.0 / 30.0,
    "stage_units_in_meters": 1.0,
}


def bootstrap_world_if_needed(**overrides):
    """Create the Isaac World singleton if it doesn't exist yet."""

    try:
        from omni.isaac.core import World
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Isaac Core World API is unavailable in the current environment.") from exc

    try:
        world = World.instance()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to query Isaac World instance: {exc}") from exc

    if world is not None:
        return world

    world_kwargs = dict(_DEFAULT_WORLD_KWARGS)
    world_kwargs.update({k: v for k, v in overrides.items() if v is not None})
    try:
        world = World(**world_kwargs)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to create Isaac World with arguments {world_kwargs}: {exc}") from exc

    carb.log_info(
        "[World] Created Isaac World (physics_dt=%s, rendering_dt=%s)."
        % (world_kwargs.get("physics_dt"), world_kwargs.get("rendering_dt"))
    )
    return world


def ensure_world(*_args, **_kwargs):
    """Return a valid omni.isaac.core.World, creating it if necessary."""

    return bootstrap_world_if_needed(**_kwargs)
