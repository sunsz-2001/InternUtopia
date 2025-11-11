from __future__ import annotations

import carb

_DEFAULT_WORLD_KWARGS = {
    "physics_dt": 1.0 / 30.0,
    "rendering_dt": 1.0 / 30.0,
    "stage_units_in_meters": 1.0,  # 默认值，会在运行时从场景读取
}


def _detect_stage_units_in_meters() -> float:
    """从当前 USD stage 中检测单位缩放

    Returns:
        float: metersPerUnit 值，默认 1.0（米）
    """
    try:
        import omni.usd
        from pxr import UsdGeom

        ctx = omni.usd.get_context()
        if not ctx:
            carb.log_warn("[World] USD context not available, using default stage units (1.0)")
            return 1.0

        stage = ctx.get_stage()
        if not stage:
            carb.log_warn("[World] USD stage not available, using default stage units (1.0)")
            return 1.0

        # 读取 stage 的 metersPerUnit 元数据
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))

        if meters_per_unit <= 0:
            carb.log_warn(f"[World] Invalid stage units ({meters_per_unit}), using default (1.0)")
            return 1.0

        # 对于非标准单位（非米），输出提示信息
        if abs(meters_per_unit - 1.0) > 1e-3:
            unit_name = "centimeters" if abs(meters_per_unit - 0.01) < 1e-3 else f"{meters_per_unit}m"
            carb.log_info(f"[World] Detected stage units: {meters_per_unit} meters per unit ({unit_name})")

        return meters_per_unit

    except Exception as exc:
        carb.log_warn(f"[World] Failed to detect stage units: {exc}, using default (1.0)")
        return 1.0


def bootstrap_world_if_needed(**overrides):
    """Create the Isaac World singleton if it doesn't exist yet.

    自动检测场景的单位缩放（metersPerUnit）并正确配置物理引擎。
    这对 GRScenes（厘米单位）等非标准单位的场景至关重要。
    """

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

    # 构建 World 参数
    world_kwargs = dict(_DEFAULT_WORLD_KWARGS)

    # 如果用户没有显式提供 stage_units_in_meters，从场景中自动检测
    if "stage_units_in_meters" not in overrides or overrides.get("stage_units_in_meters") is None:
        detected_units = _detect_stage_units_in_meters()
        world_kwargs["stage_units_in_meters"] = detected_units
    else:
        world_kwargs["stage_units_in_meters"] = overrides["stage_units_in_meters"]

    # 应用其他用户提供的覆盖参数
    world_kwargs.update({k: v for k, v in overrides.items() if v is not None})

    try:
        world = World(**world_kwargs)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to create Isaac World with arguments {world_kwargs}: {exc}") from exc

    carb.log_info(
        "[World] Created Isaac World (physics_dt=%s, rendering_dt=%s, stage_units_in_meters=%s)."
        % (
            world_kwargs.get("physics_dt"),
            world_kwargs.get("rendering_dt"),
            world_kwargs.get("stage_units_in_meters"),
        )
    )
    return world


def ensure_world(*_args, **_kwargs):
    """Return a valid omni.isaac.core.World, creating it if necessary."""

    return bootstrap_world_if_needed(**_kwargs)
