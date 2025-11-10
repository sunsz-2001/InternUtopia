import carb
import omni.kit.mesh.raycast
import omni.usd
from pxr import UsdGeom


class GroundProbe:
    """Utility for projecting a prim's position down to the nearest ground surface."""

    def __init__(
        self,
        *,
        robot_path: str,
        wheel_radius_m: float,
        prefix: str,
        probe_offset_m: float = 1.5,
        probe_depth_m: float = 100.0,
        min_increment_m: float = 0.05,
        skip_step_scale: float = 0.5,
    ):
        self._prefix = prefix
        self._robot_path = robot_path.rstrip("/") or "/"
        self._wheel_radius_m = float(wheel_radius_m)
        self._probe_offset_m = max(float(probe_offset_m), 0.0)
        self._probe_depth_m = max(float(probe_depth_m), 0.0)
        self._min_increment_m = max(float(min_increment_m), 0.0)
        self._skip_step_scale = max(float(skip_step_scale), 0.0)

        self._stage = omni.usd.get_context().get_stage()
        self._meters_per_unit = 1.0
        self._stage_units_per_meter = 1.0
        try:
            if self._stage:
                mpu = UsdGeom.GetStageMetersPerUnit(self._stage)
                if not mpu:
                    mpu = UsdGeom.GetFallbackStageMetersPerUnit()
                if mpu and mpu > 0.0:
                    self._meters_per_unit = float(mpu)
                    self._stage_units_per_meter = 1.0 / self._meters_per_unit
        except Exception:
            pass

        su = self._stage_units_per_meter
        self._wheel_radius_stage = self._wheel_radius_m * su
        self._probe_offset_stage = max(self._probe_offset_m, self._wheel_radius_m) * su
        self._probe_depth_stage = max(self._probe_depth_m * su, self._probe_offset_stage)
        self._min_increment_stage = max(self._min_increment_m * su, 0.01 * su)
        self._min_ray_length_stage = max(0.1 * su, self._min_increment_stage)
        skip_default = self._wheel_radius_stage * self._skip_step_scale
        self._skip_step_stage = max(skip_default, self._min_increment_stage)

        robot_prefix = self._robot_path.rstrip("/")
        self._robot_prefix = f"{robot_prefix}/" if robot_prefix else ""

        self._raycast_interface = None
        self._raycast_unavailable = False
        self._ground_probe_failure_count = 0

    @property
    def stage_units_per_meter(self) -> float:
        return self._stage_units_per_meter

    @property
    def wheel_radius_stage(self) -> float:
        return self._wheel_radius_stage

    def project(self, x: float, y: float, world_z: float):
        """Return (ground_z, projected_flag). Falls back to wheel-radius offset if projection fails."""
        iface = self._acquire_raycast()
        if iface is None:
            return world_z - self._wheel_radius_stage, False

        min_offset = max(self._probe_offset_stage, self._wheel_radius_stage)
        start_z = max(world_z + min_offset, world_z + self._min_increment_stage)
        direction = carb.Float3(0.0, 0.0, -1.0)
        remaining_depth = self._probe_depth_stage
        attempts = 0
        hit_ground_z = None

        max_attempts = 32
        while attempts < max_attempts and remaining_depth > 0.0:
            attempts += 1
            ray_length = max(min(remaining_depth, self._probe_depth_stage), self._min_ray_length_stage)
            start = carb.Float3(x, y, start_z)
            try:
                hit = iface.closestRaycast(start, direction, ray_length)
            except Exception as exc:
                carb.log_warn(f"{self._prefix}Ground projection raycast failed: {exc}")
                self._raycast_unavailable = True
                return world_z - self._wheel_radius_stage, False

            if not hit or hit.meshIndex == -1:
                break

            hit_z = float(hit.position[2])
            try:
                prim_path = iface.get_mesh_path_from_index(hit.meshIndex)
            except Exception:
                prim_path = None

            if prim_path:
                if prim_path == self._robot_path.rstrip("/"):
                    skip_robot = True
                else:
                    skip_robot = prim_path.startswith(self._robot_prefix)
            else:
                skip_robot = False

            travel = max(start_z - hit_z, 0.0)
            remaining_depth = max(remaining_depth - travel, 0.0)

            if skip_robot:
                start_z = hit_z - self._skip_step_stage
                remaining_depth = max(remaining_depth - self._skip_step_stage, 0.0)
                if remaining_depth <= 0.0:
                    break
                continue

            hit_ground_z = hit_z
            break

        if hit_ground_z is None:
            self._ground_probe_failure_count += 1
            if self._ground_probe_failure_count in (1, 20) or self._ground_probe_failure_count % 50 == 0:
                carb.log_warn(
                    f"{self._prefix}Ground projection failed {self._ground_probe_failure_count} time(s); using wheel offset."
                )
            return world_z - self._wheel_radius_stage, False

        if self._ground_probe_failure_count:
            carb.log_info(
                f"{self._prefix}Ground projection recovered after {self._ground_probe_failure_count} failed attempt(s)."
            )
        self._ground_probe_failure_count = 0
        return hit_ground_z, True

    def _acquire_raycast(self):
        if self._raycast_unavailable:
            return None
        if self._raycast_interface is not None:
            return self._raycast_interface
        try:
            iface = omni.kit.mesh.raycast.get_mesh_raycast_interface()
        except Exception as exc:
            carb.log_warn(f"{self._prefix}Raycast interface unavailable for ground projection: {exc}")
            self._raycast_unavailable = True
            return None
        if iface is None:
            carb.log_warn(f"{self._prefix}Raycast interface acquisition returned None; ground projection disabled.")
            self._raycast_unavailable = True
            return None
        try:
            iface.set_bvh_refresh_rate(omni.kit.mesh.raycast.BvhRefreshRate.FAST, True)
        except Exception:
            pass
        self._raycast_interface = iface
        return iface
