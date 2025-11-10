import asyncio
import copy
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import carb
import carb.input
import omni.appwindow
import omni.kit.app
import omni.timeline
import omni.usd
import torch
from isaacsim.core.prims import SingleArticulation
from pxr import UsdPhysics, UsdGeom
from tools.teleop.ground_probe import GroundProbe
from tools.teleop.pose_logger import PoseLogger

PREFIX = "[teleop.carter] "
SIGN_L = +1.0
SIGN_R = +1.0
DEFAULT_TIMESTEP = 1.0 / 60.0
LOGGING_RESERVED_KEYS = {
    "enabled",
    "path",
    "output_path",
    "file",
    "distance_threshold",
    "distance_m",
    "step_m",
    "print",
    "console",
    "stdout",
    "instruction",
    "instructions",
    "objects",
}


def build_pose_logger_config(logging: Optional[dict]) -> dict:
    log_cfg = dict(logging or {})
    log_enabled = log_cfg.get("enabled", True)

    log_path_cfg = log_cfg.get("path") if log_enabled else None
    if log_path_cfg is None:
        log_path_cfg = log_cfg.get("output_path") or log_cfg.get("file")
    if not log_enabled:
        log_path_cfg = None

    threshold_cfg = log_cfg.get("distance_threshold")
    if threshold_cfg is None:
        threshold_cfg = log_cfg.get("distance_m")
    if threshold_cfg is None:
        threshold_cfg = log_cfg.get("step_m")
    if threshold_cfg is None:
        threshold_cfg = 0.5

    log_print_cfg = bool(log_cfg.get("print") or log_cfg.get("console") or log_cfg.get("stdout"))
    instruction_cfg = log_cfg.get("instruction") or log_cfg.get("instructions") or ""

    objects_cfg = log_cfg.get("objects")
    if objects_cfg is not None:
        objects_cfg = copy.deepcopy(objects_cfg)

    extra_metadata = {
        key: copy.deepcopy(value)
        for key, value in log_cfg.items()
        if key not in LOGGING_RESERVED_KEYS
    }
    if not extra_metadata:
        extra_metadata = None

    return {
        "enabled": bool(log_enabled),
        "path": log_path_cfg,
        "distance_threshold": threshold_cfg,
        "print": log_print_cfg,
        "instruction": instruction_cfg,
        "objects": objects_cfg,
        "extra_metadata": extra_metadata,
    }


def _ensure_physics_and_timeline():
    app = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()
    if stage and not stage.GetPrimAtPath("/World/physicsScene"):
        try:
            UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        except Exception as exc:
            carb.log_warn(f"{PREFIX}Define physicsScene failed: {exc}")
    timeline = omni.timeline.get_timeline_interface()
    return app, stage, timeline


def _get_world_pos(prim_path: str):
    try:
        M = omni.usd.get_world_transform_matrix(prim_path)
        return M.ExtractTranslation()
    except Exception:
        return None


def _try_auto_track_width(root_path: str) -> Optional[float]:
    stage = omni.usd.get_context().get_stage()
    cand = [f"{root_path}/left_wheel_link", f"{root_path}/right_wheel_link"]
    if all(stage.GetPrimAtPath(p).IsValid() for p in cand):
        pL = _get_world_pos(cand[0])
        pR = _get_world_pos(cand[1])
        if pL is not None and pR is not None:
            try:
                return float((pL - pR).GetLength())
            except Exception:
                return None
    return None


def _vw_to_wheels(v: float, w: float, R: float, L: float):
    wl = (2.0 * v - w * L) / (2.0 * R) * SIGN_L
    wr = (2.0 * v + w * L) / (2.0 * R) * SIGN_R
    return wl, wr


def _make_velocity_setter(view, idx_l: int, idx_r: int):
    dev_str = getattr(view, "_device", "cpu") or "cpu"
    try:
        device = torch.device(dev_str)
    except Exception:
        device = torch.device("cpu")

    t_v12 = torch.zeros(2, dtype=torch.float32, device=device)
    t_v1x2 = t_v12.view(1, 2)
    t_idx64 = torch.tensor([idx_l, idx_r], dtype=torch.int64, device=device)
    t_idx32 = torch.tensor([idx_l, idx_r], dtype=torch.int32, device=device)

    try:
        view.set_joint_velocity_targets(t_v12, joint_indices=t_idx64)
        return lambda wl, wr: view.set_joint_velocity_targets(
            torch.tensor([wl, wr], dtype=torch.float32, device=device),
            joint_indices=t_idx64,
        )
    except Exception:
        pass
    try:
        view.set_joint_velocity_targets(t_v12, joint_indices=t_idx32)
        return lambda wl, wr: view.set_joint_velocity_targets(
            torch.tensor([wl, wr], dtype=torch.float32, device=device),
            joint_indices=t_idx32,
        )
    except Exception:
        pass
    try:
        view.set_joint_velocity_targets(t_v1x2, joint_indices=t_idx64)
        return lambda wl, wr: view.set_joint_velocity_targets(
            torch.tensor([[wl, wr]], dtype=torch.float32, device=device),
            joint_indices=t_idx64,
        )
    except Exception:
        pass
    try:
        view.set_joint_velocity_targets(t_v1x2, joint_indices=t_idx32)
        return lambda wl, wr: view.set_joint_velocity_targets(
            torch.tensor([[wl, wr]], dtype=torch.float32, device=device),
            joint_indices=t_idx32,
        )
    except Exception:
        pass
    try:
        view.set_joint_velocity_targets(t_v12)
        return lambda wl, wr: view.set_joint_velocity_targets(
            torch.tensor([wl, wr], dtype=torch.float32, device=device)
        )
    except Exception as exc:
        raise RuntimeError(f"{PREFIX}No working set_joint_velocity_targets signature: {exc}") from exc


def _log_scale_info(robot_root: str):
    stage = omni.usd.get_context().get_stage()

    def read_scale(path: str):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        x = UsdGeom.Xformable(prim)
        for op in x.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                try:
                    v = op.Get()
                    return tuple(v) if isinstance(v, (list, tuple)) else (v[0], v[1], v[2])
                except Exception:
                    return None
        return (1.0, 1.0, 1.0)

    w = read_scale("/World")
    r = read_scale(robot_root)
    carb.log_info(f"{PREFIX}Scale /World={w}  robot={robot_root} scale={r}")


class CarterKeyboardTeleop:
    def __init__(
        self,
        robot_path: str,
        left_joint: str,
        right_joint: str,
        wheel_radius: float = 0.24,
        track_width: Optional[float] = None,
        base_velocity: float = 0.6,
        base_turn_rate: float = 1.2,
        shift_multiplier: float = 3.0,
        ctrl_multiplier: float = 0.3,
        log_path: Optional[str] = None,
        log_distance_threshold: float = 0.5,
        log_print: bool = False,
        log_instruction: Optional[str] = None,
        episode_objects: Optional[Any] = None,
        episode_metadata: Optional[dict] = None,
        linear_slew_rate: Optional[float] = 4.0,
        angular_slew_rate: Optional[float] = 6.0,
    ):
        self.robot_path = robot_path
        self.left_joint = left_joint
        self.right_joint = right_joint
        self.wheel_radius = float(wheel_radius)
        self.track_width_cfg = track_width
        self.base_v = float(base_velocity)
        self.base_w = float(base_turn_rate)
        self.shift_mul = float(shift_multiplier)
        self.ctrl_mul = float(ctrl_multiplier)
        self._linear_slew_rate = (
            float(linear_slew_rate) if linear_slew_rate and linear_slew_rate > 0.0 else None
        )
        self._angular_slew_rate = (
            float(angular_slew_rate) if angular_slew_rate and angular_slew_rate > 0.0 else None
        )
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0
        self._cmd_initialized = False
        self._last_timeline_time: Optional[float] = None
        self._default_dt = DEFAULT_TIMESTEP

        self._app, self._stage, self._timeline = _ensure_physics_and_timeline()
        self._inp = carb.input.acquire_input_interface()
        self._appwin = omni.appwindow.get_default_app_window()
        self._kbd = self._appwin.get_keyboard() if self._appwin else None
        if self._kbd is None:
            raise RuntimeError(f"{PREFIX}No keyboard device available (硬失败).")

        self._pressed = set()
        self._sub_id = None
        self._robot = None
        self._idx_l = None
        self._idx_r = None
        self._set_vel = None
        self._track_width = None
        self._task = None
        self._running = False
        self._frame_idx = 0

        self._pose_failure_count = 0
        self._ground_probe = GroundProbe(
            robot_path=self.robot_path,
            wheel_radius_m=self.wheel_radius,
            prefix=PREFIX,
        )

        safe_robot_name = (self.robot_path.strip("/") or "World").replace("/", "_")
        try:
            distance_threshold_m = float(log_distance_threshold)
        except Exception:
            distance_threshold_m = 0.0
        threshold_units = max(distance_threshold_m, 0.0)
        self._log_print = bool(log_print)
        self._episode_objects = episode_objects
        self._episode_metadata = copy.deepcopy(episode_metadata) if episode_metadata else None
        instruction_text = (log_instruction or "").strip()
        self._pose_logger = PoseLogger(
            prefix=PREFIX,
            robot_path=self.robot_path,
            safe_name=safe_robot_name,
            log_path=log_path,
            distance_threshold=threshold_units,
            log_print=self._log_print,
            instruction=instruction_text,
            default_dir=Path.cwd() / "logs",
            episode_objects=self._episode_objects,
            extra_metadata=self._episode_metadata,
        )

    def _on_key(self, e):
        key = getattr(e.input, "name", e.input)
        typ = getattr(e.type, "name", e.type)
        if typ in ("KEY_PRESS", "KEY_REPEAT"):
            self._pressed.add(key)
        elif typ == "KEY_RELEASE":
            self._pressed.discard(key)

    def start(self):
        try:
            self._sub_id = self._inp.subscribe_to_keyboard_events(self._kbd, self._on_key)
        except Exception as exc:
            raise RuntimeError(f"{PREFIX}Keyboard subscription failed (硬失败): {exc}") from exc
        carb.log_info(f"{PREFIX}Keyboard subscription ready.")
        self._running = True
        self._task = asyncio.ensure_future(self._main())
        return self

    async def stop(self):
        self._running = False
        if self._task:
            try:
                if not self._task.done():
                    self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            finally:
                self._task = None
        try:
            if self._set_vel:
                self._set_vel(0.0, 0.0)
        except Exception:
            pass
        try:
            if self._sub_id is not None and self._kbd is not None:
                self._inp.unsubscribe_to_keyboard_events(self._kbd, self._sub_id)
        except Exception as exc:
            carb.log_warn(f"{PREFIX}unsubscribe failed: {exc}")
        finally:
            self._sub_id = None

        try:
            self._record_pose(self._frame_idx, 0.0, 0.0, force=True)
        except Exception as exc:
            carb.log_warn(f"{PREFIX}Final pose recording failed: {exc}")

        self._pose_logger.flush()
        self._cmd_initialized = False
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0
        self._last_timeline_time = None
        carb.log_info(f"{PREFIX}Stopped.")

    async def _drive_loop(self):
        carb.log_info(
            f"{PREFIX} Ready: R={self.wheel_radius:.3f} m, L={self._track_width:.3f} m | 聚焦视口后键盘生效"
        )
        carb.log_info(
            f"{PREFIX}↑↓ 控 v，←→ 控 ω；Shift×{self.shift_mul:g}，Ctrl×{self.ctrl_mul:g}，Space 急停"
        )

        #  在循环开始时记录初始位姿
        initial_pose_recorded = False
        self._cmd_initialized = False
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0
        self._last_timeline_time = None

        try:
            while self._running and self._app.is_running():
                if not (self._timeline and hasattr(self._timeline, "is_playing") and self._timeline.is_playing()):
                    await self._app.next_update_async()
                    continue

                #  第一次进入物理循环时记录初始位姿
                if not initial_pose_recorded:
                    carb.log_info(f"{PREFIX}Recording initial pose in drive loop...")
                    self._record_pose(self._frame_idx, 0.0, 0.0, force=True)
                    initial_pose_recorded = True

                v = float(("UP" in self._pressed) - ("DOWN" in self._pressed)) * self.base_v
                w = float(("LEFT" in self._pressed) - ("RIGHT" in self._pressed)) * self.base_w
                if "LEFT_SHIFT" in self._pressed or "RIGHT_SHIFT" in self._pressed:
                    v *= self.shift_mul
                    w *= self.shift_mul
                if "LEFT_CONTROL" in self._pressed or "RIGHT_CONTROL" in self._pressed:
                    v *= self.ctrl_mul
                    w *= self.ctrl_mul
                if "SPACE" in self._pressed:
                    v = 0.0
                    w = 0.0

                dt = self._resolve_dt()
                smoothing_enabled = "SPACE" not in self._pressed
                v_cmd, w_cmd = self._apply_slew(v, w, dt, smoothing=smoothing_enabled)

                if self._set_vel is not None:
                    wl, wr = _vw_to_wheels(v_cmd, w_cmd, self.wheel_radius, self._track_width)
                    try:
                        self._set_vel(wl, wr)
                    except Exception as exc:
                        carb.log_warn(f"{PREFIX}set velocity failed: {exc}")

                await self._app.next_update_async()
                self._frame_idx += 1
                self._record_pose(self._frame_idx, v_cmd, w_cmd)
        finally:
            try:
                self._record_pose(self._frame_idx, 0.0, 0.0, force=True)
            except Exception:
                pass
            try:
                if self._set_vel:
                    self._set_vel(0.0, 0.0)
            except Exception:
                pass
            self._cmd_initialized = False
            self._last_cmd_v = 0.0
            self._last_cmd_w = 0.0
            self._last_timeline_time = None

    def _record_pose(self, frame_idx: int, v_cmd: float, w_cmd: float, *, force: bool = False):
        if not self._pose_logger.enabled:
            return

        pose = self._resolve_robot_pose()
        if pose is None:
            self._pose_failure_count += 1
            if self._pose_failure_count == 1 or self._pose_failure_count % 20 == 0:
                carb.log_warn(f"{PREFIX}Unable to resolve pose for {self.robot_path} (attempt {self._pose_failure_count}).")
            return
        if self._pose_failure_count:
            carb.log_info(f"{PREFIX}Pose resolve recovered after {self._pose_failure_count} failed attempt(s).")
            self._pose_failure_count = 0

        x, y, z, yaw_deg = pose
        ground_z, ground_projected = self._ground_probe.project(x, y, z)

        time_sec = None
        if self._timeline and hasattr(self._timeline, 'get_current_time'):
            try:
                time_sec = float(self._timeline.get_current_time())
            except Exception:
                time_sec = None

        record = self._pose_logger.record(
            frame_idx=frame_idx,
            time_sec=time_sec,
            world_xyz=(x, y, z),
            ground_z=ground_z,
            yaw_deg=yaw_deg,
            command=(float(v_cmd), float(w_cmd)),
            force=force,
        )
        if record is None:
            return

        if self._log_print and self._pose_logger.print_enabled:
            z_info = f"xy=({record.x:.3f}, {record.y:.3f}) z={record.ground_z:.3f} yaw={record.yaw_deg:.1f}° d={record.distance_xy:.3f}m"
            if not ground_projected:
                z_info += " [fallback]"
            if time_sec is None:
                carb.log_info(f"{PREFIX}pose f={frame_idx} {z_info}")
            else:
                carb.log_info(f"{PREFIX}pose f={frame_idx} t={time_sec:.3f}s {z_info}")

        self._pose_logger.flush(incremental=True)

    def _resolve_dt(self) -> float:
        dt = None
        current_time = None
        if self._timeline and hasattr(self._timeline, "get_current_time"):
            try:
                current_time = float(self._timeline.get_current_time())
            except Exception:
                current_time = None
            if current_time is not None and self._last_timeline_time is not None:
                dt = current_time - self._last_timeline_time
        if current_time is not None:
            self._last_timeline_time = current_time
        if dt is None or dt <= 0.0 or dt > 0.5:
            dt = self._default_dt
        return dt

    def _apply_slew(self, target_v: float, target_w: float, dt: float, *, smoothing: bool) -> Tuple[float, float]:
        if not smoothing or (self._linear_slew_rate is None and self._angular_slew_rate is None):
            self._last_cmd_v = float(target_v)
            self._last_cmd_w = float(target_w)
            self._cmd_initialized = True
            return self._last_cmd_v, self._last_cmd_w

        if not self._cmd_initialized:
            self._last_cmd_v = float(target_v)
            self._last_cmd_w = float(target_w)
            self._cmd_initialized = True
            return self._last_cmd_v, self._last_cmd_w

        if self._linear_slew_rate is not None:
            max_delta_v = self._linear_slew_rate * dt
            delta_v = float(target_v) - self._last_cmd_v
            if delta_v > max_delta_v:
                delta_v = max_delta_v
            elif delta_v < -max_delta_v:
                delta_v = -max_delta_v
            self._last_cmd_v += delta_v
        else:
            self._last_cmd_v = float(target_v)

        if self._angular_slew_rate is not None:
            max_delta_w = self._angular_slew_rate * dt
            delta_w = float(target_w) - self._last_cmd_w
            if delta_w > max_delta_w:
                delta_w = max_delta_w
            elif delta_w < -max_delta_w:
                delta_w = -max_delta_w
            self._last_cmd_w += delta_w
        else:
            self._last_cmd_w = float(target_w)

        return self._last_cmd_v, self._last_cmd_w

    @staticmethod
    def _tensor_to_list(data) -> Optional[List[float]]:
        if data is None:
            return None
        try:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            if hasattr(data, 'tolist'):
                data = data.tolist()
            return [float(v) for v in data]
        except Exception:
            return None

    @staticmethod
    def _quat_to_yaw_deg(quat: Sequence[float]) -> float:
        if not quat or len(quat) < 4:
            return 0.0
        x, y, z, w = quat[:4]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

    def _resolve_robot_pose(self) -> Optional[Tuple[float, float, float, float]]:
        view = getattr(self._robot, '_articulation_view', None)
        if view is not None:
            try:
                positions, orientations = view.get_world_poses()
                if positions is not None and len(positions):
                    pos = self._tensor_to_list(positions[0])
                    ori = self._tensor_to_list(orientations[0]) if orientations is not None and len(orientations) else None
                    if pos and len(pos) >= 3:
                        yaw_deg = self._quat_to_yaw_deg(ori) if ori else 0.0
                        return float(pos[0]), float(pos[1]), float(pos[2]), yaw_deg
            except Exception as exc:
                carb.log_debug(f"{PREFIX}Articulation view pose fetch failed: {exc}")

        try:
            matrix = omni.usd.get_world_transform_matrix(self.robot_path)
        except Exception:
            matrix = None
        if matrix is None:
            return None
        translate = matrix.ExtractTranslation()
        x = float(translate[0])
        y = float(translate[1])
        z = float(translate[2])
        try:
            m00 = float(matrix[0][0])
            m10 = float(matrix[1][0])
        except Exception:
            m00 = 1.0
            m10 = 0.0
        yaw_rad = math.atan2(m10, m00)
        yaw_deg = math.degrees(yaw_rad)
        return x, y, z, yaw_deg

    async def _main(self):
        _log_scale_info(self.robot_path)

        self._robot = SingleArticulation(prim_path=self.robot_path)
        carb.log_info(f"{PREFIX}Robot object created, waiting for timeline to play...")

        while self._running and self._app.is_running():
            if self._timeline and hasattr(self._timeline, "is_playing") and self._timeline.is_playing():
                carb.log_info(f"{PREFIX}Timeline is playing, initializing robot...")
                break
            await self._app.next_update_async()

        if not self._running:
            carb.log_info(f"{PREFIX}Stopped before timeline started")
            return

        try:
            self._robot.initialize()
            carb.log_info(f"{PREFIX}Robot initialized successfully")
        except Exception as exc:
            raise RuntimeError(f"{PREFIX}Robot initialization failed: {exc}") from exc

        carb.log_info(f"{PREFIX}Waiting for articulation view to be ready...")
        for i in range(30):
            if hasattr(self._robot, "_articulation_view") and self._robot._articulation_view is not None:
                carb.log_info(f"{PREFIX}Articulation view ready after {i+1} frames")
                break
            await self._app.next_update_async()
        else:
            raise RuntimeError(f"{PREFIX}Articulation view not ready after 30 frames (硬失败)")

        view_dev = getattr(self._robot._articulation_view, "_device", None)
        ctrl_dev = getattr(self._robot._articulation_controller, "_device", None)
        carb.log_info(f"{PREFIX}view device={view_dev}, controller device={ctrl_dev}")

        meta = self._robot._articulation_view._metadata
        name_to_index = dict(meta.joint_indices)
        if self.left_joint not in name_to_index or self.right_joint not in name_to_index:
            carb.log_error(f"{PREFIX}Available joints: {list(name_to_index.keys())}")
            raise RuntimeError(
                f"{PREFIX}Joint not found: '{self.left_joint}' or '{self.right_joint}' (硬失败)"
            )
        self._idx_l = int(name_to_index[self.left_joint])
        self._idx_r = int(name_to_index[self.right_joint])
        carb.log_info(
            f"{PREFIX}Using joints: L='{self.left_joint}'(#{self._idx_l}), R='{self.right_joint}'(#{self._idx_r})"
        )

        if self.track_width_cfg is None:
            tw = _try_auto_track_width(self.robot_path)
            self._track_width = tw if (tw and tw > 0.01) else 0.54
            carb.log_info(f"{PREFIX}{'Auto' if tw else 'Fallback'} track width L={self._track_width:.3f} m")
        else:
            self._track_width = float(self.track_width_cfg)

        carb.log_info(f"{PREFIX}Binding velocity setter...")
        try:
            self._set_vel = _make_velocity_setter(self._robot._articulation_view, self._idx_l, self._idx_r)
            carb.log_info(f"{PREFIX} set_joint_velocity_targets bound successfully (Torch)")
        except Exception as exc:
            carb.log_error(f"{PREFIX} Failed to bind velocity setter: {exc}")
            raise

        self._frame_idx = 0
        # 初始位姿记录移到 _drive_loop 中，确保物理系统已准备好

        try:
            await self._drive_loop()
        finally:
            try:
                if self._set_vel:
                    self._set_vel(0.0, 0.0)
            except Exception:
                pass
            self._pose_logger.flush()


def launch(
    robot_path: str,
    left_joint: str,
    right_joint: str,
    wheel_radius: float = 0.24,
    track_width: Optional[float] = None,
    base_velocity: float = 0.6,
    base_turn_rate: float = 1.2,
    shift_multiplier: float = 3.0,
    ctrl_multiplier: float = 0.3,
    logging: Optional[dict] = None,
    linear_slew_rate: Optional[float] = 4.0,
    angular_slew_rate: Optional[float] = 6.0,
):
    """
    启动 Carter v1 键盘差速遥控。返回一个带 stop() 协程方法的句柄对象。
    键盘设备不可用/订阅失败/关节名无效将硬失败抛异常。
    """
    log_params = build_pose_logger_config(logging)
    log_enabled = log_params["enabled"]
    log_path_cfg = log_params["path"]
    threshold_cfg = log_params["distance_threshold"]
    log_print_cfg = log_params["print"]
    instruction_cfg = log_params["instruction"]
    objects_cfg = log_params["objects"]
    extra_metadata = log_params["extra_metadata"]

    carb.log_info(
        f"{PREFIX}launch params: path={log_path_cfg} threshold={threshold_cfg} enabled={log_enabled} print={log_print_cfg} instruction='{instruction_cfg}'"
    )

    teleop = CarterKeyboardTeleop(
        robot_path=robot_path,
        left_joint=left_joint,
        right_joint=right_joint,
        wheel_radius=wheel_radius,
        track_width=track_width,
        base_velocity=base_velocity,
        base_turn_rate=base_turn_rate,
        shift_multiplier=shift_multiplier,
        ctrl_multiplier=ctrl_multiplier,
        log_path=log_path_cfg,
        log_distance_threshold=threshold_cfg,
        log_print=log_print_cfg,
        log_instruction=instruction_cfg,
        episode_objects=objects_cfg,
        episode_metadata=extra_metadata,
        linear_slew_rate=linear_slew_rate,
        angular_slew_rate=angular_slew_rate,
    )
    return teleop.start()
