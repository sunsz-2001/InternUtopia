import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import carb


@dataclass
class PoseLogRecord:
    frame: int
    time_s: Optional[float]
    x: float
    y: float
    world_z: float
    ground_z: float
    yaw_deg: float
    distance_xy: float
    distance_xyz: float
    distance_total_xy: float
    command_v: float
    command_w: float


class PoseLogger:
    """
    Utility to accumulate robot poses and write them to disk in a consistent format.
    Keeps XY distance thresholds, handles incremental flushing, and falls back to a
    home-directory log if the primary location is not writable.
    """

    def __init__(
        self,
        *,
        prefix: str,
        robot_path: str,
        safe_name: str,
        log_path: Optional[Sequence[str]] = None,
        distance_threshold: float = 0.0,
        log_print: bool = False,
        instruction: str = "",
        default_dir: Optional[Path] = None,
        episode_objects: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self._prefix = prefix
        self._robot_path = robot_path
        self._safe_name = safe_name or "World"
        self._log_threshold = max(float(distance_threshold), 0.0)
        self._print_enabled = bool(log_print)
        self._instruction = instruction.strip()
        self._episode_objects = copy.deepcopy(episode_objects) if episode_objects is not None else None
        self._extra_metadata = copy.deepcopy(extra_metadata) if extra_metadata else None

        self._log_points: List[dict] = []
        self._log_last_pose: Optional[Tuple[float, float, float]] = None
        self._log_total_distance = 0.0
        self._log_write_failures = 0
        self._log_fallback_used = False
        self._first_record: Optional[PoseLogRecord] = None
        self._last_record: Optional[PoseLogRecord] = None

        default_dir = default_dir or (Path.cwd() / "logs")

        resolved_path: Optional[Path]
        if log_path:
            if isinstance(log_path, (str, os.PathLike)):
                resolved_path = Path(log_path)
            else:
                resolved_path = Path(*log_path)
            resolved_path = resolved_path.expanduser()
            if not resolved_path.is_absolute():
                resolved_path = (Path.cwd() / resolved_path).resolve()
        else:
            resolved_path = (default_dir / f"{self._safe_name}_path.json").resolve()
            self._log_fallback_used = True

        self._log_path = resolved_path
        self._log_enabled = self._log_path is not None

        if self._log_enabled and self._log_path:
            carb.log_info(
                f"{self._prefix}Pose logging enabled: path={self._log_path} threshold={self._log_threshold:.3f} print={self._print_enabled}"
            )
            if self._log_fallback_used:
                carb.log_warn(
                    f"{self._prefix}No logging.path provided; using default {self._log_path}. Set logging.path to override."
                )
        else:
            carb.log_info(f"{self._prefix}Pose logging disabled for {self._robot_path}.")

    @property
    def enabled(self) -> bool:
        return self._log_enabled and self._log_path is not None

    @property
    def print_enabled(self) -> bool:
        return self._print_enabled

    @property
    def path(self) -> Optional[Path]:
        return self._log_path if self._log_enabled else None

    @property
    def threshold(self) -> float:
        return self._log_threshold

    @property
    def total_distance_xy(self) -> float:
        return self._log_total_distance

    def record(
        self,
        *,
        frame_idx: int,
        time_sec: Optional[float],
        world_xyz: Tuple[float, float, float],
        ground_z: float,
        yaw_deg: float,
        command: Tuple[float, float],
        force: bool = False,
    ) -> Optional[PoseLogRecord]:
        if not self.enabled:
            return None

        x, y, world_z = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        last_pose = self._log_last_pose
        if last_pose is None:
            distance_xy = 0.0
            distance_xyz = 0.0
        else:
            dx = x - last_pose[0]
            dy = y - last_pose[1]
            dz = world_z - last_pose[2]
            distance_xy = (dx * dx + dy * dy) ** 0.5
            distance_xyz = (dx * dx + dy * dy + dz * dz) ** 0.5

        should_log = force or last_pose is None or distance_xy >= self._log_threshold
        if not should_log:
            return None

        if last_pose is None:
            distance_xy = 0.0
            distance_xyz = 0.0

        self._log_total_distance += distance_xy
        self._log_last_pose = (x, y, world_z)

        v_cmd, w_cmd = float(command[0]), float(command[1])
        entry = {
            "frame": int(frame_idx),
            "time_s": time_sec,
            "position": {"x": x, "y": y, "z": float(ground_z)},
            "yaw_deg": float(yaw_deg),
            "distance": {
                "since_last_xy": float(distance_xy),
                "since_last_xyz": float(distance_xyz),
                "total_xy": float(self._log_total_distance),
            },
            "command": {"v": v_cmd, "w": w_cmd},
        }
        self._log_points.append(entry)

        record_obj = PoseLogRecord(
            frame=int(frame_idx),
            time_s=time_sec,
            x=x,
            y=y,
            world_z=world_z,
            ground_z=float(ground_z),
            yaw_deg=float(yaw_deg),
            distance_xy=float(distance_xy),
            distance_xyz=float(distance_xyz),
            distance_total_xy=float(self._log_total_distance),
            command_v=v_cmd,
            command_w=w_cmd,
        )
        if self._first_record is None:
            self._first_record = record_obj
        self._last_record = record_obj
        return record_obj

    @staticmethod
    def _build_pose_metadata(record: PoseLogRecord) -> dict:
        return {
            "xyz": [float(record.x), float(record.y), float(record.ground_z)],
            "world_z": float(record.world_z),
            "yaw_deg": float(record.yaw_deg),
        }

    def flush(self, *, incremental: bool = False):
        if not self.enabled or not self._log_points or self._log_path is None:
            return
        if incremental and len(self._log_points) > 1 and len(self._log_points) % 5 != 0:
            return

        path_entries: List[dict] = []
        for item in self._log_points:
            pos = item.get("position", {})
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            ground_z = float(pos.get("z", 0.0))

            entry = {
                "frame": int(item.get("frame", 0)),
                "xyz": [x, y, ground_z],
                "yaw_deg": float(item.get("yaw_deg", 0.0)),
            }

            time_val = item.get("time_s")
            if time_val is not None:
                try:
                    entry["time_s"] = float(time_val)
                except Exception:
                    pass

            dist = item.get("distance")
            if isinstance(dist, dict):
                entry["distance_xy"] = float(dist.get("since_last_xy", 0.0))
                entry["distance_total_xy"] = float(dist.get("total_xy", 0.0))

            cmd = item.get("command")
            if isinstance(cmd, dict):
                entry["command"] = {
                    "v": float(cmd.get("v", 0.0)),
                    "w": float(cmd.get("w", 0.0)),
                }

            path_entries.append(entry)

        metadata = {
            "robot_path": self._robot_path,
            "distance_threshold_xy": self._log_threshold,
            "distance_total_xy": self._log_total_distance,
            "sample_count": len(self._log_points),
        }
        if self._first_record is not None:
            metadata["robot_initial_pose"] = self._build_pose_metadata(self._first_record)
        if self._last_record is not None:
            metadata["robot_final_pose"] = self._build_pose_metadata(self._last_record)
        if self._episode_objects is not None:
            metadata["objects"] = copy.deepcopy(self._episode_objects)
        if self._extra_metadata:
            for key, value in self._extra_metadata.items():
                metadata[key] = copy.deepcopy(value)

        payload = {
            "instruction": self._instruction,
            "gt_path": path_entries,
            "metadata": metadata,
        }

        success = self._try_write_payload(payload)
        if success and not incremental:
            carb.log_info(f"{self._prefix}Saved {len(self._log_points)} pose samples to {self._log_path}")

    def _try_write_payload(self, payload: dict) -> bool:
        if self._log_path is None:
            return False

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._log_path.with_suffix(self._log_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
                fp.flush()
                os.fsync(fp.fileno())
            tmp_path.replace(self._log_path)
            self._log_write_failures = 0
            return True
        except Exception as exc:
            self._log_write_failures += 1
            carb.log_warn(f"{self._prefix}Failed to write pose log to {self._log_path}: {exc}")

            if not self._log_fallback_used:
                fallback_dir = Path.home() / "ReplicatorResult" / "teleop"
                try:
                    fallback_dir.mkdir(parents=True, exist_ok=True)
                    fallback_path = fallback_dir / f"{self._safe_name}_path.json"
                    carb.log_warn(f"{self._prefix}Switching pose log to fallback path {fallback_path}")
                    self._log_path = fallback_path
                    self._log_fallback_used = True
                    return self._try_write_payload(payload)
                except Exception as fallback_exc:
                    carb.log_error(
                        f"{self._prefix}Fallback log path creation failed: {fallback_exc}. Pose logging disabled."
                    )
                    self._log_enabled = False
            elif self._log_write_failures % 5 == 0:
                carb.log_error(
                    f"{self._prefix}Pose logging still failing after {self._log_write_failures} attempts. Latest error: {exc}"
                )
            return False
