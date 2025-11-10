"""Standalone entry that drives envset scenarios via InternUtopia's SimulatorRunner."""

from __future__ import annotations

import argparse
import asyncio
import copy
import time
from pathlib import Path
from typing import Any, Dict

# Note: Do NOT import Isaac Sim modules (carb, omni, etc.) here!
# They must be imported AFTER SimulationApp is initialized.

from internutopia.core.config import Config
from internutopia.core.runner import SimulatorRunner
from internutopia.core.task_config_manager.base import create_task_config_manager

from internutopia_extension import import_extensions
from internutopia_extension.envset.config_loader import EnvsetConfigLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run envset scenarios through InternUtopia's SimulatorRunner.")
    parser.add_argument("--config", required=True, type=Path, help="Path to InternUtopia YAML config.")
    parser.add_argument("--envset", required=True, type=Path, help="Path to envset JSON file.")
    parser.add_argument("--scenario", default=None, help="Scenario id inside envset JSON.")
    parser.add_argument("--headless", action="store_true", help="Force headless SimulationApp.")
    parser.add_argument(
        "--extension-path",
        action="append",
        dest="extension_paths",
        help="Additional extension search path (can be specified multiple times). "
             "Example: --extension-path /path/to/isaaclab/source",
    )
    parser.add_argument(
        "--skip-isaac-assets",
        action="store_true",
        help="Skip querying shared Isaac Sim asset root (defaults to querying).",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="Keep SimulationApp alive for N seconds (omit to keep running until window closes).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not auto-start timeline playback after setup.",
    )
    parser.add_argument(
        "--run-data",
        action="store_true",
        help="(Future) Trigger data generation via Runner. Currently not supported.",
    )
    parser.add_argument("--label", default="standalone", help="Tag recorded in Infos.ext_version for diagnostics.")
    return parser.parse_args()


def _parse_config_model(config_dict: dict) -> Config:
    try:
        return Config.model_validate(config_dict)  # pydantic v2
    except AttributeError:  # pragma: no cover - fallback for v1
        return Config.parse_obj(config_dict)


class EnvsetStandaloneRunner:
    """Ties envset configuration into InternUtopia's SimulatorRunner lifecycle."""

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._config_path = args.config.expanduser().resolve()
        self._envset_path = args.envset.expanduser().resolve()
        self._bundle = EnvsetConfigLoader(self._config_path, self._envset_path, args.scenario).load()
        self._merged_dict = copy.deepcopy(self._bundle.config)
        self._runner: SimulatorRunner | None = None
        self._data_gen = None
        self._keyboard = None
        self._keyboard_robots = []
        self._shutdown_flag = False
        self._temp_paths = [self._bundle.merged_config_path]

    def request_shutdown(self):
        self._shutdown_flag = True

    def run(self):
        print("[EnvsetStandalone] Building config model...")
        # Build config first (before SimulationApp)
        config_model = self._build_config_model()

        print("[EnvsetStandalone] Importing extensions...")
        # Import extensions (prepares robot/controller registrations)
        import_extensions()

        print("[EnvsetStandalone] Creating runner (initializing SimulationApp)...")
        # Create runner (this initializes SimulationApp)
        self._runner = self._create_runner(config_model)

        print("[EnvsetStandalone] Preparing runtime settings...")
        # Now we can use Isaac Sim modules (carb, etc.)
        self._prepare_runtime_settings()

        print("[EnvsetStandalone] Post-runner initialization...")
        self._post_runner_initialize()

        print("[EnvsetStandalone] Resetting environment...")
        # Reset and start
        self._runner.reset()

        # 等待场景和对象完全初始化
        print("[EnvsetStandalone] Waiting for scene and objects to initialize...")
        self._wait_for_initialization()

        # 确保timeline保持暂停状态
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            timeline.pause()
            print("[EnvsetStandalone] Timeline was playing, paused it.")

        if self._args.run_data:
            self._init_data_generation()
            self._run_data_generation()
        else:
            # 不再自动启动timeline，等待用户手动启动
            print("[EnvsetStandalone] Ready. Timeline is paused. Please start timeline manually when ready.")
            print("[EnvsetStandalone] Entering main loop...")
            self._main_loop()

        print("[EnvsetStandalone] Run completed.")

    def shutdown(self):
        if self._runner and self._runner.simulation_app:
            try:
                self._runner.simulation_app.close()
            except Exception:
                pass
        for path in self._temp_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    # ---------- internal helpers ----------

    def _prepare_runtime_settings(self):
        # Import Isaac Sim modules here, after runner initialization
        import carb
        import carb.settings

        # Enable required extensions before importing envset modules
        print("[EnvsetStandalone] Enabling required extensions...")
        try:
            from omni.isaac.core.utils.extensions import enable_extension

            # Core envset dependencies (based on extension.toml dependencies)
            enable_extension("omni.usd")
            enable_extension("omni.anim.retarget.core")
            enable_extension("omni.kit.scripting")
            enable_extension("omni.kit.mesh.raycast")  # Required for raycast functionality
            enable_extension("omni.services.pip_archive")
            enable_extension("isaacsim.sensors.camera")
            enable_extension("isaacsim.sensors.physics")
            enable_extension("isaacsim.sensors.rtx")
            enable_extension("isaacsim.storage.native")
            enable_extension("isaacsim.core.utils")
            enable_extension("omni.metropolis.utils")
            enable_extension("omni.anim.navigation.schema")
            enable_extension("omni.anim.navigation.core")
            enable_extension("omni.anim.navigation.meshtools")
            enable_extension("omni.anim.people")
            enable_extension("isaacsim.anim.robot")
            enable_extension("omni.replicator.core")
            enable_extension("isaacsim.replicator.incident")

            # Optional: Matterport (may not be available in all Isaac Sim versions)
            try:
                enable_extension("omni.isaac.matterport")
                carb.log_info("[EnvsetStandalone] Matterport extension enabled")
            except Exception:
                carb.log_warn("[EnvsetStandalone] Matterport extension not available - Matterport scene import will be disabled")

            carb.log_info("[EnvsetStandalone] Required extensions enabled")
        except Exception as exc:
            carb.log_warn(f"[EnvsetStandalone] Failed to enable some extensions: {exc}")

        # Now safe to import envset modules that depend on these extensions
        from internutopia_extension.envset.settings import AssetPaths, Infos
        from internutopia_extension.envset.simulation import (
            ENVSET_AUTOSTART_SETTING,
            ENVSET_PATH_SETTING,
            ENVSET_SCENARIO_SETTING,
        )

        settings_iface = carb.settings.get_settings()
        settings_iface.set(ENVSET_PATH_SETTING, str(self._envset_path))
        settings_iface.set(ENVSET_AUTOSTART_SETTING, False)
        settings_iface.set(ENVSET_SCENARIO_SETTING, self._bundle.scenario_id)
        settings_iface.set(AssetPaths.USE_ISAAC_SIM_ASSET_ROOT_SETTING, not self._args.skip_isaac_assets)

        Infos.ext_version = str(self._args.label)
        Infos.ext_path = str(Path(__file__).resolve().parent)

        try:
            import warp

            warp.init()
        except Exception as exc:
            carb.log_warn(f"[EnvsetStandalone] Warp init failed: {exc}")

        if not self._args.skip_isaac_assets:
            try:
                asyncio.run(AssetPaths.cache_paths_async())
            except Exception as exc:
                carb.log_warn(f"[EnvsetStandalone] Failed to cache asset root: {exc}")

    def _build_config_model(self) -> Config:
        merged = copy.deepcopy(self._merged_dict)
        sim_section = merged.setdefault("simulator", {})
        if self._args.headless:
            sim_section["headless"] = True

        # Add extension paths from command line and/or config
        extension_folders = sim_section.get("extension_folders", [])
        if self._args.extension_paths:
            # Extend with CLI-provided paths
            extension_folders.extend(self._args.extension_paths)
        if extension_folders:
            sim_section["extension_folders"] = extension_folders

        # Convert controller dicts back to objects before pydantic parsing
        # This ensures pydantic preserves the correct ControllerCfg subclass types
        merged = self._convert_controller_dicts_to_objects(merged)

        config_model = _parse_config_model(merged)
        return config_model

    def _convert_controller_dicts_to_objects(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        将字典中的控制器配置转换回对象，确保 pydantic 解析时保持正确的子类类型。
        这样可以像 demo 一样直接使用对象，而不是字典。
        """
        # Import all ControllerCfg subclasses to build type mapping
        try:
            from internutopia_extension.configs.controllers import (
                AliengoMoveBySpeedControllerCfg,
                DifferentialDriveControllerCfg,
                FrankaMocapTeleopControllerCfg,
                G1MoveBySpeedControllerCfg,
                GR1MoveBySpeedControllerCfg,
                GR1TeleOpControllerCfg,
                GripperControllerCfg,
                H1MoveBySpeedControllerCfg,
                InverseKinematicsControllerCfg,
                JointControllerCfg,
                LayoutEditMocapControllerCfg,
                MoveAlongPathPointsControllerCfg,
                MoveToPointBySpeedControllerCfg,
                RecoverControllerCfg,
                RMPFlowControllerCfg,
                RotateControllerCfg,
            )
            
            # Build type -> class mapping
            # Map controller type strings to their corresponding ControllerCfg subclasses
            type_to_class = {
                'AliengoMoveBySpeedController': AliengoMoveBySpeedControllerCfg,
                'DifferentialDriveController': DifferentialDriveControllerCfg,
                'FrankaMocapTeleopController': FrankaMocapTeleopControllerCfg,
                'G1MoveBySpeedController': G1MoveBySpeedControllerCfg,
                'GR1MoveBySpeedController': GR1MoveBySpeedControllerCfg,
                'GR1TeleOpController': GR1TeleOpControllerCfg,
                'GripperController': GripperControllerCfg,
                'H1MoveBySpeedController': H1MoveBySpeedControllerCfg,
                'InverseKinematicsController': InverseKinematicsControllerCfg,
                'JointController': JointControllerCfg,
                'LayoutEditMocapController': LayoutEditMocapControllerCfg,
                'MoveAlongPathPointsController': MoveAlongPathPointsControllerCfg,
                'MoveToPointBySpeedController': MoveToPointBySpeedControllerCfg,
                'RecoverController': RecoverControllerCfg,
                'RMPFlowController': RMPFlowControllerCfg,
                'RotateController': RotateControllerCfg,
            }
        except ImportError:
            # If imports fail, return as-is (fallback to dict)
            return config_dict
        
        def convert_controller_dict(ctrl_dict: Dict[str, Any]) -> Any:
            """递归转换单个控制器字典为对象"""
            if not isinstance(ctrl_dict, dict):
                return ctrl_dict
            
            ctrl_type = ctrl_dict.get('type')
            if not ctrl_type:
                return ctrl_dict
            
            # Find the corresponding ControllerCfg subclass
            cfg_class = type_to_class.get(ctrl_type)
            if not cfg_class:
                # If no matching class found, use base ControllerCfg
                from internutopia.core.config.robot import ControllerCfg
                cfg_class = ControllerCfg
            
            # Recursively convert sub_controllers
            if 'sub_controllers' in ctrl_dict and ctrl_dict['sub_controllers']:
                sub_controllers = [convert_controller_dict(sub) for sub in ctrl_dict['sub_controllers']]
                ctrl_dict = dict(ctrl_dict)
                ctrl_dict['sub_controllers'] = sub_controllers
            
            # Create the object using pydantic's model_validate or parse_obj
            try:
                return cfg_class.model_validate(ctrl_dict)  # pydantic v2
            except AttributeError:
                return cfg_class.parse_obj(ctrl_dict)  # pydantic v1
        
        # Recursively convert controllers in task_configs -> robots -> controllers
        if 'task_configs' in config_dict:
            for task in config_dict['task_configs']:
                if 'robots' in task and task['robots']:
                    for robot in task['robots']:
                        if 'controllers' in robot and robot['controllers']:
                            robot['controllers'] = [convert_controller_dict(ctrl) for ctrl in robot['controllers']]
        
        return config_dict

    def _create_runner(self, config: Config) -> SimulatorRunner:
        task_manager = create_task_config_manager(config)
        runner = SimulatorRunner(config=config, task_config_manager=task_manager)
        return runner

    def _post_runner_initialize(self):
        # Import Isaac Sim modules (can now be safely imported)
        from internutopia_extension.envset.world_utils import bootstrap_world_if_needed
        from internutopia_extension.envset.agent_manager import AgentManager
        from internutopia_extension.envset.patches import install_safe_simtimes_guard

        bootstrap_world_if_needed()
        AgentManager.get_instance()
        install_safe_simtimes_guard()

    def _init_data_generation(self):
        """Initialize DataGeneration for recording simulation data."""
        import carb

        try:
            from internutopia_extension.data_generation.data_generation import DataGeneration
        except ImportError as e:
            carb.log_error(f"[EnvsetStandalone] Failed to import DataGeneration: {e}")
            return

        self._data_gen = DataGeneration()

        # Get data generation config from envset scenario or use defaults
        scenario = self._bundle.scenario
        data_gen_cfg = scenario.get("data_generation") or {}

        # Set writer name and params
        self._data_gen.writer_name = data_gen_cfg.get("writer", "BasicWriter")
        self._data_gen.writer_params = data_gen_cfg.get("writer_params") or {
            "output_dir": "_out_envset",
            "rgb": True,
            "semantic_segmentation": False,
        }

        # Set number of frames (default to 300 if not specified)
        self._data_gen._num_frames = data_gen_cfg.get("num_frames", 300)

        # Camera path list (empty means auto-detect from stage)
        self._data_gen._camera_path_list = data_gen_cfg.get("camera_paths") or []

        carb.log_info(
            f"[EnvsetStandalone] DataGeneration initialized: writer={self._data_gen.writer_name}, "
            f"frames={self._data_gen._num_frames}"
        )

    def _run_data_generation(self):
        """Run data generation asynchronously."""
        import carb

        if self._data_gen is None:
            carb.log_error("[EnvsetStandalone] DataGeneration not initialized")
            return

        carb.log_info("[EnvsetStandalone] Starting data generation...")
        try:
            asyncio.run(self._data_gen.run_async(will_wait_until_complete=True))
            carb.log_info("[EnvsetStandalone] Data generation completed successfully")
        except Exception as e:
            carb.log_error(f"[EnvsetStandalone] Data generation failed: {e}")
            import traceback
            carb.log_error(traceback.format_exc())

        # After data generation completes, continue with normal simulation if requested
        # 注意：不再自动启动timeline，等待用户手动启动
        if not self._args.no_play:
            print("[EnvsetStandalone] Data generation completed. Timeline is paused. Please start timeline manually when ready.")
            # 确保timeline保持暂停
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                if timeline.is_playing():
                    timeline.pause()
                    print("[EnvsetStandalone] Timeline was playing, paused it.")
            except Exception:
                pass
            self._main_loop()

    def _wait_for_initialization(self):
        """等待场景和对象完全初始化"""
        import carb
        from omni.isaac.core.simulation_context import SimulationContext
        
        # 确保timeline保持暂停
        try:
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_playing():
                timeline.pause()
                carb.log_info("[EnvsetStandalone] Timeline was playing, paused it during initialization")
        except Exception:
            pass
        
        # 使用SimulationContext的step方法来等待几帧
        # 这样可以确保USD系统处理完所有变更，但物理仿真还未启动
        try:
            world = self._runner._world if hasattr(self._runner, '_world') else None
            if world:
                # 渲染几帧但不启动物理，让场景完全加载
                for _ in range(5):
                    SimulationContext.render(world)
                carb.log_info("[EnvsetStandalone] Scene initialization wait completed")
            else:
                carb.log_warn("[EnvsetStandalone] World not available, skipping initialization wait")
        except Exception as e:
            carb.log_warn(f"[EnvsetStandalone] Initialization wait failed: {e}, continuing anyway")
        
        # 再次确保timeline保持暂停
        try:
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_playing():
                timeline.pause()
                carb.log_info("[EnvsetStandalone] Timeline paused after initialization wait")
        except Exception:
            pass

    def _wait_for_async_loading(self):
        """等待异步资产加载完成 - 已移除，改为在初始化等待中处理"""
        # 此方法已不再需要，因为load_assets_to_scene是异步的，会在后台完成
        # 我们只需要确保timeline保持暂停即可
        pass

    def _start_timeline(self):
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

    def _detect_keyboard_control(self):
        """Detect if any robot requires keyboard control."""
        import carb

        scenario = self._bundle.scenario
        robots_cfg = scenario.get("robots", {})
        robot_entries = robots_cfg.get("entries", [])

        keyboard_robots = []
        for robot in robot_entries:
            control = robot.get("control", {})
            control_mode = (control.get("mode") or "").lower()

            if "keyboard" in control_mode:
                robot_name = robot.get("label") or robot.get("type", "unknown")
                robot_type = (robot.get("type") or "").lower()

                # Map robot type to controller name
                controller_name = self._get_controller_name_for_robot(robot_type)
                if controller_name:
                    keyboard_robots.append({
                        "name": robot_name,
                        "controller": controller_name,
                        "type": robot_type
                    })
                    carb.log_info(f"[EnvsetStandalone] Detected keyboard control for robot: {robot_name}")

        return keyboard_robots

    def _get_controller_name_for_robot(self, robot_type: str) -> str | None:
        """Map robot type to its base controller name."""
        # Differential drive robots
        if robot_type in {"carter", "carter_v1", "jetbot", "differential_drive"}:
            return "move_by_speed"
        # Legged robots
        if robot_type in {"aliengo", "h1", "g1", "gr1", "human"}:
            return "move_by_speed"
        # Unknown type
        return None

    def _init_keyboard(self):
        """Initialize keyboard interaction if needed."""
        import carb

        self._keyboard_robots = self._detect_keyboard_control()

        if self._keyboard_robots:
            try:
                from internutopia_extension.interactions.keyboard import KeyboardInteraction
                self._keyboard = KeyboardInteraction()
                carb.log_info(f"[EnvsetStandalone] Keyboard control initialized for {len(self._keyboard_robots)} robot(s)")
            except ImportError as e:
                carb.log_error(f"[EnvsetStandalone] Failed to import KeyboardInteraction: {e}")
                self._keyboard = None
                self._keyboard_robots = []

    def _collect_actions(self):
        """Collect actions from keyboard or return empty actions for autonomous mode."""
        if not self._keyboard or not self._keyboard_robots:
            # Autonomous mode: return empty actions for all envs
            return [{}] * self._runner.env_num

        # Read keyboard input
        command = self._keyboard.get_input()
        x_speed = float(command[0] - command[1])  # I/K keys
        y_speed = float(command[2] - command[3])  # J/L keys
        z_speed = float(command[4] - command[5])  # U/O keys

        # Build actions for all keyboard-controlled robots
        actions = []
        for env_id in range(self._runner.env_num):
            env_action = {}
            for robot_cfg in self._keyboard_robots:
                # Use the controller name as action key
                env_action[robot_cfg["controller"]] = (x_speed, y_speed, z_speed)
            actions.append(env_action)

        return actions

    def _wait_for_articulations_initialized(self, max_wait_frames: int = 100):
        """
        等待所有 articulation 初始化完成，同时检查 NavMesh 烘培状态。
        
        Args:
            max_wait_frames: 最大等待帧数，超过此帧数后放弃等待
        
        Returns:
            bool: 如果所有 articulation 都已初始化返回 True，否则返回 False
        """
        import carb
        from omni.isaac.core.simulation_context import SimulationContext
        
        if not self._runner:
            return False
        
        world = self._runner._world if hasattr(self._runner, '_world') else None
        if not world:
            carb.log_warn("[EnvsetStandalone] World not available, cannot wait for articulations")
            return False
        
        # 尝试获取 NavMesh 接口
        navmesh_interface = None
        try:
            import omni.anim.navigation.core as nav
            navmesh_interface = nav.acquire_interface()
        except Exception:
            pass
        
        carb.log_info("[EnvsetStandalone] Waiting for all articulations to initialize...")
        if navmesh_interface:
            carb.log_info("[EnvsetStandalone] Also checking NavMesh baking status...")
        
        navmesh_ready = False
        
        for frame_idx in range(max_wait_frames):
            # 渲染一帧，让物理系统有机会初始化
            SimulationContext.render(world)
            
            # 检查所有任务的机器人
            all_initialized = True
            uninitialized_robots = []
            
            for task_name, task in self._runner.current_tasks.items():
                if not hasattr(task, 'robots') or not task.robots:
                    continue
                
                for robot_name, robot in task.robots.items():
                    if not hasattr(robot, 'articulation'):
                        continue
                    
                    if not hasattr(robot.articulation, 'handles_initialized'):
                        continue
                    
                    if not robot.articulation.handles_initialized:
                        all_initialized = False
                        uninitialized_robots.append(f"{task_name}/{robot_name}")
            
            # 检查 NavMesh 状态
            navmesh_status_msg = ""
            if navmesh_interface:
                try:
                    navmesh = navmesh_interface.get_navmesh()
                    if navmesh is not None:
                        if not navmesh_ready:
                            navmesh_ready = True
                            try:
                                area_count = navmesh.get_area_count()
                                navmesh_status_msg = f", NavMesh ready (areas={area_count})"
                            except Exception:
                                navmesh_status_msg = ", NavMesh ready"
                    else:
                        navmesh_status_msg = ", NavMesh baking..."
                except Exception:
                    navmesh_status_msg = ", NavMesh status unknown"
            
            if all_initialized:
                if navmesh_ready:
                    carb.log_info(
                        f"[EnvsetStandalone] All articulations initialized and NavMesh ready after {frame_idx + 1} frames"
                    )
                else:
                    carb.log_info(
                        f"[EnvsetStandalone] All articulations initialized after {frame_idx + 1} frames"
                        f"{navmesh_status_msg}"
                    )
                return True
            
            # 每10帧打印一次状态
            if frame_idx % 10 == 0:
                status_parts = []
                if uninitialized_robots:
                    status_parts.append(
                        f"Uninitialized robots: {', '.join(uninitialized_robots[:5])}"
                        f"{'...' if len(uninitialized_robots) > 5 else ''}"
                    )
                if navmesh_status_msg:
                    status_parts.append(navmesh_status_msg.strip(', '))
                
                status_str = " | ".join(status_parts) if status_parts else "Waiting..."
                carb.log_info(
                    f"[EnvsetStandalone] Waiting... ({frame_idx + 1}/{max_wait_frames} frames) | {status_str}"
                )
        
        # 最终状态报告
        final_status = []
        if uninitialized_robots:
            final_status.append(f"Some articulations not initialized: {', '.join(uninitialized_robots[:3])}")
        if navmesh_interface:
            try:
                navmesh = navmesh_interface.get_navmesh()
                if navmesh is None:
                    final_status.append("NavMesh not ready")
            except Exception:
                pass
        
        if final_status:
            carb.log_warn(
                f"[EnvsetStandalone] Timeout after {max_wait_frames} frames. "
                f"{' | '.join(final_status)}. Continuing anyway, but errors may occur."
            )
        else:
            carb.log_warn(
                f"[EnvsetStandalone] Timeout after {max_wait_frames} frames. "
                f"Continuing anyway, but errors may occur."
            )
        return False

    def _are_articulations_ready(self) -> bool:
        """
        快速检查所有 articulation 是否已初始化。
        
        Returns:
            bool: 如果所有 articulation 都已初始化返回 True，否则返回 False
        """
        if not self._runner:
            return False
        
        for task_name, task in self._runner.current_tasks.items():
            if not hasattr(task, 'robots') or not task.robots:
                continue
            
            for robot_name, robot in task.robots.items():
                if not hasattr(robot, 'articulation'):
                    continue
                
                if not hasattr(robot.articulation, 'handles_initialized'):
                    continue
                
                if not robot.articulation.handles_initialized:
                    return False
        
        return True

    def _main_loop(self):
        import carb

        sim_app = self._runner.simulation_app if self._runner else None
        if sim_app is None:
            return

        # Initialize keyboard control if needed
        self._init_keyboard()

        # 检查 timeline 是否正在播放，如果是则等待 articulation 初始化
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            carb.log_info("[EnvsetStandalone] Timeline is playing, waiting for articulations to initialize...")
            self._wait_for_articulations_initialized()
        else:
            carb.log_info("[EnvsetStandalone] Timeline is paused. Articulations will initialize when timeline starts.")

        deadline = None
        if self._args.hold_seconds is not None:
            deadline = time.monotonic() + max(0.0, self._args.hold_seconds)

        carb.log_info(
            f"[EnvsetStandalone] Entering main loop "
            f"(keyboard={'enabled' if self._keyboard else 'disabled'})"
        )

        # 在主循环中持续检查 timeline 状态，如果从暂停变为播放，等待初始化
        timeline_was_playing = timeline.is_playing()
        
        while sim_app.is_running() and not self._shutdown_flag:
            if deadline is not None and time.monotonic() >= deadline:
                break

            # 检查 timeline 状态变化：如果从暂停变为播放，等待 articulation 初始化
            timeline_is_playing = timeline.is_playing()
            if timeline_is_playing and not timeline_was_playing:
                carb.log_info("[EnvsetStandalone] Timeline started, waiting for articulations to initialize...")
                self._wait_for_articulations_initialized()
                timeline_was_playing = True
            elif not timeline_is_playing:
                timeline_was_playing = False

            # 如果 timeline 正在播放，检查 articulation 是否已准备好
            # 如果未准备好，跳过这一步，只更新应用但不调用 runner.step()
            if timeline_is_playing:
                if not self._are_articulations_ready():
                    # Articulation 还未准备好，只更新应用，不调用 step()
                    sim_app.update()
                    continue
            
            # Collect actions (keyboard input or empty for autonomous)
            actions = self._collect_actions()

            # Use runner.step() instead of sim_app.update()
            try:
                self._runner.step(actions=actions, render=True)
            except Exception as e:
                # 如果是 articulation 未初始化的错误，等待并重试
                if "Failed to get root link transforms" in str(e) or "handles_initialized" in str(e):
                    carb.log_warn(f"[EnvsetStandalone] Articulation not ready, waiting... Error: {e}")
                    # 等待几帧让 articulation 初始化
                    for _ in range(5):
                        sim_app.update()
                    # 再次检查，如果准备好了就继续，否则跳过这一步
                    if self._are_articulations_ready():
                        carb.log_info("[EnvsetStandalone] Articulations ready, continuing...")
                        continue
                    else:
                        carb.log_warn("[EnvsetStandalone] Articulations still not ready, skipping step()")
                        sim_app.update()
                        continue
                else:
                    carb.log_error(f"[EnvsetStandalone] Error in runner.step(): {e}")
                    # Fallback to simple update on error
                    sim_app.update()


def main():
    args = _parse_args()
    if not args.config.expanduser().exists():
        raise SystemExit(f"Config file not found: {args.config}")
    if not args.envset.expanduser().exists():
        raise SystemExit(f"Envset file not found: {args.envset}")

    runner = EnvsetStandaloneRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("[EnvsetStandalone] Interrupted by user")
        runner.request_shutdown()
    except Exception as e:
        print(f"[EnvsetStandalone] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
