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

    def request_shutdown(self):
        self._shutdown_flag = True

    def _log_extension_status(self):
        """打印关键扩展的启用状态，辅助诊断。"""
        try:
            import omni  # type: ignore
        except ImportError:
            print("[EnvsetStandalone] omni 模块不可用，无法打印扩展状态。")
            return

        extensions_to_check = [
            "omni.usd",
            "omni.anim.retarget.core",
            "omni.kit.scripting",
            "omni.kit.mesh.raycast",
            "omni.services.pip_archive",
            "isaacsim.sensors.camera",
            "isaacsim.sensors.physics",
            "isaacsim.sensors.rtx",
            "isaacsim.storage.native",
            "isaacsim.core.utils",
            "omni.metropolis.utils",
            "omni.anim.navigation.schema",
            "omni.anim.navigation.core",
            "omni.anim.navigation.meshtools",
            "omni.anim.people",
            "isaacsim.anim.robot",
            "omni.replicator.core",
            "isaacsim.replicator.incident",
        ]

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        print("[EnvsetStandalone] Extension status:")
        for ext in extensions_to_check:
            enabled = ext_manager.is_extension_enabled(ext)
            print(f"  {ext:40s} -> {enabled}")

    def _print_runtime_snapshot(self, label: str):
        """打印 stage / 角色 / Agent 当前状态。"""
        try:
            import omni  # type: ignore
            import omni.timeline  # type: ignore
            from pxr import Usd  # type: ignore
            from internutopia_extension.envset.agent_manager import AgentManager
            from internutopia_extension.envset.settings import PrimPaths
        except ImportError as exc:
            print(f"[EnvsetStandalone] 无法导入调试所需模块: {exc}")
            return

        stage = omni.usd.get_context().get_stage()
        print(f"[EnvsetStandalone] === Snapshot: {label} ===")
        print(f"  Stage valid: {bool(stage)}")

        characters_root_path = None
        characters_root = None
        if stage:
            characters_root_path = PrimPaths.characters_parent_path()
            characters_root = stage.GetPrimAtPath(characters_root_path)
        print(
            f"  Characters root: {characters_root_path or 'N/A'} -> "
            f"{bool(characters_root and characters_root.IsValid())}"
        )

        skel_infos: list[dict[str, str]] = []
        if characters_root and characters_root.IsValid():
            for prim in Usd.PrimRange(characters_root):
                if prim.GetTypeName() == "SkelRoot":
                    graph_attr = prim.GetAttribute("omni:anim_graph:graph_path")
                    graph_path = graph_attr.Get() if graph_attr and graph_attr.IsValid() else None
                    scripts_attr = prim.GetAttribute("omni:scripting:scripts")
                    scripts = scripts_attr.Get() if scripts_attr and scripts_attr.IsValid() else None
                    skel_infos.append(
                        {
                            "path": str(prim.GetPath()),
                            "graph": str(graph_path) if graph_path else "",
                            "scripts": str(scripts) if scripts else "",
                        }
                    )
                    if len(skel_infos) >= 5:
                        break
        print(f"  Detected SkelRoot count: {len(skel_infos)}")
        if skel_infos:
            for info in skel_infos:
                print(
                    "    SkelRoot: {path}, anim_graph={graph}, scripts={scripts}".format(
                        path=info["path"],
                        graph=info["graph"] or "None",
                        scripts=info["scripts"] or "None",
                    )
                )

        mgr = AgentManager.get_instance()
        agents = list(mgr.get_all_agent_names())
        print(f"  Registered agents ({len(agents)}): {agents[:5]}")
        timeline = omni.timeline.get_timeline_interface()
        print(f"  Timeline playing?: {timeline.is_playing()}")

    def _debug_articulation_paths(self):
        """调试：检查articulation路径和状态"""
        import omni  # type: ignore
        from pxr import Usd, Sdf  # type: ignore

        stage = omni.usd.get_context().get_stage()

        # 1) 打印预期路径是否存在
        expect = "/World/env_0/robots/aliengo"
        prim = stage.GetPrimAtPath(Sdf.Path(expect))
        print(f"[DEBUG] expect prim exists? {prim.IsValid()} type={prim.GetTypeName() if prim.IsValid() else None}")

        # 2) 用 tensors 列举当前全场景 articulation 根
        try:
            from isaacsim.core.simulation_manager import SimulationManager  # type: ignore
            psv = SimulationManager.get_physics_sim_view()
        except Exception:
            try:
                import omni.physics.tensors as phys  # type: ignore
                psv = phys.create_simulation_view("numpy")
            except Exception:
                print("[DEBUG] Cannot create physics simulation view")
                return

        try:
            all_view = psv.create_articulation_view("/*")  # 列举全部
            print(f"[DEBUG] articulations count = {all_view.count}")
            try:
                # 有的版本能拿到 dof_paths 或 body_paths，帮助反推根
                if hasattr(all_view, 'dof_paths') and all_view.count > 0:
                    print(f"[DEBUG] sample dof_paths (first 3) = {all_view.dof_paths[:3]}")
            except Exception:
                pass
        except Exception as e:
            print(f"[DEBUG] Cannot create articulation view: {e}")

        # 3) 简单枚举 /World/env_0 下的子树，肉眼确认
        def _list(prefix="/World/env_0"):
            if not stage.GetPrimAtPath(prefix).IsValid():
                print(f"[DEBUG] no such prefix: {prefix}")
                return
            print(f"[DEBUG] children under {prefix}:")
            it = Usd.PrimRange(stage.GetPrimAtPath(prefix))
            n = 0
            for p in it:
                if p.IsA(Usd.Prim):
                    path = p.GetPath().pathString
                    if "/robots" in path or "Aliengo" in path or "aliengo" in path:
                        print(f"   {path} {p.GetTypeName()} {p.GetAppliedSchemas()}")
                        n += 1
                        if n > 10:  # 限制输出
                            break

        _list("/World")
        _list("/World/env_0")

    def run(self):
        print("[EnvsetStandalone] Building config model...")
        # Build config first (before SimulationApp)
        config_model = self._build_config_model()

        print("[EnvsetStandalone] Importing extensions...")
        # Import extensions (prepares robot/controller registrations)
        import_extensions()

        print("[EnvsetStandalone] Initializing SimulationApp...")
        # Initialize SimulationApp first (but don't create World yet)
        self._init_simulation_app(config_model)

        print("[EnvsetStandalone] Enabling critical extensions (before World creation)...")
        # Enable critical extensions IMMEDIATELY after SimulationApp creation
        # This is crucial for omni.anim.graph.core to initialize properly
        self._enable_critical_extensions()

        print("[EnvsetStandalone] Creating runner (initializing World and tasks)...")
        # Now create the full runner (this will create World)
        self._runner = self._create_runner_with_app(config_model)

        print("[EnvsetStandalone] Preparing runtime settings...")
        # Configure additional runtime settings
        self._prepare_runtime_settings()

        print("[EnvsetStandalone] Post-runner initialization...")
        self._post_runner_initialize()
        self._print_runtime_snapshot("After post-runner initialization (before reset)")

        # 调试：检查articulation路径和状态
        print("[EnvsetStandalone] Checking articulation paths and status...")
        self._debug_articulation_paths()

        print("[EnvsetStandalone] Resetting environment...")
        # Reset and start
        self._runner.reset()
        self._print_runtime_snapshot("After runner.reset()")

        # 等待场景和对象完全初始化
        print("[EnvsetStandalone] Waiting for scene and objects to initialize...")
        self._wait_for_initialization()
        self._print_runtime_snapshot("After initialization wait")

        if self._args.run_data:
            self._init_data_generation()
            self._run_data_generation()
        else:
            # 不再自动启动timeline，等待用户手动启动
            print("[EnvsetStandalone] Entering main loop...")
            self._main_loop()

        print("[EnvsetStandalone] Run completed.")

    def shutdown(self):
        if self._runner and self._runner.simulation_app:
            try:
                self._runner.simulation_app.close()
            except Exception:
                pass

    # ---------- internal helpers ----------

    def _init_simulation_app(self, config: Config):
        """Initialize SimulationApp without creating World yet."""
        from isaacsim import SimulationApp  # type: ignore
        import os

        headless = config.simulator.headless
        
        # Build launch config
        launch_config = {
            'headless': headless,
            'anti_aliasing': 0,
            'hide_ui': False,
            'multi_gpu': False
        }

        # Add custom extension paths if specified
        if hasattr(config.simulator, 'extension_folders') and config.simulator.extension_folders:
            ext_paths = config.simulator.extension_folders
            if 'ISAAC_EXTRA_EXT_PATH' in os.environ:
                existing = os.environ['ISAAC_EXTRA_EXT_PATH']
                os.environ['ISAAC_EXTRA_EXT_PATH'] = os.pathsep.join([existing] + ext_paths)
            else:
                os.environ['ISAAC_EXTRA_EXT_PATH'] = os.pathsep.join(ext_paths)

        print(f"[EnvsetStandalone] Creating SimulationApp with config: {launch_config}")
        self._external_sim_app = SimulationApp(launch_config)
        # Apply the same post-init configuration as SimulatorRunner.setup_isaacsim
        try:
            self._external_sim_app._carb_settings.set('/physics/cooking/ujitsoCollisionCooking', False)
        except Exception:
            pass
        print("[EnvsetStandalone] SimulationApp created successfully")

    def _enable_critical_extensions(self):
        """Enable critical extensions immediately after SimulationApp creation.
        
        This must happen BEFORE World creation to ensure omni.anim.graph.core
        and related extensions are properly initialized.
        """
        import carb  # type: ignore
        
        print("[EnvsetStandalone] Enabling critical extensions for animation graph...")
        try:
            from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

            # CRITICAL: Animation graph extensions MUST be enabled before World creation
            # This is required for Isaac Sim 5.0.0 where omni.anim.graph.core v107.3.0
            # needs early initialization
            enable_extension("omni.anim.graph.core")
            enable_extension("omni.anim.retarget.core")
            enable_extension("omni.anim.navigation.schema")
            enable_extension("omni.anim.navigation.core")
            enable_extension("omni.anim.navigation.meshtools")
            enable_extension("omni.anim.people")
            
            carb.log_info("[EnvsetStandalone] Critical animation extensions enabled before World creation")
        except Exception as exc:
            carb.log_error(f"[EnvsetStandalone] FAILED to enable critical extensions: {exc}")
            raise

    def _prepare_runtime_settings(self):
        """Configure additional runtime settings and enable remaining extensions."""
        import carb  # type: ignore
        import carb.settings  # type: ignore

        # Enable remaining extensions (non-critical ones)
        print("[EnvsetStandalone] Enabling remaining extensions...")
        try:
            from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

            # Core envset dependencies
            enable_extension("omni.usd")
            enable_extension("omni.kit.scripting")
            enable_extension("omni.kit.mesh.raycast")
            enable_extension("omni.services.pip_archive")
            enable_extension("isaacsim.sensors.camera")
            enable_extension("isaacsim.sensors.physics")
            enable_extension("isaacsim.sensors.rtx")
            enable_extension("isaacsim.storage.native")
            enable_extension("isaacsim.core.utils")
            enable_extension("omni.metropolis.utils")
            enable_extension("isaacsim.anim.robot")
            enable_extension("omni.replicator.core")
            enable_extension("isaacsim.replicator.incident")
            
            try:
                enable_extension("omni.physxcommands")
            except Exception:
                carb.log_warn("[EnvsetStandalone] omni.physxcommands not available (optional)")

            # Optional: Matterport
            try:
                enable_extension("omni.isaac.matterport")
                carb.log_info("[EnvsetStandalone] Matterport extension enabled")
            except Exception:
                carb.log_warn("[EnvsetStandalone] Matterport extension not available")

            carb.log_info("[EnvsetStandalone] Remaining extensions enabled")
        except Exception as exc:
            carb.log_warn(f"[EnvsetStandalone] Failed to enable some extensions: {exc}")
        finally:
            self._log_extension_status()

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
            import warp  # type: ignore

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

        # Note: task_adapter now returns RobotCfg/ControllerCfg objects directly
        # No need for dict->object conversion, Pydantic handles it automatically

        config_model = _parse_config_model(merged)
        return config_model

    def _create_runner(self, config: Config) -> SimulatorRunner:
        """Legacy method - creates runner with embedded SimulationApp."""
        task_manager = create_task_config_manager(config)
        runner = SimulatorRunner(config=config, task_config_manager=task_manager)
        return runner

    def _create_runner_with_app(self, config: Config) -> SimulatorRunner:
        """Create runner using the pre-initialized SimulationApp."""
        from internutopia.core.util import log

        if not hasattr(self, "_external_sim_app") or self._external_sim_app is None:
            raise RuntimeError("SimulationApp is not initialized. Call _init_simulation_app() first.")

        # Prepare task manager (reuse helpers from core)
        task_manager = create_task_config_manager(config)

        # Define a subclass that reuses our already-created SimulationApp
        external_app = self._external_sim_app

        class _ReusableAppRunner(SimulatorRunner):
            def __init__(self, *, _simulation_app, **kwargs):
                self._external_sim_app = _simulation_app
                super().__init__(**kwargs)

            def setup_isaacsim(self):
                # Reuse existing SimulationApp instead of creating a new one
                self._simulation_app = self._external_sim_app
                try:
                    self._simulation_app._carb_settings.set('/physics/cooking/ujitsoCollisionCooking', False)
                except Exception:
                    pass

                native = self.config.simulator.native
                webrtc = self.config.simulator.webrtc

                try:
                    from isaacsim import util  # type: ignore
                except ImportError:
                    self.setup_streaming_420(native, webrtc)
                else:
                    if native:
                        log.warning('native streaming is DEPRECATED, webrtc streaming is used instead')
                    webrtc = native or webrtc
                    self.setup_streaming_450(webrtc)

        # Instantiate custom runner
        runner = _ReusableAppRunner(
            _simulation_app=external_app,
            config=config,
            task_config_manager=task_manager,
        )

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
        import carb  # type: ignore

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
        import carb  # type: ignore

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
            self._main_loop()

    def _wait_for_initialization(self):
        import carb  # type: ignore
        from omni.isaac.core.simulation_context import SimulationContext  # type: ignore

        print("[EnvsetStandalone] Waiting for initialization...")
        # 获取 World 实例
        try:
            world = self._runner._world if hasattr(self._runner, '_world') else None
            if not world:
                carb.log_warn("[EnvsetStandalone] World not available, skipping initialization wait")
                return

            # Step 1: 执行 physics steps 让物理状态传播和稳定
            # 这对防止物体穿透地板至关重要
            print("[EnvsetStandalone] Starting physics warm-up (2 steps)...")
            for i in range(2):
                try:
                    world.step(render=False)
                    print(f"[EnvsetStandalone] Physics warm-up step {i+1}/2 completed")
                except Exception as e:
                    print(f"[EnvsetStandalone] Physics step {i+1} failed: {e}")

            # Step 2: 执行 render steps 让传感器数据更新
            # 这对相机和其他传感器的正确初始化很重要
            print("[EnvsetStandalone] Starting render warm-up (12 steps)...")
            for i in range(12):
                try:
                    SimulationContext.render(world)
                    if i % 3 == 0:  # 每 3 帧输出一次日志
                        print(f"[EnvsetStandalone] Render warm-up step {i+1}/12")
                except Exception as e:
                    print(f"[EnvsetStandalone] Render step {i+1} failed: {e}")

            print("[EnvsetStandalone] Scene initialization wait completed (2 physics + 12 render steps)")

        except Exception as e:
            print(f"[EnvsetStandalone] Initialization wait failed: {e}, continuing anyway")

    def _start_timeline(self):
        import omni.timeline  # type: ignore

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

    def _detect_keyboard_control(self):
        """Detect if any robot requires keyboard control."""
        import carb  # type: ignore

        scenario = self._bundle.scenario
        robots_cfg = scenario.get("robots", {})
        robot_entries = robots_cfg.get("entries", [])

        print(f"[DEBUG] Detecting keyboard control, found {len(robot_entries)} robots")
        keyboard_robots = []
        for robot in robot_entries:
            control = robot.get("control", {})
            control_mode = (control.get("mode") or "").lower()
            # 使用 label 作为机器人名称，这与 task.robots 字典中的键一致
            robot_name = robot.get("label") or robot.get("type", "unknown")
            print(f"[DEBUG] Robot {robot_name}: control_mode='{control_mode}'")

            if "keyboard" in control_mode:
                robot_type = (robot.get("type") or "").lower()

                # Map robot type to controller name
                controller_name = self._get_controller_name_for_robot(robot_type)
                print(f"[DEBUG] Mapped robot_type '{robot_type}' to controller '{controller_name}'")
                if controller_name:
                    keyboard_robots.append({
                        "name": robot_name,  # 这是机器人在 task.robots 中的键
                        "controller": controller_name,  # 这是控制器的名称
                        "type": robot_type
                    })
                    carb.log_info(f"[EnvsetStandalone] Detected keyboard control for robot: {robot_name}")
                else:
                    print(f"[DEBUG] No controller mapping for robot_type '{robot_type}'")

        print(f"[DEBUG] Total keyboard robots detected: {len(keyboard_robots)}")
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
        import carb  # type: ignore

        print("[DEBUG] Starting keyboard initialization...")
        self._keyboard_robots = self._detect_keyboard_control()
        print(f"[DEBUG] Keyboard robots after detection: {self._keyboard_robots}")

        if self._keyboard_robots:
            try:
                print("[DEBUG] Importing KeyboardInteraction...")
                from internutopia_extension.interactions.keyboard import KeyboardInteraction
                print("[DEBUG] Creating KeyboardInteraction...")
                self._keyboard = KeyboardInteraction()
                carb.log_info(f"[EnvsetStandalone] Keyboard control initialized for {len(self._keyboard_robots)} robot(s)")
                print("[DEBUG] Keyboard initialization successful")
            except ImportError as e:
                carb.log_error(f"[EnvsetStandalone] Failed to import KeyboardInteraction: {e}")
                print(f"[DEBUG] Keyboard import failed: {e}")
                self._keyboard = None
                self._keyboard_robots = []

        if self._keyboard:
            print("[EnvsetStandalone] Keyboard control ready!")
            print("[EnvsetStandalone] Make sure Isaac Sim window has focus to receive keyboard input")
            print("[EnvsetStandalone] Use I/K for forward/back, J/L for left/right, U/O for up/down")

    def _collect_actions(self):
        """Collect actions from keyboard or return empty actions for autonomous mode."""
        if not self._keyboard or not self._keyboard_robots:
            # Autonomous mode: return empty actions for all envs
            return [{}] * self._runner.env_num

        # Read keyboard input
        command = self._keyboard.get_input()
        print(f"[DEBUG] Keyboard command: {command}")
        x_speed = float(command[0] - command[1])  # I/K keys
        y_speed = float(command[2] - command[3])  # J/L keys
        z_speed = float(command[4] - command[5])  # U/O keys
        print(f"[DEBUG] Computed speeds: x={x_speed}, y={y_speed}, z={z_speed}")

        # Build actions for all keyboard-controlled robots
        # 关键修复：动作格式必须是 {机器人名称: {控制器名称: 动作}}
        # 这样才能匹配 runner.py 中 task.robots 字典的键
        actions = []
        for env_id in range(self._runner.env_num):
            env_action = {}
            for robot_cfg in self._keyboard_robots:
                # robot_cfg["name"] 是机器人在 task.robots 中的键（即 label）
                # robot_cfg["controller"] 是控制器的名称
                robot_name = robot_cfg["name"]
                controller_name = robot_cfg["controller"]
                
                # 构建正确的嵌套结构：{机器人名称: {控制器名称: 动作}}
                env_action[robot_name] = {
                    controller_name: (x_speed, y_speed, z_speed)
                }
                print(f"[DEBUG] Built action for robot '{robot_name}': {{'{controller_name}': ({x_speed}, {y_speed}, {z_speed})}}")
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
        import carb  # type: ignore
        from omni.isaac.core.simulation_context import SimulationContext  # type: ignore
        
        if not self._runner:
            return False
        
        world = self._runner._world if hasattr(self._runner, '_world') else None
        if not world:
            carb.log_warn("[EnvsetStandalone] World not available, cannot wait for articulations")
            return False
        
        # 尝试获取 NavMesh 接口
        navmesh_interface = None
        try:
            import omni.anim.navigation.core as nav  # type: ignore
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
        import carb  # type: ignore
        print("[EnvsetStandalone] Entering main loop...")
        sim_app = self._runner.simulation_app if self._runner else None
        if sim_app is None:
            return
        print("[EnvsetStandalone] Simulation app initialized")
        # Initialize keyboard control if needed
        self._init_keyboard()
        print("[EnvsetStandalone] Keyboard control initialized")
        # 检查 timeline 是否正在播放，如果是则等待 articulation 初始化
        import omni.timeline  # type: ignore
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            print("[EnvsetStandalone] Timeline is playing, waiting for articulations to initialize...")
            self._wait_for_articulations_initialized()
        else:
            print("[EnvsetStandalone] Timeline is paused. Articulations will initialize when timeline starts.")

        deadline = None
        if self._args.hold_seconds is not None:
            deadline = time.monotonic() + max(0.0, self._args.hold_seconds)

        print(
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
                    print("[EnvsetStandalone] Articulation not ready, skipping step()")
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
                    print(f"[EnvsetStandalone] Articulation not ready, waiting... Error: {e}")
                    # 等待几帧让 articulation 初始化
                    for _ in range(5):
                        sim_app.update()
                    # 再次检查，如果准备好了就继续，否则跳过这一步
                    if self._are_articulations_ready():
                        print("[EnvsetStandalone] Articulations ready, continuing...")
                        continue
                    else:
                        print("[EnvsetStandalone] Articulations still not ready, skipping step()")
                        sim_app.update()
                        continue
                else:
                    print(f"[EnvsetStandalone] Error in runner.step(): {e}")
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
