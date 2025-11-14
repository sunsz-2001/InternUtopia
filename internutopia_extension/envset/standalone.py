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
        self._simulation_app = None
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
                    # 检查动画图和脚本状态（用于诊断）
                    # 注意：Isaac Sim 5.0.0 中 AnimGraphSchema 不可用，我们只检查属性
                    has_api = prim.HasAPI("AnimationGraphAPI")
                    
                    scripts_attr = prim.GetAttribute("omni:scripting:scripts")
                    scripts = scripts_attr.Get() if scripts_attr and scripts_attr.IsValid() else None
                    
                    info_dict = {
                        "path": str(prim.GetPath()),
                        "has_api": str(has_api),
                        "scripts": str(scripts) if scripts else "",
                    }
                    
                    skel_infos.append(info_dict)
                    if len(skel_infos) >= 5:
                        break
        print(f"  Detected SkelRoot count: {len(skel_infos)}")
        if skel_infos:
            for info in skel_infos:
                print(
                    "    SkelRoot: {path}, has_AnimGraphAPI={has_api}, scripts={scripts}".format(
                        path=info["path"],
                        has_api=info["has_api"],
                        scripts=info["scripts"] or "None",
                    )
                )

        print("AgentManager instance exists:", AgentManager.has_instance())
        mgr = AgentManager.get_instance()
        print("AgentManager instance:", mgr)
        agents = list(mgr.get_all_agent_names())
        print("Agents:", agents)
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
        config_model = self._build_config_model()

        print("[EnvsetStandalone] Importing extensions...")
        import_extensions()

        print("[EnvsetStandalone] Initializing SimulationApp (pre-run bootstrap)...")
        self._simulation_app = self._initialize_simulation_app(config_model)

        print("[EnvsetStandalone] Preparing runtime settings...")
        self._prepare_runtime_settings()

        print("[EnvsetStandalone] Creating runner (reusing SimulationApp)...")
        self._runner = self._create_runner_with_app(config_model)

        print("[EnvsetStandalone] Post-runner initialization...")
        self._post_runner_initialize()
        print("[EnvsetStandalone] Checking articulation paths and status...")
        self._debug_articulation_paths()

        print("[EnvsetStandalone] Resetting environment (this will load scene and start timeline)...")
        # Reset and start - 这会加载场景并启动 timeline
        self._runner.reset()

        # ⚠️ 关键步骤 1：在 runner.reset() 之后烘焙 NavMesh
        # runner.reset() 已经加载了场景到 /World/env_0/scene
        # 虽然 timeline 已经启动，但虚拟人脚本还需要几帧才会真正初始化
        # 我们必须在脚本初始化之前完成 NavMesh 烘焙
        print("[EnvsetStandalone] Baking NavMesh (scene is now loaded)...")
        navmesh_success = self._bake_navmesh_sync()
        if navmesh_success:
            print("[EnvsetStandalone] NavMesh baking completed successfully")
        else:
            print("[EnvsetStandalone] NavMesh baking failed or skipped")

        # ⚠️ 关键步骤 2：在 NavMesh 烘焙完成后，初始化虚拟人物的行为脚本和动画图
        # 这个时机确保了：
        # 1. NavMesh 已经准备好
        # 2. 虚拟人物的 USD prim 已经 spawn
        # 3. Timeline 已经启动，但脚本还没有开始运行
        if navmesh_success:
            from internutopia_extension.envset.runtime_hooks import EnvsetTaskRuntime
            print("[EnvsetStandalone] Initializing virtual humans (attaching behaviors)...")
            EnvsetTaskRuntime.initialize_virtual_humans(self._bundle.scenario)
            print("[EnvsetStandalone] Virtual humans initialization completed")
        else:
            print("[EnvsetStandalone] Skipping virtual human initialization due to NavMesh failure")

        # 等待场景和对象完全初始化
        print("[EnvsetStandalone] Waiting for scene and objects to initialize...")
        self._wait_for_initialization()

        if self._args.run_data:
            self._init_data_generation()
            self._run_data_generation()
        else:
            # 不再自动启动timeline，等待用户手动启动
            print("[EnvsetStandalone] Entering main loop...")
            self._main_loop()

        print("[EnvsetStandalone] Run completed.")

    def shutdown(self):
        app = None
        if self._runner and self._runner.simulation_app:
            app = self._runner.simulation_app
        elif self._simulation_app:
            app = self._simulation_app

        if app:
            try:
                app.close()
            except Exception:
                pass

    # ---------- internal helpers ----------

    def _prepare_runtime_settings(self):
        # Import Isaac Sim modules here, after runner initialization
        import carb  # type: ignore
        import carb.settings  # type: ignore

        # Enable required extensions before importing envset modules
        print("[EnvsetStandalone] Enabling required extensions...")
        try:
            from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

            # Core envset dependencies (based on extension.toml dependencies)
            enable_extension("omni.usd")
            enable_extension("omni.anim.retarget.core")
            enable_extension("omni.kit.scripting")
            enable_extension("omni.kit.mesh.raycast")
            enable_extension("omni.services.pip_archive")
            enable_extension("isaacsim.sensors.camera")
            enable_extension("isaacsim.sensors.physics")
            enable_extension("isaacsim.sensors.rtx")
            enable_extension("isaacsim.storage.native")
            enable_extension("isaacsim.core.utils")
            enable_extension("omni.metropolis.utils")

            # ★★ 关键：完整的 Anim Graph 套件 ★★
            enable_extension("omni.anim.graph.core")
            enable_extension("omni.anim.graph.schema")   # 你现在少的就是这个
            # 可选：如果后面要用 UI 编辑图，可以顺便开
            # enable_extension("omni.anim.graph.ui")

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

            # ★★ 关键修复：在启用所有 AnimGraph/People 扩展后，重新创建干净的 stage ★★
            # 避免"先有 stage → 后 enable anim.graph"导致插件错过初始化的已知 bug
            # 参考：https://forums.developer.nvidia.com/t/isaac-sim-people-simulation-broken-in-4-1-0/301378
            try:
                import omni.usd  # type: ignore
                usd_ctx = omni.usd.get_context()
                usd_ctx.new_stage()
                carb.log_info("[EnvsetStandalone] Re-created stage after enabling AnimGraph/People extensions (bug workaround)")
            except Exception as e:
                carb.log_warn(f"[EnvsetStandalone] Failed to recreate stage after enabling AnimGraph: {e}")

            carb.log_info("[EnvsetStandalone] Required extensions enabled")
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

    def _initialize_simulation_app(self, config: Config):
        from isaacsim import SimulationApp  # type: ignore
        import os

        simulator_cfg = config.simulator

        launch_config = {
            "headless": simulator_cfg.headless,
            "anti_aliasing": 0,
            "hide_ui": False,
            "multi_gpu": False,
        }

        extension_folders = getattr(simulator_cfg, "extension_folders", None) or []
        if extension_folders:
            if "ISAAC_EXTRA_EXT_PATH" in os.environ:
                existing = os.environ["ISAAC_EXTRA_EXT_PATH"]
                os.environ["ISAAC_EXTRA_EXT_PATH"] = os.pathsep.join([existing] + extension_folders)
            else:
                os.environ["ISAAC_EXTRA_EXT_PATH"] = os.pathsep.join(extension_folders)
            launch_config["extension_folders"] = extension_folders

        sim_app = SimulationApp(launch_config)
        sim_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)

        self._configure_streaming(sim_app, simulator_cfg)
        return sim_app

    def _configure_streaming(self, sim_app, simulator_cfg):
        native = getattr(simulator_cfg, "native", False)
        webrtc = getattr(simulator_cfg, "webrtc", False)

        try:
            from isaacsim import util  # type: ignore  # noqa: F401
        except ImportError:
            self._configure_streaming_420(sim_app, native, webrtc)
        else:
            if native:
                print("[EnvsetStandalone] native streaming is deprecated, enabling webrtc instead.")
            self._configure_streaming_450(sim_app, native or webrtc)

    @staticmethod
    def _configure_streaming_420(sim_app, native: bool, webrtc: bool):
        if webrtc:
            from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

            sim_app.set_setting("/app/window/drawMouse", True)
            sim_app.set_setting("/app/livestream/proto", "ws")
            sim_app.set_setting("/app/livestream/websocket/framerate_limit", 60)
            sim_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.services.streamclient.webrtc")
        elif native:
            from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

            sim_app.set_setting("/app/window/drawMouse", True)
            sim_app.set_setting("/app/livestream/proto", "ws")
            sim_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
            sim_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.kit.streamsdk.plugins-3.2.1")
            enable_extension("omni.kit.livestream.core-3.2.0")
            enable_extension("omni.kit.livestream.native")

    @staticmethod
    def _configure_streaming_450(sim_app, enable_webrtc: bool):
        if not enable_webrtc:
            return
        from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

        sim_app.set_setting("/app/window/drawMouse", True)
        try:
            enable_extension("omni.kit.livestream.webrtc")
        except Exception:
            enable_extension("omni.services.streamclient.webrtc")

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

    def _create_runner_with_app(self, config: Config) -> SimulatorRunner:
        """创建 SimulatorRunner，复用已初始化的 SimulationApp"""
        task_manager = create_task_config_manager(config)
        if self._simulation_app is None:
            raise RuntimeError("SimulationApp must be initialized before creating SimulatorRunner.")

        # 临时替换 setup_isaacsim，让 SimulatorRunner 复用我们的 SimulationApp
        original_setup = SimulatorRunner.setup_isaacsim

        def _reuse_setup(runner_self):
            # 直接设置私有属性 _simulation_app（不能用 property setter）
            runner_self._simulation_app = self._simulation_app
            runner_self._simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
            self._reuse_streaming_configuration(runner_self)

        SimulatorRunner.setup_isaacsim = _reuse_setup
        try:
            runner = SimulatorRunner(config=config, task_config_manager=task_manager)
        finally:
            SimulatorRunner.setup_isaacsim = original_setup

        return runner

    def _reuse_streaming_configuration(self, runner: SimulatorRunner):
        native = getattr(runner.config.simulator, "native", False)
        webrtc = getattr(runner.config.simulator, "webrtc", False)

        try:
            from isaacsim import util  # type: ignore  # noqa: F401
        except ImportError:
            runner.setup_streaming_420(native, webrtc)
        else:
            if native:
                from internutopia.core.util import log

                log.warning("native streaming is deprecated, enabling webrtc instead")
            runner.setup_streaming_450(native or webrtc)

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

    def _bake_navmesh_sync(self):
        """同步烘焙 NavMesh（使用阻塞方式）"""
        import carb  # type: ignore
        from internutopia_extension.envset.navmesh_utils import ensure_navmesh_volume
        from internutopia_extension.envset.runtime_hooks import EnvsetTaskRuntime
        import omni.anim.navigation.core as nav  # type: ignore
        
        carb.log_info("[EnvsetStandalone] Starting NavMesh baking (blocking)...")
        
        try:
            scenario = self._bundle.scenario
            navmesh_cfg = scenario.get("navmesh") or {}
            scene_cfg = scenario.get("scene") or {}
            
            if not navmesh_cfg:
                carb.log_error("[EnvsetStandalone] No navmesh config, skipping bake")
                return False
            
            # 解析 root_path（与 runtime_hooks.py 中的逻辑相同）
            stage = self._runner._stage
            scenario_id = scenario.get("id")
            actual_scene_root = None
            
            if stage and scenario_id:
                candidate_path = f"/World/{scenario_id}/scene"
                prim = stage.GetPrimAtPath(candidate_path)
                if prim and prim.IsValid():
                    actual_scene_root = candidate_path
            
            if not actual_scene_root and stage:
                for env_id in range(10):
                    candidate_path = f"/World/env_{env_id}/scene"
                    prim = stage.GetPrimAtPath(candidate_path)
                    if prim and prim.IsValid():
                        actual_scene_root = candidate_path
                        break
            
            if not actual_scene_root:
                carb.log_error("[EnvsetStandalone] Cannot find scene root for NavMesh baking")
                return False
            
            root_path = actual_scene_root
            include_parent = navmesh_cfg.get("include_volume_parent") or "/World/NavMesh"
            z_padding = navmesh_cfg.get("z_padding") or 2.0
            min_size = navmesh_cfg.get("min_include_volume_size") or {}
            min_xy = navmesh_cfg.get("min_include_xy") or min_size.get("xy") or None
            min_z = navmesh_cfg.get("min_include_z") or min_size.get("z") or None
            agent_radius = navmesh_cfg.get("agent_radius") or 10.0
            
            carb.log_info(f"[EnvsetStandalone] NavMesh bake root: {root_path}")
            
            # 创建 NavMesh volume
            volumes = ensure_navmesh_volume(
                root_prim_path=root_path,
                z_padding=z_padding,
                include_volume_parent=include_parent,
                min_xy=min_xy,
                min_z=min_z,
            )
            
            if not volumes:
                carb.log_error("[EnvsetStandalone] No NavMeshVolume created")
                return False
            
            # 等待几帧让体素注册
            sim_app = self._runner.simulation_app
            for _ in range(3):
                sim_app.update()
            
            # 设置 agent radius
            try:
                import omni.kit.commands  # type: ignore
                omni.kit.commands.execute(
                    "ChangeSetting",
                    path="/exts/omni.anim.navigation.core/navMesh/config/agentRadius",
                    value=float(agent_radius),
                )
            except Exception:
                pass
            
            # 同步烘焙（阻塞调用）
            interface = nav.acquire_interface()
            carb.log_info("[EnvsetStandalone] Starting NavMesh baking (blocking)...")
            interface.start_navmesh_baking_and_wait()
            navmesh = interface.get_navmesh()
            
            if navmesh is None:
                carb.log_error("[EnvsetStandalone] NavMesh baking failed")
                return False
            else:
                carb.log_info("[EnvsetStandalone] NavMesh baking completed successfully")
                # 设置全局标志，标记 NavMesh 已就绪
                EnvsetTaskRuntime._navmesh_ready = True
                return True
                
        except Exception as exc:
            carb.log_error(f"[EnvsetStandalone] NavMesh baking exception: {exc}")
            import traceback
            carb.log_error(traceback.format_exc())
            return False

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
        # print(f"[DEBUG] Keyboard command: {command}")
        x_speed = float(command[0] - command[1])  # I/K keys
        y_speed = float(command[2] - command[3])  # J/L keys
        z_speed = float(command[4] - command[5])  # U/O keys
        #print(f"[DEBUG] Computed speeds: x={x_speed}, y={y_speed}, z={z_speed}")

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
                # print(f"[DEBUG] Built action for robot '{robot_name}': {{'{controller_name}': ({x_speed}, {y_speed}, {z_speed})}}")
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
            print("[EnvsetStandalone] Timeline is already playing, waiting for articulations to initialize...")
            self._wait_for_articulations_initialized()
            self._print_runtime_snapshot("After timeline auto-started")
            # 等待几帧让脚本有时间初始化
            for _ in range(5):
                sim_app.update()
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
