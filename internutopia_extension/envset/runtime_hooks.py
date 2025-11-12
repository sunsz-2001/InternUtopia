from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import carb
import carb.eventdispatcher
import carb.settings
import omni.kit.commands
import omni.usd
from omni.anim.people.settings import AgentEvent
from omni.metropolis.utils.semantics_util import SemanticsUtils
from pxr import NavSchema

from internutopia_extension.envset.agent_manager import AgentManager
from internutopia_extension.guards.arrival_guard import ArrivalGuard

from .navmesh_utils import ensure_navmesh_volume
from .settings import BehaviorScriptPaths, PrimPaths
from .stage_util import CharacterUtil, populate_anim_graph


class EnvsetTaskRuntime:
    _navmesh_ready = False
    _pending_routes: Dict[str, List[str]] = {}
    _route_subscription = None
    _vh_spawned = False
    _arrival_guard = ArrivalGuard()

    @classmethod
    def configure_task(cls, task):
        envset_cfg = getattr(task.config, "envset", None)
        if not envset_cfg:
            envset_cfg = getattr(task.config, "__dict__", {}).get("envset")
        if not envset_cfg:
            return
        try:
            cls._setup_navmesh(envset_cfg)
        except Exception as exc:
            carb.log_warn(f"[EnvsetRuntime] NavMesh setup skipped: {exc}")
        try:
            cls._setup_virtual_routes(envset_cfg)
        except Exception as exc:
            carb.log_warn(f"[EnvsetRuntime] Routes setup skipped: {exc}")
        try:
            cls._setup_virtual_characters(envset_cfg)
        except Exception as exc:
            carb.log_warn(f"[EnvsetRuntime] Character setup skipped: {exc}")

    @classmethod
    def _setup_navmesh(cls, envset_cfg):
        if cls._navmesh_ready:
            return
        navmesh_cfg = envset_cfg.get("navmesh") or {}
        if not navmesh_cfg:
            return
        scene_cfg = envset_cfg.get("scene") or {}
        
        # Get the configured bake root path (e.g., "/Root/Meshes/Base/ground")
        configured_bake_root = navmesh_cfg.get("bake_root_prim_path")
        configured_scene_root = scene_cfg.get("root_prim_path")  # e.g., "/Root"
        
        # Resolve the actual scene root path in the loaded stage
        # InternUtopia loads scenes to /World/{scenario_id}/scene, using scenario id from envset.json
        stage = omni.usd.get_context().get_stage()
        actual_scene_root = None
        
        # Get scenario_id from envset_cfg
        scenario_id = envset_cfg.get("scenario_id")
        
        if stage and scenario_id:
            # Use scenario_id to build the path: /World/{scenario_id}/scene
            candidate_path = f"/World/{scenario_id}/scene"
            prim = stage.GetPrimAtPath(candidate_path)
            if prim and prim.IsValid():
                actual_scene_root = candidate_path
            else:
                # Fallback: try to find any scene prim under /World/{scenario_id}/
                scenario_prim = stage.GetPrimAtPath(f"/World/{scenario_id}")
                if scenario_prim and scenario_prim.IsValid():
                    # Look for "scene" child
                    for child in scenario_prim.GetChildren():
                        if child.GetName() == "scene" and child.IsValid():
                            actual_scene_root = str(child.GetPath())
                            break
        
        # If scenario_id not available, fallback to old env_id-based search
        if not actual_scene_root and stage:
            # Try to find the scene prim loaded by InternUtopia (typically /World/env_0/scene)
            for env_id in range(10):  # Check up to 10 environments
                candidate_path = f"/World/env_{env_id}/scene"
                prim = stage.GetPrimAtPath(candidate_path)
                if prim and prim.IsValid():
                    actual_scene_root = candidate_path
                    break
        
        # If we found the InternUtopia scene path, map the bake root relative to it
        if actual_scene_root and configured_bake_root:
            # Check if bake_root is already a relative path (doesn't start with "/")
            if not configured_bake_root.startswith("/"):
                # It's already a relative path, use it directly
                root_path = f"{actual_scene_root}/{configured_bake_root}"
            elif configured_scene_root and configured_bake_root.startswith(configured_scene_root):
                # Extract the relative path from the configured scene root
                # e.g., "/Root/Meshes/Base/ground" -> "Meshes/Base/ground" (if scene_root="/Root")
                relative_path = configured_bake_root[len(configured_scene_root):].lstrip("/")
                root_path = f"{actual_scene_root}/{relative_path}" if relative_path else actual_scene_root
            else:
                # If bake_root doesn't start with scene_root, try to extract relative part
                # Extract just the last part (e.g., "Meshes/Base/ground" from "/Root/Meshes/Base/ground")
                parts = configured_bake_root.strip("/").split("/")
                if len(parts) > 1:
                    # Try to find the matching subpath under actual_scene_root
                    relative_path = "/".join(parts[1:])  # Skip the root part
                    root_path = f"{actual_scene_root}/{relative_path}"
                else:
                    root_path = configured_bake_root
        elif actual_scene_root:
            # No bake_root configured, use the actual scene root
            root_path = actual_scene_root
        else:
            # Fallback to configured paths or defaults
            root_path = configured_bake_root or configured_scene_root or "/World"
        
        include_parent = navmesh_cfg.get("include_volume_parent") or "/World/NavMesh"
        z_padding = navmesh_cfg.get("z_padding") or 2.0
        # Support both formats: direct fields or nested in min_include_volume_size
        min_size = navmesh_cfg.get("min_include_volume_size") or {}
        min_xy = navmesh_cfg.get("min_include_xy") or min_size.get("xy") or None
        min_z = navmesh_cfg.get("min_include_z") or min_size.get("z") or None
        
        carb.log_info(f"[EnvsetRuntime] NavMesh bake root resolved: {configured_bake_root} -> {root_path}")
        ensure_navmesh_volume(
            root_prim_path=root_path,
            z_padding=z_padding,
            include_volume_parent=include_parent,
            min_xy=min_xy,
            min_z=min_z,
        )
        cls._navmesh_ready = True

    @classmethod
    def _setup_virtual_characters(cls, envset_cfg):
        if cls._vh_spawned:
            return
        vh_cfg = (envset_cfg.get("virtual_humans") or {}) if envset_cfg else {}
        if not vh_cfg:
            return
        spawn_points = vh_cfg.get("spawn_points") or []
        name_sequence = vh_cfg.get("name_sequence") or []
        assets = vh_cfg.get("assets") or {}
        count = vh_cfg.get("count")
        if count is None:
            count = max(len(spawn_points), len(name_sequence), len(assets))
        try:
            count = int(count)
        except Exception:
            return
        if count <= 0:
            return

        asset_root = cls._resolve_asset_root(vh_cfg.get("asset_root"))
        fallback_asset = next(iter(assets.values()), None)

        spawned_any = False
        spawned_prims = []
        for idx in range(count):
            name = cls._resolve_character_name(name_sequence, idx)
            asset = assets.get(name) or fallback_asset
            usd_path = cls._resolve_asset_path(asset, asset_root)
            if not usd_path:
                continue
            spawn = spawn_points[idx] if idx < len(spawn_points) else {}
            pos = cls._to_float3(spawn.get("position"))
            rot = cls._safe_float(spawn.get("orientation_deg"), 0.0)
            prim = CharacterUtil.load_character_usd_to_stage(usd_path, pos, rot, name)
            if prim and prim.IsValid():
                cls._exclude_from_navmesh(prim.GetPrimPath())
                spawned_prims.append(prim)
                spawned_any = True

        if spawned_any:
            # 立即应用碰撞体到spawned的虚拟人物
            cls._apply_colliders_to_spawned_characters(spawned_prims, envset_cfg)
            cls._setup_character_behaviors()
            cls._configure_arrival_guard(envset_cfg, vh_cfg)
            cls._vh_spawned = True

    @classmethod
    def _setup_virtual_routes(cls, envset_cfg):
        vh = (envset_cfg.get("virtual_humans") or {}) if envset_cfg else {}
        routes = vh.get("routes") or []
        pending = {}
        for entry in routes:
            name = entry.get("name")
            commands = entry.get("commands") or []
            if not name or not commands:
                continue
            pending[name] = list(commands)
        if not pending:
            return
        cls._pending_routes.update(pending)
        cls._subscribe_route_events()
        cls._flush_routes()

    @classmethod
    def _subscribe_route_events(cls):
        if cls._route_subscription is not None:
            return
        dispatcher = carb.eventdispatcher.get_eventdispatcher()
        cls._route_subscription = dispatcher.observe_event(
            event_name=AgentEvent.AgentRegistered,
            on_event=cls._on_agent_registered,
            observer_name="internutopia/envset/runtime/routes",
        )

    @classmethod
    def _on_agent_registered(cls, event):
        payload = getattr(event, "payload", None) or {}
        agent_name = payload.get("agent_name")
        if not agent_name:
            return
        cls._inject_route(agent_name)

    @classmethod
    def _flush_routes(cls):
        for agent_name in list(cls._pending_routes.keys()):
            cls._inject_route(agent_name)

    @classmethod
    def _inject_route(cls, agent_name: str):
        commands = cls._pending_routes.get(agent_name)
        if not commands:
            return
        mgr = AgentManager.get_instance()
        if not mgr.agent_registered(agent_name):
            return
        mgr.inject_command(agent_name, commands, force_inject=True, instant=True)
        cls._pending_routes.pop(agent_name, None)

    # --------------------- helpers ---------------------

    @staticmethod
    def _resolve_character_name(sequence, idx):
        if sequence and idx < len(sequence) and sequence[idx]:
            return str(sequence[idx])
        return CharacterUtil.get_character_name_by_index(idx)

    @staticmethod
    def _resolve_asset_root(asset_root_cfg) -> Optional[str]:
        if not asset_root_cfg:
            return None
        settings_key = asset_root_cfg.get("settings_key")
        if settings_key:
            try:
                value = carb.settings.get_settings().get(settings_key)
                if value:
                    return str(value)
            except Exception:
                pass
        fallback = asset_root_cfg.get("fallback")
        if fallback:
            return str(fallback)
        path_val = asset_root_cfg.get("path")
        if path_val:
            return str(path_val)
        return None

    @staticmethod
    def _resolve_asset_path(asset: Optional[str], root: Optional[str]) -> Optional[str]:
        if not asset:
            return None
        asset = str(asset)
        if asset.startswith(("omniverse://", "http://", "https://", "/")):
            return asset
        if root:
            return str(Path(root).joinpath(asset.lstrip("/")))
        return asset

    @staticmethod
    def _to_float3(value) -> carb.Float3:
        if not value:
            return carb.Float3(0.0, 0.0, 0.0)
        try:
            x, y, z = float(value[0]), float(value[1]), float(value[2])
        except Exception:
            x = y = z = 0.0
        return carb.Float3(x, y, z)

    @staticmethod
    def _safe_float(value, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @classmethod
    def _apply_colliders_to_spawned_characters(cls, spawned_prims, envset_cfg):
        """立即应用碰撞体到spawned的虚拟人物"""
        if not spawned_prims:
            return
        
        try:
            from .virtual_human_colliders import VirtualHumanColliderApplier, ColliderConfig
            
            vh_cfg = envset_cfg.get("virtual_humans") or {}
            approx_shape = vh_cfg.get("collider_shape") or vh_cfg.get("approximation_shape") or "convexHull"
            kinematic_flag = vh_cfg.get("collider_kinematic")
            if kinematic_flag is None:
                kinematic_flag = True
            
            collider_cfg = ColliderConfig(
                approximation_shape=str(approx_shape),
                kinematic=bool(kinematic_flag)
            )
            
            # 构建实际spawned的prim路径列表
            character_paths = [str(prim.GetPrimPath()) for prim in spawned_prims if prim and prim.IsValid()]
            
            if character_paths:
                applier = VirtualHumanColliderApplier(
                    character_paths=character_paths,
                    collider_config=collider_cfg,
                )
                # 立即应用，不等待timeline
                applier.activate(apply_immediately=True)
                carb.log_info(f"[EnvsetRuntime] Applied colliders to {len(character_paths)} spawned characters")
        except Exception as exc:
            carb.log_warn(f"[EnvsetRuntime] Failed to apply colliders to spawned characters: {exc}")

    @staticmethod
    def _exclude_from_navmesh(prim_path):
        try:
            omni.kit.commands.execute(
                "ApplyNavMeshAPICommand", prim_path=str(prim_path), api=NavSchema.NavMeshExcludeAPI
            )
        except Exception:
            pass

    @classmethod
    def _setup_character_behaviors(cls):
        carb.log_info("[EnvsetRuntime] Setting up character behaviors...")
        biped = CharacterUtil.load_default_biped_to_stage()
        if biped is None or not biped.IsValid():
            carb.log_warn("[EnvsetRuntime] Default biped failed to load; animation graph may be unavailable.")
        else:
            carb.log_info(f"[EnvsetRuntime] Default biped prim: {biped.GetPath()}")

        character_list = CharacterUtil.get_characters_in_stage()
        carb.log_info(f"[EnvsetRuntime] Characters detected in stage: {len(character_list)}")
        if not character_list:
            return

        anim_graph = CharacterUtil.get_anim_graph_from_character(biped)
        if not anim_graph:
            carb.log_warn("[EnvsetRuntime] No animation graph found on default biped; retrying populate_anim_graph() after importing omni.anim.graph.core")
            try:
                import omni.anim.graph.core  # type: ignore  # noqa: F401
            except Exception as exc:
                carb.log_error(f"[EnvsetRuntime] Failed to import omni.anim.graph.core: {exc}")
            else:
                try:
                    populate_anim_graph()
                except Exception as retry_exc:
                    carb.log_error(f"[EnvsetRuntime] populate_anim_graph retry failed: {retry_exc}")
                finally:
                    anim_graph = CharacterUtil.get_anim_graph_from_character(biped)

        if anim_graph:
            carb.log_info(f"[EnvsetRuntime] Applying anim graph from {anim_graph.GetPath()}")
            CharacterUtil.setup_animation_graph_to_character(character_list, anim_graph)
        else:
            carb.log_error("[EnvsetRuntime] Animation graph still missing after retry; characters may not register with AgentManager.")

        script_path = BehaviorScriptPaths.behavior_script_path()
        carb.log_info(f"[EnvsetRuntime] Attaching behavior script: {script_path}")
        CharacterUtil.setup_python_scripts_to_character(character_list, script_path)
        
        # 验证脚本是否成功挂载
        for char_prim in character_list:
            scripts_attr = char_prim.GetAttribute("omni:scripting:scripts")
            if scripts_attr and scripts_attr.IsValid():
                scripts = scripts_attr.Get()
                carb.log_info(f"[EnvsetRuntime] Character {char_prim.GetPath()} scripts: {scripts}")
            else:
                carb.log_warn(f"[EnvsetRuntime] Character {char_prim.GetPath()} has no scripts attribute!")
        
        SemanticsUtils.add_update_prim_metrosim_semantics(character_list, type_value="class", name="character")

    @classmethod
    def _configure_arrival_guard(cls, envset_cfg, vh_cfg):
        tol = cls._safe_float(vh_cfg.get("arrival_tolerance_m"), 0.5)
        scene_cfg = envset_cfg.get("scene") or {}
        scene_category = scene_cfg.get("category")
        cls._arrival_guard.enable_if_grscenes(scene_category, tolerance_m=tol)
