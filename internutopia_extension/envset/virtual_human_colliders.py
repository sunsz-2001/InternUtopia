from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import carb
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Sdf, UsdPhysics


@dataclass
class ColliderConfig:
    """碰撞体配置

    Attributes:
        approximation_shape: 碰撞几何体近似方法
            - "convexHull": 凸包（快速，但对复杂形状不精确）
            - "convexDecomposition": 凸分解（更精确，推荐用于复杂形状）
            - "meshSimplification": 网格简化
        kinematic: 是否为运动学刚体（不受外力影响）
        enable_ccd: 启用连续碰撞检测（防止快速移动物体穿透）
        contact_offset: 接触偏移量（米），对象在此距离内开始生成接触
        rest_offset: 休息偏移量（米），对象休息时的碰撞边界调整
    """

    approximation_shape: str = "convexDecomposition"  # 更精确的碰撞检测
    kinematic: bool = True
    enable_ccd: bool = True  # 启用 CCD 防止穿透
    contact_offset: float = 0.02  # 2cm 接触偏移
    rest_offset: float = 0.0  # 无额外偏移


class VirtualHumanColliderApplier:
    """
    Attach PhysX rigid bodies and colliders to a given list of prim paths once the timeline starts.
    """

    DEFAULT_PREFIX = "[virtual_human_colliders] "

    def __init__(
        self,
        *,
        character_paths: Sequence[str],
        collider_config: Optional[ColliderConfig] = None,
        prefix: str = DEFAULT_PREFIX,
    ):
        self._prefix = prefix
        self._character_paths: List[str] = self._normalize_paths(character_paths)
        self._collider_cfg = collider_config or ColliderConfig()
        self._timeline = omni.timeline.get_timeline_interface()
        self._subscription = None
        self._applied_paths: Set[str] = set()

    @staticmethod
    def _normalize_paths(paths: Sequence[str]) -> List[str]:
        norm: List[str] = []
        seen: Set[str] = set()
        for item in paths or []:
            if not item:
                continue
            path = str(item).strip()
            if not path or path in seen:
                continue
            norm.append(path)
            seen.add(path)
        return norm

    def update_character_paths(self, paths: Sequence[str]):
        self._character_paths = self._normalize_paths(paths)
        self._applied_paths.clear()

    def _iter_existing_character_roots(self) -> Iterable[str]:
        """
        Iterate over existing character root prims.
        If the specified path exists, use it. Otherwise, search for character prims
        under the characters parent path to handle cases where the actual structure
        differs from the expected path.
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        
        found_paths: Set[str] = set()
        
        # First, try the exact paths specified
        for raw_path in self._character_paths:
            sdf_path = Sdf.Path(raw_path)
            prim = stage.GetPrimAtPath(sdf_path)
            if prim and prim.IsValid():
                found_paths.add(str(prim.GetPath()))
                yield str(prim.GetPath())
        
        # If no exact matches found, search for character prims under the characters parent path
        if not found_paths:
            from .stage_util import CharacterUtil, PrimPaths
            character_parent = PrimPaths.characters_parent_path()
            character_roots = CharacterUtil.get_characters_root_in_stage()
            for root_prim in character_roots:
                root_path = str(root_prim.GetPath())
                # Check if this root matches any of our expected names
                root_name = root_path.split("/")[-1]
                for expected_path in self._character_paths:
                    expected_name = expected_path.split("/")[-1]
                    if root_name == expected_name or root_path == expected_path:
                        if root_path not in found_paths:
                            found_paths.add(root_path)
                            yield root_path
                            break

    def _ensure_physics_scene(self):
        """确保物理场景存在并配置 CCD（连续碰撞检测）"""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        physics_scene_path = "/World/physicsScene"
        physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)

        # 创建物理场景（如果不存在）
        if not physics_scene_prim or not physics_scene_prim.IsValid():
            try:
                UsdPhysics.Scene.Define(stage, physics_scene_path)
                carb.log_info(f"{self._prefix}Created physics scene at {physics_scene_path}")
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(f"{self._prefix}Failed to define /World/physicsScene: {exc}")
                return

        # 配置 PhysX 场景参数（启用 CCD 和稳定化）
        try:
            from pxr import PhysxSchema

            physics_scene = UsdPhysics.Scene.Get(stage, physics_scene_path)
            if physics_scene:
                physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())

                # 启用连续碰撞检测（防止快速移动物体穿透）
                if not physx_scene_api.GetEnableCCDAttr():
                    physx_scene_api.CreateEnableCCDAttr().Set(True)
                    carb.log_info(f"{self._prefix}Enabled CCD (Continuous Collision Detection)")

                # 启用稳定化（对大时间步有帮助）
                if not physx_scene_api.GetEnableStabilizationAttr():
                    physx_scene_api.CreateEnableStabilizationAttr().Set(True)
                    carb.log_info(f"{self._prefix}Enabled stabilization pass")

        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"{self._prefix}Failed to configure PhysX scene: {exc}")

    def _apply_rigid_body_with_colliders(self, path: str):
        """应用刚体和碰撞体到指定路径，并配置高级物理参数"""
        # Step 1: 使用命令创建基本的刚体和碰撞体
        try:
            omni.kit.commands.execute(
                "SetRigidBodyCommand",
                path=path,
                approximationShape=self._collider_cfg.approximation_shape,
                kinematic=self._collider_cfg.kinematic,
            )
        except Exception:
            omni.kit.commands.execute(
                "omni.physxcommands.SetRigidBodyCommand",
                path=path,
                approximationShape=self._collider_cfg.approximation_shape,
                kinematic=self._collider_cfg.kinematic,
            )

        # Step 2: 应用高级物理参数（CCD、contact_offset、rest_offset）
        try:
            from pxr import PhysxSchema

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return

            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return

            # 启用 CCD（如果配置要求）
            if self._collider_cfg.enable_ccd:
                rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                if rigid_body_api:
                    rigid_body_api.CreateEnableCCDAttr().Set(True)

            # 设置碰撞偏移参数
            # 遍历所有子 prim 寻找碰撞体
            for child_prim in prim.GetAllChildren():
                collision_api = UsdPhysics.CollisionAPI(child_prim)
                if collision_api:
                    # 设置接触偏移
                    if self._collider_cfg.contact_offset is not None:
                        if not collision_api.GetContactOffsetAttr():
                            collision_api.CreateContactOffsetAttr().Set(self._collider_cfg.contact_offset)
                    # 设置休息偏移
                    if self._collider_cfg.rest_offset is not None:
                        if not collision_api.GetRestOffsetAttr():
                            collision_api.CreateRestOffsetAttr().Set(self._collider_cfg.rest_offset)

        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"{self._prefix}Failed to apply advanced physics params to {path}: {exc}")

    def _apply_once(self):
        self._ensure_physics_scene()
        applied: List[str] = []
        for root in self._iter_existing_character_roots():
            if root in self._applied_paths:
                continue
            try:
                self._apply_rigid_body_with_colliders(root)
                applied.append(root)
                self._applied_paths.add(root)
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(f"{self._prefix}Failed to apply collider to {root}: {exc}")
        if applied:
            carb.log_info(f"{self._prefix}Applied rigid bodies to: {applied}")
        else:
            carb.log_debug(f"{self._prefix}No character roots found to apply colliders.")

    def _on_timeline_event(self, evt):
        import omni.timeline

        if evt.type == omni.timeline.TimelineEventType.PLAY.value:
            self._apply_once()
            self.deactivate()

    def activate(self, apply_immediately: bool = False):
        """
        Activate the collider applier.
        
        Args:
            apply_immediately: If True, apply colliders immediately instead of waiting for timeline play event.
                              This is useful when you want to set up physics before timeline starts.
        """
        if self._subscription is not None:
            return
        
        # 如果要求立即应用，或者timeline已经在播放，立即应用
        if apply_immediately or self._timeline.is_playing():
            self._apply_once()
            # 如果timeline已经在播放，不需要订阅事件
            if self._timeline.is_playing():
                return
        
        # 否则订阅timeline事件，在timeline启动时应用
        stream = self._timeline.get_timeline_event_stream()
        self._subscription = stream.create_subscription_to_pop(self._on_timeline_event)

    def deactivate(self):
        if self._subscription is not None:
            try:
                self._subscription.unsubscribe()
            except Exception:
                pass
            self._subscription = None
