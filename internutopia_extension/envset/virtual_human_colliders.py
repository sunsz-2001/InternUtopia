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
    approximation_shape: str = "convexHull"
    kinematic: bool = True


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
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        if stage.GetPrimAtPath("/World/physicsScene"):
            return
        try:
            UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"{self._prefix}Failed to define /World/physicsScene: {exc}")

    def _apply_rigid_body_with_colliders(self, path: str):
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
