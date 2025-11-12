"""
诊断脚本：检查 envset 虚拟人是否正常注册（Linux 命令行运行）

用法：
    /path/to/isaac-sim/python.sh -m omni.isaac.core.scripts.python -- \
        /path/to/diag_agent_status.py
"""

import sys
from pathlib import Path

# 注：omni、carb、pxr 等模块由 Isaac Sim 提供，只有在其 Python 解释器中运行时才存在。
# 这里延后导入，以便在普通 Python 环境中给出清晰提示。
# 实际诊断逻辑会在 main() 中 import 这些模块。

_simulation_app = None


def _ensure_simulation_app():
    """确保 SimulationApp 已创建（导入 carb/omni 前必须调用）。"""
    global _simulation_app
    if _simulation_app is None:
        from isaacsim import SimulationApp  # type: ignore

        print("[Diag] Initializing SimulationApp (headless)")
        _simulation_app = SimulationApp({"headless": True})


def _ensure_path_in_sys(path: Path):
    if path and path.exists():
        sys.path.insert(0, str(path))


def main():
    # 尝试将脚本所在目录加入 sys.path（方便导入项目模块）
    script_dir = Path(__file__).resolve().parent
    _ensure_path_in_sys(script_dir)
    _ensure_path_in_sys(script_dir.parent)  # 项目根目录可按需调整

    print("=== Envset virtual human diagnostic ===")

    # 1. 检查扩展启用状态
    try:
        _ensure_simulation_app()
        import carb  # type: ignore
        import omni  # type: ignore
        import omni.timeline  # type: ignore
        import omni.usd  # type: ignore
        from pxr import Usd  # type: ignore
    except ImportError as exc:
        print(f"导入 Isaac Sim 模块失败：{exc}")
        print("请使用 Isaac Sim 自带的 python/sh 运行此脚本，例如：")
        print("  /path/to/isaac-sim/python.sh -m internutopia_extension.envset.diag_agent_status")
        return

    must_extensions = [
        "omni.anim.people",
        "omni.anim.navigation.core",
        "omni.anim.navigation.schema",
        "omni.anim.navigation.meshtools",
        "omni.physxcommands",
        "isaacsim.anim.robot",
    ]

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    print("\n[Extensions]")
    try:
        from omni.isaac.core.utils.extensions import enable_extension  # type: ignore
    except Exception as exc:
        print(f"Warning: 无法导入 enable_extension：{exc}")
        enable_extension = None  # type: ignore

    for ext in must_extensions:
        enabled = ext_manager.is_extension_enabled(ext)
        if not enabled and enable_extension:
            try:
                enable_extension(ext)
                enabled = ext_manager.is_extension_enabled(ext)
            except Exception as exc:
                print(f"  启用扩展 {ext} 失败：{exc}")
        print(f"{ext:40s} -> {enabled}")

    # 2. 获取 stage 与 World 信息
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    print("\n[Stage]")
    print("Stage valid:", bool(stage))

    if not stage:
        print("Stage is invalid，诊断终止")
        return

    # 3. Characters 结构、SkelRoot/脚本情况
    try:
        from internutopia_extension.envset.stage_util import CharacterUtil, PrimPaths
    except Exception as exc:
        print(f"导入 stage_util 失败：{exc}")
        return

    characters_root_path = PrimPaths.characters_parent_path()
    characters_root = stage.GetPrimAtPath(characters_root_path)
    print("\n[Characters]")
    print(f"Characters root path: {characters_root_path}")
    print("Root prim valid:", bool(characters_root and characters_root.IsValid()))

    skel_roots = []
    if characters_root and characters_root.IsValid():
        for prim in Usd.PrimRange(characters_root):
            if prim.GetTypeName() == "SkelRoot":
                skel_roots.append(prim)

    print(f"Detected SkelRoot count: {len(skel_roots)}")
    for prim in skel_roots:
        print(f"  SkelRoot -> {prim.GetPath()}")
        has_anim_graph_attr = prim.HasAttribute("omni:anim_graph:graph_path")
        anim_graph_path = None
        if has_anim_graph_attr:
            anim_graph_path = prim.GetAttribute("omni:anim_graph:graph_path").Get()
        scripts_attr = prim.GetAttribute("omni:scripting:scripts")
        scripts = scripts_attr.Get() if scripts_attr and scripts_attr.IsValid() else None
        print(f"    Has AnimationGraphAPI attr : {has_anim_graph_attr}, value: {anim_graph_path}")
        print(f"    Scripts attached           : {scripts}")

    # 4. AgentManager 注册情况
    try:
        from internutopia_extension.envset.agent_manager import AgentManager
    except Exception as exc:
        print(f"导入 AgentManager 失败：{exc}")
        return

    mgr = AgentManager.get_instance()
    registered_agents = list(mgr.get_all_agent_names())
    print("\n[AgentManager]")
    print("Registered agents:", registered_agents)
    print("Is 'Character' registered?:", mgr.agent_registered("Character"))
    if registered_agents:
        for name in registered_agents:
            agent_obj = mgr.get_agent_script_instance_by_name(name)
            prim_path = getattr(agent_obj, "prim_path", None)
            print(f"  {name}: prim_path={prim_path}")

    # 5. 时间轴状态
    timeline = omni.timeline.get_timeline_interface()
    print("\n[Timeline]")
    print("Timeline playing?:", timeline.is_playing())

    print("\n诊断完成。")

    # 如果是脚本内部创建的 SimulationApp，则关闭
    global _simulation_app
    if _simulation_app is not None:
        try:
            _simulation_app.close()
        except Exception:
            pass
        _simulation_app = None


if __name__ == "__main__":
    main()

