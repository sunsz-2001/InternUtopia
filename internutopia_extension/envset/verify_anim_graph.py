#!/usr/bin/env python3
"""
简化的验证脚本：检查动画图是否正确应用到角色上
在 Isaac Sim Python 环境中运行
"""

import sys
import time


def _ensure_simulation_app():
    """确保 SimulationApp 已初始化"""
    try:
        import carb  # type: ignore  # noqa: F401
        return  # 如果能导入 carb，说明 SimulationApp 已初始化
    except ImportError:
        pass

    # 需要初始化 SimulationApp
    from isaacsim import SimulationApp  # type: ignore

    launch_config = {
        "headless": False,
        "hide_ui": False,
    }
    app = SimulationApp(launch_config)
    return app


def main():
    # 初始化 SimulationApp
    sim_app = _ensure_simulation_app()

    # 导入 Isaac Sim 模块
    import carb  # type: ignore
    import omni.usd  # type: ignore
    from pxr import Usd  # type: ignore
    from omni.isaac.core.utils.extensions import enable_extension  # type: ignore
    from omni.anim.graph.schema import AnimGraphSchema  # type: ignore

    print("\n" + "=" * 80)
    print("验证动画图应用状态")
    print("=" * 80)

    # 启用必要的扩展
    print("\n[1] 启用扩展...")
    extensions = [
        "omni.anim.graph.core",
        "omni.anim.people",
        "omni.kit.mesh.raycast",
        "omni.metropolis.utils",
    ]
    for ext in extensions:
        try:
            enable_extension(ext)
            print(f"  ✓ {ext}")
        except Exception as exc:
            print(f"  ✗ {ext}: {exc}")

    # 打开 USD 场景
    print("\n[2] 打开 USD 场景...")
    stage_path = "/home/ubuntu/sunsz/IsaacAssets/Environments/Simple_Warehouse/full_warehouse.usd"
    success = omni.usd.get_context().open_stage(stage_path)
    if not success:
        print(f"  ✗ 无法打开场景: {stage_path}")
        return

    print(f"  ✓ 场景已打开: {stage_path}")

    # 等待场景加载
    for _ in range(10):
        sim_app.update()

    stage = omni.usd.get_context().get_stage()

    # 查找所有 SkelRoot
    print("\n[3] 查找 SkelRoot...")
    characters_root = stage.GetPrimAtPath("/World/Characters")
    if not characters_root or not characters_root.IsValid():
        print("  ✗ /World/Characters 不存在")
        return

    skel_roots = []
    for prim in Usd.PrimRange(characters_root):
        if prim.GetTypeName() == "SkelRoot":
            skel_roots.append(prim)

    print(f"  ✓ 找到 {len(skel_roots)} 个 SkelRoot")

    if not skel_roots:
        print("  ⚠ 没有找到任何角色，无法验证")
        return

    # 检查每个 SkelRoot 的动画图状态
    print("\n[4] 检查动画图状态...")
    for skel_root in skel_roots:
        path = skel_root.GetPath()
        print(f"\n  角色: {path}")

        # 检查是否有 AnimationGraphAPI
        has_api = skel_root.HasAPI("AnimationGraphAPI")
        print(f"    - HasAPI('AnimationGraphAPI'): {has_api}")

        if has_api:
            try:
                # 使用正确的方法获取动画图
                anim_graph_api = AnimGraphSchema.AnimationGraphAPI(skel_root)
                anim_graph_rel = anim_graph_api.GetAnimationGraphRel()

                if anim_graph_rel:
                    targets = anim_graph_rel.GetTargets()
                    if targets:
                        print(f"    - AnimationGraph 目标: {targets[0]}")
                        print(f"    - ✓ 动画图已正确应用")
                    else:
                        print(f"    - ✗ AnimationGraph 关系存在但没有目标")
                else:
                    print(f"    - ✗ AnimationGraph 关系不存在")
            except Exception as exc:
                print(f"    - ✗ 获取动画图失败: {exc}")
        else:
            print(f"    - ✗ 没有 AnimationGraphAPI")

        # 检查行为脚本
        scripts_attr = skel_root.GetAttribute("omni:scripting:scripts")
        if scripts_attr and scripts_attr.IsValid():
            scripts = scripts_attr.Get()
            if scripts:
                print(f"    - 行为脚本: {scripts[0].path if hasattr(scripts[0], 'path') else scripts[0]}")
            else:
                print(f"    - ✗ 没有行为脚本")
        else:
            print(f"    - ✗ 没有 omni:scripting:scripts 属性")

    print("\n" + "=" * 80)
    print("验证完成")
    print("=" * 80 + "\n")

    # 保持窗口打开
    print("按 Ctrl+C 退出...")
    try:
        while sim_app.is_running():
            sim_app.update()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n退出...")

    sim_app.close()


if __name__ == "__main__":
    main()

