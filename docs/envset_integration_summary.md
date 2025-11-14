```markdown
# EnvSet × InternUtopia 集成说明（当前进度）

> 目的：使用 InternUtopia 的 `SimulatorRunner` 作为唯一调度器，将原 `envset` 扩展里的场景/虚拟人/机器人/数据生成能力整合到 Runner 生命周期中，支持 headless 批量运行。

## 1. 配置与入口

- CLI：`internutopia_extension/envset/standalone.py`
  - 参数：
    - `--config`（InternUtopia YAML，基础配置文件，可使用 `config_minimal.yaml`）
    - `--envset`（envset JSON，场景配置）
    - `--scenario`（可选，指定scenario id）
    - `--headless`（无头模式）
    - `--extension-path`（可重复，添加自定义扩展搜索路径，如 isaaclab source 目录）
    - `--hold-seconds`（运行时长限制）
    - `--no-play`（不自动播放）
    - `--run-data`（数据生成模式）
    - `--skip-isaac-assets`（跳过Isaac资产查询）
    - `--label`（标签）
  - 流程：
    1. 通过 `EnvsetConfigLoader` 合并 YAML + envset，产生 `EnvsetConfigBundle`（含结构化 `scenario_data`）。
    2. `EnvsetTaskAugmentor` 将 envset 信息注入每个 `task_config`，包括 `scene`, `navmesh`, `virtual_humans`, `robots`。
       - **重要**：`EnvsetTaskAugmentor.apply()` 迭代 `task_configs` 列表并增强每个task，不会创建新task。因此YAML中必须至少有一个task条目。
    3. 启动 InternUtopia `SimulatorRunner`（创建 SimulationApp）。
    4. 启用必需的 Isaac Sim extensions（`omni.metropolis.utils`, `omni.anim.navigation.*`, `omni.anim.people`, `isaacsim.anim.robot`）。
    5. 初始化 envset 运行期组件（World/AgentManager/Patch）。
    6. `runner.reset()` 后根据 CLI 选择是否自动 `timeline.play()`，循环调用 `runner.step(actions)`。

- 引用与扩展：`import_extensions()` 会注册扩展中的 controller/object/robot/task，`bootstrap_world_if_needed()` + `AgentManager.get_instance()` 确保 envset 运行期依赖存在。

## 2. envset → TaskCfg 映射

### 2.1 架构设计（2024-11 重构）

**核心原则**：**统一使用 Pydantic 对象，避免字典⇄对象转换**

在 2024-11 的重构中，我们彻底消除了配置构建过程中的字典⇄对象转换问题，采用纯对象化架构：

**重构前的问题**：
- ❌ 混合模式：差速机器人用字典构建，四足/人形机器人用对象转字典
- ❌ 循环转换：Pydantic对象 → 字典（`model_dump()`）→ 重新构建对象（`_convert_controller_dicts_to_objects`）
- ❌ 字段丢失：`model_dump()` 序列化嵌套的 `sub_controllers` 时会丢失字段（如 `joint_names`, `policy_weights_path`）
- ❌ 类型不安全：字典操作缺少编译时类型检查
- ❌ 代码冗余：需要 87 行的类型映射和转换代码

**重构后的架构**：

```
envset JSON → RobotSpec (dataclass)
           ↓
     _build_robot_controllers() → List[ControllerCfg] (Pydantic对象)
           ↓
     _build_robot_entry() → RobotCfg (Pydantic对象)
           ↓
     _inject_robots() → 直接添加到 task["robots"]
           ↓
     Pydantic 自动验证并构建 Config
```

**关键实现细节**：

1. **`_build_robot_controllers()` 返回 Pydantic 对象**（lines 176-235 in task_adapter.py）：
   ```python
   # 差速机器人：直接构建对象
   drive_cfg = DifferentialDriveControllerCfg(
       name=f"{name}_drive",
       wheel_radius=wheel_radius,
       wheel_base=wheel_base,
   )
   goto_cfg = MoveToPointBySpeedControllerCfg(
       name=f"{name}_move",
       sub_controllers=[drive_cfg],  # 嵌套对象
   )
   return [goto_cfg]

   # 四足/人形机器人：使用 model_copy(deep=True) 克隆预定义配置
   cloned = aliengo_move_to_point_cfg.model_copy(deep=True)
   return [cloned]
   ```

2. **`_build_robot_entry()` 返回 RobotCfg 对象**（lines 115-144）：
   ```python
   robot_cfg = RobotCfg(
       name=name,
       type=robot_type,
       controllers=controllers,  # List[ControllerCfg] 对象
   )
   return robot_cfg
   ```

3. **`_inject_robots()` 直接添加对象**（lines 88-112）：
   ```python
   robot_list.append(robot_cfg)  # RobotCfg 对象，不是字典
   ```

4. **删除了转换代码**（standalone.py 中的 87 行）：
   - 删除了 `_convert_controller_dicts_to_objects()` 方法
   - 删除了 16 个控制器类型的导入和映射字典
   - Pydantic 自动处理嵌套对象的验证

**优势**：
- ✅ **类型安全**：编译时检查，IDE 自动补全
- ✅ **零序列化开销**：不需要 dict ↔ object 转换
- ✅ **字段完整性**：嵌套对象保留所有字段（包括 `joint_names`, `policy_weights_path` 等）
- ✅ **代码简洁**：减少 87 行转换代码，提高可维护性
- ✅ **一致性**：所有机器人类型使用统一的构建方式

**向后兼容**：
- `_inject_robots()` 同时支持字典和 RobotCfg 对象（用于名称冲突检测）
- 现有的 envset JSON 格式无需修改

**配置传递优化**：
- ❌ **移除了临时 YAML 文件**：之前 `_write_temp_yaml()` 会将合并后的配置写入临时文件，但该文件从未被读取
- ✅ **直接使用内存对象**：配置通过 `EnvsetConfigBundle.config` 在内存中传递，保留完整的 Pydantic 对象
- ✅ **避免序列化问题**：`yaml.safe_dump()` 无法正确序列化 Pydantic 对象（会丢失 `type` 等字段），现在完全避免了这个问题
- ✅ **性能提升**：减少了不必要的 I/O 操作

---

- `config_loader.EnvsetConfigLoader`
  - 解析 envset JSON，结构化为 `EnvsetScenarioData`（scene/navmesh/vh/robots/logging）。
  - 把 envset 场景路径、use_matterport 等写入 YAML 的 `scene` 段。
  - 调用 `EnvsetTaskAugmentor.apply()`，把 envset 数据嵌入 Task 配置。

- `task_adapter.EnvsetTaskAugmentor`
  - 把 `scenario_data` 推送到每个 `task` 的 `envset` 字段。
  - 机器人映射（当前）：
    - `type` in `{carter, carter_v1, jetbot, differential_drive}` → InternUtopia `JetbotRobot` + `DifferentialDriveController` + `MoveToPointBySpeedController`（参数来自 envset `control.params`）。
    - 其它类型暂未处理（TODO: aliengo、h1 等）。
  - `virtual_humans.routes/spawn_points/assets` 保存在 `task.config.envset.virtual_humans`，供 runtime hooks 使用。

## 3. 运行期钩子（EnvsetTaskRuntime）

在 `internutopia/core/task/task.py:104` 的 `load()` 末尾调用 `_apply_envset_runtime_hooks()`，由 `EnvsetTaskRuntime` 做以下工作：

1. **NavMesh Volume 创建**：根据 `envset.navmesh` 创建 NavMesh include volume（`ensure_navmesh_volume`），但**不立即烘焙**。
2. **虚拟人 Spawn**（2024-11 重构）：
   - 按 `virtual_humans` 的 spawn_point/name_sequence/assets 加载 USD prims；
   - 应用碰撞体配置；
   - 设置语义信息、NavMesh 排除；
   - **不立即附加行为脚本和动画图**（延后到 NavMesh 烘焙完成后）。
3. **Routes 配置**：监听 `AgentEvent.AgentRegistered`，在角色注册后调用 `AgentManager.inject_command()` 下发 envset routes（`GoTo`/`Idle` 等指令）。

### 3.1 虚拟人初始化时序（2024-11-12 重构）

**问题背景**：虚拟人物需要依赖 NavMesh 才能正确注册到 AgentManager 并执行导航命令。之前的实现存在时序问题：
1. NavMesh 烘焙在场景加载前执行（失败）
2. 行为脚本在 NavMesh 准备好之前就附加（Agent 注册失败）

**新的初始化流程**：

```
1. runner.reset()
   └─ task.load()
       ├─ 加载场景 USD 到 /World/env_0/scene
       └─ EnvsetTaskRuntime.configure_task()
           ├─ _setup_navmesh() → 只创建 volume，不烘焙
           ├─ _setup_virtual_routes() → 配置路由
           └─ _setup_virtual_characters() → 只 spawn USD prims
               ✅ 标记: _vh_spawned = True

2. standalone.py::_bake_navmesh_sync()
   ├─ 解析场景根路径（/World/env_0/scene）
   ├─ ensure_navmesh_volume() → 创建 NavMesh volume
   ├─ sim_app.update() × 3 → 等待体素注册
   ├─ interface.start_navmesh_baking_and_wait() → 同步烘焙
   └─ EnvsetTaskRuntime._navmesh_ready = True
   ✅ 返回: True (成功) / False (失败)

3. EnvsetTaskRuntime.initialize_virtual_humans()
   ├─ 检查: _vh_spawned 必须为 True
   ├─ 检查: _navmesh_ready 必须为 True
   └─ _setup_character_behaviors()
       ├─ load_default_biped_to_stage()
       │   └─ populate_anim_graph() → 创建默认动画图
       ├─ get_anim_graph_from_character()
       ├─ setup_animation_graph_to_character()
       │   └─ ApplyAnimationGraphAPICommand → 应用动画图
       └─ setup_python_scripts_to_character()
           └─ 附加 behavior_script.py 到每个角色
               → 脚本执行时会注册到 AgentManager
               → 触发 AgentEvent.AgentRegistered 事件
               → _on_agent_registered() 注入路由命令
```

**关键时序说明**：

- **为什么 NavMesh 烘焙必须在 runner.reset() 之后？**
  - `runner.reset()` 内部会调用 `task.load()`，将场景 USD 加载到 `/World/env_0/scene`
  - 此时场景几何体已经存在，可以基于它烘焙 NavMesh
  - 之前在 `runner.reset()` 之前烘焙会失败：`Root prim '/World' not found`

- **为什么虚拟人初始化必须在 NavMesh 烘焙之后？**
  - 行为脚本中的导航相关初始化需要 `interface.get_navmesh()` 返回有效对象
  - 如果 NavMesh 还没有烘焙，Agent 无法注册到 AgentManager
  - 拆分成两个阶段：spawn（创建 USD prims）→ initialize（附加脚本和动画图）

- **Timeline 启动时机的影响**：
  - `runner.reset()` 内部会调用 `SimulationContext.reset()`，这会自动启动 timeline
  - 虽然 timeline 已经启动，但行为脚本的执行有几帧延迟
  - 我们利用这个延迟窗口，在脚本真正执行之前完成 NavMesh 烘焙和脚本附加

**状态标志管理**：

```python
class EnvsetTaskRuntime:
    _navmesh_ready = False    # NavMesh 是否已烘焙完成
    _vh_spawned = False       # 虚拟人是否已 spawn
    _pending_routes = {}      # 待注入的路由命令
```

- `_vh_spawned`：在 `_setup_virtual_characters()` 中设置，标记虚拟人 USD prims 已创建
- `_navmesh_ready`：在 `_bake_navmesh_sync()` 成功后设置，标记 NavMesh 已就绪

**调试日志关键点**：

运行成功时，日志顺序应该是：
```
[EnvsetRuntime] Spawned 1 virtual humans (behaviors NOT yet initialized)
[EnvsetStandalone] NavMesh baking completed successfully
[EnvsetStandalone] Initializing virtual humans (attaching behaviors)...
[EnvsetRuntime] Setting up character behaviors...
[EnvsetRuntime] Applying anim graph from /World/Characters/Biped_Setup/CharacterAnimation/AnimationGraph
[EnvsetRuntime] Virtual humans initialization completed
```

> 注意：目前未实现随机化或 spawn shuffle——按需求明确无需支持。

### 3.2 AnimGraph/People 初始化时序修复（2024-11-14）

**问题背景**：

在 Standalone 模式下启用 `omni.anim.graph` 和 `omni.anim.people` 扩展时，虚拟人物会出现以下问题：
1. **T-pose 不解除**：角色保持初始 T-pose 姿态，动画图不生效
2. **路由命令不执行**：虽然注入成功，但角色不移动

**症状日志**：

```
[Warning] [omni.fabric.plugin] Warning: attribute animationGraph not found for path .../male_adult_medical_01
[Warning] [omni.anim.graph.core.plugin] getCharacter - .../male_adult_medical_01 is not a SkelRoot prim or does not apply AnimationGraphAPI
[Warning] [character_behavior] Command file field is empty.
[EnvsetRuntime] No pending route for Character when trying to inject
```

#### 根本原因分析

通过日志时间戳和代码流程分析，发现了两个根本性问题：

**问题 1：扩展加载顺序导致 AnimGraph 插件状态异常**

官方已知 bug（[NVIDIA Forum #301378](https://forums.developer.nvidia.com/t/isaac-sim-people-simulation-broken-in-4-1-0/301378)）：

```
SimulationApp 创建 [0ms]
  ↓ 自动生成默认 stage
enable_extension("omni.anim.graph.core") [100ms]
  ↓ 扩展在已有 stage 上启用
AnimGraph 插件初始化 [150ms]
  ↓ ⚠️ 错过了 stage 初始状态，内部 CharacterManager 未正确建立
runner.reset() [500ms]
  ↓ 加载场景、spawn 角色
ApplyAnimationGraphAPICommand [1000ms]
  ↓ USD schema 层成功应用
  ✗ 但 AnimGraph 插件内部状态已损坏
  ✗ getCharacter() 永远返回 None
  ✗ 角色保持 T-pose
```

**问题 2：路由命令过早注入并被消费**

在 `runtime_hooks.py:619-630` 的手动 Agent 注册逻辑中，路由在 behavior 真正准备好之前就被注入并从 `_pending_routes` 中移除：

```python
# 错误的时序
_register_scripts_with_agent_manager()
  ├─ inst.init_character()  # behavior 开始初始化
  ├─ inst.on_play()         # behavior on_play
  ├─ mgr.register_agent()   # 手动注册到 AgentManager
  └─ _inject_route()        # ⚠️ 立即注入路由
      └─ _pending_routes.pop('Character')  # 路由被消费

... 几帧后 ...

behavior 内部完成初始化
  └─ 发送 AgentRegistered 事件
      └─ _on_agent_registered('Character')
          └─ _inject_route('Character')
              ✗ "No pending route" # 已经被 pop 了！
```

#### 修复方案

**修复 1：Stage 重建（standalone.py:372-382）**

在启用所有 AnimGraph/People 扩展后，立即重建干净的 stage：

```python
# 在 _prepare_runtime_settings() 中
enable_extension("omni.anim.graph.core")
enable_extension("omni.anim.graph.schema")
enable_extension("omni.anim.navigation.core")
enable_extension("omni.anim.people")

# ★★ 关键修复：重建 stage ★★
import omni.usd
usd_ctx = omni.usd.get_context()
usd_ctx.new_stage()
carb.log_info("Re-created stage after enabling AnimGraph/People extensions (bug workaround)")
```

**执行顺序变更**：

```
SimulationApp 创建 [0ms]
  ↓ 生成默认 stage（将被丢弃）
enable_extension("omni.anim.graph.core") [100ms]
  ↓
new_stage() [150ms]  # ★ 创建干净的 stage
  ↓ AnimGraph 插件在新 stage 上正确初始化
runner.reset() [500ms]
  ↓ 在干净 stage 上加载场景、spawn 角色
ApplyAnimationGraphAPICommand [1000ms]
  ↓ USD schema + 插件内部状态都正常
  ✓ getCharacter() 返回有效对象
  ✓ 角色解除 T-pose，动画正常
```

**效果**：
- ✅ 解决 T-pose 问题
- ✅ AnimGraph 插件正确识别角色
- ✅ 不再出现 "animationGraph not found" 警告

**修复 2：路由注入时机调整（runtime_hooks.py:624-628）**

移除手动注册时的立即路由注入，改为等待真正的 `AgentRegistered` 事件：

```python
# 修改前（错误）
if mgr and hasattr(inst, "get_agent_name"):
    agent_name = inst.get_agent_name()
    mgr.register_agent(agent_name, inst.prim_path)
    EnvsetTaskRuntime._inject_route(agent_name)  # ✗ 过早注入

# 修改后（正确）
if mgr and hasattr(inst, "get_agent_name"):
    agent_name = inst.get_agent_name()
    mgr.register_agent(agent_name, inst.prim_path)
    # ★ 不立即注入，等待 AgentRegistered 事件触发
    print(f"Route injection will be triggered by AgentRegistered event for {agent_name}")
```

**时序修正**：

```
_register_scripts_with_agent_manager()
  ├─ inst.init_character()
  ├─ inst.on_play()
  ├─ mgr.register_agent()
  └─ 保留路由不注入  # ★ 等待正确时机

... render warm-up ...

behavior 完全初始化
  └─ 发送 AgentRegistered 事件
      └─ _on_agent_registered('Character')
          └─ _inject_route('Character')
              ✓ _pending_routes 中仍有路由
              ✓ mgr.inject_command() 成功
              ✓ 角色开始执行 GoTo 命令
```

**效果**：
- ✅ 路由在正确时机注入
- ✅ 不再出现 "No pending route" 错误
- ✅ 角色正常执行导航命令

**修复 3：路由注入错误处理增强（runtime_hooks.py:372-382）**

增加详细的日志和异常处理：

```python
@classmethod
def _inject_route(cls, agent_name: str):
    commands = cls._pending_routes.get(agent_name)
    if not commands:
        print(f"[EnvsetRuntime] No pending route for {agent_name}")
        return

    try:
        mgr.inject_command(agent_name, commands, force_inject=True, instant=True)
        cls._pending_routes.pop(agent_name, None)
        print(f"[EnvsetRuntime] ✓ Successfully injected route to agent '{agent_name}': {commands}")
    except Exception as exc:
        print(f"[EnvsetRuntime] ✗ Failed to inject route to agent '{agent_name}': {exc}")
        # 注入失败时不 pop，保留路由供后续重试
        import traceback
        print(traceback.format_exc())
```

#### 验证清单

修复成功后，日志应该显示：

1. **Stage 重建**：
   ```
   [EnvsetStandalone] Re-created stage after enabling AnimGraph/People extensions (bug workaround)
   ```

2. **AnimGraph 正常应用**（无警告）：
   ```
   [CharacterUtil] AnimGraph applied to .../male_adult_medical_01 -> [Sdf.Path('...')]
   # 不再有 "animationGraph not found" 警告
   ```

3. **路由注入成功**：
   ```
   [EnvsetRuntime] Route injection will be triggered by AgentRegistered event for Character
   [AgentManager][DEBUG] register_agent succeeded for Character
   [EnvsetRuntime] ✓ Successfully injected route to agent 'Character': ['Character GoTo -5.0 -4.0 0.0 _', ...]
   ```

4. **角色行为正常**：
   - 角色从 T-pose 折叠手臂（动画生效）
   - 角色开始沿 NavMesh 移动（导航生效）

#### 技术细节说明

**为什么 new_stage() 是必要的？**

`omni.anim.graph.core` 插件在启用时会：
1. 监听 stage 上的 prim 创建/修改事件
2. 扫描现有 stage 建立内部 CharacterManager 映射
3. 为每个 SkelRoot 创建 Fabric 绑定

如果插件在 stage 已经存在后才启用，第 2 步会在一个"半成品" stage 上执行，导致内部状态损坏。`new_stage()` 确保插件在一个干净的 stage 上初始化。

**为什么路由注入要等待事件？**

`character_behavior.py` 的初始化有多个阶段：
1. `init_character()` - 创建内部数据结构
2. `on_play()` - 注册到 AgentManager
3. **几帧延迟** - 等待 AnimGraph 完全就绪
4. 发送 `AgentRegistered` 事件 - 标记真正可以接收命令

手动注册只完成了第 2 步，此时 behavior 还没有准备好处理导航命令。等待事件确保在第 4 步之后才注入。

**参考资料**：
- [Isaac Sim People Simulation Bug](https://forums.developer.nvidia.com/t/isaac-sim-people-simulation-broken-in-4-1-0/301378)
- [AnimationGraph Standalone Issue](https://forums.developer.nvidia.com/t/animationgraph-does-not-work-with-standalone-python/293382)
- [People Animations in Standalone](https://forums.developer.nvidia.com/t/people-animations-in-standalone-app/282800)

## 4. 物理仿真稳定性修复（2024-11）

### 4.1 问题背景

在导入 GRScenes 等场景后，经常出现物体穿透地板、不断下沉的问题。这是由多个因素共同造成的：

1. **物理初始化时序问题**：PhysX 在 reset 后需要 1-2 个 physics steps 才能更新刚体状态
2. **单位缩放问题**：GRScenes 使用厘米单位（metersPerUnit=0.01），而 Isaac Sim 默认使用米单位
3. **碰撞检测不足**：缺少连续碰撞检测（CCD），快速移动物体会穿透碰撞体
4. **碰撞网格不精确**：默认的 convexHull 近似对复杂几何体不准确

### 4.2 实施的修复

#### 修复 1：增强物理初始化等待（standalone.py:314-376）

**根据 Isaac Sim 最佳实践**，在 reset 后增加 warm-up 步骤：

```python
def _wait_for_initialization(self):
    # Step 1: 执行 2 个 physics steps 让刚体状态传播和稳定
    for i in range(2):
        world.step(render=False)

    # Step 2: 执行 12 个 render steps 让传感器数据更新
    for i in range(12):
        SimulationContext.render(world)
```

**效果**：
- ✅ 防止物体在初始化时穿透地板
- ✅ 确保传感器/相机数据正确初始化
- ✅ 增加约 0.5-1 秒启动时间（可接受）

#### 修复 2：动态单位缩放检测（world_utils.py:12-97）

**自动从场景 USD 读取 metersPerUnit 并配置 World**：

```python
def _detect_stage_units_in_meters() -> float:
    """从当前 USD stage 中检测单位缩放"""
    stage = omni.usd.get_context().get_stage()
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    # 对于 GRScenes: meters_per_unit = 0.01 (厘米)
    return meters_per_unit

def bootstrap_world_if_needed(**overrides):
    # 自动检测并应用正确的单位
    detected_units = _detect_stage_units_in_meters()
    world = World(stage_units_in_meters=detected_units, ...)
```

**效果**：
- ✅ GRScenes 场景物理计算正确（重力、质量、速度）
- ✅ 碰撞检测阈值匹配场景单位
- ✅ 控制台输出单位检测日志便于调试

#### 修复 3：启用 CCD 和改进碰撞配置（virtual_human_colliders.py）

**物理场景配置（lines 112-149）**：
```python
def _ensure_physics_scene(self):
    # 启用连续碰撞检测（CCD）
    physx_scene_api.CreateEnableCCDAttr().Set(True)
    # 启用稳定化（对大时间步有帮助）
    physx_scene_api.CreateEnableStabilizationAttr().Set(True)
```

**碰撞体配置改进（lines 14-32）**：
```python
@dataclass
class ColliderConfig:
    approximation_shape: str = "convexDecomposition"  # 更精确
    enable_ccd: bool = True  # 启用 CCD
    contact_offset: float = 0.02  # 2cm 接触偏移
    rest_offset: float = 0.0
```

**虚拟人刚体配置（lines 151-202）**：
- 为刚体启用 CCD
- 设置 contact_offset 和 rest_offset 参数
- 对所有子碰撞体应用高级物理参数

**效果**：
- ✅ 防止快速移动物体穿透
- ✅ 更精确的碰撞检测（凸分解替代凸包）
- ✅ 物理仿真更稳定

### 4.3 测试建议

运行测试时观察以下日志输出：

```bash
[World] Detected stage units: 0.01 meters per unit (centimeters)
[World] Created Isaac World (..., stage_units_in_meters=0.01)
[EnvsetStandalone] Starting physics warm-up (2 steps)...
[EnvsetStandalone] Physics warm-up step 1/2 completed
[EnvsetStandalone] Physics warm-up step 2/2 completed
[EnvsetStandalone] Starting render warm-up (12 steps)...
[EnvsetStandalone] Scene initialization wait completed
[virtual_human_colliders] Enabled CCD (Continuous Collision Detection)
[virtual_human_colliders] Enabled stabilization pass
```

如果看到这些日志，说明修复已正确应用。

---

## 5. 键盘控制动作传递修复（2024-11）

### 5.1 问题背景

在实现 envset 的键盘控制功能时，发现通过 JSON 配置加载的机器人无法响应键盘输入，而直接使用 Python 脚本（如 `aliengo_keyboard.py`）则工作正常。经过深入分析，发现了两个根本性问题：

#### 问题 1：动作数据结构不匹配

**症状**：
- 机器人不响应键盘输入
- 没有报错，程序正常运行
- 日志中出现 `[WARNING] unknown controller move_by_speed in action`

**根本原因**：

`standalone.py` 和 `gym_env.py` 构建的动作数据结构不同，导致 `runner.py` 无法正确分发动作。

1. **Python 脚本路径**（正常工作）：
   ```python
   # aliengo_keyboard.py
   env_action = {'move_by_speed': (x, y, z)}
   env.step(action=env_action)
   
   # gym_env.py 自动包装
   _actions = [{self._robot_name: action}]  # {"aliengo": {"move_by_speed": (x,y,z)}}
   self._runner.step(_actions)
   ```

2. **JSON 配置路径**（问题路径）：
   ```python
   # standalone.py (修复前)
   env_action = {"move_by_speed": (x, y, z)}  # 缺少机器人名称层
   actions = [env_action]
   self._runner.step(actions)
   
   # runner.py 处理
   for name, action in action_dict.items():
       # name = "move_by_speed" (控制器名称，而非机器人名称)
       if name in task.robots:  # False! task.robots 的键是 "keyboard_aliengo"
           # 动作被跳过，机器人不动
   ```

**为什么没有报错？**

`runner.py` 的设计是"宽容"的，为了支持多机器人场景，如果动作字典中的键在 `task.robots` 中找不到，它会默默忽略，而不是抛出异常。这种设计在多智能体场景中是合理的，但在这里完美地掩盖了数据结构不匹配的问题。

#### 问题 2：控制器类型不匹配

**症状**：
- 即使修复了动作结构，仍然出现 `[WARNING] unknown controller move_by_speed in action`
- 机器人姿态与 demo 方式不同

**根本原因**：

`EnvsetTaskAugmentor` 为 aliengo 默认配置的是 `move_to_point` 控制器，而键盘发送的是 `move_by_speed` 动作：

```python
# task_adapter.py (修复前)
if robot_type == "aliengo" and aliengo_move_to_point_cfg is not None:
    return [aliengo_move_to_point_cfg]  # 名称是 "move_to_point"

# standalone.py 发送的动作
env_action = {"keyboard_aliengo": {"move_by_speed": (x,y,z)}}

# aliengo.py 处理
for controller_name, controller_action in action.items():
    if controller_name not in self.controllers:  # "move_by_speed" 不在控制器列表中
        log.warning(f'unknown controller {controller_name} in action')
```

### 5.2 修复方案

#### 修复 1：动作数据结构（standalone.py）

**修改位置**：`internutopia_extension/envset/standalone.py:471-504`

**修改前**：
```python
def _collect_actions(self):
    actions = []
    for env_id in range(self._runner.env_num):
        env_action = {}
        for robot_cfg in self._keyboard_robots:
            env_action[robot_cfg["controller"]] = (x_speed, y_speed, z_speed)
        actions.append(env_action)
    return actions
```

**修改后**：
```python
def _collect_actions(self):
    actions = []
    for env_id in range(self._runner.env_num):
        env_action = {}
        for robot_cfg in self._keyboard_robots:
            robot_name = robot_cfg["name"]  # 机器人在 task.robots 中的键
            controller_name = robot_cfg["controller"]
            # 构建正确的嵌套结构：{机器人名称: {控制器名称: 动作}}
            env_action[robot_name] = {
                controller_name: (x_speed, y_speed, z_speed)
            }
        actions.append(env_action)
    return actions
```

**效果**：
- ✅ 动作格式匹配 `runner.py` 的期望
- ✅ `runner.py` 能够正确找到机器人并调用 `apply_action()`
- ✅ 支持多机器人场景

#### 修复 2：控制器类型选择（task_adapter.py）

**修改位置**：`internutopia_extension/envset/task_adapter.py:217-227`

**修改前**：
```python
# Legged and humanoid robots - use pre-defined configurations
if robot_type == "aliengo" and aliengo_move_to_point_cfg is not None:
    return [aliengo_move_to_point_cfg]
```

**修改后**：
```python
# Legged and humanoid robots - check control mode first
if robot_type == "aliengo":
    # 如果 control.mode 包含 "move_by_speed"，使用基础速度控制器
    control_mode = (spec.control.mode or "").lower() if spec.control else ""
    if "move_by_speed" in control_mode and aliengo_move_by_speed_cfg is not None:
        return [EnvsetTaskAugmentor._clone_and_override_controller(aliengo_move_by_speed_cfg, params)]
    # 否则使用默认的 move_to_point 控制器
    elif aliengo_move_to_point_cfg is not None:
        return [EnvsetTaskAugmentor._clone_and_override_controller(aliengo_move_to_point_cfg, params)]
```

**效果**：
- ✅ 根据 JSON 配置中的 `control.mode` 选择合适的控制器
- ✅ `keyboard_move_by_speed` 模式使用 `move_by_speed` 控制器
- ✅ 其他模式使用 `move_to_point` 控制器（支持导航）
- ✅ 向后兼容：未指定 mode 时使用默认控制器

### 5.3 数据流对比

**修复后的完整数据流**：

```
1. JSON 配置
   {
     "type": "aliengo",
     "label": "keyboard_aliengo",
     "control": {"mode": "keyboard_move_by_speed"}
   }

2. EnvsetTaskAugmentor._build_robot_controllers()
   → 检测到 "move_by_speed" in mode
   → 返回 [move_by_speed_cfg]  # 名称是 "move_by_speed"

3. create_robots() 
   → robot_map["keyboard_aliengo"] = AliengoRobot(...)
   → robot.controllers = {"move_by_speed": MoveBySpeedController(...)}

4. standalone.py._collect_actions()
   → 构建 [{"keyboard_aliengo": {"move_by_speed": (1.0, 0.0, 0.0)}}]

5. runner.py.step()
   → for name, action in action_dict.items():
   →   name = "keyboard_aliengo" ✓
   →   if name in task.robots: ✓
   →     task.robots["keyboard_aliengo"].apply_action(action)

6. aliengo.py.apply_action()
   → for controller_name, controller_action in action.items():
   →   controller_name = "move_by_speed" ✓
   →   if controller_name in self.controllers: ✓
   →     controller.action_to_control(controller_action)
   →     self.articulation.apply_action(control) ✓
```

### 5.4 配置示例

**正确的 envset JSON 配置**：

```json
{
  "robots": {
    "entries": [
      {
        "type": "aliengo",
        "label": "keyboard_aliengo",
        "spawn_path": "/World/aliengo",
        "usd_path": "/path/to/aliengo_camera.usd",
        "initial_pose": {
          "position": [0.0, 0.0, 0.5],
          "orientation_deg": 0.0
        },
        "control": {
          "mode": "keyboard_move_by_speed",  // 关键：包含 "move_by_speed"
          "params": {
            "base_velocity": 1.0,
            "base_turn_rate": 2.0
          }
        }
      }
    ]
  }
}
```

**支持的控制模式**：
- `keyboard_move_by_speed` - 键盘直接控制速度（适合四足机器人）
- `keyboard_locomotion` - 同上
- 其他包含 `keyboard` 但不含 `move_by_speed` 的模式 - 使用 `move_to_point` 控制器

### 5.5 调试建议

如果遇到类似问题，可以通过以下方式排查：

1. **检查动作格式**：在 `standalone.py._collect_actions()` 中添加打印
   ```python
   print(f"[DEBUG] Actions: {actions}")
   # 应该看到：[{"keyboard_aliengo": {"move_by_speed": (x, y, z)}}]
   ```

2. **检查控制器注册**：在运行时打印
   ```python
   print(f"[DEBUG] Robot controllers: {list(robot.controllers.keys())}")
   # 应该看到：['move_by_speed']
   ```

3. **观察警告信息**：
   - `[WARNING] unknown controller move_by_speed in action` → 控制器类型不匹配
   - 无警告但机器人不动 → 动作数据结构问题

## 6. 兼容性处理

### 6.1 可选扩展支持

某些 Isaac Sim extensions 可能在不同版本中不可用，已做兼容处理：

- **omni.isaac.matterport**: 用于导入 Matterport 场景，在某些 Isaac Sim 版本中不存在
  - 处理方式：可选导入，缺失时会显示警告但不会导致启动失败
  - 影响：如果使用 Matterport 场景且扩展缺失，会报错但不影响其他功能

相关代码：
- `simulation.py:25-32` - 可选导入 matterport
- `simulation.py:1218-1221` - 使用前检查可用性
- `standalone.py:149-154` - 尝试启用但失败不影响启动

## 7. Task 终止控制（2024-11-14）

### 7.1 FiniteStepTask 终止机制

`FiniteStepTask` 提供了灵活的终止控制，支持手动、自动和自定义终止条件。

**默认行为**：
- ✅ **自动停止默认关闭** - 任务会无限运行，适合交互式调试和探索
- ✅ **手动控制** - 通过 API 随时停止任务
- ✅ **可扩展** - 支持子类自定义终止条件

#### 终止控制 API

**1. 手动终止**（推荐用于交互式场景）：

```python
# 在运行时的任何时候，可以手动请求停止
task = runner.current_tasks['0']  # 获取第一个 task
task.request_stop()
# 下一次 is_done() 检查时，task 将标记为完成
```

**2. 自动终止**（适合批量运行和基准测试）：

```python
# 启用自动停止：运行指定步数后自动结束
task = runner.current_tasks['0']
task.enable_auto_stop()  # 将在 max_steps 步后自动停止

# 动态调整步数
task.set_max_steps(5000)

# 禁用自动停止（恢复无限运行模式）
task.disable_auto_stop()
```

**3. 查询状态**：

```python
# 检查当前状态
current_step = task.get_current_step()
max_steps = task.get_max_steps()
is_auto_enabled = task.is_auto_stop_enabled()
is_manual_requested = task.is_stop_requested()

print(f"Step {current_step}/{max_steps}, auto={is_auto_enabled}")
```

**4. 自定义终止条件**（高级用法）：

```python
# 创建自定义 Task 子类
from internutopia_extension.tasks.finite_step_task import FiniteStepTask

class GoalReachedTask(FiniteStepTask):
    def _check_custom_termination_conditions(self) -> bool:
        # 自定义逻辑：检查机器人是否到达目标
        robot = self.robots.get('my_robot')
        if robot is None:
            return False

        pos = robot.get_position()
        goal = self.goal_position
        distance = np.linalg.norm(pos - goal)

        if distance < 0.1:  # 距离目标小于 10cm
            print(f"[{self.name}] Goal reached! Stopping task.")
            return True
        return False
```

#### 终止优先级

终止条件按以下优先级检查（`is_done()` 中的实现）：

```
1. 手动停止（request_stop()）         ← 最高优先级
   ↓
2. 自动停止（enable_auto_stop()）     ← 如果启用
   ↓
3. 自定义条件（_check_custom_termination_conditions()）
   ↓
4. 继续运行（返回 False）             ← 默认行为
```

#### 配置文件说明

`config_minimal.yaml` 中的 `max_steps` 参数现在**仅作为上限参考**，不会自动触发停止：

```yaml
task_configs:
  - type: FiniteStepTask
    max_steps: 3000  # 仅在调用 enable_auto_stop() 后生效
    robots: []
```

**重要**：如果你需要任务自动停止，必须在代码中显式调用 `task.enable_auto_stop()`。

#### 使用场景示例

**场景 1：交互式调试**（默认模式）
```python
# 不做任何设置，任务会无限运行
# 在 Isaac Sim 中手动观察和调试
# 需要停止时在代码中调用 task.request_stop()
```

**场景 2：定时基准测试**
```python
# 启动后立即启用自动停止
runner.reset()
for task in runner.current_tasks.values():
    task.enable_auto_stop()
    task.set_max_steps(10000)  # 运行 10000 步
# 任务会自动在 10000 步后结束
```

**场景 3：目标驱动任务**
```python
# 创建自定义 Task，重写 _check_custom_termination_conditions
# 当机器人完成任务（到达目标、抓取物体等）时自动停止
```

#### 从 standalone.py 访问 Task

```python
# 在 standalone.py 的 _main_loop 中访问 task
def _main_loop(self):
    # 获取第一个 task
    task_name = list(self._runner.current_tasks.keys())[0]
    task = self._runner.current_tasks[task_name]

    # 示例：按键控制终止
    # if keyboard_pressed('Q'):
    #     task.request_stop()

    # 示例：运行 5000 步后自动停止
    # if task.get_current_step() == 0:
    #     task.enable_auto_stop()
    #     task.set_max_steps(5000)

    while sim_app.is_running():
        # ... 主循环逻辑 ...
        pass
```

### 7.2 现存缺口 / TODO

1. ~~**机器人类型扩展**~~ ✅ **已完成**
   - ✅ 支持 `aliengo`（四足）、`h1`、`g1`、`gr1`（人形）、`franka`（机械臂）等类型
   - ✅ `EnvsetTaskAugmentor._resolve_robot_type/_build_robot_controllers` 已扩展
   - **实现说明**：
     - **差速驱动机器人**（jetbot, carter等）：从envset参数构建DifferentialDriveController + MoveToPointBySpeedController
     - **四足/人形机器人**（aliengo, h1, g1, gr1）：**直接引用预定义的完整控制器配置**（从 `internutopia_extension.configs.robots.*`），包含策略权重和关节映射，可通过envset参数覆盖速度设置
     - **机械臂**（franka）：不设置controllers，使用机器人默认配置
   - **关键修复**：解决了之前四足/人形机器人缺少底层策略控制器的问题，现在使用完整的预定义配置

2. ~~**数据生成**~~ ✅ **已完成**
   - ✅ CLI 的 `--run-data` 已实现
   - ✅ 集成 `DataGeneration` 到 `standalone.py`
   - **实现说明**：
     - 从envset scenario的`data_generation`字段读取配置（可选）
     - 支持配置项：`writer`（默认BasicWriter）、`writer_params`、`num_frames`（默认300）、`camera_paths`
     - 自动检测场景中的相机（如果未指定camera_paths）
     - 数据生成完成后可继续正常仿真（根据--no-play标志）

3. ~~**键盘控制支持**~~ ✅ **已完成**
   - ✅ 主循环从 `sim_app.update()` 改为 `runner.step(actions)`
   - ✅ 自动检测 `control.mode` 中的 "keyboard" 标识
   - ✅ 支持混合模式：键盘控制和自主导航可同时运行
   - ✅ **修复动作数据结构不匹配问题**（2024-11）
   - **实现说明**：
     - 如果envset中有 `control.mode` 包含 "keyboard"，自动初始化键盘交互
     - 键盘映射：I/K前后，J/L左右，U/O上下（适配legged robots）
     - 支持多机器人键盘控制
     - 向后兼容：无keyboard配置时等同于原有的自主运行模式
   - **关键修复**（2024-11）：
     - **问题**：`standalone.py` 构建的动作格式 `{控制器名称: 动作}` 与 `runner.py` 期望的 `{机器人名称: {控制器名称: 动作}}` 不匹配，导致动作被静默丢弃
     - **修复**：在 `standalone.py._collect_actions()` 中构建正确的嵌套结构，确保动作能传递到机器人
     - **控制器匹配**：根据 `control.mode` 中的 `move_by_speed` 标识，在 `task_adapter.py` 中为四足机器人选择对应的基础控制器（`move_by_speed` vs `move_to_point`）

3. **机器人控制策略说明**
   - 建议在文档中明确 envset `robots.entries[].control` 的字段含义（例如 `mode`, `module` 仅供回溯，实际控制交由 InternUtopia controllers，并通过 `params` 决定差速/速度等）。

4. **引用/扩展校验**
   - 新增模块引用 `ArrivalGuard`, `BehaviorScriptPaths` 等已在 `runtime_hooks.py` 中导入，需要在实际运行时确认 extension path 正确（当前 import 路径基于包内结构，满足 `python -m` 运行）。
   - 若 future 修改了包结构或路径，请同步更新。

5. ~~**物理穿透问题**~~ ✅ **已修复（2024-11）**
   - ✅ 增强物理初始化等待（2 physics steps + 12 render steps）
   - ✅ 动态检测和应用场景单位缩放（GRScenes 厘米单位支持）
   - ✅ 启用 CCD（连续碰撞检测）和物理稳定化
   - ✅ 改进碰撞网格近似（convexDecomposition）和偏移参数
   - 详见 **第 4 章：物理仿真稳定性修复**

## 8. 使用示例

### 8.0 基本运行命令

如果你的自定义扩展（如 isaaclab extensions）在特定目录：

```bash
# 基本用法
python -m internutopia_extension.envset.standalone \
  --config config_minimal.yaml \
  --envset scenario.json

# 添加扩展搜索路径（如 isaaclab 的 source 目录）
python -m internutopia_extension.envset.standalone \
  --config config_minimal.yaml \
  --envset scenario.json \
  --extension-path /path/to/isaaclab/source \
  --extension-path /another/extension/directory

# 使用 headless 模式和数据生成
python -m internutopia_extension.envset.standalone \
  --config config_minimal.yaml \
  --envset scenario.json \
  --extension-path /path/to/isaaclab/source \
  --headless \
  --run-data
```

**说明**：
- `--extension-path` 可以多次指定，添加多个扩展搜索路径
- 这些路径会被添加到 Isaac Sim 的扩展搜索系统中
- 如果你的 `omni.isaac.matterport` 等扩展在 isaaclab 的 `source` 目录，使用这个参数指定
- `source` 目录结构通常是：`isaaclab/source/omni.isaac.matterport/config/extension.toml`

### 8.1 支持的机器人类型

在 `envset.json` 的 `robots.entries[].type` 字段中可以使用以下值：

- **差速驱动机器人**：`jetbot`、`carter`、`carter_v1`、`differential_drive`
- **四足机器人**：`aliengo`
- **人形机器人**：`h1`、`g1`、`gr1`、`human`
- **机械臂**：`franka`

### 8.2 机器人控制参数

在 `robots.entries[].control` 中可配置：

```json
"control": {
    "mode": "keyboard_diff_drive",  // 控制模式：包含"keyboard"则启用键盘控制
    "params": {
        "wheel_radius": 0.24,        // 车轮半径（仅差速驱动）
        "track_width": 0.54,          // 轮距（仅差速驱动）
        "base_velocity": 0.6,         // 前进速度（所有类型）
        "base_turn_rate": 1.2         // 旋转速度（所有类型）
    }
}
```

**控制模式说明**：
- `mode` 中包含 `"keyboard"` → 启用键盘控制（如 `"keyboard_diff_drive"`）
- `mode` 为其他值或不包含 `"keyboard"` → 自主导航模式

**键盘控制键位**：
- `I` / `K` → 前进 / 后退
- `J` / `L` → 左移 / 右移（横向）
- `U` / `O` → 上升 / 下降（仅支持的机器人）

### 8.3 数据生成配置（可选）

在 `envset.json` 的 scenario 中添加 `data_generation` 字段：

```json
{
  "id": "example_scenario",
  "scene": {...},
  "data_generation": {
    "writer": "BasicWriter",
    "num_frames": 500,
    "writer_params": {
      "output_dir": "_out_data",
      "rgb": true,
      "semantic_segmentation": false
    },
    "camera_paths": []  // 留空则自动检测场景中的相机
  }
}
```

使用数据生成功能：
```bash
python -m internutopia_extension.envset.standalone \
  --config config_minimal.yaml \
  --envset scenario.json \
  --run-data
```

### 8.4 键盘控制机器人配置

在 `envset.json` 中配置键盘控制的aliengo机器人示例：

```json
{
  "robots": {
    "entries": [
      {
        "type": "aliengo",
        "label": "keyboard_aliengo",
        "spawn_path": "/World/aliengo",
        "initial_pose": {
          "position": [0.0, 0.0, 0.5],
          "orientation_deg": 0.0
        },
        "control": {
          "mode": "keyboard_locomotion",
          "params": {
            "base_velocity": 1.0,
            "base_turn_rate": 2.0
          }
        }
      }
    ]
  }
}
```

运行：
```bash
python -m internutopia_extension.envset.standalone \
  --config config_minimal.yaml \
  --envset scenario.json
  # 不要加 --headless（键盘需要窗口）
```

**注意**：
- `config_minimal.yaml` 是最小化的基础配置文件，已包含在项目根目录
- 大部分配置来自 `envset.json`，config文件提供基础的模拟器设置和至少一个task模板
- `task_configs` 必须包含至少一个task条目（即使内容为空），envset会将场景、机器人等信息注入到这个task中
- 不能使用完全空的 `task_configs: []`，因为EnvsetTaskAugmentor会迭代现有task来注入envset数据

## 9. 自定义 Extension 集成

如果你有自己的 Isaac Sim extension（带 `extension.toml`），可以通过以下方式集成：

1. **配置文件方式**（推荐）：在 YAML 中添加 `simulator.extension_folders`
   ```yaml
   simulator:
     extension_folders:
       - /path/to/your/extension
   ```

2. **代码启用方式**：在 `standalone.py` 中调用 `enable_extension("your.extension.name")`

3. **环境变量方式**：设置 `ISAACLAB_EXTENSION_PATHS` 环境变量

详细说明请参考：[自定义 Extension 集成指南](custom_extensions_guide.md)

## 10. 后续可选优化

1. **配置Schema验证**：使用Pydantic或JSON Schema验证envset.json格式
2. **文档完善**：补充CLI/文档，说明envset JSON中各字段的详细约定
3. **扩展性增强**：如需更多行为（如控制输入、RL接口），在Runner中读取envset字段即可扩展
4. **物理参数微调**：根据具体场景调整 physics_dt、contact_offset、rest_offset 等参数以获得最佳稳定性
```
