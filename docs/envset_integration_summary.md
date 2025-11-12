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

1. **NavMesh**：根据 `envset.navmesh` 创建/调整 NavMesh include volume（`ensure_navmesh_volume`）。
2. **虚拟人加载**：
   - 按 `virtual_humans` 的 spawn_point/name_sequence/assets 加载 USD；
   - 为虚拟人绑定动画图 (`CharacterUtil.setup_animation_graph_to_character`) 与行为脚本 (`BehaviorScriptPaths.behavior_script_path`);
   - 设置语义信息、NavMesh 排除；
   - 如果 `scene.category == "GRScenes"`，根据 `arrival_tolerance_m` 启用 `ArrivalGuard`。
3. **Routes 注入**：监听 `AgentEvent.AgentRegistered`，在角色注册后调用 `AgentManager.inject_command()` 下发 envset routes（`GoTo`/`Idle` 等指令）。

> 注意：目前未实现随机化、虚拟人碰撞体或 spawn shuffle——按需求明确无需支持。

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

## 7. 现存缺口 / TODO

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
