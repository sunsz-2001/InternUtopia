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

## 4. 现存缺口 / TODO

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
   - **实现说明**：
     - 如果envset中有 `control.mode` 包含 "keyboard"，自动初始化键盘交互
     - 键盘映射：I/K前后，J/L左右，U/O上下（适配legged robots）
     - 支持多机器人键盘控制
     - 向后兼容：无keyboard配置时等同于原有的自主运行模式

3. **机器人控制策略说明**
   - 建议在文档中明确 envset `robots.entries[].control` 的字段含义（例如 `mode`, `module` 仅供回溯，实际控制交由 InternUtopia controllers，并通过 `params` 决定差速/速度等）。

4. **引用/扩展校验**
   - 新增模块引用 `ArrivalGuard`, `BehaviorScriptPaths` 等已在 `runtime_hooks.py` 中导入，需要在实际运行时确认 extension path 正确（当前 import 路径基于包内结构，满足 `python -m` 运行）。
   - 若 future 修改了包结构或路径，请同步更新。

## 5. 使用示例

### 5.1 支持的机器人类型

在 `envset.json` 的 `robots.entries[].type` 字段中可以使用以下值：

- **差速驱动机器人**：`jetbot`、`carter`、`carter_v1`、`differential_drive`
- **四足机器人**：`aliengo`
- **人形机器人**：`h1`、`g1`、`gr1`、`human`
- **机械臂**：`franka`

### 5.2 机器人控制参数

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

### 5.3 数据生成配置（可选）

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

### 5.4 键盘控制机器人配置

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

## 6. 后续可选优化

1. **配置Schema验证**：使用Pydantic或JSON Schema验证envset.json格式
2. **文档完善**：补充CLI/文档，说明envset JSON中各字段的详细约定
3. **扩展性增强**：如需更多行为（如控制输入、RL接口），在Runner中读取envset字段即可扩展
```
