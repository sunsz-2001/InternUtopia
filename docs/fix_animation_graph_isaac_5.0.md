# 修复 Isaac Sim 5.0.0 动画图初始化问题

## 问题背景

在 Isaac Sim 5.0.0 中使用 `envset` 配置虚拟人时，遇到以下问题：

1. **错误现象**：
   ```
   [Warning] [omni.graph.core._impl_registration.register_python_ogn] 
   Python import process in omni.anim.graph.core failed - 'NoneType' object is not subscriptable
   ```

2. **症状**：
   - `populate_anim_graph()` 调用失败
   - 虚拟人的 `SkelRoot` 上没有动画图（`anim_graph=None`）
   - 虚拟人无法注册到 `AgentManager`
   - `GoTo` 等导航命令无法执行

3. **根本原因**：
   - Isaac Sim 5.0.0 中 `omni.anim.graph.core` 从版本 106.5.1 更新到 **107.3.0**
   - 该扩展要求在 `World` 初始化**之前**就被启用
   - 原代码在 `SimulatorRunner` 创建（包括 `World` 初始化）**之后**才启用扩展
   - 导致动画图相关的 Python 节点注册失败

## 解决方案：方案 A - 修正扩展启用时机

### 核心思路

将扩展启用提前到 `SimulationApp` 创建之后、`World` 初始化之前：

```
原流程（有问题）：
1. import_extensions()
2. _create_runner() → 创建 SimulationApp + World
3. _prepare_runtime_settings() → enable_extension("omni.anim.graph.core")  ❌ 太晚
4. populate_anim_graph()  ❌ 扩展初始化已失败

修复后的流程：
1. import_extensions()
2. _init_simulation_app() → 只创建 SimulationApp
3. _enable_critical_extensions() → enable_extension("omni.anim.graph.core")  ✅ 正确时机
4. _create_runner_with_app() → 创建 World（此时扩展已就绪）
5. populate_anim_graph()  ✅ 成功
```

### 代码修改

#### 1. 修改 `run()` 方法的执行顺序

**文件**：`internutopia_extension/envset/standalone.py`

**修改前**：
```python
def run(self):
    config_model = self._build_config_model()
    import_extensions()
    self._runner = self._create_runner(config_model)  # SimulationApp + World 一起创建
    self._prepare_runtime_settings()  # 扩展启用太晚
    self._post_runner_initialize()
    # ...
```

**修改后**：
```python
def run(self):
    config_model = self._build_config_model()
    import_extensions()
    self._init_simulation_app(config_model)  # 只创建 SimulationApp
    self._enable_critical_extensions()  # 立即启用关键扩展
    self._runner = self._create_runner_with_app(config_model)  # 创建 World
    self._prepare_runtime_settings()  # 启用其他扩展
    self._post_runner_initialize()
    # ...
```

#### 2. 新增 `_init_simulation_app()` 方法

```python
def _init_simulation_app(self, config: Config):
    """Initialize SimulationApp without creating World yet."""
    from isaacsim import SimulationApp
    import os

    headless = config.simulator.headless
    launch_config = {
        'headless': headless,
        'anti_aliasing': 0,
        'hide_ui': False,
        'multi_gpu': False
    }

    # Handle custom extension paths
    if hasattr(config.simulator, 'extension_folders') and config.simulator.extension_folders:
        ext_paths = config.simulator.extension_folders
        if 'ISAAC_EXTRA_EXT_PATH' in os.environ:
            existing = os.environ['ISAAC_EXTRA_EXT_PATH']
            os.environ['ISAAC_EXTRA_EXT_PATH'] = os.pathsep.join([existing] + ext_paths)
        else:
            os.environ['ISAAC_EXTRA_EXT_PATH'] = os.pathsep.join(ext_paths)

    self.simulation_app = SimulationApp(launch_config)
```

#### 3. 新增 `_enable_critical_extensions()` 方法

```python
def _enable_critical_extensions(self):
    """Enable critical extensions immediately after SimulationApp creation.
    
    This must happen BEFORE World creation to ensure omni.anim.graph.core
    and related extensions are properly initialized.
    """
    import carb
    from omni.isaac.core.utils.extensions import enable_extension

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
```

#### 4. 新增 `_create_runner_with_app()` 方法

```python
def _create_runner_with_app(self, config: Config) -> SimulatorRunner:
    """Create runner using the pre-initialized SimulationApp.
    
    This method creates the SimulatorRunner but skips the SimulationApp
    initialization since it was already done in _init_simulation_app().
    """
    from internutopia.core.task_config_manager import create_task_config_manager
    from internutopia.core.scene.scene import IScene
    from omni.isaac.core import World
    from internutopia.core.util import log
    
    # Create task manager
    task_manager = create_task_config_manager(config)
    
    # Create a custom runner that uses our pre-initialized SimulationApp
    runner = SimulatorRunner.__new__(SimulatorRunner)
    runner.config = config
    runner.task_config_manager = task_manager
    runner.env_num = config.env_num
    runner.simulation_app = self.simulation_app  # 使用已创建的 SimulationApp
    
    # Now create World (this is safe because extensions are already enabled)
    physics_dt = eval(config.simulator.physics_dt) if isinstance(config.simulator.physics_dt, str) else config.simulator.physics_dt
    rendering_dt = eval(config.simulator.rendering_dt) if isinstance(config.simulator.rendering_dt, str) else config.simulator.rendering_dt
    use_fabric = config.simulator.use_fabric
    
    runner.dt = physics_dt
    runner._world = World(
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
        stage_units_in_meters=1.0,
        sim_params={'use_fabric': use_fabric},
    )
    
    # Initialize remaining runner attributes
    runner._scene = IScene.create()
    runner._stage = runner._world.stage
    runner.task_name_to_env_id_map = {}
    runner.env_id_to_task_name_map = {}
    runner.finished_tasks = set()
    runner.render_interval = config.simulator.rendering_interval if config.simulator.rendering_interval is not None else 5
    runner.render_trigger = 0
    runner.loop = False
    runner._render = False
    runner.metrics_config = None
    runner.metrics_save_path = config.metrics_save_path
    
    return runner
```

#### 5. 修改 `_prepare_runtime_settings()` 方法

将关键的动画图扩展移到 `_enable_critical_extensions()` 中，只保留其他扩展的启用：

```python
def _prepare_runtime_settings(self):
    """Configure additional runtime settings and enable remaining extensions."""
    import carb
    from omni.isaac.core.utils.extensions import enable_extension

    # Enable remaining extensions (non-critical ones)
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
    
    # Optional extensions
    try:
        enable_extension("omni.physxcommands")
    except Exception:
        carb.log_warn("[EnvsetStandalone] omni.physxcommands not available (optional)")
    
    try:
        enable_extension("omni.isaac.matterport")
    except Exception:
        carb.log_warn("[EnvsetStandalone] Matterport extension not available")
    
    # ... rest of the method
```

## 验证方法

运行修改后的代码，检查以下输出：

1. **扩展启用顺序正确**：
   ```
   [EnvsetStandalone] Initializing SimulationApp...
   [EnvsetStandalone] SimulationApp created successfully
   [EnvsetStandalone] Enabling critical extensions (before World creation)...
   [EnvsetStandalone] Critical animation extensions enabled before World creation
   [EnvsetStandalone] Creating runner (initializing World and tasks)...
   [DEBUG] Before World creation (with extensions enabled)
   [DEBUG] After World creation
   ```

2. **动画图成功加载**：
   ```
   [EnvsetRuntime] Setting up character behaviors...
   [EnvsetRuntime] Default biped prim: /World/Characters/Biped_Setup/biped_demo_meters
   [EnvsetRuntime] Characters detected in stage: 1
   [EnvsetRuntime] Applying anim graph from /World/Characters/Biped_Setup/biped_demo_meters/AnimGraph
   ```

3. **虚拟人成功注册**：
   ```
   [EnvsetStandalone] === Snapshot: After initialization wait ===
   Detected SkelRoot count: 2
     SkelRoot: /World/Characters/Character/ManRoot/male_adult_medical_01, 
       anim_graph=/World/Characters/Biped_Setup/biped_demo_meters/AnimGraph,  ✅ 不再是 None
       scripts=[.../character_behavior.py]
   Registered agents (1): ['Character']  ✅ 成功注册
   ```

4. **无 `omni.graph` 错误**：
   - 不再出现 `'NoneType' object is not subscriptable` 警告

## 技术要点

### 1. 为什么必须在 World 之前启用扩展？

根据 Isaac Sim 官方文档和 5.0.0 的已知问题：
- OmniGraph 相关的图表需要在 `World` 或 `SimulationContext` 初始化之前创建
- `omni.anim.graph.core` v107.3.0 在内部注册 Python 节点时，依赖某些只在早期阶段初始化的对象
- 如果 `World` 已经创建，这些对象可能已经被锁定或初始化为 `None`

### 2. 为什么使用 `__new__()` 创建 runner？

- `SimulatorRunner.__init__()` 会自动调用 `setup_isaacsim()` 和 `create_world()`
- 我们需要跳过 `setup_isaacsim()`（因为 `SimulationApp` 已创建），但仍需要创建 `World`
- 使用 `__new__()` 可以创建实例但不调用 `__init__()`，然后手动初始化所有属性

### 3. 扩展启用的优先级

**关键扩展（必须在 World 之前）**：
- `omni.anim.graph.core`
- `omni.anim.retarget.core`
- `omni.anim.navigation.*`
- `omni.anim.people`

**其他扩展（可以在 World 之后）**：
- 传感器扩展（`isaacsim.sensors.*`）
- 工具扩展（`omni.kit.*`）
- 可选扩展（`omni.isaac.matterport`）

## 相关资源

- Isaac Sim 5.0.0 迁移指南：https://docs.robotsfan.com/isaaclab/source/refs/migration.html
- Isaac Sim 已知问题：https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/known_issues.html
- `omni.anim.graph.core` 版本变化：106.5.1 → 107.3.0

## 后续工作

1. ✅ 修复扩展启用时机
2. ⏳ 测试虚拟人 `GoTo` 命令是否正常执行
3. ⏳ 测试机器人键盘控制是否仍然正常
4. ⏳ 在不同场景下回归测试（warehouse、空场景等）
5. ⏳ 更新 `envset_integration_summary.md` 文档

## 日期

2024-11-12

