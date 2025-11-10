# 自定义 Extension 集成指南

本文档说明如何在 InternUtopia 中集成自定义的 Isaac Sim/Omniverse Kit extensions。

## 背景

Isaac Sim 使用 Omniverse Kit 的扩展系统，通过 `extension.toml` 文件定义扩展的依赖关系、Python 模块等。如果你有自己的扩展（类似 isaaclab 的扩展），可以通过以下方式集成到 InternUtopia。

## 方法 1：通过配置文件指定扩展路径（推荐）

### 1.1 在 YAML 配置中添加扩展路径

**config.yaml**:
```yaml
simulator:
  physics_dt: 0.00416666667
  rendering_dt: 0.00416666667
  headless: false
  extension_folders:
    - /path/to/your/extension/directory
    - /another/extension/directory
```

### 1.2 扩展目录结构

你的扩展目录应该包含标准的 `extension.toml`：

```
my_custom_extension/
├── config/
│   └── extension.toml          # 扩展配置文件
├── my_extension/
│   ├── __init__.py
│   ├── my_module.py
│   └── ...
└── docs/
    └── README.md
```

### 1.3 extension.toml 示例

```toml
[package]
version = "1.0.0"
authors = ["Your Name"]
title = "My Custom Extension"
description = "Description of your extension"
repository = ""
category = "Custom"
keywords = ["isaac", "custom"]

[dependencies]
"omni.usd" = {}
"isaacsim.core.utils" = {}
# 添加你需要的其他依赖

[[python.module]]
name = "my_extension"

[settings]
# 你的设置
```

### 1.4 使用示例

```bash
python -m internutopia_extension.envset.standalone \
  --config config_with_extensions.yaml \
  --envset scenario.json
```

## 方法 2：在代码中动态启用扩展

如果你的扩展已经在 Isaac Sim 的搜索路径中（例如在 `~/.local/share/ov/pkg/isaac-sim-*/exts/` 下），可以在代码中直接启用：

### 2.1 修改 standalone.py

在 `_prepare_runtime_settings()` 中添加你的扩展：

```python
def _prepare_runtime_settings(self):
    import carb
    import carb.settings

    from omni.isaac.core.utils.extensions import enable_extension

    # 启用你的自定义扩展
    enable_extension("my.custom.extension")

    # ... 其他代码
```

### 2.2 在 import_extensions() 中注册

如果你的扩展包含需要在 InternUtopia 中注册的 controllers、robots、tasks：

**internutopia_extension/__init__.py**:
```python
def import_extensions():
    import internutopia_extension.controllers
    import internutopia_extension.robots
    # ... 其他模块

    # 导入你的自定义扩展模块（如果需要注册到 InternUtopia）
    try:
        import my_extension.controllers  # 假设你的扩展有 controllers
        import my_extension.robots
    except ImportError as e:
        print(f"Custom extension not available: {e}")
```

## 方法 3：通过环境变量指定扩展路径

设置 Isaac Sim 的扩展搜索路径环境变量：

```bash
export ISAACLAB_EXTENSION_PATHS="/path/to/your/extensions:/another/path"

python -m internutopia_extension.envset.standalone \
  --config config.yaml \
  --envset scenario.json
```

然后在 `runner.py` 中读取环境变量：

```python
import os

def setup_isaacsim(self):
    # ...
    launch_config = { ... }

    # 从环境变量读取扩展路径
    ext_paths = os.environ.get('ISAACLAB_EXTENSION_PATHS')
    if ext_paths:
        launch_config['extension_folders'] = ext_paths.split(':')

    self._simulation_app = SimulationApp(launch_config)
```

## 方法 4：通过符号链接（简单但不推荐）

将你的扩展链接到 Isaac Sim 的扩展目录：

```bash
ln -s /path/to/your/extension \
  ~/.local/share/ov/pkg/isaac-sim-*/exts/my.custom.extension
```

然后在代码中启用：

```python
enable_extension("my.custom.extension")
```

## 集成示例：完整工作流

假设你有一个名为 `my_nav_extension` 的扩展，包含自定义的导航控制器。

### 1. 扩展结构

```
/home/user/my_nav_extension/
├── config/
│   └── extension.toml
└── my_nav_extension/
    ├── __init__.py
    ├── controllers/
    │   ├── __init__.py
    │   └── advanced_nav_controller.py
    └── configs/
        ├── __init__.py
        └── controllers.py
```

### 2. 控制器实现

**my_nav_extension/controllers/advanced_nav_controller.py**:
```python
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.articulation_action import ArticulationAction

@BaseController.register('AdvancedNavController')
class AdvancedNavController(BaseController):
    def __init__(self, config, robot, scene):
        super().__init__(config, robot, scene)
        # 你的初始化代码

    def forward(self, **kwargs) -> ArticulationAction:
        # 你的控制逻辑
        pass

    def action_to_control(self, action):
        return self.forward(...)
```

### 3. 配置文件

**config_with_nav_extension.yaml**:
```yaml
simulator:
  physics_dt: 0.00416666667
  rendering_dt: 0.00416666667
  headless: false
  extension_folders:
    - /home/user/my_nav_extension

env_num: 1

task_configs:
  - type: FiniteStepTask
    max_steps: 3000
    robots: []
```

### 4. 在 envset 中使用

**scenario.json**:
```json
{
  "scenarios": [{
    "id": "test_custom_nav",
    "scene": { ... },
    "robots": {
      "entries": [{
        "type": "jetbot",
        "control": {
          "mode": "custom_nav",
          "params": {
            "custom_param": 1.5
          }
        }
      }]
    }
  }]
}
```

### 5. 修改 task_adapter 支持自定义控制器

如果需要在 envset 中自动使用你的控制器，修改 `task_adapter.py`：

```python
def _build_robot_controllers(spec: RobotSpec, name: str):
    params = spec.control.params if spec.control else {}
    control_mode = (spec.control.mode or "").lower() if spec.control else ""

    # 检测自定义模式
    if "custom_nav" in control_mode:
        return [{
            "name": f"{name}_custom_nav",
            "type": "AdvancedNavController",
            "custom_param": params.get("custom_param", 1.0)
        }]

    # ... 其他标准控制器
```

### 6. 运行

```bash
python -m internutopia_extension.envset.standalone \
  --config config_with_nav_extension.yaml \
  --envset scenario.json
```

## 常见问题

### Q1: 扩展没有被加载？

**检查步骤**：
1. 确认 `extension.toml` 路径正确
2. 查看日志中的扩展加载信息
3. 检查扩展名称是否与 `extension.toml` 中的 `[package]` 匹配

### Q2: 扩展依赖缺失？

在 `extension.toml` 中添加所有依赖：

```toml
[dependencies]
"omni.usd" = {}
"omni.isaac.core" = {}
"isaacsim.core.utils" = {}
# 确保所有依赖都列出
```

### Q3: Python 模块导入失败？

确保在 `extension.toml` 中声明了 Python 模块：

```toml
[[python.module]]
name = "my_extension"
```

### Q4: 与现有代码冲突？

使用唯一的注册名称：

```python
@BaseController.register('MyCompany_AdvancedNavController')
class AdvancedNavController(BaseController):
    pass
```

## 最佳实践

1. **使用配置文件方式**：通过 YAML 的 `extension_folders` 指定路径，便于版本控制和部署

2. **保持扩展独立**：不要依赖 InternUtopia 内部实现细节，只使用公开的 API

3. **文档完善**：为你的扩展编写 README，说明如何集成和使用

4. **版本兼容**：在 `extension.toml` 中明确依赖版本，避免不兼容问题

5. **测试隔离**：创建独立的测试场景验证你的扩展功能

## 参考资源

- Isaac Sim Extension 文档: https://docs.omniverse.nvidia.com/isaacsim/latest/
- Omniverse Kit Extension 系统: https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/
- InternUtopia Extension 示例: `internutopia_extension/controllers/`, `internutopia_extension/robots/`
