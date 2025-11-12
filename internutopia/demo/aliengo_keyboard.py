from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
import sys
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.robots.aliengo import (
    AliengoRobotCfg,
    move_by_speed_cfg,
)
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg
from internutopia_extension.interactions.keyboard import KeyboardInteraction


def main():
    # On Windows, prefer GUI (keyboard needs a window); on UNIX decide by DISPLAY
    headless = False if sys.platform.startswith('win') else not has_display()

    # Avoid WebRTC to keep compatibility across Isaac Sim 4.5/5.0 by default.
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            webrtc=False,
        ),
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path='/home/ubuntu/sunsz/IsaacAssets/Environments/Simple_Warehouse/full_warehouse.usd',
                robots=[
                    AliengoRobotCfg(
                        position=(0.0, 0.0, 1.05),
                        controllers=[move_by_speed_cfg],
                    )
                ],
            ),
        ],
    )

    import_extensions()

    env = Env(config)
    env.reset()

    keyboard = KeyboardInteraction()
    i = 0
    while env.simulation_app.is_running():
        i += 1
        # map I/K, J/L, U/O into (x, y, yaw) velocities
        command = keyboard.get_input()
        x_speed = float(command[0] - command[1])
        y_speed = float(command[2] - command[3])
        z_speed = float(command[4] - command[5])
        env_action = {move_by_speed_cfg.name: (x_speed, y_speed, z_speed)}
        env.step(action=env_action)

    env.close()


if __name__ == '__main__':
    main()
