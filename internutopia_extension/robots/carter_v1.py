# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from collections import OrderedDict
from typing import Optional

import numpy as np

from internutopia.core.robot.articulation_action import ArticulationAction
from internutopia.core.robot.isaacsim.articulation import IsaacsimArticulation
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log
from internutopia_extension.configs.robots.carter_v1 import CarterV1RobotCfg
from internutopia_extension.robots.jetbot import WheeledRobot


@BaseRobot.register('CarterV1Robot')
class CarterV1Robot(BaseRobot):
    def __init__(self, config: CarterV1RobotCfg, scene: IScene):
        super().__init__(config, scene)
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f'carter_v1 {config.name} position    : ' + str(self._start_position))
        log.debug(f'carter_v1 {config.name} orientation : ' + str(self._start_orientation))

        usd_path = config.usd_path

        log.debug(f'carter_v1 {config.name} usd_path         : ' + str(usd_path))
        log.debug(f'carter_v1 {config.name} config.prim_path : ' + str(config.prim_path))
        self.prim_path = str(config.prim_path)
        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)
        # Carter V1 uses 'left_wheel' and 'right_wheel' as joint names (not 'left_wheel_joint')
        self.articulation = WheeledRobot(
            prim_path=config.prim_path,
            name=config.name,
            wheel_dof_names=['left_wheel', 'right_wheel'],
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=usd_path,
            scale=self._robot_scale,
        )

    def get_robot_scale(self):
        return self._robot_scale

    def get_pose(self):
        return self.articulation.get_pose()

    def apply_action(self, action: dict):
        """
        Args:
            action (dict): inputs for controllers.
        """
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warning(f'unknown controller {controller_name} in action')
                continue
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self.articulation.apply_action(control)

    def get_obs(self) -> OrderedDict:
        position, orientation = self.articulation.get_pose()

        # custom
        obs = {
            'position': position,
            'orientation': orientation,
            'joint_positions': self.articulation.get_joint_positions(),
            'joint_velocities': self.articulation.get_joint_velocities(),
            'controllers': {},
            'sensors': {},
        }

        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs['controllers'][c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs['sensors'][sensor_name] = sensor_obs.get_data()
        return self._make_ordered(obs)

