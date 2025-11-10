__copyright__ = "Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import unittest
from pathlib import Path

import carb.settings
import numpy as np
import omni.kit
import omni.kit.test
import omni.usd
import AnimGraphSchema
from omni.metropolis.utils.unit_test import *
from omni.kit.scripting.scripts.script_manager import ScriptManager
from isaacsim.replicator.agent.core.agent_manager import AgentManager
from isaacsim.replicator.agent.core.settings import BehaviorScriptPaths
from isaacsim.replicator.agent.core.stage_util import CharacterUtil, RobotUtil

CHARACTERS_PARENT_PRIM_PATH = "/exts/isaacsim.replicator.agent/characters_parent_prim_path"
ROBOTS_PARENT_PRIM_PATH = "/exts/isaacsim.replicator.agent/robots_parent_prim_path"


class TestCommandInjection(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    def fetch_command_name(self, command):
        """handle the updated character ID system"""
        if isinstance(command, tuple) and len(command) == 2:
            _, test_command = command
        else:
            test_command = command
        return str(test_command)

    # ======= Test Command Injection========
    async def test_command_injection(self):
        """
        Test if the Inject Command feature is indeed injecting commands into the agent's behavior script command list
        """
        # On an empty scene
        with context_create_example_sim_manager() as sim:
            # load 2 characters and 2 robots
            prop = sim.get_config_file_property("character", "num")
            prop.set_value(2)
            prop = sim.get_config_file_property("robot", "nova_carter_num")
            prop.set_value(2)
            # Wait for simulation set up done
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)

            # Agent manager controls the agent command list
            # Script manager controls all the BehaviorScript instance
            agent_manager = AgentManager.get_instance()
            script_manager = ScriptManager.get_instance()

            """Check whether the characters are successfully spawned and have the right behavior script and anim graph attached"""
            skelroot_list = CharacterUtil.get_characters_in_stage()
            self.assertEqual(len(skelroot_list), 2)
            character_script_path = BehaviorScriptPaths.behavior_script_path()
            default_biped = CharacterUtil.get_default_biped_character()
            anim_graph_path = CharacterUtil.get_anim_graph_from_character(default_biped).GetPrimPath()
            for skelroot in skelroot_list:
                attr = skelroot.GetAttribute("omni:scripting:scripts").Get()
                self.assertEqual(attr[0].path, character_script_path)
                anim_graph_ref = AnimGraphSchema.AnimationGraphAPI(skelroot).GetAnimationGraphRel()
                self.assertEqual(anim_graph_ref.GetTargets()[0], anim_graph_path)

            """Check whether the robots are successfully spawned and have the right behavior script attached"""
            robot_list = RobotUtil.get_robots_in_stage()
            self.assertEqual(len(robot_list), 2)
            robot_script_path = BehaviorScriptPaths.robot_behavior_script_path("nova_carter")
            for robot in robot_list:
                attr = robot.GetAttribute("omni:scripting:scripts").Get()
                self.assertEqual(attr[0].path, robot_script_path)

            # Need to wait 2 cycles for the script to be registered
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()
            # There should be 4 BehaviorScript instance
            self.assertEqual(len(script_manager._prim_to_scripts.keys()), 4)

            for scripts in script_manager._prim_to_scripts.values():
                for _, inst in scripts.items():
                    if inst:
                        if hasattr(inst, "init_character"):
                            # setup characters
                            inst.init_character()
                        else:
                            # setup robots
                            inst.on_play()

                        # # Manually register the agents to the AgentManager
                        agent_name = inst.get_agent_name()
                        agent_path = inst.prim_path
                        agent_manager.register_agent(agent_name, agent_path)

            await omni.kit.app.get_app().next_update_async()

            """Test whether all agents in stage are registered to AgentManager"""
            character_prim_list = CharacterUtil.get_characters_root_in_stage()
            for character_prim in character_prim_list:
                character_name = character_prim.GetName()
                result = agent_manager.agent_registered(str(character_name))
                self.assertTrue(result)
            robot_prim_list = RobotUtil.get_robots_in_stage()
            for robot_prim in robot_prim_list:
                robot_name = robot_prim.GetName()
                result = agent_manager.agent_registered(str(robot_name))
                self.assertTrue(result)

            # Insert commands to the agents
            agent_manager.inject_command_for_all_agents(
                [
                    "Nova_Carter Idle 1",
                    "Nova_Carter GoTo 0 0 0",
                    "Nova_Carter_01 Idle 2",
                    "Nova_Carter_01 GoTo 0 0 0",
                    "Character Idle 3",
                    "Character GoTo 0 0 0 0",
                    "Character_01 Idle 4",
                    "Character_01 GoTo 0 0 0 0",
                ],
                True,
            )

            """ Test whether each agent is injected with an Idle command and a GoTo command"""
            for scripts in script_manager._prim_to_scripts.values():
                for _, inst in scripts.items():
                    if inst:
                        self.assertTrue(self.fetch_command_name(inst.commands[0]).find("Idle") != -1)
                        self.assertTrue(self.fetch_command_name(inst.commands[1]).find("GoTo") != -1)
