__copyright__ = "Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import asyncio
import carb
import omni.kit
import omni.kit.test
import omni.usd
import omni.timeline
from omni.metropolis.utils.unit_test import (
    context_create_example_sim_manager,
    wait_for_simulation_set_up_done,
)
from omni.metropolis.utils.unit_test.stage import TestStage
from omni.anim.people.scripts.character_behavior import CharacterBehavior, COMMAND_CALLBCAK_CHECKPOINT
from isaacsim.replicator.agent.core.agent_manager import AgentManager
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from isaacsim.replicator.agent.core.response.core import ResponsePickAgent, CommandResponse, AgentResponseManager


class TestResponse(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_command_response(self):
        """
        Test if CommandReponses can be tracked in AgentResponseManager and if commands are passed correctly.
        """
        with context_create_example_sim_manager() as sim:
            # Set up sim with 2 characters
            prop = sim.get_config_file_property("character", "num")
            prop.set_value(2)
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)

            all_agents = [CharacterUtil.get_character_name_by_index(0), CharacterUtil.get_character_name_by_index(1)]

            # Assign some init commands
            sim.save_commands(
                [
                    f"{all_agents[0]} Idle 1000",
                    f"{all_agents[0]} Idle 1100",
                    f"{all_agents[1]} Idle 1000",
                    f"{all_agents[1]} Idle 1100",
                ]
            )

            agent_manager = AgentManager.get_instance()
            response_manager = AgentResponseManager.get_instance()

            # Start timeline
            timeline = omni.timeline.get_timeline_interface()
            while not timeline.is_playing():
                timeline.play()
                await omni.kit.app.get_app().next_update_async()

            # Wait a few frames
            for i in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Trigger the frist response
            response_01 = CommandResponse(
                name="test response 01",
                priority=1,
                pick_agent=ResponsePickAgent.FIRST_AVAILABLE,
                resume=False,
                position=carb.Float3(0, 0, 0),
                commands=["Idle 2000", "LookAround 2000"],
            )
            response_manager.trigger_response(response_01)

            # Wait a few frames
            for i in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Test if AgentRespondManger has tracked the first response
            self.assertEqual(len(response_manager.active_response_agent_dict), 1)
            self.assertTrue(response_01 in response_manager.active_response_agent_dict.keys())

            # Test if pickd agent has been tracked
            response_01_agent_list = response_manager.active_response_agent_dict[response_01]
            self.assertEqual(len(response_01_agent_list), 1)

            agent = response_01_agent_list[0]
            other_agent = all_agents[0] if agent == all_agents[1] else all_agents[1]

            script_instance: CharacterBehavior = agent_manager.get_agent_script_instance_by_name(agent)
            other_script_instance: CharacterBehavior = agent_manager.get_agent_script_instance_by_name(other_agent)

            # Test if commands has been replaced for first agent
            self.assertEqual(len(script_instance.commands), 3)  # response_01.resume = False
            _, command = script_instance.commands[0]
            self.assertListEqual(command, ["Idle", "2000"])
            _, command = script_instance.commands[1]
            self.assertListEqual(command, ["LookAround", "2000"])
            command_id, command = script_instance.commands[2]
            self.assertEqual(command_id, response_01.name)
            self.assertListEqual(command, [COMMAND_CALLBCAK_CHECKPOINT])

            # Wait a few frames
            for i in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Trigger second response
            response_02 = CommandResponse(
                name="test response 02",
                priority=10,
                pick_agent=ResponsePickAgent.ALL,
                resume=True,
                position=carb.Float3(3, 0, 0),
                commands=["LookAround 3000"],
            )
            response_manager.trigger_response(response_02)

            # Wait a few more frames
            for i in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Test if AgentRespondManger has tracked the second response
            self.assertEqual(len(response_manager.active_response_agent_dict), 2)
            self.assertTrue(response_02 in response_manager.active_response_agent_dict.keys())

            # Test if agents are assigned correctly (pick_agent = all, priority = 10)
            response_02_agent_list = response_manager.active_response_agent_dict[response_02]
            self.assertEqual(len(response_02_agent_list), 2)
            self.assertTrue(agent in response_02_agent_list)
            self.assertTrue(other_agent in response_02_agent_list)

            # Test if first-response agent has expected commands
            self.assertEqual(len(script_instance.commands), 4)
            _, command = script_instance.commands[0]
            self.assertListEqual(command, ["LookAround", "3000"])
            command_id, command = script_instance.commands[1]
            self.assertTrue(command_id, response_02.name)
            self.assertListEqual(command, [COMMAND_CALLBCAK_CHECKPOINT])
            _, command = script_instance.commands[2]
            self.assertListEqual(command, ["LookAround", "2000"])  # response_01.resume = True
            command_id, command = script_instance.commands[3]
            self.assertTrue(command_id, response_01.name)
            self.assertListEqual(command, [COMMAND_CALLBCAK_CHECKPOINT])

            # Test if second-response agent has expected commands
            self.assertEqual(len(other_script_instance.commands), 3)
            _, command = other_script_instance.commands[0]
            self.assertListEqual(command, ["LookAround", "3000"])
            command_id, command = other_script_instance.commands[1]
            self.assertTrue(command_id, response_02.name)
            self.assertListEqual(command, [COMMAND_CALLBCAK_CHECKPOINT])
            _, command = other_script_instance.commands[2]
            self.assertListEqual(command, ["Idle", "1100"])

            timeline.stop()

    async def test_response_callback(self):
        """
        Test if CommandReponses can be cleared when commands finish execution.
        """
        with context_create_example_sim_manager() as sim:
            # Set up sim with 1 character
            prop = sim.get_config_file_property("character", "num")
            prop.set_value(1)
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)

            agent = CharacterUtil.get_character_name_by_index(0)

            # Start timeline
            timeline = omni.timeline.get_timeline_interface()
            while not timeline.is_playing():
                timeline.play()
                await omni.kit.app.get_app().next_update_async()

            # Wait a few frames
            for i in range(10):
                await omni.kit.app.get_app().next_update_async()

            agent_manager = AgentManager.get_instance()
            response_manager = AgentResponseManager.get_instance()

            # Trigger a short response
            response = CommandResponse(
                name="test response callback",
                priority=1,
                pick_agent=ResponsePickAgent.FIRST_AVAILABLE,
                resume=False,
                position=carb.Float3(0, 0, 0),
                commands=["LookAround 3"],
            )
            response_manager.trigger_response(response)

            # Wait a few frames
            for i in range(10):
                await omni.kit.app.get_app().next_update_async()

            self.assertEqual(len(response_manager.active_response_agent_dict), 1)

            # Wait with a timeout until response
            sleep = 3
            time = 0
            timeout = 30
            while not len(response_manager.active_response_agent_dict) == 0 and time < timeout:
                await asyncio.sleep(sleep)
                print("Wait for response finish timeout: {0}/{1}".format(time, timeout))
                time = time + sleep

            self.assertEqual(len(response_manager.active_response_agent_dict), 0)

            timeline.stop()
