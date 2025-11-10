__copyright__ = "Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
import asyncio
import os
import sys

import numpy as np
import omni.kit
import omni.kit.test
import omni.usd
from omni.metropolis.utils.unit_test import (
    MINIMAL_STAGE_URL,
    TestStage,
    StageSetupOptions,
)
from omni.metropolis.utils.carb_util import CarbUtil
from isaacsim.replicator.agent.core.settings import Infos
from isaacsim.replicator.agent.core.randomization.character_randomizer import CharacterRandomizer
from isaacsim.replicator.agent.core.randomization.randomizer import Randomizer

# This determines number of agent
TEST_ITERATION = 10


class TestRandomization(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_character_randomization(self):
        self.randomizer = CharacterRandomizer(0)
        async with TestStage(MINIMAL_STAGE_URL, StageSetupOptions.BAKE_NAV_MESH):
            _agent_pos_list = self._test_character_position()
            await self._test_character_command(_agent_pos_list)

    def _test_character_position(self):
        area_list = ["Walkable"]
        pos_list = [self.randomizer.get_random_position(i, area_list) for i in range(TEST_ITERATION)]

        # Test if the distance of any two agents are always greater than 1
        # Test if seed overflow is handled
        for seed in range(1 + sys.maxsize, TEST_ITERATION):
            randomizer = Randomizer(seed)
            pos_list_test = [randomizer.get_random_position(i, area_list) for i in range(TEST_ITERATION)]
            for i, pos_i in enumerate(pos_list_test):
                for j in range(i + 1, len(pos_list_test)):
                    self.assertEqual(CarbUtil.dist3(pos_i, pos_list_test[j]) > 1, True)

            # Test if a different seed results in a different result
            list_different = False
            for i, pos_i in enumerate(pos_list):
                if not CarbUtil.equal3(pos_i, pos_list_test[i]):
                    list_different = True
            self.assertEqual(list_different, True)

            # Disable the following tests until changes from https://jirasw.nvidia.com/browse/OMPE-32674 are applied.

            # Test if randomizers with the same seed produce the same result
            # randomizer.reset()
            # randomizer.update_seed(0)
            # pos_list_test = [randomizer.get_random_position(i) for i in range(TEST_ITERATION)]
            # list_different = False
            # for i in range(len(pos_list)):
            #     if not CarbUtil.equal3(pos_list[i], pos_list_test[i]):
            #         list_different = True
            # self.assertEqual(list_different, False)

            # Test if two seperate random position calls produce the same end result as one
            # pos_list_test = [randomizer.get_random_position(i) for i in range(TEST_ITERATION // 2)] + [
            #     randomizer.get_random_position(i) for i in range(TEST_ITERATION // 2, TEST_ITERATION)
            # ]
            # list_different = False
            # for i in range(len(pos_list)):
            #     if not CarbUtil.equal3(pos_list[i], pos_list_test[i]):
            #         list_different = True
            # self.assertEqual(list_different, False)

        return pos_list

    async def _test_character_command(self, pos_list):
        agent_list = {}
        command_duration = 1800
        area_list = ["Walkable"]

        # Load example transition map (convegence vector = [0.38456, 0.29746, 0.31796, 0])
        ext_path = Infos.ext_path
        file_path = os.path.join(ext_path, "test_data", "example_character_command_transition_map.json")
        self.randomizer.load_command_transition_map(file_path)

        for idx in range(TEST_ITERATION):
            agent_list["a" + str(idx)] = pos_list[idx]
        gather_result = await asyncio.gather(
            self.randomizer.generate_commands(0, command_duration, agent_list, area_list)
        )
        commands = gather_result[0]

        agent_idx = 0  # noqa
        goto_count = 0
        idle_count = 0
        lookaround_count = 0
        sit_count = 0

        # Seperate the commands for each agent
        agent_command_list = {}
        for command in commands:
            command = command.split()
            agent_name = command[0]
            if agent_name not in agent_command_list:
                agent_command_list[agent_name] = []
            agent_command_list[agent_name].append(command[1:])

        # Test each agent has enough commands to last for the given duration
        for agent in agent_command_list.keys():  # noqa
            commands = agent_command_list[agent]
            duration = 0
            for command in commands:
                # Get the count for each command for testing the random distribution
                if command[0] == "GoTo":
                    goto_count += 1
                elif command[0] == "Idle":
                    idle_count += 1
                elif command[0] == "LookAround":
                    lookaround_count += 1
                elif command[0] == "Sit":
                    sit_count += 1

                # Get the command duration
                if command[0] == "GoTo":
                    duration += np.linalg.norm(
                        pos_list[agent_idx][:2] - np.array([float(command[1]), float(command[2])])
                    )
                    pos_list[agent_idx] = [float(command[1]), float(command[2]), float(command[3])]
                else:
                    duration += float(command[1])

            print(f"{agent} calculated command duration = {duration} ")
            self.assertEqual(duration > command_duration, True)
            agent_idx += 1

        # Test whether the commands are generated according to the Markov Chain
        total_commands = goto_count + lookaround_count + idle_count + sit_count
        print(f"Goto calculated convegence = {goto_count / total_commands}")
        print(f"Idle calculated convegence = {idle_count / total_commands}")
        print(f"LookAround calculated convegence = {lookaround_count / total_commands}")
        print(f"Sit calculated convegence = {sit_count / total_commands}")
        self.assertEqual(abs(goto_count / total_commands - 0.38456) < 0.01, True)
        self.assertEqual(abs(idle_count / total_commands - 0.29746) < 0.01, True)
        self.assertEqual(abs(lookaround_count / total_commands - 0.31796) < 0.01, True)
        self.assertEqual(abs(sit_count / total_commands - 0) < 0.01, True)
