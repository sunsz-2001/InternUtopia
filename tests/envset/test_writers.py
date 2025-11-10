__copyright__ = "Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
import os
import carb
import numpy as np
import omni.kit
import omni.kit.test
from isaacsim.replicator.agent.core.stage_util import CameraUtil, CharacterUtil
from omni.metropolis.utils.data_capture_util import CameraDataCaptureHelper
from isaacsim.replicator.agent.core.data_generation.object_info_manager import AgentInfo, ObjectInfo
from isaacsim.replicator.agent.core.data_generation.annotator_data_processor import AnnotatorDataProcessor
from typing import Dict, Any, List, Tuple, Optional
from isaacsim.replicator.agent.core.agent_manager import AgentManager
from isaacsim.core.utils.bounds import compute_obb_corners
from omni.metropolis.utils.usd_util import USDUtil
from functools import partial
import omni.usd
import copy
from pxr import Usd, UsdGeom
from omni.metropolis.utils.unit_test import (
    TestStage,
    ANNOTATED_BOX_URL,
    context_create_example_sim_manager,
    wait_for_simulation_set_up_done,
)
from isaacsim.core.utils import prims
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
class RawDataCapture:
    """capture the raw data from annotator"""

    _default_camera_resolution: Tuple[int, int] = (1920, 1080)
    _default_annotator_list: List[str] = [
        "rgb",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "camera_params",
    ]
    _default_width_theshold = 0.6
    _default_height_threshold = 0.6

    @classmethod
    def fetch_raw_annotator_data_fn(
        cls, camera_path: str, annotator_dict: Dict[str, Any], data_container: Dict[str, Dict[str, Any]]
    ):
        """fetch the single frame raw annotator data"""
        target_annotator_dict = data_container[camera_path]
        for annotator_name in annotator_dict.keys():
            # check whether the data is our target code
            if annotator_name not in target_annotator_dict:
                continue
            annotator_data = annotator_dict.get(annotator_name, None)
            # Check whether the target annotator is generated.
            if annotator_data is None:
                continue
            # Cache the annotator data, use the do_array_copy to do deep copy, so the data would not be deprecated/rewritten during the time.
            annotator_data_value = annotator_data.get_data()
            target_annotator_dict.update({annotator_name: copy.deepcopy(annotator_data_value)})

    # NOTE :: this part of the code is used to test the existing camera based data caption
    @classmethod
    async def check_camera_annotator_fetching(
        cls, annotator_name_list: List[str], loading_frame: Optional[int] = 10
    ) -> Dict[str, Dict[str, Any]]:
        """test whether the annotator fetching can work properly"""

        carb.log_warn("start test camera annotator fetching")
        # get camera prim lists in the stage
        camera_prim_list = CameraUtil.get_cameras_in_stage()
        camera_path_list = [str(camera_prim.GetPrimPath()) for camera_prim in camera_prim_list]
        # generate the data container base on the input.

        data_container: Dict[str, Dict[str, Any]] = {}
        for camera_path in camera_path_list:
            camera_data_container = {}
            for annotator in annotator_name_list:
                camera_data_container[annotator] = None
            data_container[camera_path] = camera_data_container

        post_processing_fn = partial(cls.fetch_raw_annotator_data_fn, data_container=data_container)
        await CameraDataCaptureHelper.capture_static_data_async(
            camera_path_list=camera_path_list,
            annotator_name_list=annotator_name_list,
            post_processing_fn=post_processing_fn,
            camera_resolution=cls._default_camera_resolution,
            loading_frame=loading_frame,
        )
        return data_container


class WriterDataChecker:
    """Helper methods for checking annotator data"""

    object_detection_test_scene_name = "test_ira_writer.usd"
    _test_scene_folder = "data/test_scenes"

    @staticmethod
    def compare_mae(array_a: np.ndarray, array_b: np.ndarray) -> float:
        """compare two different np array"""
        assert array_a.shape == array_b.shape, f"Shape mismatch: {array_a.shape} vs {array_b.shape}"
        return np.mean(np.abs(array_a.astype(np.float32) - array_b.astype(np.float32)))

    @staticmethod
    def highlight_and_get_bbox(instance_seg_data, target_id: int):
        """given target instance id and instance segmentation array, extract the tight bbox information"""
        instance_seg_data = np.array(instance_seg_data, dtype=np.uint32)

        # Create mask for the target_id
        mask = instance_seg_data == target_id

        # If target_id is not present at all
        if not np.any(mask):
            return None

        # Get coordinates where mask is True
        y_coords, x_coords = np.where(mask)
        # Compute bounding box
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        return ((x_min, x_max), (y_min, y_max))

    @staticmethod
    def check_overlap_ratio(
        loose_box_scope: List[Tuple[int, int]],
        tight_box_scope: List[Tuple[int, int]],
        width_threshold: float = 0.5,
        height_threshold: float = 0.5,
    ):
        """calculate the overlap ratio of two bbox"""
        # Unpack coordinates
        (x_min_l, x_max_l), (y_min_l, y_max_l) = loose_box_scope
        (x_min_t, x_max_t), (y_min_t, y_max_t) = tight_box_scope

        # Calculate box widths and heights
        loose_width = x_max_l - x_min_l
        loose_height = y_max_l - y_min_l

        # Calculate overlap region
        overlap_x_min = max(x_min_l, x_min_t)
        overlap_x_max = min(x_max_l, x_max_t)
        overlap_y_min = max(y_min_l, y_min_t)
        overlap_y_max = min(y_max_l, y_max_t)

        # Overlap dimensions
        overlap_width = max(0, overlap_x_max - overlap_x_min)
        overlap_height = max(0, overlap_y_max - overlap_y_min)

        # Avoid divide by zero
        if loose_width == 0 or loose_height == 0:
            return False, False

        width_ratio = overlap_width / loose_width
        height_ratio = overlap_height / loose_height

        # check the horizon and verteix occlusion  threshold

        width_check = width_ratio >= width_threshold
        height_check = height_ratio >= height_threshold

        return width_check, height_check

    @staticmethod
    def compare_bbox_vertices_np_only(calculated_vertices, ground_truth_vertices):
        """
        Compare two sets of 3D bounding box vertices (unordered) using NumPy only.
        Greedily matches nearest vertices and computes distances.

        Args:
            calculated_vertices (np.ndarray): (8, 3) array of predicted bbox vertices.
            ground_truth_vertices (np.ndarray): (8, 3) array of ground truth bbox vertices.

        Returns:
            dict: {
                "average_distance": float,
                "max_distance": float,
                "matched_pairs": List[Tuple[int, int]],
                "distance_per_pair": List[float]
            }
        """
        calculated = np.asarray(calculated_vertices).reshape(-1, 3)
        ground_truth = np.asarray(ground_truth_vertices).reshape(-1, 3)

        if calculated.shape != (8, 3) or ground_truth.shape != (8, 3):
            raise ValueError("Expected 8x3 vertex arrays")

        used_gt_indices = set()
        matched_pairs = []
        distance_per_pair = []

        for i, calc_point in enumerate(calculated):
            # Compute distances to all ground truth points not yet matched
            dists = np.linalg.norm(ground_truth - calc_point, axis=1)
            for j in sorted(used_gt_indices):
                dists[j] = np.inf  # Mask out already matched ground truth points

            # Find the closest unmatched ground truth point
            best_match = np.argmin(dists)
            matched_pairs.append((i, best_match))
            distance_per_pair.append(dists[best_match])
            used_gt_indices.add(best_match)

        avg_dist = np.mean(distance_per_pair)
        max_dist = np.max(distance_per_pair)

        return {
            "average_distance": avg_dist,
            "max_distance": max_dist,
            "matched_pairs": matched_pairs,
            "distance_per_pair": distance_per_pair,
        }

    @staticmethod
    def get_test_stage_path(test_scene_file_name: str):
        """fetch the test stage url path"""
        EXT_PATH = (
            omni.kit.app.get_app()
            .get_extension_manager()
            .get_extension_path_by_module("isaacsim.replicator.agent.core")
        )
        test_scene_file_path = os.path.join(EXT_PATH, WriterDataChecker._test_scene_folder, test_scene_file_name)
        return test_scene_file_path

class TestIRAWriters(omni.kit.test.AsyncTestCase):
    _class_name = "TestIRAWriters"

    # Before running each test
    async def setUp(self):
        pass

    # After running each test
    async def tearDown(self):
        pass

    async def test_irabasicwriter_3d_bbox(self):
        """generate the ground truth data for 3d bbox, compare the data with the ground truth"""
        # check out the 3d bbox of the objects in the stage, compare with the extra

        # get the test stage url path
        test_scene_file_name = WriterDataChecker.object_detection_test_scene_name
        test_scene_file_path = WriterDataChecker.get_test_stage_path(test_scene_file_name=test_scene_file_name)
        # load the stage and set up cameras
        async with TestStage(stage_path=test_scene_file_path):
            with context_create_example_sim_manager() as sim:
                prop = sim.get_config_file_property("character", "num")
                prop.set_value(0)
                prop_group = sim.get_config_file_property_group("sensor", "camera_group")
                prop_group.get_property("camera_num").set_value(3)
                sim.set_up_simulation_from_config_file()
                await wait_for_simulation_set_up_done(sim)
                # test the ira basic writer's bbox fetching

                stage = omni.usd.get_context().get_stage()
                annotator_name_list = ["bounding_box_3d"]
                # match camera path to the target bbox information captured in the scene.
                camera_path_to_bbox = await RawDataCapture.check_camera_annotator_fetching(
                    loading_frame=5, annotator_name_list=annotator_name_list
                )
                # test the post process function of the basic writer.
                # default writer do not have post process, compare the transform directly.
                camera_prim_list = CameraUtil.get_cameras_in_stage()
                camera_path_list = [str(camera_prim.GetPrimPath()) for camera_prim in camera_prim_list]
                for camera_path in camera_path_list:
                    bbox_3d_anno = camera_path_to_bbox[camera_path]["bounding_box_3d"]
                    bbox_3d_info = bbox_3d_anno["info"]
                    bbox_3d_data = bbox_3d_anno["data"]

                    prim_path_list = bbox_3d_info["primPaths"]

                    for i in range(len(prim_path_list)):
                        # there is knowned issue related to the path
                        # therefore the corners of the bbox are calculated and compared
                        bbox_data = bbox_3d_data[i]
                        prim_path = prim_path_list[i]
                        # get target object prim path
                        target_prim = stage.GetPrimAtPath(prim_path)
                        # fetch the ground truth object transform
                        anno_transform = np.reshape(bbox_data["transform"], (4, 4))
                        ground_truth_transform = np.reshape(omni.usd.get_world_transform_matrix(target_prim), (4, 4))
                        # compute the difference between the captured data and the ground truth.
                        diff = WriterDataChecker.compare_mae(array_a=anno_transform, array_b=ground_truth_transform)
                        self.assertLessEqual(diff, 0.2)

    async def test_tao_writer_3d_bbox_filter(self):
        """test whether the tao writer post processing can calculate correct 3d corner values"""
        # get the test stage url path
        test_scene_file_name = WriterDataChecker.object_detection_test_scene_name
        test_scene_file_path = WriterDataChecker.get_test_stage_path(test_scene_file_name=test_scene_file_name)
        # load the stage and set up cameras
        # Get original focusing filter setting:
        original_focusing_label_setting  = SyntheticData.Get().get_instance_mapping_semantic_filter()
        default_focusing_label_setting  = "*:*"
        SyntheticData.Get().set_instance_mapping_semantic_filter(default_focusing_label_setting)

        carb.log_warn("start the test")

        async with TestStage(stage_path=test_scene_file_path):
            with context_create_example_sim_manager() as sim:
                prop = sim.get_config_file_property("character", "num")
                prop.set_value(0)
                prop_group = sim.get_config_file_property_group("sensor", "camera_group")
                prop_group.get_property("camera_num").set_value(3)
                sim.set_up_simulation_from_config_file()
                await wait_for_simulation_set_up_done(sim)
                # test the ira basic writer's bbox fetching

                carb.log_warn("scene finish loading")
                stage = omni.usd.get_context().get_stage()

                annotator_name_list = [
                    "camera_params",
                    "bounding_box_3d_fast",
                    "bounding_box_2d_tight_fast",
                    "bounding_box_2d_loose_fast",
                    # "instance_segmentation_fast",
                ]

                # match camera path to the target bbox information captured in the scene.
                camera_path_to_annotators = await RawDataCapture.check_camera_annotator_fetching(
                    loading_frame=30, annotator_name_list=annotator_name_list
                )

                carb.log_warn("fetched the target annotators")
                # test the post process function of the basic writer.
                # default writer do not have post process, compare the transform directly.
                camera_prim_list = CameraUtil.get_cameras_in_stage()
                camera_path_list = [str(camera_prim.GetPrimPath()) for camera_prim in camera_prim_list]

                for camera_path in camera_path_list:
                    # create a sample object info container to store the objects
                    info_dict: Dict[str, ObjectInfo] = {}
                    occlusion_result_dict: Dict[str, bool] = {}
                    if camera_path not in camera_path_to_annotators.keys():
                        continue
                    target_camera_annotators = camera_path_to_annotators[camera_path]
                    carb.log_warn("successfully fetched camera annotator list" + str(target_camera_annotators.keys()))
                    # attempt to fetch the camera parameter
                    camera_params = target_camera_annotators.get("camera_params", None)
                    if camera_params is None:
                        # fail to fetch the camera parameter parameter for target camera
                        carb.log_error(f"Fail to fetch camera parameter for target camera {camera_path}")
                        continue
                    # fetch the instance segementation annotator as the ground truth
                    # instance_segmentation_anno = target_camera_annotators.get("instance_segmentation_fast", None)
                    # if instance_segmentation_anno is None:
                    #     # fail to fetch the instance segementation anno
                    #     carb.log_error(f"Fail to fetch the instance segmentation anno for target camera {camera_path}")
                    #     continue

                    object_annotator_list = [
                        "bounding_box_3d_fast",
                        "bounding_box_2d_tight_fast",
                        "bounding_box_2d_loose_fast",
                    ]

                    check_object_info = True
                    for object_annotator in object_annotator_list:
                        if not object_annotator in target_camera_annotators.keys():
                            carb.log_error(f"fail to fetch the {object_annotator}")
                            check_object_info = False

                    if not check_object_info:
                        continue

                    ## Test the data recorder.
                    for annotator_name in object_annotator_list:
                        annotator = target_camera_annotators[annotator_name]
                        # extract annotator info
                        annotator_info = annotator["info"]
                        carb.log_warn(f"This current{annotator_name} 's data structure {annotator.keys()}")
                        carb.log_warn(f"This is current info structure {annotator_info.keys()}")
                        object_prim_paths = annotator_info["primPaths"]
                        id_to_labels = annotator_info["idToLabels"]
                        # attempts to normailize the key to int from the char format.
                        id_to_labels = {int(k): v for k, v in id_to_labels.items()}
                        # extract annotator data
                        annotator_data = annotator["data"]
                        for idx, prim_path in enumerate(object_prim_paths):
                            object_data = annotator_data[idx]
                            semantic_id = object_data["semanticId"]
                            label = id_to_labels[semantic_id]
                            # Determine the appropriate info dictionary (agent or object)
                            if prim_path not in info_dict:
                                # store the object info in the cache
                                object_info = ObjectInfo(label=label, prim_path=prim_path)
                                info_dict[prim_path] = object_info
                            info_dict[prim_path].update_annotator_data(annotator_name, object_data)

                    # after the object info has been fully recorded:
                    # fetch the default occlusion ratio.
                    width_threshold, height_theshold = (
                        RawDataCapture._default_width_theshold,
                        RawDataCapture._default_height_threshold,
                    )
                    # fetch the bbox cache of the existing objects in the stage
                    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])

                    carb.log_warn("Fetch the bbox cache")

                    for object_prim_path, object_info in info_dict.items():
                        # check whether the point's visualization meet the threshold value.
                        threshold_check_result = AnnotatorDataProcessor.validate_and_process_object(
                            object_info=object_info,
                            camera_params=camera_params,
                            width_threshold=width_threshold,
                            height_threshold=height_theshold,
                        )

                        occlusion_result_dict[object_prim_path] = threshold_check_result
                        if not threshold_check_result:
                            continue
                        # Then compare the calculated 3d bbox corner's location with the ground truth values
                        box_3d_information = object_info.get_annotator_info("bounding_box_3d_fast")
                        carb.log_warn("This is the box 3d information" + str(box_3d_information.keys()))
                        calculated_bbox_vertexs = box_3d_information["vertex"]["translations_3d"]
                        ground_truth_corners = compute_obb_corners(bbox_cache=bbox_cache, prim_path=object_prim_path)

                        diff_info = WriterDataChecker.compare_bbox_vertices_np_only(
                            calculated_vertices=calculated_bbox_vertexs, ground_truth_vertices=ground_truth_corners
                        )
                        avg_dist = diff_info["average_distance"]
                        max_dist = diff_info["max_distance"]
                        # scene the test stage is fixed. This step can help user distinguish which object has issue.
                        if avg_dist > 0.3 or max_dist > 0.5:
                            carb.log_error(
                                f"The distance between {object_info.get_prim_path()} 's calculated bbox vertexs and ground truth vertes is too large. Average: {avg_dist}. Max: {max_dist} "
                            )

                        if max_dist > 0.5 or avg_dist > 0.3:
                            carb.log_error(
                                f"Target prim {object_prim_path} has mistake larger than the expectation MAX_Dist{max_dist} and AVG Dist {avg_dist}"
                            )
                        self.assertLessEqual(avg_dist, 0.3)
                        self.assertLessEqual(max_dist, 0.5)

        # recover the filter type back to the original value
        SyntheticData.Get().set_instance_mapping_semantic_filter(original_focusing_label_setting)

    async def test_nested_agent_and_object(self):
        """test the nested placed agent and objects, writer should still distinguish objects and agents correctly"""
        # get the test stage url path
        test_scene_file_name = WriterDataChecker.object_detection_test_scene_name
        test_scene_file_path = WriterDataChecker.get_test_stage_path(test_scene_file_name=test_scene_file_name)
        # load the stage and set up cameras

        # fetch the prim path of the test object
        async with TestStage(stage_path=test_scene_file_path):
            with context_create_example_sim_manager() as sim:
                prop = sim.get_config_file_property("character", "num")
                prop.set_value(1)
                prop_group = sim.get_config_file_property_group("sensor", "camera_group")
                prop_group.get_property("camera_num").set_value(3)
                sim.set_up_simulation_from_config_file()
                await wait_for_simulation_set_up_done(sim)
                # test the ira basic writer's bbox fetching
                stage = omni.usd.get_context().get_stage()
                # Then add the object to the stage under the first character
                character_path = str(CharacterUtil.get_characters_root_in_stage()[0].GetPrimPath())
                test_object_path = omni.usd.get_stage_next_free_path(
                    stage,
                    f"{character_path}/test_object",
                    False,
                )

                test_object_prim = prims.create_prim(test_object_path, "Xform", usd_path=ANNOTATED_BOX_URL)
                self.assertTrue(USDUtil.is_valid_prim(test_object_prim))

                annotator_name_list = [
                    "bounding_box_3d_fast",
                    "bounding_box_2d_tight_fast",
                    "bounding_box_2d_loose_fast",
                ]
                # match camera path to the target bbox information captured in the scene.
                camera_path_to_annotators = await RawDataCapture.check_camera_annotator_fetching(
                    loading_frame=10, annotator_name_list=annotator_name_list
                )
                # test the post process function of the basic writer.
                # default writer do not have post process, compare the transform directly.
                camera_prim_list = CameraUtil.get_cameras_in_stage()
                camera_path_list = [str(camera_prim.GetPrimPath()) for camera_prim in camera_prim_list]
                agent_manager = AgentManager.get_instance()

                for camera_path in camera_path_list:
                    # create a sample object info container to store the objects
                    info_dict: Dict[str, ObjectInfo] = {}
                    target_camera_annotators = camera_path_to_annotators[camera_path]
                    # fetch the instance segementation annotator as the ground truth
                    object_annotator_list = [
                        "bounding_box_3d_fast",
                        "bounding_box_2d_tight_fast",
                        "bounding_box_2d_loose_fast",
                    ]

                    for annotator_name in object_annotator_list:
                        agent_list = []
                        annotator = target_camera_annotators[annotator_name]
                        # extract annotator info
                        annotator_info = annotator["info"]
                        object_prim_paths = annotator_info["primPaths"]
                        id_to_labels = annotator_info["idToLabels"]
                        # attempts to normailize the key to int from the char format.
                        id_to_labels = {int(k): v for k, v in id_to_labels.items()}
                        # extract annotator data
                        for idx, prim_path in enumerate(object_prim_paths):
                            # Determine the appropriate info dictionary (agent or object)
                            if prim_path not in info_dict:
                                # store the object info in the cache
                                if agent_manager.is_agent_semantic_prim_path(prim_path):
                                    agent_list.append(prim_path)

                        self.assertLessEqual(len(agent_list), 1)

    async def test_tao_writer_character_threshold(self):
        """test whether the character threshold value is calculated properly"""
        # TODO:: implement the test senario helper
        pass
