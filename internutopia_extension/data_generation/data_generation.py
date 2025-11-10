import asyncio

from pxr import Usd
import carb
import carb.events
import omni.kit.ui
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils import prims
from omni.metropolis.utils.config_file.core import ConfigFile
from internutopia_extension.envset.settings import Settings
from internutopia_extension.envset.stage_util import CameraUtil, LidarCamUtil, RobotUtil
from omni.metropolis.utils.debug_util import DebugPrint

FRAME_RATE = 30

dp = DebugPrint(Settings.DEBUG_PRINT, "DataGeneration")


class DataGeneration:
    """
    Class to handle all components of data generation - creating render products, registering writers,
    starting and stopping data generation.
    """

    def __init__(self, config_file: ConfigFile = None):
        self.writer_name = ""
        self.writer_params = {}
        self._writer = None
        self._write_robot_data = False
        self._num_frames = 0
        self._data_generation_done_callback = []

        self._num_cameras = 0
        # self._num_lidars = 0
        self._render_products = []
        self._camera_list = []
        # self._lidar_list = []
        self._camera_path_list = []
        # self._lidar_path_list = []

        # Viewport Information
        self._show_grid = None
        self._show_outline = None
        self._show_navMesh = None
        self._show_camera = None
        self._show_light = None
        self._show_audio = None
        self._show_skeleton = None
        self._show_meshes = None

        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: ConfigFile):
        self.config_file = config_file
        # Global
        prop = config_file.get_property("global", "simulation_length")
        self._num_frames = prop.get_resolved_value()
        self._num_frames += Settings.extend_data_generation_length()
        # Sensor
        camera_group: OrPropertyGroup = config_file.get_property_group("sensor", "camera_group")
        if camera_group.get_mode() == 0:
            self._num_cameras = camera_group.get_property("camera_num").get_resolved_value()
            # self._num_lidars = camera_group.get_property("lidar_num").get_resolved_value()
        else:
            self._camera_path_list = camera_group.get_property("camera_list").get_resolved_value()
            # self._lidar_path_list = camera_group.get_property("lidar_list").get_resolved_value()
        # Replicator
        writer_selection_group: SelectionPropertyGroup = config_file.get_property_group(
            "replicator", "writer_selection"
        )
        self.writer_name = writer_selection_group.selection_prop.get_resolved_value()
        self.writer_params = writer_selection_group.content_prop.get_resolved_value()

        self._write_robot_data = config_file.get_property("robot", "write_data").get_resolved_value()

    def register_recorder_done_callback(self, fn: callable):
        self._data_generation_done_callback.append(fn)
 
    # async def run_async(self, will_wait_until_complete):
    #     """简化版本的数据生成运行方法"""
        
    #     # 初始化recorder - 修复布尔比较写法
    #     if not self._init_recorder():
    #         carb.log_error("Init recorder fails.")
    #         self._clear_recorder()
    #         return

    #     # 计算总帧数 - 移除冗余检查
    #     skip_frames = carb.settings.get_settings().get(
    #         "/persistent/exts/isaacsim.replicator.agent/skip_starting_frames"
    #     ) or 0
        
    #     # 确保_num_frames有效，简化验证逻辑
    #     if self._num_frames is None or self._num_frames <= 0:
    #         self._num_frames = 1
        
    #     total_frames = self._num_frames + skip_frames
    #     carb.log_info(f"Data generation starting with {total_frames} total frames")

    #     # 获取timeline接口并保存原始状态
    #     timeline = omni.timeline.get_timeline_interface()
    #     original_timecode = timeline.get_time_codes_per_second()
        
    #     # 验证timecode有效性，但不手动修复Stage元数据
    #     if original_timecode is None or original_timecode <= 0:
    #         carb.log_warn(f"Invalid timecode detected: {original_timecode}, using default 30 fps")
    #         original_timecode = 30.0

    #     try:
    #         # 设置timeline为30fps并清理之前状态
    #         timeline.set_time_codes_per_second(30)
    #         timeline.commit_silently()
    #         await omni.kit.app.get_app().next_update_async()

    #         # 清理并启动Replicator
    #         rep.orchestrator.stop()
    #         rep.orchestrator.set_capture_on_play(True)
    #         await rep.orchestrator.run_async(num_frames=total_frames, start_timeline=True)

    #         # 可选等待完成
    #         if will_wait_until_complete:
    #             await rep.orchestrator.wait_until_complete_async()

    #     finally:
    #         # 确保清理工作在任何情况下都能执行
    #         rep.orchestrator.stop()
    #         rep.orchestrator.set_capture_on_play(False)
            
    #         # 恢复原始timeline设置 - 复用同一个timeline实例
    #         timeline.set_time_codes_per_second(original_timecode)
    #         timeline.commit_silently()
            
    #         # 只在最后等待一次更新确保状态同步
    #         await omni.kit.app.get_app().next_update_async()
            
    #         self._clear_recorder()
            
    async def run_async(self, will_wait_until_complete):
        """简化版本的数据生成运行方法"""
        
        # 初始化recorder - 修复布尔比较写法
        if self._init_recorder() == False:
            carb.log_error("Init recorder fails.")
            self._clear_recorder()
            return

        # 计算总帧数 - 移除冗余检查
        skip_frames = carb.settings.get_settings().get(
            "/persistent/exts/isaacsim.replicator.agent/skip_starting_frames"
        )
        total_frames = self._num_frames + skip_frames
        
        # 确保_num_frames有效，简化验证逻辑
        if self._num_frames is None or self._num_frames <= 0:
            self._num_frames = 1
        
        total_frames = self._num_frames + skip_frames
        carb.log_info(f"Data generation starting with {total_frames} total frames")

        # 获取timeline接口并保存原始状态
        timeline = omni.timeline.get_timeline_interface()
        original_timecode = timeline.get_time_codes_per_second()

        # 设置timeline为30fps并清理之前状态
        timeline.set_time_codes_per_second(30)
        timeline.commit_silently()
        await omni.kit.app.get_app().next_update_async()

        rep.orchestrator.set_capture_on_play(True)
        await omni.kit.app.get_app().next_update_async()
        await rep.orchestrator.run_async(num_frames=total_frames, start_timeline=True)

        # 允许 orchestrator 有一帧时间把时间轴切换到播放态
        await omni.kit.app.get_app().next_update_async()
        timeline_if = omni.timeline.get_timeline_interface()
        if not timeline_if.is_playing():
            # 如果仍未播放则主动触发一次播放，避免 capture_on_play 关闭后 timeline 仍停在 0 帧
            timeline_if.play()
            timeline_if.commit()

        rep.orchestrator.set_capture_on_play(False)

        # 可选等待完成
        if will_wait_until_complete:
            await rep.orchestrator.wait_until_complete_async()

        # Resume timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_time_codes_per_second(original_timecode)
        timeline.commit_silently()
        await omni.kit.app.get_app().next_update_async()

        self._clear_recorder()

    def _init_recorder(self) -> bool:
        if self._writer is None:
            try:
                self._writer = rep.WriterRegistry.get(self.writer_name)
            except Exception as e:
                carb.log_error(f"Could not create writer {self.writer_name}: {e}")
                return False
        try:
            self._writer.initialize(**self.writer_params)
        except Exception as e:
            carb.log_error(f"Could not initialize writer {self.writer_name}: {e}")
            return False

        # Fetch from stage if config file did not specify a camera list
        if not self._camera_path_list:
            self._camera_list = self._get_camera_list(self._num_cameras)
            self._camera_path_list = [prims.get_prim_path(cam) for cam in self._camera_list]
            # self._lidar_list = self._get_lidar_list(self._num_lidars)
            # if self._lidar_list:
            #     self._lidar_path_list = [prims.get_prim_path(ldr) for ldr in self._lidar_list]
        self._render_products = self.create_render_product_list()

        # Hide debugging visualizations like navmesh and grid in the viewport
        hide_visualization = carb.settings.get_settings().get(
            "/persistent/exts/isaacsim.replicator.agent/hide_visualization"
        )
        if hide_visualization:
            self.store_viewport_settings()
            self.set_viewport_settings()

        if not self._render_products:
            carb.log_error("No valid render products found to initialize the writer.")
            return False

        # 验证并初始化render products以确保它们有有效数据
        carb.log_info(f"Validating {len(self._render_products)} render products before writer attachment...")

        try:
            self._writer.attach(self._render_products)
        except Exception as e:
            carb.log_error(f"Could not attach render products to writer: {e}")
            return False

        return True

    def _clear_recorder(self):
        if self._writer:
            self._writer.detach()
            self._writer = None
        for rp in self._render_products:
            rp.destroy()
        self._render_products.clear()
        # Recover viewport state if debugging visualizations were automatically hidden in data generation
        hide_visualization = carb.settings.get_settings().get(
            "/persistent/exts/isaacsim.replicator.agent/hide_visualization"
        )
        if hide_visualization:
            self.recover_viewport_settings()
        # Unsubscribe events
        self._sub_stage_event = None
        # Done callback
        for callback in self._data_generation_done_callback:
            callback()

    def store_viewport_settings(self):
        self._show_grid = carb.settings.get_settings().get("/app/viewport/grid/enabled")
        self._show_outline = carb.settings.get_settings().get("/app/viewport/outline/enabled")
        self._show_navMesh = carb.settings.get_settings().get(
            "/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh"
        )
        self._show_camera = carb.settings.get_settings().get("/app/viewport/show/cameras")
        self._show_light = carb.settings.get_settings().get("/app/viewport/show/lights")
        self._show_audio = carb.settings.get_settings().get("/app/viewport/show/audio")
        self._show_skeleton = carb.settings.get_settings().get("/app/viewport/usdcontext-/scene/skeletons/visible")
        self._show_meshes = carb.settings.get_settings().get("/app/viewport//usdcontext-/scene/meshes/visible")

    def set_viewport_settings(self):
        carb.settings.get_settings().set("/app/viewport/grid/enabled", False)
        carb.settings.get_settings().set("/app/viewport/outline/enabled", False)
        carb.settings.get_settings().set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        carb.settings.get_settings().set("/app/viewport/show/cameras", False)
        carb.settings.get_settings().set("/app/viewport/show/lights", False)
        carb.settings.get_settings().set("/app/viewport/show/audio", False)
        carb.settings.get_settings().set("/app/viewport/usdcontext-/scene/skeletons/visible", False)
        carb.settings.get_settings().set("/app/viewport/usdcontext-/scene/meshes/visible", True)

    def recover_viewport_settings(self):
        if self._show_grid is not None:
            carb.settings.get_settings().set("/app/viewport/grid/enabled", self._show_grid)
        if self._show_outline is not None:
            carb.settings.get_settings().set("/app/viewport/outline/enabled", self._show_outline)
        if self._show_navMesh is not None:
            carb.settings.get_settings().set(
                "/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", self._show_navMesh
            )
        if self._show_camera is not None:
            carb.settings.get_settings().set("/app/viewport/show/cameras", self._show_camera)
        if self._show_light is not None:
            carb.settings.get_settings().set("/app/viewport/show/lights", self._show_light)
        if self._show_audio is not None:
            carb.settings.get_settings().set("/app/viewport/show/audio", self._show_audio)
        if self._show_skeleton is not None:
            carb.settings.get_settings().set("/app/viewport/usdcontext-/scene/skeletons/visible", self._show_skeleton)
        if self._show_meshes is not None:
            carb.settings.get_settings().set("/app/viewport//usdcontext-/scene/meshes/visible", self._show_meshes)

    # def create_render_product_list(self):
    #     render_product_list = []

    #     carb.log_info(f"Creating render products for {len(self._camera_path_list)} camera paths")

    #     # 统一使用相同的API调用格式，并增强验证
    #     for i, path in enumerate(self._camera_path_list):
    #         if path:  # 确保路径不为空
    #             try:
    #                 # 检查相机prim是否存在
    #                 stage = omni.usd.get_context().get_stage()
    #                 if stage:
    #                     camera_prim = stage.GetPrimAtPath(path)
    #                     if not camera_prim.IsValid():
    #                         carb.log_error(f"Camera prim at path {path} is not valid, skipping")
    #                         continue

    #                     carb.log_info(f"Creating RenderProduct for valid camera at {path}")

    #                 rp = rep.create.render_product(path, resolution=(1920, 1080))
    #                 if rp:  # 确保RenderProduct创建成功
    #                     render_product_list.append(rp)
    #                     carb.log_info(f"Successfully created RenderProduct {i} for {path}")
    #                     dp.print(f"Create RenderProduct {rp} with {path}.")
    #                 else:
    #                     carb.log_error(f"Failed to create RenderProduct for path: {path} (returned None)")

    #             except Exception as e:
    #                 carb.log_error(f"Exception creating RenderProduct for {path}: {e}")
    #         else:
    #             carb.log_warn(f"Camera path {i} is empty, skipping")

    #     if self._write_robot_data:
    #         # Add cameras as render product
    #         try:
    #             robot_cameras = RobotUtil.get_n_robot_cameras(2)
    #             carb.log_info(f"Adding {len(robot_cameras)} robot cameras to render products")
    #             for c in robot_cameras:
    #                 cam_path = prims.get_prim_path(c)
    #                 if cam_path:
    #                     try:
    #                         rp = rep.create.render_product(cam_path, resolution=(1920, 1080))
    #                         if rp:
    #                             render_product_list.append(rp)
    #                             carb.log_info(f"Successfully created Robot RenderProduct for {cam_path}")
    #                             dp.print(f"Create Robot RenderProduct {rp} with {cam_path}.")
    #                         else:
    #                             carb.log_error(f"Failed to create Robot RenderProduct for path: {cam_path} (returned None)")
    #                     except Exception as e:
    #                         carb.log_error(f"Exception creating Robot RenderProduct for {cam_path}: {e}")
    #         except Exception as e:
    #             carb.log_warn(f"Failed to get robot cameras: {e}")

    #     carb.log_info(f"Created {len(render_product_list)} total render products")
    #     return render_product_list
    
    def create_render_product_list(self):
        render_product_list = []
        for path in self._camera_path_list:
            rp = rep.create.render_product(path, (1920, 1080))
            render_product_list.append(rp)
            dp.print(f"Create RenderProduct {rp} with {path}.")
        if self._write_robot_data:
            # Add cameras as render product
            for c in RobotUtil.get_n_robot_cameras(2):
                cam_path = prims.get_prim_path(c)
                rp = rep.create.render_product(cam_path, resolution=(1920, 1080))
                render_product_list.append(rp)
                dp.print(f"Create Robot RenderProduct {rp} with {cam_path}.")
        # for path in self._lidar_path_list:
        #     lidar_rp = rep.create.render_product(path, resolution=[1, 1])
        #     render_product_list.append(lidar_rp)
        return render_product_list

    # def _get_camera_list(self, num_cameras):
    #     cameras_in_stage = CameraUtil.get_cameras_in_stage()

    #     carb.log_info(f"找到 {len(cameras_in_stage)} 个相机在场景中")

    #     # 强制修复：对于Matterport场景，使用所有找到的相机
    #     if len(cameras_in_stage) > 0:
    #         carb.log_info("强制使用场景中的所有相机进行数据生成")
    #         return cameras_in_stage

    #     # 原有逻辑保留作为后备
    #     if num_cameras == -1:
    #         return cameras_in_stage
    #     if num_cameras > len(cameras_in_stage):
    #         num_cameras = len(cameras_in_stage)
    #         carb.log_warn(
    #             "Camera Number is greater than the cameras in the stage. Only the cameras in the stage will have output."
    #         )
    #     return cameras_in_stage[:num_cameras]
    
    def _get_camera_list(self, num_cameras):
        cameras_in_stage = CameraUtil.get_cameras_in_stage()
        # If num_cam == -1, we honor whatever cameras are in the stage
        if num_cameras == -1:
            return cameras_in_stage
        if num_cameras > len(cameras_in_stage):
            num_cameras = len(cameras_in_stage)
            carb.log_warn(
                "Camera Number is greater than the cameras in the stage. Only the cameras in the stage will have output."
            )
        return cameras_in_stage[:num_cameras]

    def _get_lidar_list(self, num_lidar):
        return None

    def _lidar_fusion_renderproduct_prune(self):
        return
