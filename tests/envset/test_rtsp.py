__copyright__ = "Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
import asyncio
import doctest
import inspect
import os
from unittest.mock import patch

import carb
import numpy as np
import omni.kit
import omni.kit.test
import omni.replicator.core as rep
from isaacsim.replicator.agent.core.data_generation.writers.rtsp import RTSPCamera, RTSPWriter
from omni.replicator.core import Writer, WriterRegistry
from omni.syntheticdata import SyntheticData


class TestRTSPCamera(omni.kit.test.AsyncTestCase):
    _default_camera_path = "/World/Cameras/Camera"
    _default_rtsp_stream_url = "rtsp://localhost:8554/RTSPWriter"

    # Before running each test
    async def setUp(self):
        pass

    # After running each test
    async def tearDown(self):
        pass

    @patch("isaacsim.replicator.agent.core.data_generation.writers.rtsp.which", return_value="/usr/bin/ffmpeg")
    async def test_initialize(self, mock_which):
        # ---Test---
        # stream_url and prim_path have to be specified.
        with self.assertRaises(ValueError):
            camera = RTSPCamera(rtsp_stream_url=self._default_rtsp_stream_url)
        # ---Test---

        # ---Test---
        camera = RTSPCamera(rtsp_stream_url=self._default_rtsp_stream_url, prim_path=self._default_camera_path)
        self.assertEqual(camera.device, camera._default_device)
        self.assertEqual(camera.fps, camera._default_fps)
        self.assertEqual(camera.get_resolution(), camera._default_resolution)
        self.assertEqual(camera.bitrate, camera._default_bitrate)
        self.assertEqual(camera.annotator, camera._default_annotator)

        # Default command
        whole_rtsp_stream_url = camera.rtsp_stream_url + self._default_camera_path.replace('/', "_") + "_" + camera.annotator
        vcodec = "hevc_nvenc"  # 'h264_nvenc'
        command = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-hwaccel_device",
            str(camera._default_device),
            "-re",
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "rawvideo",
            # '-thread_queue_size', '4096',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
            # '-vcodec', 'rawvideo',
            "-pix_fmt",
            camera.pixel_fmt,  # 'bgr24',
            # '-src_range', '1',
            "-s",
            f"{camera._default_resolution[0]}x{camera._default_resolution[1]}",
            # '-r', str(fps),
            # '-stream_loop', '-1', # Loop infinite times.
            "-i",
            "-",
            "-c:a",
            "copy",
            "-c:v",
            vcodec,
            "-preset",
            "ll",
            # '-pix_fmt', 'yuv420p',
            # '-preset', 'ultrafast',
            # '-vbr', '5', # Variable Bit Rate (VBR). Valid values are 1 to 5. 5 has the best quality.
            # '-b:v', f'{bitrate}k',
            "-maxrate:v",
            f"{camera._default_bitrate}k",
            "-bufsize:v",
            "64M",  # Buffering is probably required
            # passthrough (0) - Each frame is passed with its timestamp from the demuxer to the muxer.
            # -vsync 0 cannot be applied together with -r/-fpsmax.
            # cfr (1) - Frames will be duplicated and dropped to achieve exactly the requested constant frame rate.
            # vfr (2) - Frames are passed through with their timestamp or dropped so as to prevent 2 frames from having the same timestamp.
            # '-vsync', 'passthrough',
            "-vsync",
            "cfr",
            "-r",
            str(camera._default_fps),
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",  # udp is the most performant. But udp does not support NAT/firewall nor encryption
            # '-vf', 'scale=in_range=full:out_range=full',
            # '-dst_range', '1', '-color_range', '2',
            whole_rtsp_stream_url,
        ]
        self.assertEqual(camera.command, command)
        # ---Test---

        # ---Test---
        # Fully speced parameters
        device = 1
        fps = 45
        resolution = (1024, 512)
        bitrate = 1234567
        annotator = "rgb"  # NOTE: test parameter has been changed to fit current settings
        camera = RTSPCamera(
            device=device,
            fps=fps,
            resolution=resolution,
            bitrate=bitrate,
            annotator=annotator,
            rtsp_stream_url=self._default_rtsp_stream_url,
            prim_path=self._default_camera_path,
        )
        self.assertEqual(camera.device, device)
        self.assertEqual(camera.fps, fps)
        self.assertEqual(camera.get_resolution(), resolution)
        self.assertEqual(camera.bitrate, bitrate)
        self.assertEqual(camera.annotator, annotator)
        self.assertEqual(camera.prim_path, self._default_camera_path)

        # Default command
        whole_rtsp_stream_url = camera.rtsp_stream_url + self._default_camera_path.replace('/', "_") + "_" + camera.annotator
        vcodec = "hevc_nvenc"  # 'h264_nvenc'
        command = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-hwaccel_device",
            str(camera.device),
            "-re",
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "rawvideo",
            # '-thread_queue_size', '4096',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
            # '-vcodec', 'rawvideo',
            "-pix_fmt",
            camera.pixel_fmt,  # 'bgr24',
            # '-src_range', '1',
            "-s",
            f"{camera.resolution[0]}x{camera.resolution[1]}",
            # '-r', str(fps),
            # '-stream_loop', '-1', # Loop infinite times.
            "-i",
            "-",
            "-c:a",
            "copy",
            "-c:v",
            vcodec,
            "-preset",
            "ll",
            # '-pix_fmt', 'yuv420p',
            # '-preset', 'ultrafast',
            # '-vbr', '5', # Variable Bit Rate (VBR). Valid values are 1 to 5. 5 has the best quality.
            # '-b:v', f'{bitrate}k',
            "-maxrate:v",
            f"{camera.bitrate}k",
            "-bufsize:v",
            "64M",  # Buffering is probably required
            # passthrough (0) - Each frame is passed with its timestamp from the demuxer to the muxer.
            # -vsync 0 cannot be applied together with -r/-fpsmax.
            # cfr (1) - Frames will be duplicated and dropped to achieve exactly the requested constant frame rate.
            # vfr (2) - Frames are passed through with their timestamp or dropped so as to prevent 2 frames from having the same timestamp.
            # '-vsync', 'passthrough',
            "-vsync",
            "cfr",
            "-r",
            str(camera.fps),
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",  # udp is the most performant. But udp does not support NAT/firewall nor encryption
            # '-vf', 'scale=in_range=full:out_range=full',
            # '-dst_range', '1', '-color_range', '2',
            whole_rtsp_stream_url,
        ]
        self.assertEqual(camera.command, command)
        # ---Test---

        # ---Test---
        # Test parameter autocorrection
        camera = RTSPCamera(device=-1, rtsp_stream_url=self._default_rtsp_stream_url, prim_path=self._default_camera_path)
        self.assertEqual(camera.device, camera._default_device)
        # test wrong fps
        camera = RTSPCamera(fps=-30, rtsp_stream_url=self._default_rtsp_stream_url, prim_path=self._default_camera_path)
        self.assertEqual(camera.fps, camera._default_fps)
        # test wrong resolution
        camera = RTSPCamera(resolution=(-30, -20), rtsp_stream_url=self._default_rtsp_stream_url, prim_path=self._default_camera_path)
        self.assertEqual(camera.resolution, camera._default_resolution)
        # test wrong bitrate
        camera = RTSPCamera(bitrate=-30, rtsp_stream_url=self._default_rtsp_stream_url, prim_path=self._default_camera_path)
        self.assertEqual(camera.bitrate, camera._default_bitrate)
        # ---Test---

        # ---Test---
        # Invalid annotator
        annotator = "wrong_annotator"
        with self.assertRaises(ValueError):
            camera = RTSPCamera(annotator=annotator, rtsp_stream_url=self._default_rtsp_stream_url, prim_path=self._default_camera_path)
