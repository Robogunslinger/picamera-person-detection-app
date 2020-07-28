# python3
#
# Copyright 2020 Peter-YJ-LIU. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A picamera recording helper library that records video up to designated length
"""
import io
import picamera

class CameraRecorder:
    """Utility for recording picamera video"""

    def __init__(
        self, 
        camera: picamera.PiCamera, 
        duration_in_sec: int=20, 
        format: str='h264'):
        """Initializes ObjectDetector parameters.

        Args:
            camera: picamera object
            duration_in_sec: desired video recording duration in seconds, default as 20 seconds
            format: desired video format, default as 'h264'
        """
        try:
            self._camera = camera
            self._duration_limit = duration_in_sec
            self._stream = picamera.PiCameraCircularIO(camera, seconds=duration_in_sec)
            self._format = format
        except Exception as ex:
            raise ex 
    
    def start_camera(self):
        """Starts recording into stream, before saving into actual files. When starting, allow 1 second timeout for any exception.
        """
        self._camera.start_recording(self._stream, format=self._format)
        try:
            self._camera.wait_recording(1)
        except Exception as ex:
            print(f"Failed to start recording")
            self.stop_camera()
            raise ex 
    
    def get_camera_stream(self):
        '''Returns the camera stream
        
        Returns: the camera stream
        '''
        return self._stream

    def record_video(self, output_path: str):
        """Record the video in the stream into disk. 

        Args:
            output_path: the output file path in String
        """
        # Keep recording for n // 2 seconds and only then write the video stream to disk
        try:
            self._camera.wait_recording(self._duration_limit // 2)
            with self._stream.lock:
                # Find the first header frame in the video
                for frame in self._stream.frames:
                    if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                        self._stream.seek(frame.position)
                        break
                # Write the rest of the stream to disk
                with io.open(output_path, 'wb') as output:
                    output.write(self._stream.read())
        except Exception as ex:
            raise ex

    def stop_camera(self):
        """Stops recording into stream.
        """
        try:
            self._camera.stop_recording()
        except Exception as ex:
            raise ex
