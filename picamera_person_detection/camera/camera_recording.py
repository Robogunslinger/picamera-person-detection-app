import io
import random
import picamera

class CameraRecorder:
    """Utility for recording camera videos"""

    def __init__(
        self, 
        camera: picamera.PiCamera, 
        duration_in_sec: int=20, 
        format: str='h264'):
        """Initializes ObjectDetector parameters.

        Args:
            camera:
            duration_in_sec:
        """
        try:
            self._camera = camera
            self._duration_limit = duration_in_sec
            self._stream = picamera.PiCameraCircularIO(camera, seconds=duration_in_sec)
            self._format = format
        except Exception as ex:

            raise ex 
    
    def start_camera(self):

        self._camera.start_recording(self._stream, format=self._format)
        try:
            self._camera.wait_recording(1)
        except Exception as ex:
            print(f"Failed to start recording")
            self.stop_camera()
            raise ex 

    def record_video(self, output_path ):
        # Keep recording for 10 seconds and only then write the
        # stream to disk
        try:
            camera.wait_recording(self._duration_limit // 2)
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
        finally:
            self.stop_recording()

    def stop_recording(self):
        try:
            self._camera.stop_recording()
        except Exception as ex:
            raise ex
