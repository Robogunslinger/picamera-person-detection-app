import picamera

class Camera():

    @classmethod
    def get_camera(framerate=5, resolution=None):
        if resolution is None:
            camera = picamera.PiCamera()
        else:
            camera = picamera.PiCamera(resolution=resolution)
        return camera

    @classmethod
    def stop(camera):
        camera.close()
