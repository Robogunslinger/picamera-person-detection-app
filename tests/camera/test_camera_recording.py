import pytest
import os
import picamera
from picamera_person_detection.camera import camera_recording
from verify import verify_video, verify_image, RAW_FORMATS

@pytest.fixture
def video_format():
    return 'h264'

@pytest.fixture
def video_resolution():
    return (1280, 720)

@pytest.fixture
def camera_recorder(video_resolution, video_format):
    pi_camera = picamera.PiCamera(resolution=video_resolution)
    recorder = camera_recording.CameraRecorder(pi_camera, duration_in_sec=20, format=video_format)
    yield recorder
    pi_camera.close()

def test_fixture_video_format(video_format):
    assert video_format == 'h264'

def test_fixture_video_resolution(video_resolution):
    assert video_resolution == (1280, 720)

def test_fixture_camera_recorder(camera_recorder):
    assert isinstance(camera_recorder, camera_recording.CameraRecorder)
    
def test_record_video(tmpdir, camera_recorder, video_format, video_resolution):
    output_path = os.path.join(tmpdir, 'motion.h264')
    camera_recorder.start_camera()
    camera.record_video(output_path)
    camera.stop_camera()

    verify_video(output_path, video_format, video_resolution)
