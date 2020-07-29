import os
import argparse
from datetime import datetime
from PIL import Image

from picamera_person_detection.camera import camera_recording
from picamera_person_detection.camera import camera_setup
from picamera_person_detection.object_detection import object_detection

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__ ))

def prepare_tensorflow_model():

    model_path = os.path.join(CURRENT_FOLDER,  'resources/detect.tflite')
    label_path = os.path.join(CURRENT_FOLDER, 'resources/coco_labels.txt')
    object_detector = object_detection.ObjectDetector(model_path, label_path)
    return object_detector

def detect_person(camera):

    stream = io.BytesIO()
    camera.capture(stream, format="jpeg", use_video_port=True)

    stream.seek(0)
    image = Image.open(stream).convert('RGB')

    object_detector = prepare_tensorflow_model()
    results = object_detector.detect_objects(image, threshold=0.4)
    results_labels = [ object_detector.get_label_name(result['class_id']) for result in results ]

    if "person" in results_labels:
        print('Detected person')
        return True
    return False

def prepare_video_output_path(format='h264'):
    utc_now = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    file_output_path = os.path.join(CURRENT_FOLDER, f"UTC{utc_now}.{format}")
    return file_output_path

def run_camera_detection():

    camera = camera_setup.Camera.get_camera()
    camera_recorder = camera_recording.CameraRecorder(camera, duration_in_sec=20, format='h264')
    camera_recorder.start_camera()
    
    try:
        while True:
            has_person = detect_person(camera)
            
            if has_person:
                file_output_path = prepare_video_output_path(format='h264')
                camera_recorder.record_video(file_output_path)
    finally:
        camera_recorder.stop_camera()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--threshold',
        help='Score threshold for detected objects.',
        required=False,
        type=float,
        default=0.4)
    args = parser.parse_args()

    run_camera_detection()

if __name__ == "__main__":
    main()
