from object_detection import object_detection

import os
from datetime import datetime
from PIL import Image
from picamera_person_detection.camera import camera_recording
from picamera_person_detection.camera import camera_setup

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__ ))

def prepare_tensorflow_model():

    model_path = os.path.join(CURRENT_FOLDER,  'resources/detect.tflite')
    label_path = os.path.join(CURRENT_FOLDER, 'resources/coco_labels.txt')
    object_detector = object_detection.ObjectDetector(model_path, label_path)
    return object_detector

def detect_person(stream):
    image = Image.open(stream).convert('RGB')

    results = object_detector.detect_objects(image, threshold=0.4)
    results_labels = [ object_detector.get_label_name(result['class_id']) for result in results ]

    if "person" in results_labels:
        return True
    return False

def prepare_video_output_path(format='h264'):
    utc_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    file_output_path = os.path.join(CURRENT_FOLDER, f"UTC{utc_now}.{format}")
    return file_output_path

def run_camera_detection():
    
    object_detector = prepare_tensorflow_model()

    camera = camera_setup.get_camera()
    camera_recorder = camera_recording.CameraRecorder(camera, duration_in_sec=20, format='h264')
    camera_recorder.start_camera()
    
    try:
        stream = camera_recorder.get_camera_stream()
        while True:
            stream.seek(0)
            has_person = detect_person(stream)
            
            if has_person:
                file_output_path = prepare_video_output_path(format='h264')
                
                camera_recorder.record_video(file_output_path)
            else:
                stream.seek(0)
                stream.truncate()
    finally:
        camera_recorder.stop_camera()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    parser.add_argument(
        '--threshold',
        help='Score threshold for detected objects.',
        required=False,
        type=float,
        default=0.4)
    args = parser.parse_args()

    run_camera_detection()