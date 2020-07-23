import pytest
import os
from distutils import dir_util
from PIL import Image
from picamera_person_detection.object_detection import object_detection

@pytest.fixture
def test_folder():
    current_folder = os.path.dirname(os.path.abspath(__file__ ))
    return current_folder

@pytest.fixture
def object_detector(test_folder):
    model_path = os.path.join(test_folder,  'detect.tflite')
    label_path = os.path.join(test_folder, 'object_labels.txt')
    return object_detection.ObjectDetector(model_path, label_path)

@pytest.fixture
def image(test_folder):
    image_path = os.path.join(test_folder, 'person_object.jpg')
    return Image.open(image_path)

def test_fixture_test_folder(test_folder):
    assert isinstance(test_folder, str)

def test_fixture_object_detector(object_detector):
    assert isinstance(object_detector, object_detection.ObjectDetector)

def test_fixture_image(image):
    assert isinstance(image, Image.Image)

def test_get_label_name(object_detector):
    assert object_detector.get_label_name(0) == "person"

def test_resize(object_detector, image):
    model_width, model_height = object_detector.get_model_width_height()
    resized_image = object_detector.resize(image)

    assert resized_image.width == model_width
    assert resized_image.height == model_height

def test_detect_objects(object_detector, image):
    results = object_detector.detect_objects(image, threshold=0.4)
    results_labels = [ object_detector.get_label_name(result['class_id']) for result in results ]
    assert "person" in results_labels
