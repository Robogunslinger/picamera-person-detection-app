# python3
#
# Copyright 2020 Peter-YJ-LIU and The TensorFlow Authors. All Rights Reserved.
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
"""An objection detection library that uses the tensorflow lite model to detect objects from the image.
"""

from PIL import Image

import numpy as np
import re
from tflite_runtime.interpreter import Interpreter

class ObjectDetector:
    """Utility for detecting objects"""

    def __init__(self, model_path: str, label_path: str):
        """Initializes ObjectDetector parameters.

        Args:
            model_path: model file path in string
            label_path: model label file path in string
        """
        try:
            self._labels = self.init_labels(label_path)
            self._interpreter = self.init_interpreter(model_path)
            self._input_height, self._input_width = self.init_height_width()
        except Exception as ex:
            print(f"Model initialization failed!")
            raise ex 

    def init_labels(self, path: str) -> dict:
        """Loads the labels file. Supports files with or without index numbers.
        
        Args:
            path: model label file path in string

        Returns:
            the label in dictionary format
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}

            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
        return labels

    def init_interpreter(self, path: str) -> Interpreter:
        """Initialize the interpreter
        
        Args:
            path: model file path in string

        Returns:
            the model interpreter
        """
        interpreter = Interpreter(path)
        interpreter.allocate_tensors()
        return interpreter

    def init_height_width(self) -> tuple:
        """Get the initialized input height and width
        
        Returns:
            input height and input width in a tuple
        """
        _, input_height, input_width, _ = self._interpreter.get_input_details()[0]['shape']
        return input_height, input_width

    def set_input_tensor(self, image):
        """Sets the input tensor based on the image.
        
        Args:
            image: the image for the input tensor
        """
        tensor_index = self._interpreter.get_input_details()[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index) -> np.ndarray:
        """Returns the output tensor at the given index.
        
        Args:
            index: index position for the output tensor
        
        Returns:
            the output tensor
        """
        output_details = self._interpreter.get_output_details()[index]
        tensor = np.squeeze(self._interpreter.get_tensor(output_details['index']))

        return tensor
    
    def get_label_name(self, class_id: float) -> str:
        """Returns the label name based on the class id
        
        Args:
            class_id: the class id of label
        
        Returns: 
            the label name
        """
        return self._labels[class_id]
    
    def get_model_width_height(self) -> tuple:
        """Returns the width and height from the model
        
        Returns:
            the model input width and height in a tuple
        """
        return self._input_width, self._input_height

    def resize(self, image: Image.Image) -> Image.Image:
        """Resizes the image based on the model input height and width
        
        Args:
            image:
        
        Returns:
            the resized image
        """
        rgb_converted = image.convert('RGB')
        resized_image = rgb_converted.resize(
            (self._input_width, self._input_height), Image.ANTIALIAS)

        return resized_image

    def detect_objects(self, image: Image.Image, threshold: float) -> list:
        """Returns a list of detection results, each a dictionary of object info.
        
        Args:
            image: 
            threshold:

        Returns:
            the result in a list of dictionaries, with each 
            dictionary containing the object class_id, score, bounding_box
        """
        resized_image = self.resize(image)

        self.set_input_tensor(resized_image)
        self._interpreter.invoke()

        # Get all output details
        boxes = self.get_output_tensor(0)
        classes = self.get_output_tensor(1)
        scores = self.get_output_tensor(2)
        count = int(self.get_output_tensor(3))

        results = []
        for i in range(count):
            if scores[i] >= threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)

        return results
