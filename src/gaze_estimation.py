import time
from openvino.inference_engine import IENetwork, IECore
import cv2
import math
import logging as log
import sys

class Model_Gaze_Estimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions = extensions

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Check if modeld path is correct")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        global net
        #load the model using IECore()
        core = IECore()
        self.check_model(core)
        net = core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
        return net

    def predict(self, left_eye_image, right_eye_image, pose_angles):
        #Process images
        processed_left_eye_image = self.preprocess_input(left_eye_image)
        processed_righ_eye_image = self.preprocess_input(right_eye_image)

        #get ouput from net
        outputs = net.infer({"head_pose_angles":pose_angles, "left_eye_image":processed_left_eye_image, "right_eye_image":processed_righ_eye_image})
        #Get new mouse co-ordinate and gaze vector
        mouse_coordinate, gaze_vector = self.preprocess_output(outputs, pose_angles)
        #return new mouse co-ordinate and gaze vector
        return mouse_coordinate, gaze_vector
        

    def check_model(self, core):
        # Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            core.add_extension(self.extensions, self.device)

        ###: Check for supported layers ###
        if "CPU" in self.device:
            supported_layers = core.query_network(self.model, "CPU")
            not_supported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(self.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
                sys.exit(1)

    def preprocess_input(self, image):
        #Get Input shape 
        n, c, h, w = self.model.inputs[self.input_name].shape

        #Pre-process the image ###
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        return image

    def preprocess_output(self, outputs, pose_angles):
        gaze_vector = outputs[self.output_name[0]].tolist()[0]
        roll_angle = pose_angles["angle_r_fc"]
        cos = math.cos(roll_angle * math.pi / 180.0)
        sin = math.sin(roll_angle * math.pi / 180.0)
        
        x = gaze_vector[0] * cos + gaze_vector[1] * sin
        y = -gaze_vector[0] *  sin+ gaze_vector[1] * cos
        return (x,y), gaze_vector
