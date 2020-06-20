import time
from openvino.inference_engine import IENetwork, IECore
import cv2
import math

class Gaze_Estimation:
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
        

    def check_model(self):
        raise NotImplementedError

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
