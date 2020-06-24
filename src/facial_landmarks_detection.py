import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import logging as log

class Model_Facial_Landmarks_Detection:
    '''
    Class for the Facial Landmark Detection Model.
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

    def predict(self, image):
        
        left_eye = []
        right_eye = []
        processed_image = self.preprocess_input(image)
        # Start asynchronous inference for specified request
        net.start_async(request_id=0,inputs={self.input_name: processed_image})
        # Wait for the result
        if net.requests[0].wait(-1) == 0:
            #get out put
            outputs = net.requests[0].outputs[self.output_name]
            outputs= outputs[0]
            left_eye, right_eye = self.draw_outputs(outputs, image)
            
        return left_eye, right_eye, outputs
    
    def draw_outputs(self, outputs, image):
        #get image width and hight
        initial_h = image.shape[0]
        initial_w = image.shape[1]
        
        xl,yl = outputs[0][0]*initial_w,outputs[1][0]*initial_h
        xr,yr = outputs[2][0]*initial_w,outputs[3][0]*initial_h
        # make box for left eye 
        xlmin = int(xl-20)
        ylmin = int(yl-20)
        xlmax = int(xl+20)
        ylmax = int(yl+20)
        #draw boudning box on left eye
        cv2.rectangle(image, (xlmin, ylmin), (xlmax, ylmax), (0,55,255), 2)
        #get left eye image
        left_eye =  image[ylmin:ylmax, xlmin:xlmax]
        
        # make box for right eye 
        xrmin = int(xr-20)
        yrmin = int(yr-20)
        xrmax = int(xr+20)
        yrmax = int(yr+20)
        #draw boinding box on right eye
        cv2.rectangle(image, (xrmin, yrmin), (xrmax, yrmax), (0,55,255), 2)
        #get righ eye image
        right_eye = image[yrmin:yrmax, xrmin:xrmax]

        return left_eye, right_eye

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