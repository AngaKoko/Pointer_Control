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

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    extensions=args.extensions

    start_model_load_time=time.time()
    fld= Model_Facial_Landmarks_Detection(model, device, extensions)
    fld.load_model()
    total_model_load_time = time.time() - start_model_load_time
    print("Total model load time = "+str(total_model_load_time))

    # Checks for live feed
    if video_file == 'CAM':
        input_stream = 0
        single_image_mode = False
    # Checks for input image
    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') :
        single_image_mode = True
        input_stream = video_file
    # Checks for video file
    else:
        input_stream = video_file
        assert os.path.isfile(video_file), "Specified input file doesn't exist"
        single_image_mode = False

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    if input_stream:
        cap.open(video_file)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    counter=0
    start_inference_time=time.time()

    #loop until stream is over
    while cap.isOpened:
        #  Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(50)
        #increament counter
        counter += 1

        output = fld.predict(frame)

        ### Write an output image if `single_image_mode` ###
        ### Send the frame to the FFMPEG server ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
        else:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #Break if escape key pressed
            if key_pressed == 27:
                break
            
    #get total inference time and frame per second
    total_time = time.time()-start_inference_time
    total_inference_time = round(total_time, 1)
    fps = counter/total_inference_time

    print("Total inference time = "+str(total_inference_time))
    print("Frames per second = "+str(fps))

    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("-m", '--model', default="models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009", help="location of model to be used")
    parser.add_argument("-d", '--device', default='CPU', help="device to run inference")
    parser.add_argument("-v", '--video', default="bin/demo.mp4", help="video location")
    parser.add_argument("-e", '--extensions', default=None, help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    
    args=parser.parse_args()

    main(args)