import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import logging as log
import os
import cv2
import argparse
import sys

class Model_Face_Detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
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
        processed_image = self.preprocess_input(image)
        # Start asynchronous inference for specified request
        cropped_image = image
        net.start_async(request_id=0,inputs={self.input_name: processed_image})
        # Wait for the result
        if net.requests[0].wait(-1) == 0:
            #get out put
            outputs = net.requests[0].outputs[self.output_name]
            coords = self.preprocess_output(outputs)
            bounding_box, image = self.draw_outputs(coords, image)
            bounding_box = bounding_box[0] 
            cropped_image = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]         
        return cropped_image

    def draw_outputs(self, coords, image):
        #get image width and hight
        initial_h = image.shape[0]
        initial_w = image.shape[1]
        bounding_box = []
        for value in coords:
            # Draw bounding box on detected objects
            xmin = int(value[3] * initial_w)
            ymin = int(value[4] * initial_h)
            xmax = int(value[5] * initial_w)
            ymax = int(value[6] * initial_h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,55,255), 2)
            bounding_box.append([xmin, ymin, xmax, ymax])
        return bounding_box, image

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

    def preprocess_output(self, outputs):
        bounding_box = []
        
        for value in outputs[0][0]:
            #check if confidence is greater than probability threshold
            if value[2] > self.threshold:
                bounding_box.append(value)
        return bounding_box

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    threshold=args.threshold
    extensions = args.extensions

    start_model_load_time=time.time()
    fd= Model_Face_Detection(model, device, extensions, threshold)
    fd.load_model()
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

    try:
        #loop until stream is over
        while cap.isOpened:
            #  Read from the video capture ###
            flag, frame = cap.read()
            if not flag:
                break

            key_pressed = cv2.waitKey(50)
            #increament counter
            counter += 1

            image= fd.predict(frame)

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
                

    except Exception as e:
        print("Could not run inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("-m", '--model', default="models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001", help="location of model to be used")
    parser.add_argument("-d", '--device', default='CPU', help="device to run inference")
    parser.add_argument("-v", '--video', default="bin/demo.mp4", help="video location")
    parser.add_argument("-e", '--extensions', default=None, help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-pt", '--threshold', default=0.60, type=float, help="Probability threshold for model")
    
    args=parser.parse_args()

    main(args)