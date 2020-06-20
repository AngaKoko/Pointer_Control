import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

class Facial_Landmarks_Detection:
    '''
    Class for the Facial Landmark Detection Model.
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
        net = core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
        return net

    def predict(self, image):
        
        eye_points = []
        processed_image = self.preprocess_input(image)
        # Start asynchronous inference for specified request
        net.start_async(request_id=0,inputs={self.input_name: processed_image})
        # Wait for the result
        if net.requests[0].wait(-1) == 0:
            #get out put
            output = net.requests[0].outputs[self.output_name]
            left_eye_point_x = output[0][0][0]
            left_eye_point_y = output[0][2][0]
            right_eye_poit_x = output[0][3][0]
            right_eye_poit_y = output[0][4][0]

            eye_points.append(left_eye_point_x)
            eye_points.append(left_eye_point_y)
            eye_points.append(right_eye_poit_x)
            eye_points.append(right_eye_poit_y)
          
        return eye_points
    
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

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    threshold=args.threshold

    start_model_load_time=time.time()
    fld= Facial_Landmarks_Detection(model, device, threshold)
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
    parser.add_argument("-m", '--model', default="models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009", help="location of model to be used")
    parser.add_argument("-d", '--device', default='CPU', help="device to run inference")
    parser.add_argument("-v", '--video', default="bin/demo.mp4", help="video location")
    parser.add_argument("-e", '--extensions', default=None)
    parser.add_argument("-pt", '--threshold', default=0.60, help="Probability threshold for model")
    
    args=parser.parse_args()

    main(args)