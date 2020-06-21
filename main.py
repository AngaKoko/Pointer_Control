import numpy as np
import time
import os
import cv2
import argparse
import sys

from openvino.inference_engine import IENetwork, IECore
from src.face_detection import Model_Face_Detection
from src.facial_landmarks_detection import Model_Facial_Landmarks_Detection
from src.gaze_estimation import Model_Gaze_Estimation
from src.head_pose_estimation import Model_Head_Pose_Estimation

def main(args):
    fd_model=args.model_face_detection
    flm_model = args.model_facial_landmark
    ge_model = args.model_gaze_estimation
    hpe_model = args.model_head_pose_estimation
    device=args.device
    video_file=args.video
    threshold=args.threshold
    extenstions = args.cpu_extensions

    #Load all model
    #Get time it takes to load each model
    start_model_load_time=time.time()
    #load face detection model
    fd= Model_Face_Detection(model_name = fd_model, device = device, extensions=extenstions, threshold=threshold)
    fd.load_model()
    fd_model_load_time = time.time() - start_model_load_time
    
    #load facial landmark detection model
    flm_start_time = time.time()
    fldm = Model_Facial_Landmarks_Detection(model_name = flm_model, device = device, extensions=extenstions)
    fldm.load_model()
    fldm_load_time = time.time() - flm_start_time

    #load gaze estimation model
    gem_start_time = time.time()
    gem = Model_Gaze_Estimation(model_name = ge_model, device = device, extensions=extenstions)
    gem.load_model()
    gem_load_time = time.time() - gem_start_time

    #load head pose estimation model
    hpm_start_time = time.time()
    hpm = Model_Head_Pose_Estimation(model_name = hpe_model, device = device, extensions=extenstions)
    hpm.load_model()
    hpm_load_time = time.time() - hpm_start_time

    #calculate total time it took to load all models
    total_model_time = time.time() - start_model_load_time
    
    #print time it took to load each model
    print("Time to load face detection model = "+str(fd_model_load_time))
    print("Time to load facial landmark detection model = "+str(fldm_load_time))
    print("Time to load gaze estimation model = "+str(gem_load_time))
    print("Time to load head pose estimation model = "+str(hpm_load_time))
    print("Total model load time = "+str(total_model_time))

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
    parser.add_argument("-mfd", '--model_face_detection', default="models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001", help="location of face detection model model to be used")
    parser.add_argument("-mflm", '--model_facial_landmark', default="models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009", help="location of facial landmark model to be used")
    parser.add_argument("-mge", '--model_gaze_estimation', default="models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002", help="location of gaze estimation model to be used")
    parser.add_argument("-mhpe", '--model_head_pose_estimation', default="models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001", help="location of head pose estimation model to be used")
    parser.add_argument("-d", '--device', default='CPU', help="device to run inference")
    parser.add_argument("-v", '--video', default="bin/demo.mp4", help="video location")
    parser.add_argument("-l", '--cpu_extensions', default=None, help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-pt", '--threshold', default=0.60, help="Probability threshold for model")
    
    args=parser.parse_args()

    main(args)