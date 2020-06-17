# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

For this project we will be using 4 models: 

* [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

The models can be downloaded using `model downloader`. To download a model, naviagate to the directory containing the model downloader 
`cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader` . 
Use `--name` argument for model name, `-o` argument to specify your output directory

**Download Face Detection model**

    sudo ./downloader.py --name face-detection-adas-binary-0001 -o /home/anga/Documents/Pointer_Control/models

**Download Head Pose Estimation model**

    sudo ./downloader.py --name head-pose-estimation-adas-0001 -o /home/anga/Documents/Pointer_Control/models

**Download Facial Landmarks Detection model**

    sudo ./downloader.py --name landmarks-regression-retail-0009 -o /home/anga/Documents/Pointer_Control/models
    
**Download Gaze Estimation model**

    sudo ./downloader.py --name gaze-estimation-adas-0002 -o /home/anga/Documents/Pointer_Control/models

## Install Dependencies
**Numpy**

    pip3 install numpy



## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
