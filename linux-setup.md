# Linux - Initial Setup

### Install Intel® Distribution of OpenVINO™ Toolkit

Refer to [this page](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) for more information about how to install and setup the Intel® Distribution of OpenVINO™ Toolkit.

You will need the OpenCL™ Runtime Package if you plan to run inference on the GPU. It is not mandatory for CPU inference. 

### Install the following dependencies

```
sudo apt update
sudo apt-get install python3-pip
pip3 install numpy
sudo apt install ffmpeg
sudo apt-get install cmake
```

If you’re prompted to upgrade pip, do not update.
