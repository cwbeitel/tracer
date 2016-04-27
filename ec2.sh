
# The following describes how to set up a new image on EC2 with Cuda 7.5, CuDNN, and Tensor Flow
# The default Ubuntu 14.04 kernel needs to be reconfigured in order to work propperly (see below).
# The result of this available as an AMI: ami-ff3a2095 
# Another available AMI: ami-e191b38b 
# Via http://ramhiser.com/2016/01/05/installing-tensorflow-on-an-aws-ec2-instance-with-gpu-support/
# and https://github.com/jasimpson/tensorflow-on-aws
# This works as-is but when upgrading this image to tensor flow 0.8.0 I get the error:
# "/usr/bin/python: libcudart.so.7.5: cannot open shared object file: No such file or directory"

# Launch a new GPU instance on EC2
# 16gb storage wasn't enough, 32gb was enough


# Basics
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get install -y python-pip python-dev
#sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn zip
sudo pip install -U pip

# Install tensorflow
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

# Installing CUDA 7.5
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

# CuDNN (did I use the wrong version here? Needs to be 4.0?)
# navigate to: https://developer.nvidia.com/rdp/form/cudnn-download-survey and get a copy of CuDNN
# Save the file linked from "cuDNN v4 Library for Linux" to your machine
scp -i [your keyfile] [thecuDNN_file_just_saved] ubuntu@[your machine]:~/
#tar -xf cudnn-6.5-linux-x64-v2.tar
tar -xvf cudnn-70-linux-x64-v40.tgz
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
# Note that I found that the CNN example did not work when I originally used cuDNN 6.5 (v4)
# and did work when I switched to 7.0.

# Environment variables
echo >> .bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> .bashrc
echo "export CUDA_ROOT=/usr/local/cuda" >> .bashrc
echo "export PATH=$PATH:/usr/local/cuda/bin:$HOME/bin" >> .bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> .bashrc

# ------
# try verifying the CUDA installation

sudo nvidia-smi
dmesg
#[  121.046222] nvidia: Unknown symbol drm_open (err 0)
#[  121.046234] nvidia: Unknown symbol drm_poll (err 0)
#[  121.046245] nvidia: Unknown symbol drm_pci_init (err 0)
#[  121.046296] nvidia: Unknown symbol drm_gem_prime_handle_to_fd (err 0)
#[  121.046308] nvidia: Unknown symbol drm_gem_private_object_init (err 0)
#[  121.046320] nvidia: Unknown symbol drm_gem_mmap (err 0)
#[  121.046324] nvidia: Unknown symbol drm_ioctl (err 0)
#....
# So clearly there is a problem with drm. 

# You'll see something similar with dmesg when trying a tensorflow example
python -m tensorflow.models.image.mnist.convolutional
#...
#modprobe: ERROR: could not insert 'nvidia_352': Unknown symbol in module, or unknown parameter (see dmesg)
#E tensorflow/stream_executor/cuda/cuda_driver.cc:481] failed call to cuInit: CUDA_ERROR_NO_DEVICE
#I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:102] kernel driver does not appear to be running on this host (ip-172-31-10-7): /proc/driver/nvidia/version does not exist
#I tensorflow/core/common_runtime/gpu/gpu_init.cc:81] No GPU devices available on machine.
#...
#[  566.289383] nvidia: Unknown symbol drm_open (err 0)
#[  566.289396] nvidia: Unknown symbol drm_poll (err 0)
#[  566.289407] nvidia: Unknown symbol drm_pci_init (err 0)
#[  566.289457] nvidia: Unknown symbol drm_gem_prime_handle_to_fd (err 0)
#[  566.289470] nvidia: Unknown symbol drm_gem_private_object_init (err 0)


# We need to build a kernel that has built-in DRM.
# Thanks @lealyf: https://devtalk.nvidia.com/default/topic/769719/drm-ko-missing-on-ubuntu-14-04-1-lts-aws-ec2-g2-2xlarge-instance/
sudo apt-get build-dep linux-image-$(uname -r)
apt-get source linux-image-`uname -r`
cd linux-3.13.0
chmod a+x debian/scripts/*
chmod a+x debian/scripts/misc/*
fakeroot debian/rules clean
sudo apt-get install -y libncurses-dev

# With the following command, the first prompt will be the following:
# Do you want to edit config: amd64/config.flavour.generic? [Y/n] 
# #Say yes, then navigate to Devices > Graphics Support > Direct Rendering Manager
# Select the option to exit, then to save
fakeroot debian/rules editconfigs

# Next, we need to build the new kernel. This will take (from 1:29 to )
fakeroot debian/rules clean
fakeroot debian/rules binary-headers binary-generic

# Three .deb files will be present in the directory above if the build is successful
cd ..
ls *.deb
#linux-headers-..._all.deb
#linux-headers-..._amd64.deb
#linux-image-..._amd64.deb

# Install the new packages and reboot
sudo dpkg -i linux*.deb
sudo reboot

# Lastly, patch the missing linux-cloud-tools dependency
sudo apt-get -f install
#sudo apt-get -f install linux-tools-3.13.0-85-generic 
#sudo apt-get install linux-cloud-tools-3.13.0-85
#linux-cloud-tools-3.13.0-85-generic

# Try again to verify the CUDA installation, this time it should work
sudo nvidia-smi
#+------------------------------------------------------+                       
#| NVIDIA-SMI 352.63     Driver Version: 352.63         |                       
#|-------------------------------+----------------------+----------------------+
#| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#|===============================+======================+======================|
#|   0  GRID K520           Off  | 0000:00:03.0     Off |                  N/A |
#| N/A   35C    P0    35W / 125W |     11MiB /  4095MiB |      0%      Default |
#+-------------------------------+----------------------+----------------------+
#                                                                               
#+-----------------------------------------------------------------------------+
#| Processes:                                                       GPU Memory |
#|  GPU       PID  Type  Process name                               Usage      |
#|=============================================================================|
#|  No running processes found                                                 |
#+-----------------------------------------------------------------------------+

# Then, try running the Tensor Flow CNN example
python -m tensorflow.models.image.mnist.convolutional
#I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcublas.so locally
#I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcudnn.so locally
#I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcufft.so locally
#I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcuda.so.1 locally
#I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcurand.so locally
#Extracting data/train-images-idx3-ubyte.gz
#Extracting data/train-labels-idx1-ubyte.gz
#Extracting data/t10k-images-idx3-ubyte.gz
#Extracting data/t10k-labels-idx1-ubyte.gz
#I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:900] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
#I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties: 
#name: GRID K520
#major: 3 minor: 0 memoryClockRate (GHz) 0.797
#pciBusID 0000:00:03.0
#Total memory: 4.00GiB
#Free memory: 3.95GiB
#I tensorflow/core/common_runtime/gpu/gpu_init.cc:126] DMA: 0 
#I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 0:   Y 
#I tensorflow/core/common_runtime/gpu/gpu_device.cc:755] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GRID K520, pci bus id: 0000:00:03.0)
#Initialized!
#Step 0 (epoch 0.00), 8.0 ms
#Minibatch loss: 12.054, learning rate: 0.010000
#Minibatch error: 90.6%
#Validation error: 84.6%
#Step 100 (epoch 0.12), 25.1 ms
#Minibatch loss: 3.288, learning rate: 0.010000
#Minibatch error: 6.2%
#Validation error: 7.1%
#Step 200 (epoch 0.23), 25.0 ms
#Minibatch loss: 3.456, learning rate: 0.010000
#Minibatch error: 14.1%
#Validation error: 3.7%

# I compared these step times to those on my laptop and there appeared to be around an 8-10x speedup.

