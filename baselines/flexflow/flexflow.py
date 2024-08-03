# sudo apt-get -y install cudnn9-cuda-12
# sudo apt install curl
# apt install cmake ninja-build
# specify CUDNN_DIR on .bashrc
# curl https://sh.rustup.rs -sSf | sh -s -- -y
# pip install flexflow
import os
# Set the CUDA version to 12.0 for flexflow
cuda_version = "12.0"
CUDA_HOME = "/usr/local/cuda"
# Append the new env variables to the existing environment variables
os.environ["PATH"] = f"{CUDA_HOME}-{cuda_version}/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = f"{CUDA_HOME}-{cuda_version}/lib64:" + os.environ["LD_LIBRARY_PATH"]
