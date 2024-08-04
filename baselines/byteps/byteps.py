# sudo apt install libnccl2=2.20.3-1+cuda12.3 libnccl-dev=2.20.3-1+cuda12.3
# dpkg -L libnccl2 libnccl-dev
# sudo find / -name "*libnccl*"
# cp /usr/include/nccl.h /usr/local/cuda-12.3/include/nccl.h
# ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda-12.3/nccl/lib/libnccl.so.2
# export BYTEPS_NCCL_HOME=/usr
# pip3 install byteps