FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
#FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN mkdir /app
WORKDIR /app

#RUN apt-get update && apt-get -y upgrade && apt-get install -y wget
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcublas-dev_10.1.0.105-1_amd64.deb
#RUN dpkg -i libcublas-dev_10.1.0.105-1_amd64.deb

# RUN cd /home/ && git clone https://github.com/NVIDIA/apex.git apex && cd apex && python setup.py install --cuda_ext --cpp_ext
RUN git clone https://github.com/NVIDIA/apex.git apex && cd apex && git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
ADD requirements.txt ./
RUN pip install -r requirements.txt

RUN git clone https://github.com/mapillary/inplace_abn.git
RUN cd inplace_abn && python setup.py install
RUN python3 -c "from inplace_abn import InPlaceABN;"

