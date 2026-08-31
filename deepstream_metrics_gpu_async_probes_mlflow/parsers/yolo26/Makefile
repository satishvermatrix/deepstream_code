# Build inside ds8-dev. No TensorRT custom engine — parse-only shared lib.
CXX ?= g++
CUDA_VER ?= 13.0
DS_INC ?= /opt/nvidia/deepstream/deepstream/sources/includes
CUDA_INC ?= /usr/local/cuda-$(CUDA_VER)/include
TARGET = libnvdsinfer_custom_impl_Yolo26.so

all: $(TARGET)

$(TARGET): nvdsparsebbox_yolo26.cpp
	$(CXX) -shared -fPIC -std=c++11 -O2 -I$(DS_INC) -I$(CUDA_INC) -o $@ $<

clean:
	rm -f $(TARGET)
