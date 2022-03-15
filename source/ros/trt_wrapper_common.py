from itertools import chain
import argparse
import os
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import threading
import time
import math
import torch
import imageio
import random
import string
    
class TRTInference:
    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    def __init__(self, trt_engine_path, batch_size, multiThread=False):
        if multiThread:
            self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        # img information 
        self.batch_size = batch_size
        self.multiThread = multiThread

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        with open(trt_engine_path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        context = engine.create_execution_context()

        #inputs, outputs, bindings, stream = self.allocate_buffers(engine, self.batch_size)
        inputs, outputs, bindings, stream = self.allocate_buffers(engine, 1)

        # store
        self.stream  = stream
        self.context = context
        self.engine  = engine

        self.bindings = bindings
        self.inputs = inputs 
        self.outputs = outputs
    
    def allocate_buffers(self, engine, batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:

            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dims = engine.get_binding_shape(binding)
        
            # in case batch dimension is -1 (dynamic)
            if dims[0] < 0:
                size *= -1

            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def infer(self, inImg):


        # img information
        multiThread = self.multiThread
        batch_size = self.batch_size

        if multiThread:
            self.cfx.push()
        
        # restore
        stream  = self.stream
        context = self.context

        inputs = self.inputs
        outputs = self.outputs
        bindings = self.bindings

        if batch_size == 1:
            # read image
            inputs[0].host = inImg[0].ravel()
        else:
            # read image
            imgs1d = [img.ravel() for img in inImg]
            img_in = np.concatenate(imgs1d, axis=0)
            inputs[0].host = img_in

        # inference
        trt_outputs = self.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        if multiThread:
            self.cfx.pop()
        
        return trt_outputs
    
    def destory(self):
        self.cfx.pop()
        
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()