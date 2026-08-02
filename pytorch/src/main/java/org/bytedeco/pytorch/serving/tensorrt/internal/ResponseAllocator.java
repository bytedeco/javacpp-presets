package org.bytedeco.pytorch.serving.tensorrt.internal;

import org.bytedeco.pytorch.serving.tensorrt.TRTEngine;

/**
 * Placeholder for custom {@code nvinfer1::IOutputAllocator} integration (Phase 2).
 *
 * <p>MVP {@link TRTEngine} pre-allocates device output buffers
 * from resolved shapes after {@code setInputShape}, which matches the common
 * Python sample pattern of fixed output bindings. Dynamic-size outputs that
 * require {@code IOutputAllocator} reallocation are deferred.
 *
 * @see org.bytedeco.tensorrt.nvinfer.IOutputAllocator
 */

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.tensorrt.nvinfer.IOutputAllocator;
import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.tensorrt.nvinfer.Dims64;

/**
 * ResponseAllocator implements {@code nvinfer1::IOutputAllocator} for TensorRT Python-style API.
 *
 * <p>MVP: returns pre-allocated device buffers from
 * Dynamic output reallocation (Phase 2) can reuse this skeleton.
 *
 * <p>Methods are virtual via JavaCPP (see {@link org.bytedeco.tensorrt.nvinfer.IOutputAllocator}).
 */
public final class ResponseAllocator  {

    private final Pointer deviceBuffer;

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     *
     * @param deviceBuffer
     */
    public ResponseAllocator(Pointer deviceBuffer) {
//        super(deviceBuffer);
        this.deviceBuffer = deviceBuffer;
    }

//    @Override
    public Pointer reallocateOutput(String tensorName, Pointer currentMemory, long size, long alignment) {
        // Deprecated method - fall back to Async
        return reallocateOutputAsync(tensorName, currentMemory, size, alignment, null);
    }

//    @Override
    public Pointer reallocateOutputAsync(String tensorName, Pointer currentMemory, long size, long alignment, CUstream_st stream) {
        if (currentMemory != null && currentMemory.address() == deviceBuffer.address()) {
            return currentMemory; // reuse pre-allocated
        }
        // Dynamic case: could free current and allocate new, but MVP uses pre-alloc
        return null;
    }

//    @Override
    public void notifyShape(String tensorName, Dims64 dims) {
        // Shape known; pre-allocation already done in TRTEngine
    }
}


