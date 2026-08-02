package org.bytedeco.pytorch.serving.tensorrt.internal;

import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer; // used by cudaMalloc out-param
import org.bytedeco.pytorch.serving.tensorrt.TRTEngine;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtAllocationException;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;

import static org.bytedeco.cuda.global.cudart.*;

/**
 * Thin CUDA device buffer / stream helpers used by {@link TRTEngine}.
 *
 * <p>Corresponds to the host↔device copies Python samples perform before
 * {@code execute_async_v3} / {@code enqueueV3}.
 */
public final class CudaBuffers {
    private CudaBuffers() {}

    public static Pointer mallocDevice(long bytes) {
        if (bytes < 0) {
            throw new TrtInvalidArgumentException("device alloc size must be >= 0");
        }
        if (bytes == 0) {
            return new Pointer();
        }
        PointerPointer<Pointer> devPtr = new PointerPointer<>(1);
        try {
            NativeError.checkCuda(cudaMalloc(devPtr, bytes), "cudaMalloc(" + bytes + ")");
            Pointer p = devPtr.get(0);
            if (p == null || p.isNull()) {
                throw new TrtAllocationException(
                        "cudaMalloc returned null for " + bytes + " bytes");
            }
            // Retain a stable Pointer instance owning the address.
            return new Pointer(p);
        } finally {
            devPtr.deallocate();
        }
    }

    public static void freeDevice(Pointer devicePtr) {
        if (devicePtr == null || devicePtr.isNull()) {
            return;
        }
        NativeError.checkCuda(cudaFree(devicePtr), "cudaFree");
    }

    public static void copyHostToDevice(Pointer deviceDst, Pointer hostSrc, long bytes) {
        if (bytes == 0) {
            return;
        }
        NativeError.checkCuda(
                cudaMemcpy(deviceDst, hostSrc, bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D");
    }

    public static void copyDeviceToHost(Pointer hostDst, Pointer deviceSrc, long bytes) {
        if (bytes == 0) {
            return;
        }
        NativeError.checkCuda(
                cudaMemcpy(hostDst, deviceSrc, bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H");
    }

    public static byte[] copyDeviceToHostBytes(Pointer deviceSrc, long bytes) {
        if (bytes < 0 || bytes > Integer.MAX_VALUE) {
            throw new TrtInvalidArgumentException("invalid D2H size: " + bytes);
        }
        byte[] out = new byte[(int) bytes];
        if (bytes == 0) {
            return out;
        }
        BytePointer bp = new BytePointer(bytes);
        try {
            copyDeviceToHost(bp, deviceSrc, bytes);
            bp.position(0).limit(bytes).get(out);
            return out;
        } finally {
            bp.deallocate();
        }
    }

    /**
     * Create a CUDA stream ({@code cudaStreamCreate}).
     *
     * <p>JavaCPP maps {@code @ByPtrPtr CUstream_st} to a single {@link CUstream_st}
     * out-param (filled in-place).
     */
    public static CUstream_st createStreamPtr() {
        CUstream_st out = new CUstream_st();
        NativeError.checkCuda(cudaStreamCreate(out), "cudaStreamCreate");
        if (out.isNull()) {
            throw new TrtAllocationException("cudaStreamCreate returned null");
        }
        return out;
    }

    public static void destroyStream(CUstream_st stream) {
        if (stream == null || stream.isNull()) {
            return;
        }
        NativeError.checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }

    public static void synchronize(CUstream_st stream) {
        if (stream == null || stream.isNull()) {
            NativeError.checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            return;
        }
        NativeError.checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}
