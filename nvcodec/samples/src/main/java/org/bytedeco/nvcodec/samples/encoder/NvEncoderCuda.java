package org.bytedeco.nvcodec.samples.encoder;

import org.bytedeco.nvcodec.samples.util.NvCodecUtil;
import org.bytedeco.nvcodec.samples.exceptions.CudaException;
import org.bytedeco.nvcodec.samples.exceptions.NvCodecException;
import org.bytedeco.cuda.cudart.CUDA_MEMCPY2D_v2;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.nvcodec.nvencodeapi.NV_ENC_CUSTREAM_PTR;

import java.util.Vector;

import static org.bytedeco.nvcodec.global.nvencodeapi.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.nvcodec.samples.util.CudaUtil.*;

public class NvEncoderCuda extends NvEncoder {
    private SizeTPointer cudaPitch;
    protected CUctx_st cudaContext;

    /**
     * @brief This is a static function to copy input data from host memory to device memory.
     * This function assumes YUV plane is a single contiguous memory segment.
     */
    public static void copyToDeviceFrame(CUctx_st device, Pointer srcFrame, int srcPitch, LongPointer dstFrame, int dstPitch, int width, int height, int srcMemoryType, int pixelFormat, final int[] dstChromaOffsets, int numChromaPlanes) {
        copyToDeviceFrame(device, srcFrame, srcPitch, dstFrame, dstPitch, width, height, srcMemoryType, pixelFormat, dstChromaOffsets, numChromaPlanes, false, null);
    }

    public static void copyToDeviceFrame(CUctx_st device, Pointer srcFrame, int srcPitch, LongPointer dstFrame, int dstPitch, int width, int height, int srcMemoryType, int pixelFormat, final int[] dstChromaOffsets, int numChromaPlanes, boolean unAlignedDeviceCopy, CUstream_st stream) {
        if (srcMemoryType != CU_MEMORYTYPE_HOST && srcMemoryType != CU_MEMORYTYPE_DEVICE) {
            System.err.println("Invalid source memory type for copy");
        }

        try {
            checkCudaApiCall(cuCtxPushCurrent(device));
        } catch (CudaException e) {
            e.printStackTrace();
        }

        srcPitch = srcPitch != 0 ? srcPitch : getWidthInBytes(pixelFormat, width);

        CUDA_MEMCPY2D_v2 mem = new CUDA_MEMCPY2D_v2();
        mem.srcXInBytes(0);
        mem.srcMemoryType(srcMemoryType);
        if (srcMemoryType == CU_MEMORYTYPE_HOST) {
            mem.srcHost(srcFrame);
        } else {
            mem.srcDevice(srcFrame.address());
        }
        mem.srcPitch(srcPitch);
        mem.dstMemoryType(CU_MEMORYTYPE_DEVICE);
        mem.dstDevice(dstFrame.address());
        mem.dstPitch(dstPitch);
        mem.WidthInBytes(getWidthInBytes(pixelFormat, width));
        mem.Height(height);

        if (unAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE) {
            try {
                checkCudaApiCall(cuMemcpy2DUnaligned(mem));
            } catch (CudaException e) {
                e.printStackTrace();
            }
        } else {
            try {
                checkCudaApiCall(stream == null ? cuMemcpy2D(mem) : cuMemcpy2DAsync(mem, stream));
            } catch (CudaException e) {
                e.printStackTrace();
            }
        }

        Vector<Integer> srcChromaOffsets = new Vector<>();
        getChromaSubPlaneOffsets(pixelFormat, srcPitch, height, srcChromaOffsets);

        int chromaHeight = getChromaHeight(pixelFormat, height);
        int destChromaPitch = getChromaPitch(pixelFormat, dstPitch);
        int srcChromaPitch = getChromaPitch(pixelFormat, srcPitch);
        int chromaWidthInBytes = getChromaWidthInBytes(pixelFormat, width);

        for (int index = 0; index < numChromaPlanes; ++index) {
            if (chromaHeight != 0) {
                if (srcMemoryType == CU_MEMORYTYPE_HOST) {
                    mem.srcHost(srcFrame.getPointer(srcChromaOffsets.get(index)));
                } else {
                    mem.srcDevice(srcFrame.address() + dstChromaOffsets[index]);
                }

                mem.srcPitch(srcChromaPitch);

                mem.dstDevice(dstFrame.address() + dstChromaOffsets[index]);
                mem.dstPitch(destChromaPitch);
                mem.WidthInBytes(chromaWidthInBytes);
                mem.Height(chromaHeight);

                if (unAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE) {
                    try {
                        checkCudaApiCall(cuMemcpy2DUnaligned(mem));
                    } catch (CudaException e) {
                        e.printStackTrace();
                    }
                } else {
                    try {
                        checkCudaApiCall(stream == null ? cuMemcpy2D(mem) : cuMemcpy2DAsync(mem, stream));
                    } catch (CudaException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        try {
            checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }

    public static void copyToDeviceFrame(CUctx_st device, Pointer srcFrame, int srcPitch, LongPointer dstFrame, int dstPitch, int width, int height, int srcMemoryType, int pixelFormat, LongPointer[] dstChromaDevicePointers, int dstChromaPitch, int numChromaPlanes) {
        copyToDeviceFrame(device, srcFrame, srcPitch, dstFrame, dstPitch, width, height, srcMemoryType, pixelFormat, dstChromaDevicePointers, dstChromaPitch, numChromaPlanes, false);
    }

    public static void copyToDeviceFrame(CUctx_st device, Pointer srcFrame, int srcPitch, LongPointer dstFrame, int dstPitch, int width, int height, int srcMemoryType, int pixelFormat, LongPointer[] dstChromaDevicePointers, int dstChromaPitch, int numChromaPlanes, boolean unAlignedDeviceCopy) {
        if (srcMemoryType != CU_MEMORYTYPE_HOST && srcMemoryType != CU_MEMORYTYPE_DEVICE) {
            System.err.println("Invalid source memory type for copy");
        }

        try {
            checkCudaApiCall(cuCtxPushCurrent(device));
        } catch (CudaException e) {
            e.printStackTrace();
        }

        CUDA_MEMCPY2D_v2 mem = new CUDA_MEMCPY2D_v2();
        mem.srcXInBytes(0);
        mem.srcMemoryType(srcMemoryType);

        if (srcMemoryType == CU_MEMORYTYPE_HOST) {
            mem.srcHost(srcFrame);
        } else {
            mem.srcDevice(srcFrame.address());
        }
        mem.srcPitch(srcPitch);
        mem.dstMemoryType(CU_MEMORYTYPE_DEVICE);
        mem.dstDevice(dstFrame.get());
        mem.dstPitch(dstPitch);
        mem.WidthInBytes(getWidthInBytes(pixelFormat, width));
        mem.Height(height);

        if (unAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE) {
            try {
                checkCudaApiCall(cuMemcpy2DUnaligned(mem));
            } catch (CudaException e) {
                e.printStackTrace();
            }
        } else {
            try {
                checkCudaApiCall(cuMemcpy2D(mem));
            } catch (CudaException e) {
                e.printStackTrace();
            }
        }

        Vector<Integer> srcChromaOffsets = new Vector<>();
        getChromaSubPlaneOffsets(pixelFormat, srcPitch, height, srcChromaOffsets);
        int chromaHeight = getChromaHeight(pixelFormat, height);
        int srcChromaPitch = getChromaPitch(pixelFormat, srcPitch);
        int chromaWidthInBytes = getChromaWidthInBytes(pixelFormat, width);

        for (int index = 0; index < numChromaPlanes; ++index) {
            if (chromaHeight != 0) {
                if (srcMemoryType == CU_MEMORYTYPE_HOST) {
                    mem.srcHost(srcFrame.getPointer(srcChromaOffsets.get(index)));
                } else {
                    mem.srcDevice(srcFrame.address() + srcChromaOffsets.get(index));
                }
                mem.srcPitch(srcChromaPitch);
                mem.dstDevice(dstChromaDevicePointers[index].get());
                mem.dstPitch(dstChromaPitch);
                mem.WidthInBytes(chromaWidthInBytes);
                mem.Height(chromaHeight);

                if (unAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE) {
                    try {
                        checkCudaApiCall(cuMemcpy2DUnaligned(mem));
                    } catch (CudaException e) {
                        e.printStackTrace();
                    }
                } else {
                    try {
                        checkCudaApiCall(cuMemcpy2D(mem));
                    } catch (CudaException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        try {
            checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }

    public NvEncoderCuda(CUctx_st cudaContext, int width, int height, int bufferFormat) {
        this(cudaContext, width, height, bufferFormat, 3, false, false);

    }

    public NvEncoderCuda(CUctx_st cudaContext, int width, int height, int bufferFormat, int extraOutputDelay, boolean motionEstimationOnly, boolean outputInVideoMemory) {
        super(NV_ENC_DEVICE_TYPE_CUDA, cudaContext, width, height, bufferFormat, extraOutputDelay, motionEstimationOnly, outputInVideoMemory);
        this.cudaPitch = new SizeTPointer(1);
        this.cudaContext = cudaContext;

        if (this.encoder == null || this.encoder.isNull()) {
            System.err.println("Encoder Initialization failed");
        }

        if (this.cudaContext == null || this.cudaContext.isNull()) {
            System.err.println("Invalid Cuda Context");
        }
    }

    @Override
    protected void allocateInputBuffers(int numInputBuffers) {
        if (!this.isHWEncoderInitialized()) {
            System.err.println("Encoder intialization failed");
        }

        // for MEOnly mode we need to allocate seperate set of buffers for reference frame
        int numCount = this.motionEstimationOnly ? 2 : 1;

        for (int count = 0; count < numCount; count++) {
            try {
                checkCudaApiCall(cuCtxPushCurrent(this.cudaContext));
            } catch (CudaException e) {
                e.printStackTrace();
            }
            Vector<Pointer> inputFrames = new Vector<>();
            int pixelFormat = this.getPixelFormat();

            for (int index = 0; index < numInputBuffers; index++) {
                final LongPointer deviceFramePointer = new LongPointer(1);

                int chromaHeight = getNumChromaPlanes(pixelFormat) * getChromaHeight(pixelFormat, this.getMaxEncodedHeight());

                if (pixelFormat == NV_ENC_BUFFER_FORMAT_YV12 || pixelFormat == NV_ENC_BUFFER_FORMAT_IYUV) {
                    chromaHeight = getChromaHeight(pixelFormat, getMaxEncodedHeight());
                }

                try {
                    checkCudaApiCall(cuMemAllocPitch(deviceFramePointer, this.cudaPitch, getWidthInBytes(pixelFormat, getMaxEncodeWidth()), getMaxEncodedHeight() + chromaHeight, 16));
                } catch (CudaException e) {
                    e.printStackTrace();
                }

                inputFrames.add(new Pointer() {{
                    address = deviceFramePointer.get();
                }});
            }
            try {
                checkCudaApiCall(cuCtxPopCurrent(null));
            } catch (CudaException e) {
                e.printStackTrace();
            }

            this.registerInputResources(inputFrames, NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, this.getMaxEncodeWidth(), this.getMaxEncodedHeight(), (int) this.cudaPitch.get(), pixelFormat, count == 1);
        }
    }


    public void setIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream, NV_ENC_CUSTREAM_PTR outputStream) {
        try {
            NvCodecUtil.checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncSetIOCudaStreams().call(this.encoder, inputStream, outputStream));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void releaseInputBuffers() {
        this.releaseCudaResources();
    }

    private void releaseCudaResources() {
        if (this.encoder != null && !this.encoder.isNull()) {
            if (this.cudaContext != null && !this.cudaContext.isNull()) {
                this.unregisterInputResources();

                cuCtxPushCurrent(this.cudaContext);

                for (NvEncoderInputFrame inputFrame : this.inputFrames) {
                    Pointer pointer = inputFrame.getInputPointer();
                    if (pointer != null && !pointer.isNull()) {
                        cudaFree(pointer);
                    }
                }
                this.inputFrames.clear();

                for (NvEncoderInputFrame referenceFrame : this.referenceFrames) {
                    Pointer pointer = referenceFrame.getInputPointer();
                    if (pointer != null && !pointer.isNull()) {
                        cudaFree(pointer);
                    }
                }
                this.referenceFrames.clear();

                cuCtxPopCurrent(null);
                this.cudaContext = null;
            }
        }
    }

    @Override
    public void dispose() {
        this.releaseCudaResources();

        super.dispose();
    }
}
