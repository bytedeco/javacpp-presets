package org.bytedeco.nvcodec.samples.encoder;

import org.bytedeco.nvcodec.samples.exceptions.CudaException;
import org.bytedeco.nvcodec.samples.exceptions.NvCodecException;
import org.bytedeco.nvcodec.samples.util.VectorEx;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.nvcodec.global.nvencodeapi.*;
import org.bytedeco.nvcodec.nvencodeapi.*;

import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import static org.bytedeco.nvcodec.global.nvencodeapi.NV_ENC_OUTPUT_MOTION_VECTOR;
import static org.bytedeco.nvcodec.samples.util.CudaUtil.checkCudaApiCall;
import static org.bytedeco.nvcodec.samples.util.NvCodecUtil.checkNvCodecApiCall;
import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.nvcodec.global.nvencodeapi.*;

public class NvEncoderOutputInVidMemCuda extends NvEncoderCuda {
    private VectorEx<NV_ENC_OUTPUT_PTR> mappedOutputBuffers;
    private Vector<NV_ENC_OUTPUT_PTR> outputBuffers;
    private Vector<NV_ENC_REGISTERED_PTR> registeredResourcesOutputBuffer;

    public NvEncoderOutputInVidMemCuda(CUctx_st cudaContext, int width, int height, int bufferFormat) {
        this(cudaContext, width, height, bufferFormat, false);
    }

    public NvEncoderOutputInVidMemCuda(CUctx_st cudaContext, int width, int height, int bufferFormat, boolean motionEstimationOnly) {
        super(cudaContext, width, height, bufferFormat, 0, motionEstimationOnly, true);
        this.mappedOutputBuffers = new VectorEx<>();
        this.outputBuffers = new Vector<>();
        this.registeredResourcesOutputBuffer = new Vector<>();
    }

    private int alignUp(int s, int a) {
        return (s + a - 1) & ~(a - 1);
    }

    public int getOutputBufferSize() {
        int bufferSize = 0;
        if (this.motionEstimationOnly) {
            int encodeWidthInMbs = (this.getEncodeWidth() + 15) >> 4;
            int encodeHeightInMbs = (this.getEncodeHeight() + 15) >> 4;
            bufferSize = encodeWidthInMbs * encodeHeightInMbs + Pointer.sizeof(NV_ENC_H264_MV_DATA.class);
        } else {
            // 2-times the input size
            bufferSize = this.getFrameSize() * 2;

            bufferSize += Pointer.sizeof(NV_ENC_ENCODE_OUT_PARAMS.class);
        }
        return this.alignUp(bufferSize, 4);
    }

    /**
     * @brief This function is used to allocate output buffers in video memory for storing
     * encode or motion estimation output.
     */
    private void allocateOutputBuffers(int numOutputBuffers) {
        int size = this.getOutputBufferSize();

        try {
            checkCudaApiCall(cuCtxPushCurrent(this.cudaContext));
        } catch (CudaException e) {
            e.printStackTrace();
        }

        for (int index = 0; index < numOutputBuffers; index++) {
            final LongPointer deviceFramePointer = new LongPointer(1);

            try {
                checkCudaApiCall(cuMemAlloc(deviceFramePointer, size));
            } catch (CudaException e) {
                e.printStackTrace();
            }

            this.outputBuffers.add(new NV_ENC_OUTPUT_PTR() {
                {
                    address = deviceFramePointer.get();
                }
            });
        }

        try {
            checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException e) {
            e.printStackTrace();
        }

        this.registerOutputResources(size);
    }

    /**
     * @brief This function is used to release output buffers.
     */
    private void releaseOutputBuffers() {
        if (this.encoder != null && !this.encoder.isNull()) {
            this.unregisterOutputResources();

            for (Pointer pointer : this.outputBuffers) {
                cudaFree(pointer);
            }

            this.outputBuffers.clear();
        }
    }

    /**
     * @brief This function is used to register output buffers with NvEncodeAPI.
     */
    private void registerOutputResources(int bfrSize) {
        int bufferUsage = this.motionEstimationOnly ? NV_ENC_OUTPUT_MOTION_VECTOR : NV_ENC_OUTPUT_BITSTREAM;

        for (NV_ENC_OUTPUT_PTR pointer : this.outputBuffers) {
            if (pointer != null && !pointer.isNull()) {
                NV_ENC_REGISTERED_PTR registeredPointer = this.registerResource(pointer, NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, bfrSize, 1, bfrSize, NV_ENC_BUFFER_FORMAT_U8, bufferUsage);

                this.registeredResourcesOutputBuffer.add(registeredPointer);
            }
        }
    }

    /**
     * @brief This function is used to unregister output resources which had been previously registered for encoding
     * using RegisterOutputResources() function.
     */
    private void unregisterOutputResources() {
        for (Pointer pointer : this.mappedOutputBuffers) {
            if (pointer != null && !pointer.isNull()) {
                this.nvEncodeApiFunctionList.nvEncUnmapInputResource().call(this.encoder, (NV_ENC_INPUT_PTR) pointer);
            }
        }

        this.mappedOutputBuffers.clear();

        for (NV_ENC_REGISTERED_PTR pointer : this.registeredResourcesOutputBuffer) {
            if (pointer != null && !pointer.isNull()) {
                this.nvEncodeApiFunctionList.nvEncUnregisterResource().call(this.encoder, pointer);
            }
        }

        this.registeredResourcesOutputBuffer.clear();
    }

    /**
     * @brief This function is used to initialize the encoder session.
     * Application must call this function to initialize the encoder, before
     * starting to encode or motion estimate any frames.
     */
    @Override
    public void createEncoder(NV_ENC_INITIALIZE_PARAMS encoderParams) {
        super.createEncoder(encoderParams);

        this.allocateOutputBuffers(this.encoderBuffer);
        this.mappedOutputBuffers.resize(this.encoderBuffer, null);
    }

    /**
     * @brief This function is used to map the input and output buffers to NvEncodeAPI.
     */
    @Override
    protected void mapResources(int index) {
        super.mapResources(index);

        //map output surface
        NV_ENC_MAP_INPUT_RESOURCE mapInputResourceBitstreamBuffer = new NV_ENC_MAP_INPUT_RESOURCE();
        mapInputResourceBitstreamBuffer.version(NV_ENC_MAP_INPUT_RESOURCE_VER);
        mapInputResourceBitstreamBuffer.registeredResource(this.registeredResourcesOutputBuffer.get(index));

        try {
            checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncMapInputResource().call(this.encoder, mapInputResourceBitstreamBuffer));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        this.mappedOutputBuffers.set(index, new NV_ENC_OUTPUT_PTR(mapInputResourceBitstreamBuffer.mappedResource()));
    }


    public void encodeFrame(List<NV_ENC_OUTPUT_PTR> outputBuffer) {
        this.encodeFrame(outputBuffer, null);
    }

    /**
     * @brief This function is used to encode a frame.
     * Applications must call EncodeFrame() function to encode the uncompressed
     * data, which has been copied to an input buffer obtained from the
     * GetNextInputFrame() function.
     * This function returns video memory buffer pointers containing compressed data
     * in pOutputBuffer. If there is buffering enabled, this may return without
     * any data in pOutputBuffer.
     */
    public void encodeFrame(List<NV_ENC_OUTPUT_PTR> outputBuffer, NV_ENC_PIC_PARAMS picParams) {
        outputBuffer.clear();
        if (!this.isHWEncoderInitialized()) {
            System.err.println("Encoder device not found");
        }

        int index = this.toSend % this.encoderBuffer;

        this.mapResources(index);

        int nvStatus = this.doEncode(this.mappedInputBuffers.get(index), this.mappedOutputBuffers.get(index), picParams);

        if (nvStatus == NV_ENC_SUCCESS || nvStatus == NV_ENC_ERR_NEED_MORE_INPUT) {
            this.toSend++;
            this.getEncodedPacket(outputBuffer, true);
        } else {
            System.err.println("nvEncEncodePicture API failed");
        }
    }

    /**
     * @brief This function to flush the encoder queue.
     * The encoder might be queuing frames for B picture encoding or lookahead;
     * the application must call EndEncode() to get all the queued encoded frames
     * from the encoder. The application must call this function before destroying
     * an encoder session. Video memory buffer pointer containing compressed data
     * is returned in pOutputBuffer.
     */
    public void endEncode(List<NV_ENC_OUTPUT_PTR> outputBuffer) {
        if (!this.isHWEncoderInitialized()) {
            System.err.println("Encoder device not initialized");
        }
        this.sendEOS();

        this.getEncodedPacket(outputBuffer, false);
    }

    /**
     * @brief This function is used to run motion estimation.
     * This is used to run motion estimation on a a pair of frames. The
     * application must copy the reference frame data to the buffer obtained
     * by calling GetNextReferenceFrame(), and copy the input frame data to
     * the buffer obtained by calling GetNextInputFrame() before calling the
     * RunMotionEstimation() function.
     * This function returns video memory buffer pointers containing
     * motion vector data in pOutputBuffer.
     */
    public void runMotionEstimation(List<NV_ENC_OUTPUT_PTR> outputBuffer) {
        outputBuffer.clear();

        if (this.encoder != null && !this.encoder.isNull()) {
            int index = this.toSend % this.encoderBuffer;

            this.mapResources(index);

            int nvStatus = this.doMotionEstimation(this.mappedInputBuffers.get(index), this.mappedRefBuffers.get(index), this.mappedOutputBuffers.get(index));

            if (nvStatus == NV_ENC_SUCCESS) {
                this.toSend++;
                this.getEncodedPacket(outputBuffer, true);
            } else {
                System.err.println("nvEncRunMotionEstimationOnly API failed");
            }
        } else {
            System.err.println("Encoder Initialization failed");
        }
    }

    /**
     * @brief This is a private function which is used to get video memory buffer pointer containing compressed data
     * or motion estimation output from the encoder HW.
     * This is called by EncodeFrame() function. If there is buffering enabled,
     * this may return without any output data.
     */
    private void getEncodedPacket(List<NV_ENC_OUTPUT_PTR> outputBuffer, boolean outputDelay) {
        int end = outputDelay ? this.toSend - this.outputDelay : this.toSend;

        for (; this.got < end; this.got++) {
            int index = this.got % this.encoderBuffer;
            Pointer pointer = this.mappedOutputBuffers.get(index);

            if (pointer != null && !pointer.isNull()) {
                try {
                    checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncUnmapInputResource().call(this.encoder, new NV_ENC_INPUT_PTR(pointer)));
                } catch (NvCodecException e) {
                    e.printStackTrace();
                }
                this.mappedOutputBuffers.set(index, null);
            }

            pointer = this.mappedInputBuffers.get(index);

            if (pointer != null && !pointer.isNull()) {
                try {
                    checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncUnmapInputResource().call(this.encoder, (NV_ENC_INPUT_PTR) pointer));
                } catch (NvCodecException e) {
                    e.printStackTrace();
                }
                this.mappedOutputBuffers.set(index, null);
            }

            if (this.motionEstimationOnly) {
                pointer = this.mappedRefBuffers.get(index);

                if (pointer != null && !pointer.isNull()) {
                    try {
                        checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncUnmapInputResource().call(this.encoder, (NV_ENC_INPUT_PTR) pointer));
                    } catch (NvCodecException e) {
                        e.printStackTrace();
                    }
                    this.mappedOutputBuffers.set(index, null);
                }
            }

            outputBuffer.add(this.outputBuffers.get(index));
        }
    }

    /**
     * @brief This function is used to flush the encoder queue.
     */
    private void flushEncoder() {
        if (this.encoder != null && !this.encoder.isNull()) {
            if (!this.motionEstimationOnly) {
                List<NV_ENC_OUTPUT_PTR> outputBuffer = new LinkedList<>();

                this.endEncode(outputBuffer);
            }
        }
    }

    @Override
    public void destroyEncoder() {
        if (this.encoder != null && !this.encoder.isNull()) {
            // Incase of error it is possible for buffers still mapped to encoder.
            // flush the encoder queue and then unmapped it if any surface is still mapped
            this.flushEncoder();
            this.releaseOutputBuffers();

            super.destroyEncoder();
        }
    }

    @Override
    public void dispose() {
        this.flushEncoder();
        this.releaseOutputBuffers();

        super.dispose();
    }
}
