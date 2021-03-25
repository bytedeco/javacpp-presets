package org.bytedeco.nvcodec.samples.encoder;

import org.bytedeco.nvcodec.samples.dispose.Disposable;
import org.bytedeco.nvcodec.samples.exceptions.CudaException;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.nvcodec.nvencodeapi.NV_ENC_CUSTREAM_PTR;

import static org.bytedeco.nvcodec.samples.util.CudaUtil.checkCudaApiCall;
import static org.bytedeco.cuda.global.cudart.*;

public class NvCUStream implements Disposable {
    private CUctx_st device;
    private CUstream_st inputStream;
    private CUstream_st outputStream;

    public CUstream_st getInputStream() {
        return inputStream;
    }

    public CUstream_st getOutputStream() {
        return outputStream;
    }

    public NvCUStream(CUctx_st device, int cuStreamType, NvEncoderOutputInVidMemCuda encoder) {
        this.device = device;

        this.inputStream = new CUstream_st();
        this.outputStream = new CUstream_st();

        try {
            checkCudaApiCall(cuCtxPushCurrent(this.device));

            // Create CUDA streams
            if (cuStreamType == 1) {
                checkCudaApiCall(cuStreamCreate(this.inputStream, CU_STREAM_DEFAULT));

                this.outputStream = new CUstream_st(this.inputStream);
            } else if (cuStreamType == 2) {
                checkCudaApiCall(cuStreamCreate(this.inputStream, CU_STREAM_DEFAULT));
                checkCudaApiCall(cuStreamCreate(this.outputStream, CU_STREAM_DEFAULT));
            }

            checkCudaApiCall(cuCtxPopCurrent(null));

            PointerPointer pi = new PointerPointer<>(1);
            pi.put(this.inputStream);

            PointerPointer po = new PointerPointer<>(1);
            po.put(this.outputStream);

            encoder.setIOCudaStreams(new NV_ENC_CUSTREAM_PTR(pi), new NV_ENC_CUSTREAM_PTR(po));
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void dispose() {
        try {
            checkCudaApiCall(cuCtxPushCurrent(device));

            if (this.inputStream.address() == this.outputStream.address()) {
                if (this.inputStream != null)
                    checkCudaApiCall(cuStreamDestroy(this.inputStream));
            } else {
                if (this.inputStream != null)
                    checkCudaApiCall(cuStreamDestroy(this.inputStream));

                if (this.outputStream != null)
                    checkCudaApiCall(cuStreamDestroy(this.outputStream));
            }

            checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }
}
