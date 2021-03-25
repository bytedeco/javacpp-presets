package org.bytedeco.nvcodec.samples.cuda;

import org.bytedeco.nvcodec.samples.dispose.Disposable;
import org.bytedeco.nvcodec.samples.exceptions.CudaException;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.cuda.cudart.CUfunc_st;
import org.bytedeco.cuda.cudart.CUmod_st;
import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.javacpp.*;
import org.bytedeco.nvcodec.nvencodeapi.NV_ENC_OUTPUT_PTR;

import static org.bytedeco.nvcodec.samples.util.CudaUtil.checkCudaApiCall;
import static org.bytedeco.cuda.global.cudart.*;

public class CRC implements Disposable {
    private CUmod_st module;
    private CUctx_st device;
    private CUfunc_st cudaFunction;
    private LongPointer crcVidMem;

    public CRC(CUctx_st device, int bufferSize) {
        this.device = device;
        this.crcVidMem = new LongPointer(1);

        this.module = new CUmod_st();
        try {
            checkCudaApiCall(cuCtxPushCurrent(this.device));

            // Allocate video memory buffer to store CRC of encoded frame
            checkCudaApiCall(cuMemAlloc(this.crcVidMem, bufferSize));

            String path = getClass().getClassLoader().getResource("modules/crc.ptx").getPath().substring(1);
            checkCudaApiCall(cuModuleLoad(this.module, new BytePointer(path)));

            checkCudaApiCall(cuCtxPopCurrent(null));

            this.cudaFunction = new CUfunc_st();

            checkCudaApiCall(cuModuleGetFunction(this.cudaFunction, this.module, "_Z16ComputeCRCKernelPhPj"));
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }

    public LongPointer getCrcVidMem() {
        return crcVidMem;
    }

    public void getCRC(NV_ENC_OUTPUT_PTR videoMemBfr, CUstream_st outputStream) {
        this.computeCRC(videoMemBfr, this.crcVidMem, outputStream);
    }

    public void computeCRC(NV_ENC_OUTPUT_PTR buffer, LongPointer crcValue, CUstream_st outputCUStream) {
        Pointer[] pointers = new Pointer[]{
                new LongPointer(new long[]{
                        buffer.address()
                }),
                new LongPointer(new long[]{
                        crcValue.get()
                })
        };
        PointerPointer<Pointer> kernelParameters = new PointerPointer<Pointer>(pointers);

        try {
            checkCudaApiCall(cuLaunchKernel(this.cudaFunction, 1, 1, 1, 1, 1, 1, 0, outputCUStream, kernelParameters, null));
        } catch (CudaException e) {
            e.printStackTrace();
        }

        kernelParameters.deallocate();
    }

    @Override
    public void dispose() {
        try {
            checkCudaApiCall(cuCtxPushCurrent(this.device));
            checkCudaApiCall(cuMemFree(this.crcVidMem.get()));
            checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }
}
