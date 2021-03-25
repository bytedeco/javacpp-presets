package org.bytedeco.nvcodec.samples.encoder;

import org.bytedeco.nvcodec.samples.exceptions.CudaException;
import org.bytedeco.nvcodec.samples.util.CudaUtil;
import org.bytedeco.nvcodec.samples.dispose.Disposable;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.nvcodec.nvencodeapi.NV_ENC_ENCODE_OUT_PARAMS;

import java.io.*;

import static org.bytedeco.cuda.global.cudart.*;

public class DumpVidMemOutput implements Disposable {
    private int bfrSize;
    private boolean useCRC;
    private String crcFile;

    private CUctx_st device;

    private Pointer hostMemEncOp;
    private IntPointer hostMemCRC;
    private FileWriter writer;

    public DumpVidMemOutput(CUctx_st device, final int bfrSize, String outputFilePath, boolean useCUStream) {
        this.device = device;
        this.bfrSize = bfrSize;
        this.useCRC = useCUStream;
        this.hostMemCRC = new IntPointer();
        this.hostMemEncOp = new Pointer();
        try {
            CudaUtil.checkCudaApiCall(cuCtxPushCurrent(this.device));

            // Allocate host memory buffer to copy encoded output and CRC
            CudaUtil.checkCudaApiCall(cuMemAllocHost(this.hostMemEncOp, (this.bfrSize + (useCUStream ? 4 : 0))));

            CudaUtil.checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException e) {
            e.printStackTrace();
        }

        // Open file to dump CRC
        if (this.useCRC) {
            this.crcFile = outputFilePath + "_crc.txt";

            try {
                this.writer = new FileWriter(this.crcFile);
            } catch (IOException e) {
                e.printStackTrace();
            }

            this.hostMemCRC = new IntPointer() {
                {
                    address = hostMemEncOp.address() + bfrSize;
                }
            };

        }
    }

    public void dumpOutputToFile(LongPointer encFrameBfr, LongPointer crcBfr, FileOutputStream outputStream, int frame) {
        try {
            CudaUtil.checkCudaApiCall(cuCtxPushCurrent(this.device));

            // Copy encoded frame from video memory buffer to host memory buffer
            CudaUtil.checkCudaApiCall(cuMemcpyDtoH(this.hostMemEncOp, encFrameBfr.address(), this.bfrSize));

            // Copy encoded frame CRC from video memory buffer to host memory buffer
            if (this.useCRC) {
                CudaUtil.checkCudaApiCall(cuMemcpyDtoH(this.hostMemCRC, crcBfr.get(), 4));
            }

            CudaUtil.checkCudaApiCall(cuCtxPopCurrent(null));

            // Write encoded bitstream in file

            int offset = Pointer.sizeof(NV_ENC_ENCODE_OUT_PARAMS.class);
            int bitstreamSize = new NV_ENC_ENCODE_OUT_PARAMS(this.hostMemEncOp).bitstreamSizeInBytes();

            Pointer pointer = this.hostMemEncOp.getPointer(offset);
            pointer.limit(bitstreamSize + pointer.position());

            byte[] data = new byte[bitstreamSize];
            pointer.asByteBuffer().get(data);

            try {
                outputStream.write(data);
            } catch (IOException e) {
                e.printStackTrace();
            }

            // Write CRC in file
            if (this.useCRC) {
                if (frame == 0) {
                    this.writer.write(String.format("Frame num       CRC %n"));
                }
                this.writer.write(String.format("%5d          %s%n", frame, Integer.toHexString(this.hostMemCRC.get())));
            }
        } catch (CudaException | IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void dispose() {
        try {
            CudaUtil.checkCudaApiCall(cuCtxPushCurrent(this.device));

            CudaUtil.checkCudaApiCall(cuMemFreeHost(this.hostMemEncOp));

            CudaUtil.checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException e) {
            e.printStackTrace();
        }

        if (this.useCRC) {
            try {
                this.writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
