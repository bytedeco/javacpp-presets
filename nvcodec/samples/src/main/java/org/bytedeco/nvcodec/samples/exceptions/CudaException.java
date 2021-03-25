package org.bytedeco.nvcodec.samples.exceptions;

public class CudaException extends Exception {
    public CudaException(int result) {
        super("CUDA error number : " + result);
    }
}
