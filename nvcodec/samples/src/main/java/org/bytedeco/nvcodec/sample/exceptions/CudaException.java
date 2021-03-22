package org.bytedeco.nvcodec.sample.exceptions;

public class CudaException extends Exception {
    public CudaException(int result) {
        super("CUDA error number : " + result);
    }
}
