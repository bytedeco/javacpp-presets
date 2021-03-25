package org.bytedeco.nvcodec.samples.util;

import org.bytedeco.nvcodec.samples.exceptions.CudaException;

public class CudaUtil {
    public static void checkCudaApiCall(int result) throws CudaException {
        if (result != 0) {
            throw new CudaException(result);
        }
    }
}
