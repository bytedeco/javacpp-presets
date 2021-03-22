package org.bytedeco.nvcodec.sample.util;

import org.bytedeco.nvcodec.sample.exceptions.CudaException;

public class CudaUtil {
    public static void checkCudaApiCall(int result) throws CudaException {
        if (result != 0) {
            throw new CudaException(result);
        }
    }
}
