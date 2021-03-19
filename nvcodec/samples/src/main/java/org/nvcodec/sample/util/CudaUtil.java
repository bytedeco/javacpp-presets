package org.nvcodec.sample.util;

import org.nvcodec.sample.exceptions.CudaException;

public class CudaUtil {
    public static void checkCudaApiCall(int result) throws CudaException {
        if (result != 0) {
            throw new CudaException(result);
        }
    }
}
