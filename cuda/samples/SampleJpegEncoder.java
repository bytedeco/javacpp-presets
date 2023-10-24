/*
 * Copyright (C) 2023 Jeonghwan Park
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.bytedeco.cuda.cudart.*;
import org.bytedeco.cuda.nvjpeg.*;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.PointerPointer;

import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.cuda.global.nvjpeg.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class SampleJpegEncoder {
    static class devMalloc extends tDevMalloc {
        final static devMalloc instance = new devMalloc().retainReference();

        @Override
        public int call(PointerPointer pointerPointer, long l) {
            return cudaMalloc(pointerPointer, l);
        }
    }

    static class devFree extends tDevFree {
        final static devFree instance = new devFree().retainReference();

        @Override
        public int call(Pointer pointer) {
            return cudaFree(pointer);
        }
    }

    static class hostMalloc extends tPinnedMalloc {
        final static hostMalloc instance = new hostMalloc().retainReference();

        @Override
        public int call(PointerPointer pointerPointer, long l, int i) {
            return cudaHostAlloc(pointerPointer, l, i);
        }
    }

    static class hostFree extends tPinnedFree {
        final static hostFree instance = new hostFree().retainReference();

        @Override
        public int call(Pointer pointer) {
            return cudaFreeHost(pointer);
        }
    }

    public static void CHECK_CUDA(String functionName, int result) {
        if (result != CUDA_SUCCESS) {
            throw new IllegalStateException(String.format("%s returned '%d'", functionName, result));
        }
    }

    public static void CHECK_NVJPEG(String functionName, int result) {
        if (result != NVJPEG_STATUS_SUCCESS) {
            throw new IllegalStateException(String.format("%s returned '%d'", functionName, result));
        }
    }

    public static void main(String[] args) throws Exception {
        int imageWidth = 1280;
        int imageHeight = 720;

        nvjpegDevAllocator_t devAllocator = new nvjpegDevAllocator_t();
        devAllocator.dev_malloc(devMalloc.instance);
        devAllocator.dev_free(devFree.instance);

        nvjpegPinnedAllocator_t pinnedAllocator = new nvjpegPinnedAllocator_t();
        pinnedAllocator.pinned_malloc(hostMalloc.instance);
        pinnedAllocator.pinned_free(hostFree.instance);

        // Initialize cuda
        CUctx_st context = new CUctx_st();
        CHECK_CUDA("cuInit", cuInit(0));
        CHECK_CUDA("cuCtxCreate", cuCtxCreate(context, CU_CTX_SCHED_BLOCKING_SYNC, 0));

        // Create cuda stream
        CUstream_st stream = new CUstream_st();
        CHECK_CUDA("cuStreamCreate", cuStreamCreate(stream, 0));

        // Create handle
        nvjpegHandle nvjpegHandle = new nvjpegHandle();
        CHECK_NVJPEG("nvjpegCreateEx", nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, devAllocator, pinnedAllocator, NVJPEG_FLAGS_DEFAULT, nvjpegHandle));

        // Create encoder components
        nvjpegEncoderState nvjpegEncoderState = new nvjpegEncoderState();
        nvjpegEncoderParams nvjpegEncoderParams = new nvjpegEncoderParams();
        CHECK_NVJPEG("nvjpegEncoderParamsCreate", nvjpegEncoderParamsCreate(nvjpegHandle, nvjpegEncoderParams, stream));
        CHECK_NVJPEG("nvjpegEncoderStateCreate", nvjpegEncoderStateCreate(nvjpegHandle, nvjpegEncoderState, stream));
        CHECK_NVJPEG("nvjpegEncoderParamsSetSamplingFactors", nvjpegEncoderParamsSetSamplingFactors(nvjpegEncoderParams, NVJPEG_CSS_444, stream));

        // Create jpeg image
        nvjpegImage_t nvjpegImage = new nvjpegImage_t();

        long channelSize = imageWidth * imageHeight;

        // Fill image to blue
        for (int i = 0; i < 3; i++) {
            nvjpegImage.pitch(i, imageWidth);
            BytePointer deviceMemory = new BytePointer();

            CHECK_CUDA("cudaMalloc", cudaMalloc(deviceMemory, channelSize));
            CHECK_CUDA("cudaMemset", cudaMemset(deviceMemory, i == 0 ? 255 : 0, channelSize));

            nvjpegImage.channel(i, deviceMemory);
        }

        // Compress image
        CHECK_NVJPEG("nvjpegEncodeImage", nvjpegEncodeImage(nvjpegHandle, nvjpegEncoderState, nvjpegEncoderParams, nvjpegImage, NVJPEG_INPUT_BGR, imageWidth, imageHeight, stream));

        // Get compressed size
        SizeTPointer jpegSize = new SizeTPointer(1);
        CHECK_NVJPEG("nvjpegEncodeRetrieveBitstream", nvjpegEncodeRetrieveBitstream(nvjpegHandle, nvjpegEncoderState, (BytePointer) null, jpegSize, stream));

        // Retrieve bitstream
        BytePointer jpegBytePointer = new BytePointer(jpegSize.get());
        CHECK_NVJPEG("nvjpegEncodeRetrieveBitstream", nvjpegEncodeRetrieveBitstream(nvjpegHandle, nvjpegEncoderState, jpegBytePointer, jpegSize, stream));

        // Synchronize cuda stream
        CHECK_CUDA("cudaStreamSynchronize", cudaStreamSynchronize(stream));

        // Get bitstream to java side array
        byte[] bytes = new byte[(int) jpegSize.get()];
        jpegBytePointer.get(bytes);

        // Write
        Files.write(new File("out.jpg").toPath(), bytes);
    }
}
