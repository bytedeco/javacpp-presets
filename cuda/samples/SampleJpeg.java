/*
 * Copyright (C) 2022 Park JeongHwan
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

import org.bytedeco.cuda.nvjpeg.*;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;

import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.cuda.global.nvjpeg.*;

public class SampleJpeg {
    static class dev_malloc extends tDevMalloc {
        final static dev_malloc instance = new dev_malloc().retainReference();

        @Override
        public int call(PointerPointer pointerPointer, long l) {
            return cudaMalloc(pointerPointer, l);
        }
    }

    static class dev_free extends tDevFree {
        final static dev_free instance = new dev_free().retainReference();

        @Override
        public int call(Pointer pointer) {
            return cudaFree(pointer);
        }
    }

    static class host_malloc extends tPinnedMalloc {
        final static host_malloc instance = new host_malloc().retainReference();

        @Override
        public int call(PointerPointer pointerPointer, long l, int i) {
            return cudaHostAlloc(pointerPointer, l, i);
        }
    }

    static class host_free extends tPinnedFree {
        final static host_free instance = new host_free().retainReference();

        @Override
        public int call(Pointer pointer) {
            return cudaFreeHost(pointer);
        }
    }

    public static void CHECK_NVJPEG(String functionName, int result) {
        if (result != NVJPEG_STATUS_SUCCESS) {
            throw new IllegalStateException(String.format("%s returned '%d'", functionName, result));
        }
    }

    public static void main(String[] args) {
        nvjpegDevAllocator_t devAllocator = new nvjpegDevAllocator_t();
        devAllocator.dev_malloc(dev_malloc.instance);
        devAllocator.dev_free(dev_free.instance);

        nvjpegPinnedAllocator_t pinnedAllocator = new nvjpegPinnedAllocator_t();
        pinnedAllocator.pinned_malloc(host_malloc.instance);
        pinnedAllocator.pinned_free(host_free.instance);

        nvjpegHandle handle = new nvjpegHandle();
        nvjpegJpegState state = new nvjpegJpegState();

        nvjpegJpegDecoder decoder = new nvjpegJpegDecoder();
        nvjpegJpegState decoder_state = new nvjpegJpegState();

        nvjpegBufferPinned[] pinned_buffers = new nvjpegBufferPinned[]{new nvjpegBufferPinned(), new nvjpegBufferPinned()};
        nvjpegBufferDevice bufferDevice = new nvjpegBufferDevice();

        nvjpegJpegStream[] streams = new nvjpegJpegStream[]{new nvjpegJpegStream(), new nvjpegJpegStream()};

        nvjpegDecodeParams params = new nvjpegDecodeParams();

        // Create Components
        CHECK_NVJPEG("nvjpegCreateEx", nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, devAllocator, pinnedAllocator, NVJPEG_FLAGS_DEFAULT, handle));
        CHECK_NVJPEG("nvjpegJpegStateCreate", nvjpegJpegStateCreate(handle, state));

        CHECK_NVJPEG("nvjpegDecoderCreate", nvjpegDecoderCreate(handle, NVJPEG_BACKEND_DEFAULT, decoder));
        CHECK_NVJPEG("nvjpegDecoderStateCreate", nvjpegDecoderStateCreate(handle, decoder, decoder_state));

        CHECK_NVJPEG("nvjpegBufferPinnedCreate", nvjpegBufferPinnedCreate(handle, null, pinned_buffers[0]));
        CHECK_NVJPEG("nvjpegBufferPinnedCreate", nvjpegBufferPinnedCreate(handle, null, pinned_buffers[1]));
        CHECK_NVJPEG("nvjpegBufferDeviceCreate", nvjpegBufferDeviceCreate(handle, null, bufferDevice));

        CHECK_NVJPEG("nvjpegJpegStreamCreate", nvjpegJpegStreamCreate(handle, streams[0]));
        CHECK_NVJPEG("nvjpegJpegStreamCreate", nvjpegJpegStreamCreate(handle, streams[1]));

        CHECK_NVJPEG("nvjpegDecodeParamsCreate", nvjpegDecodeParamsCreate(handle, params));

        // Destroy Components
        CHECK_NVJPEG("nvjpegDecodeParamsDestroy", nvjpegDecodeParamsDestroy(params));

        CHECK_NVJPEG("nvjpegJpegStreamDestroy", nvjpegJpegStreamDestroy(streams[0]));
        CHECK_NVJPEG("nvjpegJpegStreamDestroy", nvjpegJpegStreamDestroy(streams[1]));

        CHECK_NVJPEG("nvjpegBufferPinnedDestroy", nvjpegBufferPinnedDestroy(pinned_buffers[0]));
        CHECK_NVJPEG("nvjpegBufferPinnedDestroy", nvjpegBufferPinnedDestroy(pinned_buffers[1]));
        CHECK_NVJPEG("nvjpegBufferDeviceDestroy", nvjpegBufferDeviceDestroy(bufferDevice));

        CHECK_NVJPEG("nvjpegJpegStateDestroy", nvjpegJpegStateDestroy(decoder_state));
        CHECK_NVJPEG("nvjpegDecoderDestroy", nvjpegDecoderDestroy(decoder));

        CHECK_NVJPEG("nvjpegJpegStateDestroy", nvjpegJpegStateDestroy(state));
        CHECK_NVJPEG("nvjpegDestroy", nvjpegDestroy(handle));
    }
}
