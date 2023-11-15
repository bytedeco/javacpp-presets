/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.cuda.global.nvcomp;
import org.bytedeco.cuda.nvcomp.*;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;

import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.cuda.global.nvcomp.*;

// https://github.com/NVIDIA/nvcomp/blob/main/examples/high_level_quickstart_example.cpp
public class nvcompLZ4Example {
    private static void decomp_compressed_with_manager_factory_example(BytePointer device_input_ptrs, long input_buffer_len) {
        CUstream_st stream = new CUstream_st();
        int cuda_error = cudaStreamCreate(stream);

        long chunk_size = 1 << 16;

        nvcompBatchedLZ4Opts_t format_opts = new nvcompBatchedLZ4Opts_t();
        format_opts.data_type(NVCOMP_TYPE_CHAR);
        LZ4Manager nvcomp_manager = new LZ4Manager(chunk_size, format_opts, stream, 0, nvcomp.NoComputeNoVerify);
        CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

        BytePointer comp_buffer = new BytePointer();
        cuda_error = cudaMalloc(comp_buffer, comp_config.max_compressed_buffer_size());

        nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

        // Construct a new nvcomp manager from the compressed buffer.
        // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a manager
        // for the use case where a buffer is received and the user doesn't know how it was compressed
        // Also note, creating the manager in this way synchronizes the stream, as the compressed buffer must be read to
        // construct the manager
        nvcompManagerBase decomp_nvcomp_manager = create_manager(comp_buffer, stream, 0, NoComputeNoVerify);

        DecompressionConfig decomp_config = decomp_nvcomp_manager.configure_decompression(comp_buffer);
        BytePointer res_decomp_buffer = new BytePointer();
        cuda_error = cudaMalloc(res_decomp_buffer, decomp_config.decomp_data_size());

        decomp_nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

        cuda_error = cudaFree(comp_buffer);
        cuda_error = cudaFree(res_decomp_buffer);
        cuda_error = cudaStreamSynchronize(stream);
        cuda_error = cudaStreamDestroy(stream);
    }

    public static void main(String[] args) {
        Loader.load(nvcomp.class);

        // Initialize a random array of chars
        int input_buffer_len = 1000000;
        byte[] uncompressed_data = new byte[input_buffer_len];

        for (int i = 0; i < input_buffer_len; i++) {
            uncompressed_data[i] = (byte) (Math.random() * 26 + 'a');
        }

        BytePointer uncompressed_data_ptr = new BytePointer(uncompressed_data);

        BytePointer device_input_ptrs = new BytePointer();

        int cuda_error = cudaMalloc(device_input_ptrs, input_buffer_len);
        cuda_error = cudaMemcpy(device_input_ptrs, uncompressed_data_ptr, input_buffer_len, cudaMemcpyDefault);

        decomp_compressed_with_manager_factory_example(device_input_ptrs, input_buffer_len);
    }
}
