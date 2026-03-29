/*
 * Copyright (C) 2023-2025 Institute for Human and Machine Cognition, Samuel Audet
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

package org.bytedeco.cuda.presets;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(inherit = cudart.class, value = {
    @Platform(value = {"linux-x86_64", "linux-arm64", "windows-x86_64"},
            include = {"<nvcomp/shared_types.h>", "<nvcomp.h>", "<nvcomp.hpp>", "<nvcomp/nvcompManager.hpp>", "<nvcomp/nvcompManagerFactory.hpp>",
                    "<nvcomp/ans.h>", "<nvcomp/ans.hpp>", "<nvcomp/bitcomp.h>", "<nvcomp/bitcomp.hpp>", "<nvcomp/cascaded.h>",
                    "<nvcomp/crc32.h>", "<nvcomp/deflate.h>", "<nvcomp/deflate.hpp>", "<nvcomp/gdeflate.h>", "<nvcomp/gdeflate.hpp>",
                    "<nvcomp/gzip.h>", "<nvcomp/lz4.h>", "<nvcomp/lz4.hpp>", "<nvcomp/snappy.h>", "<nvcomp/snappy.hpp>", "<nvcomp/zstd.h>",
                    "<nvcomp/zstd.hpp>"}, link = {"nvcomp@.5", "nvcomp_cpu@.5"}),
    @Platform(value = "windows-x86_64", preload = {"nvcomp64_5", "nvcomp_cpu64_5"})},
    target = "org.bytedeco.cuda.nvcomp", global = "org.bytedeco.cuda.global.nvcomp")
public class nvcomp implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("NVCOMP_EXPORT").cppTypes().annotations())
               .put(new Info("std::runtime_error").pointerTypes("Pointer"))
               .put(new Info("nvcompCRC32IgnoredInputChunkBytes").javaText("@MemberGetter public static native @Cast(\"const size_t*\") SizeTPointer nvcompCRC32IgnoredInputChunkBytes();"))
               .put(new Info("nvcomp::detail::PimplManager").purify())
               .put(new Info("nvcompDecompressGetMetadata",
                             "nvcompDecompressDestroyMetadata",
                             "nvcompDecompressGetTempSize",
                             "nvcompDecompressGetOutputSize",
                             "nvcompDecompressGetType",
                             "nvcompDecompressAsync",
                             "nvcomp::detail::PinnedPtrPool",
                             "nvcomp::detail::nvcompManagerInternalBase",
                             "nvcomp::CompressionConfig::CompressionConfigImpl",
                             "nvcomp::DecompressionConfig::impl",
                             "nvcomp::PimplManager::impl",
                             "nvcomp::set_scratch_allocators",
                             "PinnedPtrPool",
                             // TODO: Fix bitcomp symbols
                             "nvcompBatchedGzipCompressAsync",
                             "nvcompBatchedGzipCompressGetTempSize",
                             "nvcompBatchedGzipCompressGetTempSizeEx",
                             "nvcompBatchedGzipCompressGetMaxOutputChunkSize",
                             "nvcompBitcompDecompressConfigure",
                             "nvcompBitcompCompressAsync",
                             "nvcompBitcompCompressConfigure",
                             "nvcompIsBitcompData",
                             "nvcompBitcompDestroyMetadata",
                             "nvcompBitcompDecompressAsync",
                             "nvcompBitcompFormatOpts").skip());
    }
}
