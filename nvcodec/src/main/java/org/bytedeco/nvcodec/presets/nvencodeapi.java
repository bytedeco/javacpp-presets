/*
 * Copyright (C) 2021-2023 Jeonghwan Park, Samuel Audet
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

package org.bytedeco.nvcodec.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.nvcodec.presets.*;

/**
 *
 * @author Jeonghwan Park
 */
@Properties(
    inherit = nvcuvid.class,
    value = {
        @Platform(
            value = {"linux-arm64", "linux-ppc64le", "linux-x86_64", "windows-x86_64"},
            include = "nvEncodeAPI.h",
            link = "nvidia-encode"
        ),
        @Platform(
            value = "windows-x86_64",
            link = "nvencodeapi"
        )
    },
    target = "org.bytedeco.nvcodec.nvencodeapi",
    global = "org.bytedeco.nvcodec.global.nvencodeapi"
)

public class nvencodeapi implements InfoMapper {
    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("NVENCAPI").cppTypes().annotations())

               .put(new Info("NvEncInvalidateRefFrames",
                             "NvEncRegisterAsyncEvent",
                             "NvEncCreateBitstreamBuffer",
                             "NvEncOpenEncodeSession",
                             "NvEncSetIOCudaStreams",
                             "NvEncDestroyMVBuffer",
                             "NvEncDestroyBitstreamBuffer",
                             "NvEncUnregisterResource",
                             "NvEncDestroyEncoder",
                             "NvEncDestroyInputBuffer",
                             "NvEncGetEncodeProfileGUIDCount",
                             "NvEncCreateMVBuffer",
                             "NvEncGetSequenceParams",
                             "NvEncGetSequenceParamEx",
                             "NvEncGetEncodePresetCount",
                             "NvEncRunMotionEstimationOnly",
                             "NvEncGetEncodeGUIDs",
                             "NvEncReconfigureEncoder",
                             "NvEncGetInputFormats",
                             "NvEncOpenEncodeSessionEx",
                             "NvEncUnregisterAsyncEvent",
                             "NvEncEncodePicture",
                             "NvEncMapInputResource",
                             "NvEncGetInputFormatCount",
                             "NvEncGetEncodeGUIDCount",
                             "NvEncGetEncodeStats",
                             "NvEncUnlockBitstream",
                             "NvEncGetEncodePresetGUIDs",
                             "NvEncGetLastErrorString",
                             "NvEncGetEncodeCaps",
                             "NvEncGetEncodePresetConfig",
                             "NvEncUnmapInputResource",
                             "NvEncRegisterResource",
                             "NvEncLockBitstream",
                             "NvEncCreateInputBuffer",
                             "NvEncLockInputBuffer",
                             "NvEncUnlockInputBuffer",
                             "NvEncInitializeEncoder",
                             "NvEncGetEncodeProfileGUIDs",
                             "NvEncGetEncodePresetConfigEx",
                             "NvEncLookaheadPicture",
                             "NvEncRestoreEncoderState").skip())
               .put(new Info("NV_ENC_H264_SEI_PAYLOAD", "NV_ENC_AV1_OBU_PAYLOAD").cppText("").pointerTypes("NV_ENC_SEI_PAYLOAD"))
               .put(new Info("nvEncodeAPI.h").linePatterns(
                       " \\* \\\\union _NV_ENC_PIC_PARAMS_H264_EXT", " \\* H264 extension  picture parameters",
                       "#define NV_ENC_PARAMS_RC_VBR_MINQP.*", "#define NV_ENC_PARAMS_RC_CBR2.*",
                       "#define NV_ENC_BUFFER_FORMAT_NV12_PL.*", "#define NV_ENC_BUFFER_FORMAT_YUV444_PL.*").skip());
    }
}
