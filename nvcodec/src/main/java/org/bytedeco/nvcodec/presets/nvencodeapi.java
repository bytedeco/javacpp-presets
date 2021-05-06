/*
 * Copyright (C) 2021 Park JeongHwan
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
 * @author Park JeongHwan
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

               .put(new Info("NvEncInvalidateRefFrames").skip())
               .put(new Info("NvEncRegisterAsyncEvent").skip())
               .put(new Info("NvEncCreateBitstreamBuffer").skip())
               .put(new Info("NvEncOpenEncodeSession").skip())
               .put(new Info("NvEncSetIOCudaStreams").skip())
               .put(new Info("NvEncDestroyMVBuffer").skip())
               .put(new Info("NvEncDestroyBitstreamBuffer").skip())
               .put(new Info("NvEncUnregisterResource").skip())
               .put(new Info("NvEncDestroyEncoder").skip())
               .put(new Info("NvEncDestroyInputBuffer").skip())
               .put(new Info("NvEncGetEncodeProfileGUIDCount").skip())
               .put(new Info("NvEncCreateMVBuffer").skip())
               .put(new Info("NvEncGetSequenceParams").skip())
               .put(new Info("NvEncGetSequenceParamEx").skip())
               .put(new Info("NvEncGetEncodePresetCount").skip())
               .put(new Info("NvEncRunMotionEstimationOnly").skip())
               .put(new Info("NvEncGetEncodeGUIDs").skip())
               .put(new Info("NvEncReconfigureEncoder").skip())
               .put(new Info("NvEncGetInputFormats").skip())
               .put(new Info("NvEncOpenEncodeSessionEx").skip())
               .put(new Info("NvEncUnregisterAsyncEvent").skip())
               .put(new Info("NvEncEncodePicture").skip())
               .put(new Info("NvEncMapInputResource").skip())
               .put(new Info("NvEncGetInputFormatCount").skip())
               .put(new Info("NvEncGetEncodeGUIDCount").skip())
               .put(new Info("NvEncGetEncodeStats").skip())
               .put(new Info("NvEncUnlockBitstream").skip())
               .put(new Info("NvEncGetEncodePresetGUIDs").skip())
               .put(new Info("NvEncGetLastErrorString").skip())
               .put(new Info("NvEncGetEncodeCaps").skip())
               .put(new Info("NvEncGetEncodePresetConfig").skip())
               .put(new Info("NvEncUnmapInputResource").skip())
               .put(new Info("NvEncRegisterResource").skip())
               .put(new Info("NvEncLockBitstream").skip())
               .put(new Info("NvEncCreateInputBuffer").skip())
               .put(new Info("NvEncLockInputBuffer").skip())
               .put(new Info("NvEncUnlockInputBuffer").skip())
               .put(new Info("NvEncInitializeEncoder").skip())
               .put(new Info("NvEncGetEncodeProfileGUIDs").skip())
               .put(new Info("NvEncGetEncodePresetConfigEx").skip())

               .put(new Info("NV_ENC_H264_SEI_PAYLOAD").cppText(""))
               .put(new Info("nvEncodeAPI.h").linePatterns(
                       " \\* \\\\union _NV_ENC_PIC_PARAMS_H264_EXT", " \\* H264 extension  picture parameters",
                       "#define NV_ENC_PARAMS_RC_VBR_MINQP.*", "#define NV_ENC_PARAMS_RC_CBR2.*",
                       "#define NV_ENC_BUFFER_FORMAT_NV12_PL.*", "#define NV_ENC_BUFFER_FORMAT_YUV444_PL.*").skip());
    }
}
