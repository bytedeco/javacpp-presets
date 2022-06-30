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

package org.bytedeco.cuda.presets;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Park JeongHwan
 */
@Properties(inherit = cudart.class, value = {
    @Platform(include = "<nvjpeg.h>", link = "nvjpeg@.11"),
    @Platform(value = "windows-x86_64", preload = "nvjpeg64_11")},
        target = "org.bytedeco.cuda.nvjpeg", global = "org.bytedeco.cuda.global.nvjpeg")
@NoException
public class nvjpeg implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("nvjpegHandle_t").valueTypes("nvjpegHandle").pointerTypes("@ByPtrPtr nvjpegHandle"))

               .put(new Info("nvjpegJpegDecoder_t").valueTypes("nvjpegJpegDecoder").pointerTypes("@ByPtrPtr nvjpegJpegDecoder"))
               .put(new Info("nvjpegJpegState_t").valueTypes("nvjpegJpegState").pointerTypes("@ByPtrPtr nvjpegJpegState"))
               .put(new Info("nvjpegJpegStream_t").valueTypes("nvjpegJpegStream").pointerTypes("@ByPtrPtr nvjpegJpegStream"))

               .put(new Info("nvjpegDecodeParams_t").valueTypes("nvjpegDecodeParams").pointerTypes("@ByPtrPtr nvjpegDecodeParams"))

               .put(new Info("nvjpegEncoderState_t").valueTypes("nvjpegEncoderState").pointerTypes("@ByPtrPtr nvjpegEncoderState"))
               .put(new Info("nvjpegEncoderParams_t").valueTypes("nvjpegEncoderParams").pointerTypes("@ByPtrPtr nvjpegEncoderParams"))

               .put(new Info("nvjpegBufferPinned_t").valueTypes("nvjpegBufferPinned").pointerTypes("@ByPtrPtr nvjpegBufferPinned"))
               .put(new Info("nvjpegBufferDevice_t").valueTypes("nvjpegBufferDevice").pointerTypes("@ByPtrPtr nvjpegBufferDevice"));
    }
}
