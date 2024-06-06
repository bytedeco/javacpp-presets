/*
* Copyright (C) 2024 Zaki Ahmed
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
*
*/

package org.bytedeco.shakapackager.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
/**
 * Wrapper for <a href="https://github.com/shaka-project/shaka-packager/">ShakaPackager</a> library.
 *
 * @author Zaki Ahmed
 */
@Properties(inherit = javacpp.class,
        target = "org.bytedeco.shakapackager",
        global = "org.bytedeco.shakapackager.global.packager",
        value = {
                @Platform(
                       value = {"linux", "macosx"},
                       define = {"NDEBUG 1","SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std"},
                       compiler = "cpp17",
                       include = {
                                "packager/status.h",
                                "packager/mpd_params.h",
                                "packager/mp4_output_params.h",
                                "packager/hls_params.h",
                                "packager/file.h",
                                "packager/macros/classes.h",
                                "packager/export.h",
                                "packager/chunking_params.h",
                                "packager/buffer_callback_params.h",
                                "packager/crypto_params.h",
                                "packager/ad_cue_generator_params.h",
                                "packager/packager.h"
                       },
                       
                       link  = "packager"

                ),
                @Platform(
                       value = {"windows"},
                       define = {"_CRT_SECURE_NO_WARNINGS","SHAKA_EXPORT"},
                       compiler = "cpp17",
                       include = {
                                "packager/status.h",
                                "packager/mpd_params.h",
                                "packager/mp4_output_params.h",
                                "packager/hls_params.h",
                                "packager/file.h",
                                "packager/macros/classes.h",
                                "packager/export.h",
                                "packager/chunking_params.h",
                                "packager/buffer_callback_params.h",
                                "packager/crypto_params.h",
                                "packager/ad_cue_generator_params.h",
                                "packager/packager.h"
                       },
                       
                       link  = "libpackager"

                )
        })
public class packager implements InfoMapper {
    static {
        Loader.checkVersion("org.bytedeco", "shakapackager");
    }
    public void map(InfoMap infoMap) {

        infoMap
       .put(new Info().enumerate().friendly())
       .put(new Info("SHAKA_EXPORT").cppTypes().annotations())
       .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("boolean[]", "BoolPointer"))
       .put(new Info("const char").pointerTypes("String", "@Cast(\"const char*\") BytePointer"))
       .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
       .put(new Info().javaText("import org.bytedeco.shakapackager.functions.*;"))
       .put(new Info("std::string").annotations("@StdString").valueTypes("String","BytePointer").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
       .put(new Info("std::optional<double>").pointerTypes("DoubleOptional").define())
       .put(new Info("std::optional<uint32_t>").pointerTypes("IntOptional").define())
       .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
       .put(new Info("shaka::EncryptionParams::EncryptedStreamAttributes").pointerTypes("EncryptionParams.EncryptedStreamAttributes").purify())
       .put(new Info("std::function<std::string(const shaka::EncryptionParams::EncryptedStreamAttributes&)>").pointerTypes("StringEncryptedStreamAttributes"))
       .put(new Info("shaka::BufferCallbackParams").pointerTypes("BufferCallbackParams"))
       .put(new Info("shaka::RawKeyParams::KeyInfo").pointerTypes("RawKeyParams.KeyInfo"))
       .put(new Info("std::map<std::string,shaka::RawKeyParams::KeyInfo>").pointerTypes("StringKeyInfoMap").define())
       .put(new Info("std::function<int64_t(const std::string&,const void*,uint64_t)>").pointerTypes("Write_BufferParamsCallback"))
       .put(new Info("std::function<int64_t(const std::string&,void*,uint64_t)>").pointerTypes("Read_BufferParamsCallback"));

    }
}
