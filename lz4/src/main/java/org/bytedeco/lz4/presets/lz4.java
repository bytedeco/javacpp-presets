/*
 * Copyright (C) 2021 Benjamin Wilhelm
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

package org.bytedeco.lz4.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.lz4.LZ4FFrameInfo;
import org.bytedeco.lz4.LZ4FPreferences;

/**
 * {@link InfoMapper} for the C lz4 compression library.
 *
 * @see <a href="https://github.com/lz4/lz4/">https://github.com/lz4/lz4/</a>
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
@Properties(inherit = javacpp.class, //
        value = { //
                @Platform(include = {"<lz4.h>", "<lz4hc.h>", "<lz4frame.h>"}, link = "lz4@.1") //
        }, //
        target = "org.bytedeco.lz4", //
        global = "org.bytedeco.lz4.global.lz4" //
)
public class lz4 implements InfoMapper {

    static {
        Loader.checkVersion("org.bytedeco", "lz4");
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("defined(__cplusplus)").define());

        // LZ4
        infoMap // Skip stuff
                .put(new Info("LZ4_LIB_VERSION").skip()) //
                .put(new Info("LZ4LIB_VISIBILITY", "LZ4LIB_API").cppTypes().annotations()) //
                .put(new Info("lz4.h").linePatterns( //
                        "^#ifdef LZ4_STATIC_LINKING_ONLY$", // Skip static API
                        "^#endif   /\\* LZ4_STATIC_LINKING_ONLY \\*/$", //
                        "^#ifndef LZ4_H_98237428734687$", // Skip private definitions and obsolete functions
                        "^#endif /\\* LZ4_H_98237428734687 \\*/" //
                ).skip()) //
                // Change types
                .put(new Info("LZ4_VERSION_STRING").cppTypes("const char*")) //
                // Rename types
                .put(new Info("LZ4_stream_t").pointerTypes("LZ4Stream")) //
                .put(new Info("LZ4_streamDecode_t").pointerTypes("LZ4StreamDecode")) //
        ;

        // LZ4 HC
        infoMap // Skip stuff
                .put(new Info("lz4hc.h").linePatterns( //
                        "^#define LZ4HC_DICTIONARY_LOGSIZE 16$", // Skip private definitions + deprecated
                        "^LZ4LIB_API void LZ4_resetStreamHC \\(LZ4_streamHC_t\\* streamHCPtr, int compressionLevel\\);$", //
                        "^#define LZ4_STATIC_LINKING_ONLY   /\\* LZ4LIB_STATIC_API \\*/$", // Skip static linking only
                        "^#endif   /\\* LZ4_HC_STATIC_LINKING_ONLY \\*/$" //
                ).skip()) //
                // Rename types
                .put(new Info("LZ4_streamHC_t").pointerTypes("LZ4StreamHC")) //
        ;

        // LZ4 FRAME
        infoMap // Skip stuff
                .put(new Info("LZ4F_INIT_FRAMEINFO").skip()) // Manually implemented
                .put(new Info("LZ4F_INIT_PREFERENCES").skip()) // Manually implemented
                .put(new Info("LZ4F_ENABLE_OBSOLETE_ENUMS").define(false)) //
                .put(new Info("LZ4FLIB_VISIBILITY", "LZ4FLIB_API").cppTypes().annotations()) //
                .put(new Info("lz4frame.h").linePatterns( //
                        "^#if defined\\(LZ4F_STATIC_LINKING_ONLY\\).*$", // Ignore static API
                        "^#endif  /\\* defined\\(LZ4F_STATIC_LINKING_ONLY\\).*$" //
                ).skip()) //
                // Rename types
                .put(new Info("LZ4F_cctx").pointerTypes("LZ4FCompressionContext")) //
                .put(new Info("LZ4F_dctx").pointerTypes("LZ4FDecompressionContext")) //
                .put(new Info("LZ4F_compressOptions_t").pointerTypes("LZ4FCompressOptions")) //
                .put(new Info("LZ4F_decompressOptions_t").pointerTypes("LZ4FDecompressOptions")) //
                .put(new Info("LZ4F_frameInfo_t").pointerTypes("LZ4FFrameInfo")) //
                .put(new Info("LZ4F_preferences_t").pointerTypes("LZ4FPreferences")) //
        ;
    }

    /** Init the {@link LZ4FFrameInfo} object with the default values. */
    public static LZ4FFrameInfo LZ4F_INIT_FRAMEINFO(LZ4FFrameInfo info) {
        info.blockSizeID(org.bytedeco.lz4.global.lz4.LZ4F_default);
        info.blockMode(org.bytedeco.lz4.global.lz4.LZ4F_blockLinked);
        info.blockMode(org.bytedeco.lz4.global.lz4.LZ4F_blockLinked);
        info.contentChecksumFlag(org.bytedeco.lz4.global.lz4.LZ4F_noContentChecksum);
        info.frameType(org.bytedeco.lz4.global.lz4.LZ4F_frame);
        info.contentSize(0);
        info.dictID(0);
        info.blockChecksumFlag(org.bytedeco.lz4.global.lz4.LZ4F_noBlockChecksum);
        return info;
    }

    /** Init the {@link LZ4FPreferences} object with the default values. */
    public static LZ4FPreferences LZ4F_INIT_PREFERENCES(LZ4FPreferences pref) {
        pref.frameInfo(LZ4F_INIT_FRAMEINFO(new LZ4FFrameInfo()));
        pref.compressionLevel(0);
        pref.autoFlush(0);
        pref.favorDecSpeed(0);
        pref.reserved(0);
        return pref;
    }
}
