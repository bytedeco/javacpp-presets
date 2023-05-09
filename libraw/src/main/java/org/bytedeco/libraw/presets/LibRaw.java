/*
 * Copyright (C) 2022-2023 Jarek Sacha, Samuel Audet
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

package org.bytedeco.libraw.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Wrapper for <a href="https://www.libraw.org/">LibRaw</a> library.
 *
 * @author Jarek Sacha
 */
@Properties(inherit = javacpp.class,
        target = "org.bytedeco.libraw",
        global = "org.bytedeco.libraw.global.LibRaw",
        value = {
                @Platform(
                        include = {
                                "libraw/libraw_const.h",
                                "libraw/libraw_version.h",
                                "libraw/libraw_types.h",
                                "libraw/libraw_datastream.h",
                                "libraw/libraw.h",
                        },
                        link = "raw_r@.23",
                        preload = "gomp@.1"
                ),
                @Platform(value = "windows",
                        link = "libraw_static",
                        define = {
                                // To avoid errors like: "winsock2.h error C2011 'struct' type redefinition"
                                "WIN32_LEAN_AND_MEAN",
                                // To compile libraw_streams.h
                                "LIBRAW_WIN32_CALLS"
                        }
                )

        })
public class LibRaw implements InfoMapper {
    static {
        Loader.checkVersion("org.bytedeco", "libraw");
    }

    public void map(InfoMap infoMap) {
        infoMap
                .put(new Info().enumerate())
                .put(new Info("signed __int8").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
                //
                // libraw_const.h
                //
                // Skip to avoid unqualified referencing enum values in
                .put(new Info("LIBRAW_EXIFTOOLTAGTYPE_int8u",
                        "LIBRAW_EXIFTOOLTAGTYPE_string",
                        "LIBRAW_EXIFTOOLTAGTYPE_int16u",
                        "LIBRAW_EXIFTOOLTAGTYPE_int32u",
                        "LIBRAW_EXIFTOOLTAGTYPE_rational64u",
                        "LIBRAW_EXIFTOOLTAGTYPE_int8s",
                        "LIBRAW_EXIFTOOLTAGTYPE_undef",
                        "LIBRAW_EXIFTOOLTAGTYPE_binary",
                        "LIBRAW_EXIFTOOLTAGTYPE_int16s",
                        "LIBRAW_EXIFTOOLTAGTYPE_int32s",
                        "LIBRAW_EXIFTOOLTAGTYPE_rational64s",
                        "LIBRAW_EXIFTOOLTAGTYPE_float",
                        "LIBRAW_EXIFTOOLTAGTYPE_double",
                        "LIBRAW_EXIFTOOLTAGTYPE_ifd",
                        "LIBRAW_EXIFTOOLTAGTYPE_unicode",
                        "LIBRAW_EXIFTOOLTAGTYPE_complex",
                        "LIBRAW_EXIFTOOLTAGTYPE_int64u",
                        "LIBRAW_EXIFTOOLTAGTYPE_int64s",
                        "LIBRAW_EXIFTOOLTAGTYPE_ifd64"
                ).skip())
                // Skip[ issues with generation of "enum LibRaw_dng_processing"
                .put(new Info("LIBRAW_DNG_ALL", "LIBRAW_DNG_DEFAULT").skip())
                // Skip[ issues with generation of "enum LibRaw_processing_options"
                .put(new Info("LibRaw_processing_options").skip())

                //
                // libraw_version.h
                //
                //
                .put(new Info("LIBRAW_VERSION_TAIL").skip())
                .put(new Info("LIBRAW_VERSION_STR").skip())
                .put(new Info("LIBRAW_VERSION").skip())
                .put(new Info("LIBRAW_VERSION_TAIL").skip())

                //
                // "libraw_datastream.h"
                //
                .put(new Info("LIBRAW_WIN32_DLLDEFS").define(false))
                //
                .put(new Info("USE_DNGSDK").define(false))
                //
                .put(new Info("LibRaw_windows_datastream").skip(true))
                //  "error C2039: 'wfname': is not a member of 'LibRaw_file_datastream'"
                .put(new Info("LIBRAW_WIN32_UNICODEPATHS").define(false))
                // libraw_datastream.h(207): error C3861: 'getc_unlocked': identifier not found
                .put(new Info("libraw_nikon_makernotes_t::AFFineTuneAdj").skip())

                //
                // libraw.h
                //
                .put(new Info("LibRaw::get_internal_data_pointer").skip(true))

                //
                // To build on non-Windows
                //
                .put(new Info("defined(_WIN32) || defined(WIN32)").define(false))

                .put(new Info("LIBRAW_LIBRARY_BUILD",
                              "LIBRAW_OLD_VIDEO_SUPPORT",
                              "defined (LIBRAW_LIBRARY_BUILD) && defined(__cplusplus)",
                              "defined (LIBRAW_NO_IOSTREAMS_DATASTREAM)  && defined (LIBRAW_WIN32_CALLS)").define(false))
                .put(new Info("LIBRAW_NO_IOSTREAMS_DATASTREAM").define(true))
        ;
    }
}
