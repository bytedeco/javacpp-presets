/*
 * Copyright (C) 2017-2020 Jeremy Apthorp, Samuel Audet
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

package org.bytedeco.skia.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.*;

@Properties(
    inherit = javacpp.class,
    target = "org.bytedeco.skia",
    global = "org.bytedeco.skia.global.Skia",
    value = {
        @Platform(
            value = {"ios", "linux-x86", "macosx"},
            include = {
                "sk_types.h",
                "gr_context.h",
                "sk_bitmap.h",
                "sk_canvas.h",
                "sk_codec.h",
                "sk_colorfilter.h",
                "sk_colortable.h",
                "sk_data.h",
                "sk_document.h",
                "sk_image.h",
                "sk_imagefilter.h",
                "sk_mask.h",
                "sk_maskfilter.h",
                "sk_matrix.h",
                "sk_paint.h",
                "sk_path.h",
                "sk_patheffect.h",
                "sk_picture.h",
                "sk_pixmap.h",
                "sk_region.h",
                "sk_shader.h",
                "sk_stream.h",
                "sk_string.h",
                "sk_surface.h",
                "sk_svg.h",
                "sk_typeface.h",
                "sk_vertices.h",
                "sk_xml.h"
            },
            compiler = "cpp11",
            link = "skia",
            preload = "libskia"
        )
    }
)
public class Skia implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "skia"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("SK_API", "SK_C_API", "VKAPI_CALL", "VKAPI_PTR", "gr_mtl_handle_t").cppTypes().annotations())
            .put(new Info("SK_C_PLUS_PLUS_BEGIN_GUARD").cppText("#define SK_C_PLUS_PLUS_BEGIN_GUARD"))
            .put(new Info("SK_C_PLUS_PLUS_END_GUARD").cppText("#define SK_C_PLUS_PLUS_END_GUARD"))
            // TODO: There's probably a better way to skip these declarations,
            // but I couldn't find it.
            .put(new Info("gr_context.h").linePatterns(
                ".*gr_glinterface_assemble_interface.*", ".*",
                ".*gr_glinterface_assemble_gl_interface.*", ".*",
                ".*gr_glinterface_assemble_gles_interface.*", ".*"
            ).skip())
            .put(new Info("sk_types.h").linePatterns(".*gr_gl_func_ptr.*").skip())
            ;
    }
}
