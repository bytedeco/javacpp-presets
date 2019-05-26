/*
 * Copyright (C) 2018 Samuel Audet
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

package org.bytedeco.cminpack.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.openblas.presets.openblas;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = openblas.class,
    value = @Platform(
        define = "__cminpack_double__",
        include = "cminpack.h",
        link = "cminpack"),
    global = "org.bytedeco.cminpack.global.cminpack")
@NoException
public class cminpack implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "cminpack"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("__cminpack_double__").define(true));
        mapCommon(infoMap);
    }

    static void mapCommon(InfoMap infoMap) {
        infoMap.put(new Info("__cminpack_long_double__",
                             "__cminpack_double__",
                             "__cminpack_float__",
                             "__cminpack_half__",
                             "defined(__CUDA_ARCH__) || defined(__CUDACC__)").define(false))
               .put(new Info("CMINPACK_DECLSPEC_EXPORT",
                             "CMINPACK_DECLSPEC_IMPORT",
                             "CMINPACK_EXPORT",
                             "__cminpack_attr__",
                             "__cminpack_real__",
                             "__cminpack_type_fcn_nn__",
                             "__cminpack_type_fcnder_nn__",
                             "__cminpack_type_fcn_mn__",
                             "__cminpack_type_fcnder_mn__",
                             "__cminpack_type_fcnderstr_mn__",
                             "__cminpack_decl_fcn_nn__",
                             "__cminpack_decl_fcnder_nn__",   
                             "__cminpack_decl_fcn_mn__",
                             "__cminpack_decl_fcnder_mn__",
                             "__cminpack_decl_fcnderstr_mn__",
                             "__cminpack_param_fcn_nn__",
                             "__cminpack_param_fcnder_nn__",
                             "__cminpack_param_fcn_mn__",
                             "__cminpack_param_fcnder_mn__",
                             "__cminpack_param_fcnderstr_mn__").annotations().cppTypes().define(true));
    }
}
