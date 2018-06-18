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

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    value = {
        @Platform(
            value = {"linux-x86_64", "macosx-x86_64", "windows-x86_64"},
            include =  {"mkl.h", "mkl_version.h", "mkl_types.h", /*"mkl_blas.h",*/ "mkl_trans.h", "mkl_cblas.h",
                        "mkl_dnn_types.h", "mkl_dnn.h", /*"mkl_lapack.h", "mkl_lapacke.h", "mkl_service.h",
                        "mkl_vml.h", "mkl_vml_defines.h", "mkl_vml_types.h", "mkl_vml_functions.h",
                        "mkl_vsl.h", "mkl_vsl_defines.h", "mkl_vsl_types.h", "mkl_vsl_functions.h", "i_malloc.h"*/},
            link = "mklml_intel", preload = "iomp5"),
        @Platform(
            value = "macosx-x86_64",
            link = "mklml"),
        @Platform(
            value = "windows-x86_64",
            link = "mklml",
            preload = {"msvcp140", "vcruntime140", "api-ms-win-crt-stdio-l1-1-0", "api-ms-win-crt-math-l1-1-0",
                       "api-ms-win-crt-heap-l1-1-0", "api-ms-win-crt-runtime-l1-1-0", "api-ms-win-crt-convert-l1-1-0",
                       "msvcr120", "libiomp5md"},
            preloadpath = {"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x64/Microsoft.VC140.CRT/",
                           "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x64/"})},
    target = "org.bytedeco.javacpp.mklml")
@NoException
public class mklml implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("MKL_INT64", "MKL_UINT64", "MKL_INT", "MKL_UINT", "MKL_LONG",
                             "MKL_UINT8", "MKL_INT8", "MKL_INT16", "MKL_INT32",
                             "MKL_DECLSPEC", "MKL_CALL_CONV", "INTEL_API_DEF",

                             "mkl_simatcopy", "mkl_dimatcopy", "mkl_cimatcopy", "mkl_zimatcopy",
                             "mkl_somatcopy", "mkl_domatcopy", "mkl_comatcopy", "mkl_zomatcopy",
                             "mkl_somatcopy2", "mkl_domatcopy2", "mkl_comatcopy2", "mkl_zomatcopy2",
                             "mkl_somatadd", "mkl_domatadd", "mkl_comatadd", "mkl_zomatadd",

                             "CBLAS_INDEX").cppTypes().annotations())

               .put(new Info("MKL_Simatcopy", "MKL_Dimatcopy", "MKL_Cimatcopy", "MKL_Zimatcopy",
                             "MKL_Somatcopy2", "MKL_Domatcopy2", "MKL_Comatcopy2", "MKL_Zomatcopy2",
                             "MKL_Somatadd", "MKL_Domatadd", "MKL_Comatadd", "MKL_Zomatadd",

                             "cblas_ctrsm_batch", "cblas_dtrsm_batch", "cblas_strsm_batch", "cblas_ztrsm_batch").skip())

               .put(new Info("dnnPrimitive_t").valueTypes("_uniPrimitive_s").pointerTypes("@ByPtrPtr _uniPrimitive_s"))
               .put(new Info("dnnLayout_t").valueTypes("_dnnLayout_s").pointerTypes("@ByPtrPtr _dnnLayout_s"));
    }
}
