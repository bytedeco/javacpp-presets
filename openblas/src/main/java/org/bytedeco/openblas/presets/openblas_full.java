/*
 * Copyright (C) 2016-2025 Samuel Audet
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

package org.bytedeco.openblas.presets;

import java.util.List;
import java.util.ListIterator;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;

/**
 *
 * @author Samuel Audet
 * @author Dragan Djuric
 */
@Properties(inherit = openblas.class, global = "org.bytedeco.openblas.global.openblas_full", value = {
    @Platform(include = {"openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h", "lapack.h", "lapacke.h", "lapacke_utils.h"})})
@NoException
public class openblas_full extends openblas {

  @Override public void map(InfoMap infoMap) {
        infoMap.put(new Info("lapack.h", "lapacke.h").linePatterns(".*LAPACK_GLOBAL.*").skip())
               .put(new Info("OPENBLAS_PTHREAD_CREATE_FUNC", "OPENBLAS_BUNDERSCORE", "OPENBLAS_FUNDERSCORE", "DOUBLE_DEFINED", "xdouble",
                             "FLOATRET", "OPENBLAS_CONST", "CBLAS_INDEX", "LAPACK_IFMT", "FORTRAN_STRLEN", "lapack_int", "lapack_logical").cppTypes().annotations())
               .put(new Info("OPENBLAS_QUAD_PRECISION", "defined OPENBLAS_EXPRECISION", "OPENBLAS_USE64BITINT",
                             "defined(LAPACK_COMPLEX_STRUCTURE)", "defined(LAPACK_COMPLEX_C99)", "OPENBLAS_OS_LINUX").define(false).translate(true))
               .put(new Info("((defined(__STDC_IEC_559_COMPLEX__) || __STDC_VERSION__ >= 199901L ||"
                       + "      (__GNUC__ >= 3 && !defined(__cplusplus))) && !(defined(FORCE_OPENBLAS_COMPLEX_STRUCT))) && !defined(_MSC_VER)",
                             "defined(LAPACK_COMPLEX_CPP)", "LAPACK_COMPLEX_CUSTOM", "LAPACK_FORTRAN_STRLEN_END").define())
               .put(new Info("openblas_complex_float", "lapack_complex_float").cast().pointerTypes("FloatPointer", "FloatBuffer", "float[]"))
               .put(new Info("openblas_complex_double", "lapack_complex_double").cast().pointerTypes("DoublePointer", "DoubleBuffer", "double[]"));

        String[] functions = {
            
            // deprecated
            "cgegs",   "cggsvd",  "ctzrqf",  "dgeqpf",  "dlatzm",  "sgelsx",  "slahrd",  "zgegv",   "zggsvp",
            "cgegv",   "cggsvp",  "dgegs",   "dggsvd",  "dtzrqf",  "sgeqpf",  "slatzm",  "zgelsx",  "zlahrd",
            "cgelsx",  "clahrd",  "dgegv",   "dggsvp",  "sgegs",   "sggsvd",  "stzrqf",  "zgeqpf",  "zlatzm",
            "cgeqpf",  "clatzm",  "dgelsx",  "dlahrd",  "sgegv",   "sggsvp",  "zgegs",   "zggsvd",  "ztzrqf"};
          
        for (String f : functions) {
            infoMap.put(new Info(f, "LAPACK_" + f, "LAPACK_" + f + "_base", "LAPACKE_" + f, "LAPACKE_" + f + "_work").skip());
        }
    }

}
