/*
 * Copyright (C) 2016-2017 Samuel Audet
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

import java.util.Iterator;
import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(target = "org.bytedeco.javacpp.openblas", value = {@Platform(define = {"__OPENBLAS 1", "LAPACK_COMPLEX_CPP"},
              include = {"openblas_config.h", "cblas.h", "blas_extra.h", "lapacke_config.h", "lapacke_mangling.h", "lapacke.h", "lapacke_utils.h"},
              link    =  "openblas@.0", resource = {"include", "lib"},
              preload = {"iomp5", "mkl_avx", "mkl_avx2", "mkl_avx512", "mkl_avx512_mic", "mkl_def", "mkl_mc", "mkl_mc3", "mkl_core", "mkl_gnu_thread",
                         "mkl_intel_lp64", "mkl_intel_thread", "mkl_rt", "mkl_rt#openblas@.0", "gcc_s@.1", "quadmath@.0", "gfortran@.3"}, compiler = "fastfpu",
              preloadpath = {"/opt/intel/lib/", "/opt/intel/mkl/lib/"}),
    @Platform(value = "android", include = {"openblas_config.h", "cblas.h", "blas_extra.h" /* no LAPACK */}, link = "openblas", preload = "", compiler = "fastfpu"),
    @Platform(value = "macosx",  link = "openblas",
                                 preload = {"iomp5", "mkl_avx", "mkl_avx2", "mkl_avx512", "mkl_avx512_mic", "mkl_def", "mkl_mc", "mkl_mc3", "mkl_core", "mkl_gnu_thread",
                                            "mkl_intel_lp64", "mkl_intel_thread", "mkl_rt", "mkl_rt#openblas", "gcc_s@.1", "quadmath@.0", "gfortran@.3"}),
    @Platform(value = "windows", preload = {"libiomp5md#libiomp5md", "mkl_avx#mkl_avx", "mkl_avx2#mkl_avx2", "mkl_avx512#mkl_avx512", "mkl_avx512_mic#mkl_avx512_mic",
                                            "mkl_def#mkl_def", "mkl_mc#mkl_mc", "mkl_mc3#mkl_mc3", "mkl_core#mkl_core", "mkl_intel_lp64#mkl_intel_lp64",
                                            "mkl_intel_thread#mkl_intel_thread", "mkl_rt#mkl_rt", "mkl_rt#libopenblas", "libopenblas"}),
    @Platform(value = "windows-x86",    preloadpath = {"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/compiler/",
                                                       "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/mkl/"}),
    @Platform(value = "windows-x86_64", preloadpath = {"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler/",
                                                       "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/"}),
    @Platform(value = "linux",          preloadpath = {"/usr/lib/", "/usr/lib32/", "/usr/lib64/"}),
    @Platform(value = "linux-armhf",    preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}),
    @Platform(value = "linux-x86",      preloadpath = {"/lib32/", "/lib/", "/usr/lib32/", "/usr/lib/", "/opt/intel/lib/ia32/", "/opt/intel/mkl/lib/ia32/"}),
    @Platform(value = "linux-x86_64",   preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/", "/opt/intel/lib/intel64/", "/opt/intel/mkl/lib/intel64/"}),
    @Platform(value = "linux-ppc64",    preloadpath = {"/usr/lib/powerpc64-linux-gnu/", "/usr/lib/powerpc64le-linux-gnu/"}),
    @Platform(value = "ios", include = {"cblas.h", "clapack.h", "blas_extra.h" /* no LAPACK */},
          includepath = {"/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers/"})
   // in case openblas is available and preferred: @Platform(value = "ios", include = {"openblas_config.h", "cblas.h", "blas_extra.h" /* no LAPACK */}, preload = "libopenblas")
 })
public class openblas implements LoadEnabled, InfoMapper {

    @Override public void init(ClassProperties properties) {
        String s = System.getProperty("org.bytedeco.javacpp.openblas.nomkl", "false").toLowerCase();
        if (s.equals("true") || s.equals("t") || s.equals("")) {
            List<String> preloads = properties.get("platform.preload");
            Iterator<String> it = preloads.iterator();
            while (it.hasNext()) {
                String p = it.next();
                if (p.contains("iomp5") || p.contains("mkl")) {
                    it.remove();
                }
            }
        }
    }

    @Override public void map(InfoMap infoMap) {
        // skip LAPACK 3.7.0 until supported by MKL, at least
        infoMap.put(new Info("lapacke.h").linePatterns(".*LAPACKE_ssysv_aa.*", ".*LAPACK_ssysv_aa.*",
                                                       ".*LAPACK_ssysv_aa.*",  "#ifdef __cplusplus",
                                                       ".*LAPACK_GLOBAL.*").skip())
               .put(new Info("OPENBLAS_PTHREAD_CREATE_FUNC", "OPENBLAS_BUNDERSCORE", "OPENBLAS_FUNDERSCORE", "DOUBLE_DEFINED", "xdouble",
                             "FLOATRET", "OPENBLAS_CONST", "CBLAS_INDEX", "lapack_int", "lapack_logical").cppTypes().annotations())
               .put(new Info("OPENBLAS_QUAD_PRECISION", "defined OPENBLAS_EXPRECISION", "OPENBLAS_USE64BITINT",
                             "defined(LAPACK_COMPLEX_STRUCTURE)", "defined(LAPACK_COMPLEX_C99)").define(false))
               .put(new Info("((defined(__STDC_IEC_559_COMPLEX__) || __STDC_VERSION__ >= 199901L ||"
                       + "      (__GNUC__ >= 3 && !defined(__cplusplus))) && !(defined(FORCE_OPENBLAS_COMPLEX_STRUCT)))",
                             "defined(LAPACK_COMPLEX_CPP)", "LAPACK_COMPLEX_CUSTOM").define())
               .put(new Info("openblas_complex_float", "lapack_complex_float").cast().pointerTypes("FloatPointer", "FloatBuffer", "float[]"))
               .put(new Info("openblas_complex_double", "lapack_complex_double").cast().pointerTypes("DoublePointer", "DoubleBuffer", "double[]"));

        String[] functions = {
            // not exported by OpenBLAS
            "cblas_cgemm3m", "cblas_zgemm3m", "cblas_xerbla",
            // not implemented by MKL
            "openblas_set_num_threads", "goto_set_num_threads", "openblas_get_num_threads", "openblas_get_num_procs",
            "openblas_get_config", "openblas_get_corename", "openblas_get_parallel", "cblas_cdotc", "cblas_cdotu", "cblas_cgeadd",
            "cblas_cimatcopy", "cblas_comatcopy", "cblas_dgeadd", "cblas_dimatcopy", "cblas_domatcopy", "cblas_sgeadd",
            "cblas_simatcopy", "cblas_somatcopy", "cblas_zdotc", "cblas_zdotu", "cblas_zgeadd", "cblas_zimatcopy", "cblas_zomatcopy",
            // deprecated
            "cgegs",   "cggsvd",  "ctzrqf",  "dgeqpf",  "dlatzm",  "sgelsx",  "slahrd",  "zgegv",   "zggsvp",
            "cgegv",   "cggsvp",  "dgegs",   "dggsvd",  "dtzrqf",  "sgeqpf",  "slatzm",  "zgelsx",  "zlahrd",
            "cgelsx",  "clahrd",  "dgegv",   "dggsvp",  "sgegs",   "sggsvd",  "stzrqf",  "zgeqpf",  "zlatzm",
            "cgeqpf",  "clatzm",  "dgelsx",  "dlahrd",  "sgegv",   "sggsvp",  "zgegs",   "zggsvd",  "ztzrqf",
            // extended
            "cgbrfsx", "cporfsx", "dgerfsx", "sgbrfsx", "ssyrfsx", "zherfsx", "cgerfsx", "csyrfsx", "dporfsx", "sgerfsx", "zgbrfsx", "zporfsx",
            "cherfsx", "dgbrfsx", "dsyrfsx", "sporfsx", "zgerfsx", "zsyrfsx", "cgbsvxx", "cposvxx", "dgesvxx", "sgbsvxx", "ssysvxx", "zhesvxx",
            "cgesvxx", "csysvxx", "dposvxx", "sgesvxx", "zgbsvxx", "zposvxx", "chesvxx", "dgbsvxx", "dsysvxx", "sposvxx", "zgesvxx", "zsysvxx"};
        for (String f : functions) {
            infoMap.put(new Info(f, "LAPACK_" + f, "LAPACKE_" + f, "LAPACKE_" + f + "_work").skip());
        }
    }
}
