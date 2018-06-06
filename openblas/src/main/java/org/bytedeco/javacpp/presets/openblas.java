/*
 * Copyright (C) 2016-2018 Samuel Audet
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

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Pointer;
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
@Properties(target = "org.bytedeco.javacpp.openblas", value = {@Platform(define = {"__OPENBLAS 1", "LAPACK_COMPLEX_CPP"},
              include = {"openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h", "lapacke.h", "lapacke_utils.h"},
              link    =  "openblas@.0", resource = {"include", "lib"},
              preload = {"gcc_s@.1", "quadmath@.0", "gfortran@.3"},
              preloadpath = {"/opt/intel/lib/", "/opt/intel/mkl/lib/"}),
    @Platform(value = "android", include = {"openblas_config.h", "cblas.h" /* no LAPACK */}, link = "openblas", preload = ""),
    @Platform(value = "macosx",  link = "openblas"),
    @Platform(value = "windows", preload = "libopenblas"),
    @Platform(value = "windows-x86",    preloadpath = {"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/compiler/",
                                                       "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/mkl/"}),
    @Platform(value = "windows-x86_64", preloadpath = {"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler/",
                                                       "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/"}),
    @Platform(value = "linux",          preloadpath = {"/usr/lib/", "/usr/lib32/", "/usr/lib64/"}),
    @Platform(value = "linux-armhf",    preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}),
    @Platform(value = "linux-x86",      preloadpath = {"/lib32/", "/lib/", "/usr/lib32/", "/usr/lib/", "/opt/intel/lib/ia32/", "/opt/intel/mkl/lib/ia32/"}),
    @Platform(value = "linux-x86_64",   preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/", "/opt/intel/lib/intel64/", "/opt/intel/mkl/lib/intel64/"}),
    @Platform(value = "linux-ppc64",    preloadpath = {"/usr/powerpc64-linux-gnu/lib/", "/usr/powerpc64le-linux-gnu/lib/",
                                                       "/usr/lib/powerpc64-linux-gnu/", "/usr/lib/powerpc64le-linux-gnu/"}),
    @Platform(value = "ios", include = {"openblas_config.h", "cblas.h" /* no LAPACK */}, preload = "libopenblas") })
@NoException
public class openblas implements LoadEnabled, InfoMapper {

    @Override public void init(ClassProperties properties) {
        String platform = Loader.getPlatform();
        List<String> preloads = properties.get("platform.preload");

        // try to load MKL by default, but let users disable it
        String s = System.getProperty("org.bytedeco.javacpp.openblas.nomkl", "false").toLowerCase();
        if (s.equals("false") || s.equals("f") || s.equals("")) {
            String[] mkl = {"iomp5", "libiomp5md", "mkl_core", "mkl_avx", "mkl_avx2", "mkl_avx512", "mkl_avx512_mic",
                            "mkl_def", "mkl_mc", "mkl_mc3", "mkl_intel_lp64", "mkl_intel_thread", "mkl_rt"};
            for (int i = 0; i < mkl.length; i++) {
                preloads.add(i, mkl[i] + "#" + mkl[i]);
            }
            if (platform.startsWith("linux")) {
                preloads.add(mkl.length, "mkl_rt#openblas@.0");
            } else if (platform.startsWith("macosx")) {
                preloads.add(mkl.length, "mkl_rt#openblas");
            } else if (platform.startsWith("windows")) {
                preloads.add(mkl.length, "mkl_rt#libopenblas");
            }
        }

        // let users enable loading of arbitrary library (for Accelerate, etc)
        String lib = System.getProperty("org.bytedeco.javacpp.openblas.load", "").toLowerCase();
        if (lib.length() > 0) {
            if (platform.startsWith("linux")) {
                preloads.add(0, lib + "#openblas@.0");
            } else if (platform.startsWith("macosx")) {
                preloads.add(0, lib + "#openblas");
            } else if (platform.startsWith("windows")) {
                preloads.add(0, lib + "#libopenblas");
            }
        }
    }

    @Override public void map(InfoMap infoMap) {
        // skip LAPACK 3.7.0 and 3.8.0 until fully supported by MKL, at least
        infoMap.put(new Info("lapacke.h").linePatterns(".*LAPACKE_ssysv_aa\\(.*", ".*LAPACK_.*",
                                                       ".*LAPACK_ssysv_aa\\(.*",  ".*LAPACKE_get_nancheck.*",
                                                       ".*LAPACK_GLOBAL.*").skip())
               .put(new Info("OPENBLAS_PTHREAD_CREATE_FUNC", "OPENBLAS_BUNDERSCORE", "OPENBLAS_FUNDERSCORE", "DOUBLE_DEFINED", "xdouble",
                             "FLOATRET", "OPENBLAS_CONST", "CBLAS_INDEX", "lapack_int", "lapack_logical").cppTypes().annotations())
               .put(new Info("OPENBLAS_QUAD_PRECISION", "defined OPENBLAS_EXPRECISION", "OPENBLAS_USE64BITINT",
                             "defined(LAPACK_COMPLEX_STRUCTURE)", "defined(LAPACK_COMPLEX_C99)").define(false))
               .put(new Info("((defined(__STDC_IEC_559_COMPLEX__) || __STDC_VERSION__ >= 199901L ||"
                       + "      (__GNUC__ >= 3 && !defined(__cplusplus))) && !(defined(FORCE_OPENBLAS_COMPLEX_STRUCT))) && !defined(_MSC_VER)",
                             "defined(LAPACK_COMPLEX_CPP)", "LAPACK_COMPLEX_CUSTOM").define())
               .put(new Info("openblas_complex_float", "lapack_complex_float").cast().pointerTypes("FloatPointer", "FloatBuffer", "float[]"))
               .put(new Info("openblas_complex_double", "lapack_complex_double").cast().pointerTypes("DoublePointer", "DoubleBuffer", "double[]"));

        String[] functions = {
            // not available in Accelerate
            "cblas_caxpby", "cblas_daxpby", "cblas_saxpby", "cblas_zaxpby",
            // not exported by OpenBLAS
            "cblas_cgemm3m", "cblas_zgemm3m", "cblas_xerbla",
            // not implemented by MKL
            "openblas_set_num_threads", "goto_set_num_threads", "openblas_get_num_threads", "openblas_get_num_procs",
            "openblas_get_config", "openblas_get_corename", "openblas_get_parallel", "cblas_cdotc", "cblas_cdotu", "cblas_cgeadd",
            "cblas_cimatcopy", "cblas_comatcopy", "cblas_dgeadd", "cblas_dimatcopy", "cblas_domatcopy", "cblas_sgeadd",
            "cblas_simatcopy", "cblas_somatcopy", "cblas_zdotc", "cblas_zdotu", "cblas_zgeadd", "cblas_zimatcopy", "cblas_zomatcopy",
            "clacrm", "dlacrm", "slacrm", "zlacrm", "clarcm", "dlarcm", "slarcm", "zlarcm", "classq", "dlassq", "slassq", "zlassq",
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

    static int maxThreads = -1;
    static int vendor = 0;

    public static class SetNumThreads extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    SetNumThreads(Pointer p) { super(p); }
        protected SetNumThreads() { allocate(); }
        private native void allocate();
        public native void call(int nth);
        public native Pointer get();
        public native SetNumThreads put(Pointer address);
    }

    public static class MklSetNumThreadsLocal extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    MklSetNumThreadsLocal(Pointer p) { super(p); }
        protected MklSetNumThreadsLocal() { allocate(); }
        private native void allocate();
        public native int call(int nth);
        public native Pointer get();
        public native MklSetNumThreadsLocal put(Pointer address);
    }

    public static class MklDomainSetNumThreads extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    MklDomainSetNumThreads(Pointer p) { super(p); }
        protected MklDomainSetNumThreads() { allocate(); }
        private native void allocate();
        public native int call(int nth, int MKL_DOMAIN);
        public native Pointer get();
        public native MklDomainSetNumThreads put(Pointer address);
    }

    public static void blas_set_num_threads(int num) {
        Loader.load(openblas.class);

        Pointer mklSetNumThreads = Loader.addressof("MKL_Set_Num_Threads");
        Pointer mklSetNumThreadsLocal = Loader.addressof("MKL_Set_Num_Threads_Local");
        Pointer mklDomainSetNumThreads = Loader.addressof("MKL_Domain_Set_Num_Threads");
        Pointer openblasSetNumThreads = Loader.addressof("openblas_set_num_threads");

        vendor = 0;
        if (mklSetNumThreads != null) {
            new SetNumThreads().put(mklSetNumThreads).call(num);
            vendor = 3;
        }
        if (mklSetNumThreadsLocal != null) {
            new MklSetNumThreadsLocal().put(mklSetNumThreadsLocal).call(num);
            vendor = 3;
        }
        if (mklDomainSetNumThreads != null) {
            MklDomainSetNumThreads f = new MklDomainSetNumThreads().put(mklDomainSetNumThreads);
            f.call(num, 0); // DOMAIN_ALL
            f.call(num, 1); // DOMAIN_BLAS
            vendor = 3;
        }
        if (openblasSetNumThreads != null) {
            new SetNumThreads().put(openblasSetNumThreads).call(num);
            vendor = 2;
        }

        if (vendor != 0) {
            maxThreads = num;
        } else {
            System.out.println("Unable to tune runtime. Please set OMP_NUM_THREADS manually.");
        }
    }

    public static int blas_get_num_threads() {
        return maxThreads;
    }

    /**
     *  0 - Unknown
     *  1 - cuBLAS
     *  2 - OpenBLAS
     *  3 - MKL
     */
    public static int blas_get_vendor() {
        return vendor;
    }
}
