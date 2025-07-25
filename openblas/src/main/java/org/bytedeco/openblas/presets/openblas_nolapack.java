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
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 * @author Dragan Djuric
 */
@Properties(inherit = javacpp.class, global = "org.bytedeco.openblas.global.openblas_nolapack", value = {
    @Platform(define = {"__OPENBLAS 1", "LAPACK_COMPLEX_CPP"},
              include = {"openblas_config.h", "cblas.h"},
              link    =  "openblas_nolapack@.0", resource = {"include", "lib"},
              preload = {"gcc_s@.1.1", "gcc_s@.1", "quadmath@.0", "gfortran@.5", "gfortran@.4", "gfortran@.3", "openblas@.0#openblas_nolapack@.0"},
              preloadpath = {"/opt/intel/oneapi/mkl/latest/lib/", "/opt/intel/oneapi/compiler/latest/mac/compiler/lib/"}),
    @Platform(value = "android", link = "openblas", preload = ""),
    @Platform(value = "macosx",  preloadpath = {"/usr/local/lib/gcc/13/", "/usr/local/lib/gcc/12/", "/usr/local/lib/gcc/11/", "/usr/local/lib/gcc/10/", "/usr/local/lib/gcc/9/",
                                                "/usr/local/lib/gcc/8/", "/usr/local/lib/gcc/7/", "/usr/local/lib/gcc/6/", "/usr/local/lib/gcc/5/"}),
    @Platform(value = "windows", preload = "libopenblas#libopenblas_nolapack"),
    @Platform(value = "windows-x86",    preloadpath = {"C:/Program Files (x86)/Intel/oneAPI/mkl/latest/bin32/",
                                                       "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin32/"}),
    @Platform(value = "windows-x86_64", preloadpath = {"C:/Program Files (x86)/Intel/oneAPI/mkl/latest/bin/",
                                                       "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/"}),
    @Platform(value = "linux",          preloadpath = {"/usr/lib/", "/usr/lib32/", "/usr/lib64/"}),
    @Platform(value = "linux-armhf",    preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}),
    @Platform(value = "linux-arm64",    preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"}),
    @Platform(value = "linux-x86",      preloadpath = {"/lib32/", "/lib/", "/usr/lib32/", "/usr/lib/",
                                                       "/opt/intel/oneapi/mkl/latest/lib32/", "/opt/intel/oneapi/compiler/latest/lib32/"}),
    @Platform(value = "linux-x86_64",   preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/",
                                                       "/opt/intel/oneapi/mkl/latest/lib/", "/opt/intel/oneapi/compiler/latest/lib/"}),
    @Platform(value = "linux-ppc64",    preloadpath = {"/usr/powerpc64-linux-gnu/lib/", "/usr/powerpc64le-linux-gnu/lib/",
                                                       "/usr/lib/powerpc64-linux-gnu/", "/usr/lib/powerpc64le-linux-gnu/"}),
    @Platform(value = "ios", preload = "libopenblas") })
@NoException
public class openblas_nolapack implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "openblas"); }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        List<String> links = properties.get("platform.link");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");
        String linkName = getLinkName(); // "openblas_nolapack" or "openblas"

        // Let users enable loading of arbitrary library (for Accelerate, MKL, etc) or "none"
        String lib = System.getProperty("org.bytedeco." + linkName + ".load", "").toLowerCase();
        if (lib.length() == 0) {
            lib = System.getProperty("org.bytedeco.openblas.load", "").toLowerCase();
        }

        if (lib.equals("none")) {
            links.clear();
            preloads.clear();
            properties.setProperty("platform.library", "#none#");
            return;
        }

        // Only apply this at load time for this class only, without inheriting, since MKLML,
        // for example, doesn't come with all of LAPACK, the user might want to use it with
        // openblas_nolapack.class, but have openblas.class load something else.
        if (!Loader.isLoadLibraries() || getClass() != properties.getEffectiveClasses().get(0)) {
            return;
        }

        int i = 0;
        if (platform.startsWith("linux-x86") || platform.startsWith("macosx-x86") || platform.startsWith("windows-x86")) {
            if (lib.equals("mklml") || lib.equals("mklml_intel")) {
                String[] libs = {"iomp5", "libiomp5md", "msvcr120", "mklml_intel", "mklml"};
                for (i = 0; i < libs.length; i++) {
                    preloads.add(i, libs[i] + "#" + libs[i]);
                }
                lib = platform.startsWith("linux") ? "mklml_intel" : "mklml";
                resources.add("/org/bytedeco/mkldnn/");
            } else if (lib.equals("mkl") || lib.equals("mkl_rt")) {
                String[] libs = {"iomp5", "libiomp5md", "mkl_core@.2", "mkl_avx@.2", "mkl_avx2@.2", "mkl_avx512@.2", "mkl_avx512_mic@.2",
                                 "mkl_def@.2", "mkl_mc@.2", "mkl_mc3@.2", "mkl_intel_lp64@.2", "mkl_intel_thread@.2", "mkl_gnu_thread@.2", "mkl_rt@.2"};
                for (i = 0; i < libs.length; i++) {
                    preloads.add(i, libs[i] + "#" + libs[i]);
                }
                lib = "mkl_rt@.2";
                resources.add("/org/bytedeco/mkl/");
            }
        }

        if (lib.length() > 0) {
            if (platform.startsWith("linux")) {
                preloads.add(i, lib + "#" + linkName + "@.0");
            } else if (platform.startsWith("macosx")) {
                preloads.add(i, lib + "#" + linkName + ".0");
            } else if (platform.startsWith("windows")) {
                preloads.add(i, lib + "#lib" + linkName);
            }
        }
    }

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
            // not available in Accelerate
            "cblas_caxpby", "cblas_daxpby", "cblas_saxpby", "cblas_zaxpby", "cblas_caxpyc", "cblas_zaxpyc",
            "cblas_sgemmt", "cblas_dgemmt",
            "cblas_cgemm_batch", "cblas_dgemm_batch", "cblas_sgemm_batch", "cblas_zgemm_batch",
            "cblas_samax", "cblas_damax", "cblas_scamax", "cblas_dzamax",
            "cblas_samin", "cblas_damin", "cblas_scamin", "cblas_dzamin",
            // not exported by OpenBLAS
            "cblas_xerbla", "cblas_icamin", "cblas_idamin", "cblas_isamin", "cblas_izamin",
            "cblas_ssum", "cblas_dsum", "cblas_scsum", "cblas_dzsum",
            "cblas_ismax", "cblas_idmax", "cblas_icmax", "cblas_izmax",
            "cblas_ismin", "cblas_idmin", "cblas_icmin", "cblas_izmin",
            "cblas_csrot", "cblas_zdrot", "cblas_crotg", "cblas_zrotg",
            // not implemented by MKL
            "openblas_set_num_threads", "goto_set_num_threads", "openblas_get_num_threads", "openblas_get_num_procs",
            "openblas_get_config", "openblas_get_corename", "openblas_get_parallel", "cblas_cdotc", "cblas_cdotu",
            "cblas_cimatcopy", "cblas_comatcopy", "cblas_dimatcopy", "cblas_domatcopy", "cblas_cgeadd", "cblas_dgeadd", "cblas_sgeadd",
            "cblas_simatcopy", "cblas_somatcopy", "cblas_zdotc", "cblas_zdotu", "cblas_zgeadd", "cblas_zimatcopy", "cblas_zomatcopy",
            "clacrm", "dlacrm", "slacrm", "zlacrm", "clarcm", "dlarcm", "slarcm", "zlarcm", "classq", "dlassq", "slassq", "zlassq",
            "lapack_make_complex_double", "lapack_make_complex_float",
            "cgetsqrhrt", "dgetsqrhrt", "sgetsqrhrt", "zgetsqrhrt", "dorgtsqr_row", "sorgtsqr_row", "cungtsqr_row", "zungtsqr_row",
            "ctz_trans", "dtz_trans", "stz_trans", "ztz_trans", "ctz_nancheck", "dtz_nancheck", "stz_nancheck", "ztz_nancheck",
            "sorhr_col", "dorhr_col", "cunhr_col", "zunhr_col",
            // deprecated
            "cgegs",  "ctzrqf",  "dgeqpf",  "dlatzm",  "sgelsx",  "slahrd",  "zgegv",
            "cgegv",  "dgegs",   "dtzrqf",  "sgeqpf",  "slatzm",  "zgelsx",  "zlahrd",
            "cgelsx", "clahrd",  "dgegv",   "sgegs",   "stzrqf",  "zgeqpf",  "zlatzm",
            "cgeqpf", "clatzm",  "dgelsx",  "dlahrd",  "sgegv",   "zgegs",   "ztzrqf"};
        for (String f : functions) {
            infoMap.put(new Info(f, "LAPACK_" + f, "LAPACK_" + f + "_base", "LAPACKE_" + f, "LAPACKE_" + f + "_work").skip(skipFunctions()));
        }

        String[] brokenFunctions = {
            // not implemented by MKL
            "cgesvdq", "dgesvdq", "sgesvdq", "zgesvdq", "clangb", "dlangb", "slangb", "zlangb",
            "ctrsyl3", "dtrsyl3", "strsyl3", "ztrsyl3",
            "sgedmd", "dgedmd", "cgedmd", "zgedmd", "sgedmdq", "dgedmdq", "cgedmdq", "zgedmdq",
            // deprecated
            "cggsvd", "dggsvd", "sggsvd", "zggsvd", "zggsvp", "cggsvp", "dggsvp", "sggsvp",
            // extended
            "cgbrfsx", "cporfsx", "dgerfsx", "sgbrfsx", "ssyrfsx", "zherfsx", "cgerfsx", "csyrfsx", "dporfsx", "sgerfsx", "zgbrfsx", "zporfsx",
            "cherfsx", "dgbrfsx", "dsyrfsx", "sporfsx", "zgerfsx", "zsyrfsx", "cgbsvxx", "cposvxx", "dgesvxx", "sgbsvxx", "ssysvxx", "zhesvxx",
            "cgesvxx", "csysvxx", "dposvxx", "sgesvxx", "zgbsvxx", "zposvxx", "chesvxx", "dgbsvxx", "dsysvxx", "sposvxx", "zgesvxx", "zsysvxx",
            //broken on android and Windows
            "cblas_cgemmt", "cblas_zgemmt",
            "cblas_sbstobf16", "cblas_sbdtobf16", "cblas_sbf16tos", "cblas_dbf16tod",
            "cblas_sbdot",  "cblas_sbgemv", "cblas_sbgemm", "cblas_sbgemm_batch",
            "cblas_cgemm3m", "cblas_zgemm3m",
            // broken on Windows
            "openblas_set_num_threads_local", "openblas_set_threads_callback_function"};
        for (String f : brokenFunctions) {
            infoMap.put(new Info(f, "LAPACK_" + f, "LAPACK_" + f + "_base", "LAPACKE_" + f, "LAPACKE_" + f + "_work").skip(true));
        }
    }

    protected boolean skipFunctions() {
        return true;
    }

    protected String getLinkName() {
        return getClass().getSimpleName();
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
        Loader.load(openblas_nolapack.class);

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
