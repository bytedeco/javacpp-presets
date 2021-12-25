/*
 * Copyright (C) 2017-2021 Samuel Audet
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

package org.bytedeco.mkl.presets;

import org.bytedeco.javacpp.Loader;
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
 */
@Properties(inherit = javacpp.class, global = "org.bytedeco.mkl.global.mkl_rt", names = {"linux-x86", "macosx-x86", "windows-x86"}, value = {
    @Platform(include = {"mkl.h", "mkl_version.h", "mkl_types.h", /*"mkl_blas.h",*/ "mkl_trans.h", "mkl_cblas.h", "mkl_spblas.h", /*"mkl_lapack.h",*/ "mkl_lapacke.h",
        "mkl_dss.h", "mkl_pardiso.h", "mkl_sparse_handle.h", "mkl_service.h", "mkl_rci.h", "mkl_vml.h", "mkl_vml_defines.h", "mkl_vml_types.h", "mkl_vml_functions.h",
        "mkl_vsl.h", "mkl_vsl_defines.h", "mkl_vsl_types.h", "mkl_vsl_functions.h", "mkl_df.h", "mkl_df_defines.h", "mkl_df_types.h", "mkl_df_functions.h",
        "mkl_dfti.h", "mkl_trig_transforms.h", "mkl_poisson.h", "mkl_solvers_ee.h", /*"mkl_direct_types.h", "mkl_direct_blas.h", "mkl_direct_lapack.h", "mkl_direct_call.h",
        "mkl_dnn_types.h", "mkl_dnn.h", "mkl_blacs.h", "mkl_pblas.h", "mkl_scalapack.h", "mkl_cdft_types.h", "mkl_cdft.h", "i_malloc.h" */},
              compiler = {"fastfpu", "nodeprecated"}, includepath = "/opt/intel/oneapi/mkl/latest/include/",
              linkpath = {"/opt/intel/oneapi/mkl/latest/lib/", "/opt/intel/oneapi/compiler/latest/mac/compiler/lib/"}, link = "mkl_rt@.2",
              preload = {"mkl_core@.2", "iomp5", "libiomp5md", "mkl_gnu_thread@.2", "mkl_intel_lp64@.2", "mkl_intel_thread@.2",
                         "mkl_def@.2", "mkl_mc@.2", "mkl_mc3@.2", "mkl_p4@.2", "mkl_p4m@.2", "mkl_p4m3@.2", "mkl_avx@.2", "mkl_avx2@.2", "mkl_avx512@.2", "mkl_avx512_mic@.2",
                         "mkl_vml_def@.2", "mkl_vml_ia@.2", "mkl_vml_mc@.2", "mkl_vml_mc2@.2", "mkl_vml_mc3@.2", "mkl_vml_p4@.2", "mkl_vml_p4m@.2", "mkl_vml_p4m2@.2", "mkl_vml_p4m3@.2",
                         "mkl_vml_avx@.2", "mkl_vml_avx2@.2", "mkl_vml_avx512@.2", "mkl_vml_avx512_mic@.2", "mkl_vml_cmpt@.2"}, resource = {"include", "lib"}),
    @Platform(value = "linux-x86",    linkpath = {"/opt/intel/oneapi/mkl/latest/lib/ia32/", "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/ia32_lin/"}),
    @Platform(value = "linux-x86_64", linkpath = {"/opt/intel/oneapi/mkl/latest/lib/intel64/", "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/"}),
    @Platform(value = "windows",     includepath = "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include/"),
    @Platform(value = "windows-x86",    linkpath = "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/ia32/",
                                     preloadpath = {"C:/Program Files (x86)/Intel/oneAPI/mkl/latest/redist/ia32/",
                                                    "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/redist/ia32_win/compiler/"}),
    @Platform(value = "windows-x86_64", linkpath = "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64/",
                                     preloadpath = {"C:/Program Files (x86)/Intel/oneAPI/mkl/latest/redist/intel64/",
                                                    "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/redist/intel64_win/compiler/"}) })
@NoException
public class mkl_rt implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "mkl"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("MKL_INT", "MKL_INT64", "MKL_UINT", "MKL_UINT64", "MKL_LONG", "MKL_DECLSPEC", "MKL_CALL_CONV", "INTEL_API_DEF",
                             "MKL_UINT8", "MKL_INT8", "MKL_INT16", "MKL_BF16", "MKL_INT32", "MKL_F16", "NOTHROW",

                             "mkl_simatcopy", "mkl_dimatcopy", "mkl_cimatcopy", "mkl_zimatcopy", "mkl_somatcopy", "mkl_domatcopy", "mkl_comatcopy", "mkl_zomatcopy",
                             "mkl_somatcopy2", "mkl_domatcopy2", "mkl_comatcopy2", "mkl_zomatcopy2", "mkl_somatadd", "mkl_domatadd", "mkl_comatadd", "mkl_zomatadd",
                             "mkl_simatcopy_batch_strided", "mkl_dimatcopy_batch_strided", "mkl_cimatcopy_batch_strided", "mkl_zimatcopy_batch_strided",
                             "mkl_somatcopy_batch_strided", "mkl_domatcopy_batch_strided", "mkl_comatcopy_batch_strided", "mkl_zomatcopy_batch_strided",

                             "CBLAS_INDEX", "lapack_int", "lapack_logical", "lapack_complex_float", "lapack_complex_double", "_INTEGER_t", "_DoubleComplexType",

                             "mkl_get_version", "mkl_get_version_string", "mkl_free_buffers", "mkl_thread_free_buffers", "mkl_mem_stat", "mkl_peak_mem_usage",
                             "mkl_malloc", "mkl_calloc", "mkl_realloc", "mkl_free", "mkl_disable_fast_mm", "mkl_get_cpu_clocks", "mkl_get_cpu_frequency",
                             "mkl_get_max_cpu_frequency", "mkl_get_clocks_frequency", "mkl_set_num_threads_local", "mkl_set_num_threads", "mkl_get_max_threads",
                             "mkl_set_num_stripes", "mkl_get_num_stripes", "mkl_domain_set_num_threads", "mkl_domain_get_max_threads", "mkl_set_dynamic",
                             "mkl_get_dynamic", "mkl_enable_instructions", "mkl_set_interface_layer", "mkl_set_threading_layer", "mkl_mic_enable", "mkl_mic_disable",
                             "mkl_mic_get_device_count", "mkl_mic_get_cpuinfo", "mkl_mic_get_meminfo", "mkl_mic_set_resource_limit", "mkl_mic_get_resource_limit",
                             "mkl_mic_set_workdivision", "mkl_mic_get_workdivision", "mkl_mic_set_max_memory", "mkl_mic_free_memory", "mkl_mic_set_offload_report",
                             "mkl_mic_set_device_num_threads", "mkl_mic_set_flags", "mkl_mic_get_flags", "mkl_mic_get_status", "mkl_mic_clear_status", "mkl_cbwr_get",
                             "mkl_cbwr_set", "mkl_cbwr_get_auto_branch", "mkl_set_env_mode", "mkl_verbose", "mkl_verbose_output_file", "mkl_set_exit_handler",
                             "mkl_mic_register_memory", "mkl_set_mpi", "mkl_set_memory_limit", "mkl_finalize",

                             "DFTI_DFT_Desc_struct", "DFTI_Descriptor_struct", "DFTI_Descriptor",

                             "d_init_Helmholtz_2D", "d_commit_Helmholtz_2D", "d_Helmholtz_2D", "free_Helmholtz_2D",
                             "d_init_Helmholtz_3D", "d_commit_Helmholtz_3D", "d_Helmholtz_3D", "free_Helmholtz_3D",
                             "s_init_Helmholtz_2D", "s_commit_Helmholtz_2D", "s_Helmholtz_2D",
                             "s_init_Helmholtz_3D", "s_commit_Helmholtz_3D", "s_Helmholtz_3D",

                             "mkl_dc_type", "mkl_dc_real_type", "mkl_dc_native_type", "mkl_dc_veclen", "MKL_DC_PREC_LETTER",
                             "mkl_dc_gemm", "mkl_dc_syrk", "mkl_dc_trsm", "mkl_dc_axpy", "mkl_dc_dot", "MKL_DC_DOT_CONVERT",
                             "mkl_dc_getrf", "mkl_dc_lapacke_getrf_convert", "mkl_dc_getri", "mkl_dc_lapacke_getri_convert", "mkl_dc_getrs", "mkl_dc_lapacke_getrs_convert",
                             "__inline", "MKL_DIRECT_CALL_INIT_FLAG", "mkl_jit_create_dgemm", "mkl_jit_create_sgemm", "mkl_jit_create_cgemm", "mkl_jit_create_zgemm").cppTypes().annotations())

               .put(new Info("MKL_DEPRECATED").cppText("#define MKL_DEPRECATED deprecated").cppTypes())
               .put(new Info("MKL_DEPRECATED_C").cppText("#define MKL_DEPRECATED_C deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("MKL_CBWR_UNSET_ALL", "MKL_BLACS_LASTMPI").translate(false))

               .put(new Info("sparse_matrix_t").valueTypes("sparse_matrix").pointerTypes("@ByPtrPtr sparse_matrix"))
               .put(new Info("sparse_vector_t").valueTypes("sparse_vector").pointerTypes("@ByPtrPtr sparse_vector"))

               .put(new Info("MKL_Complex8").define().cast().pointerTypes("FloatPointer", "FloatBuffer", "float[]"))
               .put(new Info("MKL_Complex16").define().cast().pointerTypes("DoublePointer", "DoubleBuffer", "double[]"))

               .put(new Info("dss_delete_").javaText("public static native int dss_delete_(@Cast(\"void**\") @ByPtrPtr _MKL_DSS_HANDLE_t arg0, IntPointer arg1);"))

               .put(new Info("!defined(MKL_CALL_CONV)").define(false))
               .put(new Info("!defined(_WIN32) & !defined(_WIN64)").define(true))

               .put(new Info("VSL_SS_DNAN").cppTypes("long long"))
               .put(new Info("VSLStreamStatePtr").valueTypes("VSLStreamStatePtr").pointerTypes("@Cast(\"void**\") @ByPtrPtr VSLStreamStatePtr"))
               .put(new Info("VSLConvTaskPtr").valueTypes("VSLConvTaskPtr").pointerTypes("@Cast(\"void**\") @ByPtrPtr VSLConvTaskPtr"))
               .put(new Info("VSLCorrTaskPtr").valueTypes("VSLCorrTaskPtr").pointerTypes("@Cast(\"void**\") @ByPtrPtr VSLCorrTaskPtr"))

               .put(new Info("DFTI_DESCRIPTOR_HANDLE").valueTypes("DFTI_DESCRIPTOR").pointerTypes("@ByPtrPtr DFTI_DESCRIPTOR"))
               .put(new Info("DftiCreateDescriptor").define().javaText("public static native long DftiCreateDescriptor(@ByPtrPtr DFTI_DESCRIPTOR desc,\n" +
                                                                       "                              @Cast(\"DFTI_CONFIG_VALUE\") int precision,\n" +
                                                                       "                              @Cast(\"DFTI_CONFIG_VALUE\") int domain,\n" +
                                                                       "                              long dimension, long length);\n" +
                                                                       "public static native long DftiCreateDescriptor(@ByPtrPtr DFTI_DESCRIPTOR desc,\n" +
                                                                       "                              @Cast(\"DFTI_CONFIG_VALUE\") int precision,\n" +
                                                                       "                              @Cast(\"DFTI_CONFIG_VALUE\") int domain,\n" +
                                                                       "                              long dimension, CLongPointer sizes);"))

               .put(new Info("dnnPrimitive_t").valueTypes("_uniPrimitive_s").pointerTypes("@ByPtrPtr _uniPrimitive_s"))
               .put(new Info("dnnLayout_t").valueTypes("_dnnLayout_s").pointerTypes("@ByPtrPtr _dnnLayout_s"))

               .put(new Info("DFTI_DESCRIPTOR_DM_HANDLE").valueTypes("_DFTI_DESCRIPTOR_DM").pointerTypes("@ByPtrPtr _DFTI_DESCRIPTOR_DM"))

               .put(new Info("cblas_caxpby", "cblas_daxpby", "cblas_saxpby", "cblas_zaxpby", "mklfreetls", "MKLFREETLS", "MKLFreeTls",
                             "mkl_sparse_c_create_vector", "mkl_sparse_d_create_vector", "mkl_sparse_s_create_vector", "mkl_sparse_z_create_vector",
                             "mkl_sparse_c_export_vector", "mkl_sparse_d_export_vector", "mkl_sparse_s_export_vector", "mkl_sparse_z_export_vector",
                             "mkl_sparse_destroy_vector", "mkl_sparse_c_spmspvd", "mkl_sparse_d_spmspvd", "mkl_sparse_s_spmspvd", "mkl_sparse_z_spmspvd",
                             "mkl_sparse_set_spmspvd_hint", "replace_operation", "PardisopivotEntry", "MKL_Verbose_Output_File",
                             "MKL_MIC_Enable", "MKL_MIC_Disable", "MKL_MIC_Get_Device_Count", "MKL_MIC_Get_Cpuinfo", "MKL_MIC_Get_Meminfo",
                             "MKL_MIC_Set_Workdivision", "MKL_MIC_Get_Workdivision", "MKL_MIC_Set_Max_Memory", "MKL_MIC_Free_Memory",
                             "MKL_MIC_Set_Offload_Report", "MKL_MIC_Set_Device_Num_Threads", "MKL_MIC_Set_Resource_Limit", "MKL_MIC_Get_Resource_Limit",
                             "VMDAcospi", "VMDAsinpi", "VMDAtanpi", "VMDCosd", "VMDCospi", "VMDSind", "VMDSinpi", "VMDTand", "VMDTanpi",
                             "MKL_MIC_Get_Flags", "MKL_MIC_Set_Flags", "MKL_MIC_Get_Status", "MKL_MIC_Clear_Status",
                             "VSLCONVCopyTask", "VSLCONVDeleteTask", "VSLCORRCopyTask", "VSLCORRDeleteTask",
                             "vclog2i", "vmclog2i", "vmzlog2i", "vzlog2i",
                             "vcLog2I", "vmcLog2I", "vmzLog2I", "vzLog2I",
                             "VCLOG2I", "VMCLOG2I", "VMZLOG2I", "VZLOG2I").skip());
    }
}
