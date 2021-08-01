/*
 * Copyright (C) 2014-2019 Samuel Audet
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

package org.bytedeco.gsl.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.MemberGetter;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.Opaque;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.openblas.presets.openblas;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = openblas.class, target = "org.bytedeco.gsl", global = "org.bytedeco.gsl.global.gsl", value = {
    @Platform(include = {"gsl/gsl_types.h", "gsl/gsl_errno.h", "gsl/gsl_ieee_utils.h", "gsl/gsl_inline.h", "gsl/gsl_message.h", "gsl/gsl_complex.h",
        "gsl/gsl_complex_math.h", "gsl/gsl_check_range.h", "gsl/gsl_sys.h", "gsl/gsl_machine.h", "gsl/gsl_precision.h", "gsl/gsl_nan.h", "gsl/gsl_pow_int.h",
        "gsl/gsl_math.h", "gsl/gsl_min.h", "gsl/gsl_minmax.h", "gsl/gsl_mode.h", "gsl/gsl_test.h", "gsl/gsl_version.h",

        "gsl/gsl_block.h", /*"gsl/gsl_block_complex_long_double.h"*/ "gsl/gsl_block_complex_double.h", "gsl/gsl_block_complex_float.h",
        /*"gsl/gsl_block_long_double.h", */ "gsl/gsl_block_double.h", "gsl/gsl_block_float.h", "gsl/gsl_block_ulong.h", "gsl/gsl_block_long.h",
        "gsl/gsl_block_uint.h", "gsl/gsl_block_int.h", "gsl/gsl_block_ushort.h", "gsl/gsl_block_short.h", "gsl/gsl_block_uchar.h", "gsl/gsl_block_char.h",

        "gsl/gsl_vector_complex.h", "gsl/gsl_vector.h", /*"gsl/gsl_vector_complex_long_double.h",*/ "gsl/gsl_vector_complex_double.h",
        "gsl/gsl_vector_complex_float.h", /*"gsl_vector_long_double.h",*/ "gsl/gsl_vector_double.h", "gsl/gsl_vector_float.h", "gsl/gsl_vector_ulong.h",
        "gsl/gsl_vector_long.h", "gsl/gsl_vector_uint.h", "gsl/gsl_vector_int.h", "gsl/gsl_vector_ushort.h", "gsl/gsl_vector_short.h",
        "gsl/gsl_vector_uchar.h", "gsl/gsl_vector_char.h",

        "gsl/gsl_matrix.h", "gsl/gsl_blas_types.h", /*"gsl/gsl_matrix_complex_long_double.h",*/ "gsl/gsl_matrix_complex_double.h", "gsl/gsl_matrix_complex_float.h",
        /*"gsl/gsl_matrix_long_double.h",*/ "gsl/gsl_matrix_double.h", "gsl/gsl_matrix_float.h", "gsl/gsl_matrix_ulong.h", "gsl/gsl_matrix_long.h",
        "gsl/gsl_matrix_uint.h", "gsl/gsl_matrix_int.h", "gsl/gsl_matrix_ushort.h", "gsl/gsl_matrix_short.h", "gsl/gsl_matrix_uchar.h", "gsl/gsl_matrix_char.h",

        /*"gsl/gsl_cblas.h",*/ "gsl/gsl_blas.h", "gsl/gsl_bspline.h", "gsl/gsl_cdf.h", "gsl/gsl_chebyshev.h", "gsl/gsl_combination.h",
        "gsl/gsl_deriv.h", "gsl/gsl_dht.h", "gsl/gsl_diff.h", "gsl/gsl_eigen.h", "gsl/gsl_fit.h", "gsl/gsl_permutation.h", "gsl/gsl_heapsort.h",
        "gsl/gsl_histogram2d.h", "gsl/gsl_histogram.h", "gsl/gsl_integration.h", "gsl/gsl_interp.h", "gsl/gsl_linalg.h", "gsl/gsl_poly.h", "gsl/gsl_rng.h",
        "gsl/gsl_qrng.h", "gsl/gsl_randist.h", "gsl/gsl_roots.h",  "gsl/gsl_siman.h", "gsl/gsl_spline.h", "gsl/gsl_sum.h", "gsl/gsl_wavelet.h",
        "gsl/gsl_wavelet2d.h",

        "gsl/gsl_multilarge.h", "gsl/gsl_rstat.h", "gsl/gsl_spmatrix.h", "gsl/gsl_spblas.h", "gsl/gsl_splinalg.h", "gsl/gsl_interp2d.h", "gsl/gsl_spline2d.h",
        "gsl/gsl_bst_avl.h", "gsl/gsl_bst_rb.h", "gsl/gsl_bst_types.h", "gsl/gsl_bst.h",
        /*"gsl/gsl_spmatrix_complex_long_double.h",*/ "gsl/gsl_spmatrix_complex_double.h", "gsl/gsl_spmatrix_complex_float.h", /*"gsl/gsl_spmatrix_long_double.h",*/
        "gsl/gsl_spmatrix_double.h", "gsl/gsl_spmatrix_float.h", "gsl/gsl_spmatrix_ulong.h", "gsl/gsl_spmatrix_long.h", "gsl/gsl_spmatrix_uint.h",
        "gsl/gsl_spmatrix_int.h", "gsl/gsl_spmatrix_ushort.h", "gsl/gsl_spmatrix_short.h", "gsl/gsl_spmatrix_uchar.h", "gsl/gsl_spmatrix_char.h",

        "gsl/gsl_const.h", "gsl/gsl_const_num.h", "gsl/gsl_const_cgs.h", "gsl/gsl_const_mks.h", "gsl/gsl_const_cgsm.h", "gsl/gsl_const_mksa.h",

        "gsl/gsl_fft.h", "gsl/gsl_fft_complex_float.h", "gsl/gsl_fft_complex.h", "gsl/gsl_fft_halfcomplex_float.h", "gsl/gsl_fft_halfcomplex.h",
        "gsl/gsl_fft_real_float.h", "gsl/gsl_fft_real.h", "gsl/gsl_dft_complex_float.h", "gsl/gsl_dft_complex.h",

        "gsl/gsl_monte.h", "gsl/gsl_monte_plain.h", "gsl/gsl_monte_miser.h", "gsl/gsl_monte_vegas.h",

        "gsl/gsl_multifit.h", "gsl/gsl_multifit_nlin.h", "gsl/gsl_multimin.h", "gsl/gsl_multiroots.h", "gsl/gsl_multiset.h", "gsl/gsl_ntuple.h",
        "gsl/gsl_odeiv2.h", "gsl/gsl_odeiv.h",

        "gsl/gsl_permute.h", /*"gsl/gsl_permute_complex_long_double.h",*/ "gsl/gsl_permute_complex_double.h", "gsl/gsl_permute_complex_float.h",
        /*"gsl/gsl_permute_long_double.h",*/ "gsl/gsl_permute_double.h", "gsl/gsl_permute_float.h", "gsl/gsl_permute_ulong.h", "gsl/gsl_permute_long.h",
        "gsl/gsl_permute_uint.h", "gsl/gsl_permute_int.h", "gsl/gsl_permute_ushort.h", "gsl/gsl_permute_short.h", "gsl/gsl_permute_uchar.h",
        "gsl/gsl_permute_char.h",   

        "gsl/gsl_permute_vector.h", /*"gsl/gsl_permute_vector_complex_long_double.h",*/ "gsl/gsl_permute_vector_complex_double.h",
        "gsl/gsl_permute_vector_complex_float.h", /*"gsl/gsl_permute_vector_long_double.h",*/ "gsl/gsl_permute_vector_double.h",
        "gsl/gsl_permute_vector_float.h", "gsl/gsl_permute_vector_ulong.h", "gsl/gsl_permute_vector_long.h", "gsl/gsl_permute_vector_uint.h",
        "gsl/gsl_permute_vector_int.h", "gsl/gsl_permute_vector_ushort.h", "gsl/gsl_permute_vector_short.h", "gsl/gsl_permute_vector_uchar.h",
        "gsl/gsl_permute_vector_char.h",

        "gsl/gsl_specfunc.h", "gsl/gsl_sf.h", "gsl/gsl_sf_result.h", "gsl/gsl_sf_airy.h", "gsl/gsl_sf_bessel.h", "gsl/gsl_sf_clausen.h",
        "gsl/gsl_sf_coulomb.h", "gsl/gsl_sf_coupling.h", "gsl/gsl_sf_dawson.h", "gsl/gsl_sf_debye.h", "gsl/gsl_sf_dilog.h", "gsl/gsl_sf_elementary.h",
        "gsl/gsl_sf_ellint.h", "gsl/gsl_sf_elljac.h", "gsl/gsl_sf_erf.h", "gsl/gsl_sf_exp.h", "gsl/gsl_sf_expint.h", "gsl/gsl_sf_fermi_dirac.h",
        "gsl/gsl_sf_gamma.h", "gsl/gsl_sf_gegenbauer.h",  "gsl/gsl_sf_hyperg.h", "gsl/gsl_sf_laguerre.h", "gsl/gsl_sf_lambert.h", "gsl/gsl_sf_legendre.h",
        "gsl/gsl_sf_log.h", "gsl/gsl_sf_mathieu.h", "gsl/gsl_sf_pow_int.h", "gsl/gsl_sf_psi.h", "gsl/gsl_sf_synchrotron.h", "gsl/gsl_sf_transport.h",
        "gsl/gsl_sf_trig.h", "gsl/gsl_sf_zeta.h",

        "gsl/gsl_sort.h", /*"gsl/gsl_sort_long_double.h",*/ "gsl/gsl_sort_double.h", "gsl/gsl_sort_float.h", "gsl/gsl_sort_ulong.h", "gsl/gsl_sort_long.h",
        "gsl/gsl_sort_uint.h", "gsl/gsl_sort_int.h", "gsl/gsl_sort_ushort.h", "gsl/gsl_sort_short.h", "gsl/gsl_sort_uchar.h", "gsl/gsl_sort_char.h",

        "gsl/gsl_sort_vector.h", /*"gsl/gsl_sort_vector_long_double.h",*/ "gsl/gsl_sort_vector_double.h", "gsl/gsl_sort_vector_float.h",
        "gsl/gsl_sort_vector_ulong.h", "gsl/gsl_sort_vector_long.h", "gsl/gsl_sort_vector_uint.h", "gsl/gsl_sort_vector_int.h",
        "gsl/gsl_sort_vector_ushort.h", "gsl/gsl_sort_vector_short.h", "gsl/gsl_sort_vector_uchar.h", "gsl/gsl_sort_vector_char.h",

        "gsl/gsl_statistics.h", /*"gsl/gsl_statistics_long_double.h",*/ "gsl/gsl_statistics_double.h", "gsl/gsl_statistics_float.h",
        "gsl/gsl_statistics_ulong.h", "gsl/gsl_statistics_long.h", "gsl/gsl_statistics_uint.h", "gsl/gsl_statistics_int.h",
        "gsl/gsl_statistics_ushort.h", "gsl/gsl_statistics_short.h", "gsl/gsl_statistics_uchar.h", "gsl/gsl_statistics_char.h"},
     define = {"__GSL_CBLAS_H__", "GSL_COMPLEX_LEGACY"}, link = "gsl@.25"),
    @Platform(value = "android", link = "gsl"),
    @Platform(value = "windows", preload = "libgsl-25") })
@NoException
public class gsl implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "gsl"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("__cplusplus").define())
               .put(new Info("FILE").pointerTypes("FILE"))
               .put(new Info("INFINITY", "defined(HUGE_VAL)", "NAN", "defined(INFINITY)",
                       " !defined(GSL_COMPLEX_LEGACY) &&"
                     + "     defined(_Complex_I) &&"
                     + "     defined(complex) &&"
                     + "     defined(I) &&"
                     + "     defined(__STDC__) && (__STDC__ == 1) &&"
                     + "     defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)").define(false))
               .put(new Info("GSL_VAR", "__BEGIN_DECLS", "__END_DECLS", "INLINE_DECL", "INLINE_FUN", "GSL_RANGE_CHECK", "CBLAS_INDEX").cppTypes().annotations())
               .put(new Info("gsl_complex_packed_array", "gsl_complex_packed_ptr").cast().valueTypes("DoublePointer", "DoubleBuffer", "double[]"))
               .put(new Info("gsl_complex_packed_array_float").cast().valueTypes("FloatPointer", "FloatBuffer", "float[]"))
               .put(new Info("gsl_complex").base("DoublePointer"))
               .put(new Info("gsl_complex_float").base("FloatPointer"))
               .put(new Info("GSL_COMPLEX_ONE", "GSL_COMPLEX_ZERO", "GSL_COMPLEX_NEGONE").cppTypes("gsl_complex"))
               .put(new Info("GSL_MACH_EPS", "GSL_POSINF", "GSL_NEGINF", "GSL_NAN").cppTypes("double").translate())
               .put(new Info("gsl_bst_type").purify())
               .put(new Info("gsl_function_struct").pointerTypes("gsl_function"))
               .put(new Info("gsl_function_fdf_struct").pointerTypes("gsl_function_fdf"))
               .put(new Info("gsl_function_vec_struct").pointerTypes("gsl_function_vec"))
               .put(new Info("gsl_block_complex_struct").pointerTypes("gsl_block_complex"))
               .put(new Info("gsl_block_complex_float_struct").pointerTypes("gsl_block_complex_float"))
               .put(new Info("gsl_block_struct").pointerTypes("gsl_block"))
               .put(new Info("gsl_block_float_struct").pointerTypes("gsl_block_float"))
               .put(new Info("gsl_block_ulong_struct").pointerTypes("gsl_block_ulong"))
               .put(new Info("gsl_block_long_struct").pointerTypes("gsl_block_long"))
               .put(new Info("gsl_block_uint_struct").pointerTypes("gsl_block_uint"))
               .put(new Info("gsl_block_int_struct").pointerTypes("gsl_block_int"))
               .put(new Info("gsl_block_ushort_struct").pointerTypes("gsl_block_ushort"))
               .put(new Info("gsl_block_short_struct").pointerTypes("gsl_block_short"))
               .put(new Info("gsl_block_uchar_struct").pointerTypes("gsl_block_uchar"))
               .put(new Info("gsl_block_char_struct").pointerTypes("gsl_block_char"))
               .put(new Info("gsl_cheb_series_struct").pointerTypes("gsl_cheb_series"))
               .put(new Info("gsl_combination_struct").pointerTypes("gsl_combination"))
               .put(new Info("gsl_dht_struct").pointerTypes("gsl_dht"))
               .put(new Info("gsl_permutation_struct").pointerTypes("gsl_permutation"))
               .put(new Info("gsl_monte_function_struct").pointerTypes("gsl_monte_function"))
               .put(new Info("gsl_multifit_function_struct").pointerTypes("gsl_multifit_function"))
               .put(new Info("gsl_multifit_function_fdf_struct").pointerTypes("gsl_multifit_function_fdf"))
               .put(new Info("gsl_multimin_function_struct").pointerTypes("gsl_multimin_function"))
               .put(new Info("gsl_multimin_function_fdf_struct").pointerTypes("gsl_multimin_function_fdf"))
               .put(new Info("gsl_multiroot_function_struct").pointerTypes("gsl_multiroot_function"))
               .put(new Info("gsl_multiroot_function_fdf_struct").pointerTypes("gsl_multiroot_function_fdf"))
               .put(new Info("gsl_multiset_struct").pointerTypes("gsl_multiset"))
               .put(new Info("gsl_odeiv2_step_struct").pointerTypes("gsl_odeiv2_step"))
               .put(new Info("gsl_odeiv2_control_struct").pointerTypes("gsl_odeiv2_control"))
               .put(new Info("gsl_odeiv2_evolve_struct").pointerTypes("gsl_odeiv2_evolve"))
               .put(new Info("gsl_odeiv2_driver_struct").pointerTypes("gsl_odeiv2_driver"))
               .put(new Info("gsl_sf_result_struct").pointerTypes("gsl_sf_result"))
               .put(new Info("gsl_sf_result_e10_struct").pointerTypes("gsl_sf_result_e10"))
               .put(new Info("gsl_sf_legendre_Plm_array", "gsl_sf_legendre_Plm_deriv_array", "gsl_sf_legendre_sphPlm_array", "gsl_sf_legendre_sphPlm_deriv_array",
                             "gsl_sf_legendre_array_size", "gsl_bspline_deriv_alloc", "gsl_bspline_deriv_free", "gsl_multifit_fdfsolver_dif_fdf",
                             "gsl_matrix_char_norm1", "gsl_matrix_uchar_norm1", "gsl_matrix_ushort_norm1", "gsl_matrix_uint_norm1", "gsl_matrix_ulong_norm1",
                             "gsl_spmatrix_char_norm1", "gsl_spmatrix_uchar_norm1", "gsl_spmatrix_ushort_norm1", "gsl_spmatrix_ulong_norm1", "gsl_spmatrix_uint_norm1",
                             "gsl_sf_coupling_6j_INCORRECT", "gsl_sf_coupling_6j_INCORRECT_e", "gsl_spmatrix_cumsum").skip());
    }

    @Opaque public static class FILE extends Pointer {
        /** Empty constructor. Calls {@code super((Pointer)null)}. */
        public FILE() { super((Pointer)null); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public FILE(Pointer p) { super(p); }
    }

    public static native int fclose(FILE stream);
    public static native FILE fopen(String pathname, String mode);
    public static native int fread(Pointer ptr, long size, long nmemb, FILE stream);
    public static native int fwrite(@Const Pointer ptr, long size, long nmemb, FILE stream);
    public static native int fprintf(FILE stream, String format);
    public static native int fscanf(FILE stream, String format, Pointer p);
    public static native @MemberGetter FILE stdin();
    public static native @MemberGetter FILE stdout();
    public static native @MemberGetter FILE stderr();
}
