/*
 * Copyright (C) 2014 Samuel Audet
 *
 * This file is part of JavaCPP.
 *
 * JavaCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCPP.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(target="org.bytedeco.javacpp.gsl", value={
    @Platform(include={"gsl/gsl_types.h", "gsl/gsl_errno.h", "gsl/gsl_ieee_utils.h", "gsl/gsl_inline.h", "gsl/gsl_message.h", "gsl/gsl_complex.h",
        "gsl/gsl_complex_math.h", "gsl/gsl_check_range.h", "gsl/gsl_sys.h", "gsl/gsl_machine.h", "gsl/gsl_precision.h", "gsl/gsl_nan.h", "gsl/gsl_pow_int.h",
        "gsl/gsl_min.h", "gsl/gsl_minmax.h", "gsl/gsl_math.h", "gsl/gsl_mode.h", "gsl/gsl_test.h", "gsl/gsl_version.h",

        "gsl/gsl_block.h", /*"gsl/gsl_block_complex_long_double.h"*/ "gsl/gsl_block_complex_double.h", "gsl/gsl_block_complex_float.h",
        /*"gsl/gsl_block_long_double.h", */ "gsl/gsl_block_double.h", "gsl/gsl_block_float.h", "gsl/gsl_block_ulong.h", "gsl/gsl_block_long.h",
        "gsl/gsl_block_uint.h", "gsl/gsl_block_int.h", "gsl/gsl_block_ushort.h", "gsl/gsl_block_short.h", "gsl/gsl_block_uchar.h", "gsl/gsl_block_char.h",

        "gsl/gsl_vector_complex.h", "gsl/gsl_vector.h", /*"gsl/gsl_vector_complex_long_double.h",*/ "gsl/gsl_vector_complex_double.h",
        "gsl/gsl_vector_complex_float.h", /*"gsl_vector_long_double.h",*/ "gsl/gsl_vector_double.h", "gsl/gsl_vector_float.h", "gsl/gsl_vector_ulong.h",
        "gsl/gsl_vector_long.h", "gsl/gsl_vector_uint.h", "gsl/gsl_vector_int.h", "gsl/gsl_vector_ushort.h", "gsl/gsl_vector_short.h",
        "gsl/gsl_vector_uchar.h", "gsl/gsl_vector_char.h",

        "gsl/gsl_matrix.h", /*"gsl/gsl_matrix_complex_long_double.h",*/ "gsl/gsl_matrix_complex_double.h", "gsl/gsl_matrix_complex_float.h",
        /*"gsl/gsl_matrix_long_double.h",*/ "gsl/gsl_matrix_double.h", "gsl/gsl_matrix_float.h", "gsl/gsl_matrix_ulong.h", "gsl/gsl_matrix_long.h",
        "gsl/gsl_matrix_uint.h", "gsl/gsl_matrix_int.h", "gsl/gsl_matrix_ushort.h", "gsl/gsl_matrix_short.h", "gsl/gsl_matrix_uchar.h", "gsl/gsl_matrix_char.h",

        "gsl/gsl_cblas.h", "gsl/gsl_blas_types.h", "gsl/gsl_blas.h", "gsl/gsl_bspline.h", "gsl/gsl_cdf.h", "gsl/gsl_chebyshev.h", "gsl/gsl_combination.h",
        "gsl/gsl_deriv.h", "gsl/gsl_dht.h", "gsl/gsl_diff.h", "gsl/gsl_eigen.h", "gsl/gsl_fit.h", "gsl/gsl_permutation.h", "gsl/gsl_heapsort.h",
        "gsl/gsl_histogram2d.h", "gsl/gsl_histogram.h", "gsl/gsl_integration.h", "gsl/gsl_interp.h", "gsl/gsl_linalg.h", "gsl/gsl_poly.h", "gsl/gsl_rng.h",
        "gsl/gsl_qrng.h", "gsl/gsl_randist.h", "gsl/gsl_roots.h",  "gsl/gsl_siman.h", "gsl/gsl_spline.h", "gsl/gsl_sum.h", "gsl/gsl_wavelet.h",
        "gsl/gsl_wavelet2d.h",

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
                               link={"gslcblas@.0", "gsl@.0"}),
    @Platform(value="windows", link={"libgslcblas-0", "libgsl-0"}) })
public class gsl implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("__cplusplus").define())
               .put(new Info("INFINITY", "defined(HUGE_VAL)", "NAN", "defined(INFINITY)").define(false))
               .put(new Info("GSL_VAR", "__BEGIN_DECLS", "__END_DECLS", "INLINE_DECL", "INLINE_FUN", "GSL_RANGE_CHECK", "CBLAS_INDEX").cppTypes().annotations())
               .put(new Info("gsl_complex_packed_array", "gsl_complex_packed_ptr").cast().valueTypes("DoublePointer", "DoubleBuffer", "double[]"))
               .put(new Info("gsl_complex_packed_array_float").cast().valueTypes("FloatPointer", "FloatBuffer", "float[]"))
               .put(new Info("gsl_complex").base("DoublePointer"))
               .put(new Info("gsl_complex_float").base("FloatPointer"))
               .put(new Info("GSL_COMPLEX_ONE", "GSL_COMPLEX_ZERO", "GSL_COMPLEX_NEGONE").cppTypes("gsl_complex"))
               .put(new Info("GSL_MACH_EPS").cppTypes("double").translate());
    }
}
