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
@Properties(target="org.bytedeco.javacpp.fftw3", value={
    @Platform(include="<fftw3.h>", link={"fftw3@.3", "fftw3f@.3"}),
    @Platform(value="windows", link={"libfftw3-3", "libfftw3f-3"}) })
public class fftw3 implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("!defined(FFTW_NO_Complex) && defined(_Complex_I) && defined(complex) && defined(I)").define(false))
               .put(new Info("FFTW_EXTERN").cppTypes().annotations())
               .put(new Info("fftw_plan_s").pointerTypes("fftw_plan")).put(new Info("fftw_plan").valueTypes("fftw_plan"))
               .put(new Info("fftwf_plan_s").pointerTypes("fftwf_plan")).put(new Info("fftwf_plan").valueTypes("fftwf_plan"))
               .put(new Info("fftwl_plan_s").pointerTypes("fftwl_plan")).put(new Info("fftwl_plan").valueTypes("fftwl_plan"))
               .put(new Info("fftwq_plan_s").pointerTypes("fftwq_plan")).put(new Info("fftwq_plan").valueTypes("fftwq_plan"))
               .put(new Info("fftw_version").annotations("@Platform(not=\"windows\")").javaNames("fftw_version"))
               .put(new Info("fftw_cc").annotations("@Platform(not=\"windows\")").javaNames("fftw_fftw_cc"))
               .put(new Info("fftw_codelet_optim").annotations("@Platform(not=\"windows\")").javaNames("fftw_codelet_optim"))
               .put(new Info("fftwf_version").annotations("@Platform(not=\"windows\")").javaNames("fftwf_version"))
               .put(new Info("fftwf_cc").annotations("@Platform(not=\"windows\")").javaNames("fftwf_cc"))
               .put(new Info("fftwf_codelet_optim").annotations("@Platform(not=\"windows\")").javaNames("fftwf_codelet_optim"))
               .put(new Info("FFTW_DEFINE_API(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)",
                             "FFTW_DEFINE_API(FFTW_MANGLE_QUAD, __float128, fftwq_complex)").skip());
    }
}
