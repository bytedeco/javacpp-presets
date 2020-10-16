/*
 * Copyright (C) 2014-2020 Samuel Audet
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

package org.bytedeco.fftw.presets;

import org.bytedeco.javacpp.Loader;
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
 */
@Properties(inherit = javacpp.class, global = "org.bytedeco.fftw.global.fftw3", value = {
    @Platform(include = "<fftw3.h>", link = {"fftw3@.3", "fftw3f@.3"}),
    @Platform(value = "android", link = {"fftw3", "fftw3f"}),
    @Platform(value = "windows", preload = {"libfftw3-3", "libfftw3f-3"}) })
@NoException
public class fftw3 implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "fftw"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("!defined(FFTW_NO_Complex) && defined(_Complex_I) && defined(complex) && defined(I)").define(false))
               .put(new Info("FFTW_EXTERN", "FFTW_CDECL").cppTypes().annotations())
               .put(new Info("fftw_plan_s").pointerTypes("fftw_plan")).put(new Info("fftw_plan").valueTypes("fftw_plan"))
               .put(new Info("fftwf_plan_s").pointerTypes("fftwf_plan")).put(new Info("fftwf_plan").valueTypes("fftwf_plan"))
               .put(new Info("fftwl_plan_s").pointerTypes("fftwl_plan")).put(new Info("fftwl_plan").valueTypes("fftwl_plan"))
               .put(new Info("fftwq_plan_s").pointerTypes("fftwq_plan")).put(new Info("fftwq_plan").valueTypes("fftwq_plan"))
               .put(new Info("fftw_iodim_do_not_use_me", "fftwf_iodim").pointerTypes("fftw_iodim"))
               .put(new Info("fftw_iodim64_do_not_use_me", "fftwf_iodim64").pointerTypes("fftw_iodim64"))
               .put(new Info("fftw_version").annotations("@Platform(not=\"windows\")").javaNames("fftw_version"))
               .put(new Info("fftw_cc").annotations("@Platform(not=\"windows\")").javaNames("fftw_cc"))
               .put(new Info("fftw_codelet_optim").annotations("@Platform(not=\"windows\")").javaNames("fftw_codelet_optim"))
               .put(new Info("fftwf_version").annotations("@Platform(not=\"windows\")").javaNames("fftwf_version"))
               .put(new Info("fftwf_cc").annotations("@Platform(not=\"windows\")").javaNames("fftwf_cc"))
               .put(new Info("fftwf_codelet_optim").annotations("@Platform(not=\"windows\")").javaNames("fftwf_codelet_optim"))
               .put(new Info("FFTW_DEFINE_API(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)",
                             "FFTW_DEFINE_API(FFTW_MANGLE_QUAD, __float128, fftwq_complex)").skip());
    }

    /** To be used only with fftw_export_wisdom_to_string(). */
    public native static void free(Pointer p);
}
