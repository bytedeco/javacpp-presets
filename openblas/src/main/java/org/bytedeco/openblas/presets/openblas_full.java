/*
 * Copyright (C) 2025 Samuel Audet, Dragan Djuric
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

import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.InfoMap;

/**
 *
 * @author Samuel Audet
 * @author Dragan Djuric
 */
@Properties(inherit = openblas.class, global = "org.bytedeco.openblas.global.openblas_full", value = {
    @Platform(
        include = {"openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h", "lapack.h", "lapacke.h", "lapacke_utils.h"})})
@NoException
public class openblas_full extends openblas {

    @Override public void map(InfoMap infoMap) {
        openblas_nolapack.mapCommon(infoMap);

        String[] functions = {
            // not implemented by MKL
            "cgesvdq", "dgesvdq", "sgesvdq", "zgesvdq", "clangb", "dlangb", "slangb", "zlangb",
            "ctrsyl3", "dtrsyl3", "strsyl3", "ztrsyl3",
            // deprecated
            "sgedmd", "dgedmd", "cgedmd", "zgedmd", "sgedmdq", "dgedmdq", "cgedmdq", "zgedmdq",
            "cggsvd", "dggsvd", "sggsvd", "zggsvd", "zggsvp", "cggsvp", "dggsvp", "sggsvp",
            // extended
            "cgbrfsx", "cporfsx", "dgerfsx", "sgbrfsx", "ssyrfsx", "zherfsx", "cgerfsx", "csyrfsx", "dporfsx", "sgerfsx", "zgbrfsx", "zporfsx",
            "cherfsx", "dgbrfsx", "dsyrfsx", "sporfsx", "zgerfsx", "zsyrfsx", "cgbsvxx", "cposvxx", "dgesvxx", "sgbsvxx", "ssysvxx", "zhesvxx",
            "cgesvxx", "csysvxx", "dposvxx", "sgesvxx", "zgbsvxx", "zposvxx", "chesvxx", "dgbsvxx", "dsysvxx", "sposvxx", "zgesvxx", "zsysvxx"};

        for (String f : functions) {
            infoMap.put(new Info(f, "LAPACK_" + f, "LAPACK_" + f + "_base")).skip();
        }

    }

}
