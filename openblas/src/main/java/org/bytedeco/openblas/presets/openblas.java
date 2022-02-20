/*
 * Copyright (C) 2016-2021 Samuel Audet
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

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = openblas_nolapack.class, global = "org.bytedeco.openblas.global.openblas", value = {
    @Platform(
        include = {"openblas_config.h", "cblas.h"}),
    @Platform(
        value = {"linux", "macosx-x86_64", "windows"},
        include = {"openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h", "lapack.h", "lapacke.h", "lapacke_utils.h"})})
@NoException
public class openblas extends openblas_nolapack {

    @Override public void init(ClassProperties properties) {
        super.init(properties);

        List<String> links = properties.get("platform.link");
        List<String> preloads = properties.get("platform.preload");

        // Replace all occurences of "openblas_nolapack" with "openblas" (with LAPACK)
        for (List<String> l : new List[] {links, preloads}) {
            ListIterator<String> it = l.listIterator();
            while (it.hasNext()) {
                String s = it.next();
                if (s.contains("openblas_nolapack")) {
                    it.remove();
                    s = s.replace("openblas_nolapack", "openblas");
                    if (!l.contains(s)) {
                        it.add(s);
                    }
                }
            }
        }
    }
}
