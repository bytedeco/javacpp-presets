/*
 * Copyright (C) 2023 Mark Kittisopikul
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

package org.bytedeco.hdf5;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

import org.bytedeco.hdf5.presets.*;

/**
 * This is only a placeholder to facilitate loading the {@code hdf5_java} module with JavaCPP.
 * <p>
 * Call {@code Loader.load(hdf5_java.class)} before using the API in the {@code hdf.hdf5group} namespace.
 *
 * @author Mark Kittisopikul
 */
@Properties(
    inherit = {
        hdf5.class,
    },
    value = {
        @Platform(preload = {"hdf5_java"}),
    }
)
public class hdf5_java {
    static { Loader.load(); }
}
