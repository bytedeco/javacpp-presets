/*
 * Copyright (C) 2026 Samuel Audet
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

package org.bytedeco.hdf5_plugins.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;

/**
 * Packages the HDF5 filter plugins (zstd, lz4, jpeg, bzip2, bshuf, blosc, blosc2) as
 * resources. They are loaded at runtime by HDF5 itself, see {@link org.bytedeco.hdf5_plugins.HDF5Plugins}.
 *
 * @author Samuel Audet
 */
@Properties(inherit = javacpp.class, value = {
    @Platform(value = {"linux", "macosx", "windows", "android"}, resource = {"lib/plugin"}) })
public class hdf5_plugins {
    static { Loader.checkVersion("org.bytedeco", "hdf5_plugins"); }
}
