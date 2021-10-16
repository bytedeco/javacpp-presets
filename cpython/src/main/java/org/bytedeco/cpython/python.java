/*
 * Copyright (C) 2019-2021 Samuel Audet
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

package org.bytedeco.cpython;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

/**
 * With this class, we can extract easily the {@code python} program ready for execution.
 * For example, we can print the module search path from Java in a portable fashion this way:
 * <pre>{@code
 *     String python = Loader.load(org.bytedeco.cpython.python.class);
 *     ProcessBuilder pb = new ProcessBuilder(python, "-m", "site");
 *     pb.inheritIO().start().waitFor();
 * }</pre>
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = org.bytedeco.cpython.presets.python.class,
    value = {
        @Platform(
            executable = "python3.10"
        ),
        @Platform(
            value = "macosx",
            executable = "python3.10"
        ),
        @Platform(
            value = "windows",
            executable = "python"
        ),
    }
)
public class python {
    static {
        try {
            org.bytedeco.cpython.presets.python.cachePackage();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
