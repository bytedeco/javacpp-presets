/*
 * Copyright (C) 2020 Samuel Audet
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

package org.bytedeco.llvm.program;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

/**
 * With this class, we can extract easily the {@code clang} programs ready for execution:
 * <pre>{@code
 *     String clang = Loader.load(org.bytedeco.llvm.program.clang.class);
 *     ProcessBuilder pb = new ProcessBuilder(clang, ...);
 *     pb.inheritIO().start().waitFor();
 * }</pre>
 *
 * @author Samuel Audet
 */
@Properties(
//    inherit = org.bytedeco.llvm.presets.clang.class,
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            preload = {"LLVM-11", "clang-cpp@.11"},
            executable = {
                "clang",
                "clang-check",
                "clang-extdef-mapping",
                "clang-format",
                "clang-import-test",
                "clang-offload-bundler",
                "clang-offload-wrapper",
                "clang-refactor",
                "clang-rename",
                "clang-scan-deps",
                "diagtool",
                "git-clang-format",
                "scan-build",
                "scan-view"
            }
        ),
        @Platform(
            value = {"macosx", "windows"},
            preload = {"LLVM", "clang-cpp"}
        )
    }
)
public class clang {
    static { Loader.load(); }
}
