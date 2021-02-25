/*
 * Copyright (C) 2020-2021 Samuel Audet
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
 * With this class, we can extract easily the {@code llvm} programs ready for execution:
 * <pre>{@code
 *     String llvm = Loader.load(org.bytedeco.llvm.program.llvm.class);
 *     ProcessBuilder pb = new ProcessBuilder(llvm, ...);
 *     pb.inheritIO().start().waitFor();
 * }</pre>
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = org.bytedeco.llvm.presets.LLVM.class,
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            executable = {
                "dsymutil",
                "hmaptool",
                "llc",
                "lli",
                "llvm-ar",
                "llvm-as",
                "llvm-bcanalyzer",
                "llvm-cat",
                "llvm-cfi-verify",
                "llvm-config",
                "llvm-cov",
                "llvm-c-test",
                "llvm-cvtres",
                "llvm-cxxdump",
                "llvm-cxxfilt",
                "llvm-cxxmap",
                "llvm-diff",
                "llvm-dis",
                "llvm-dwarfdump",
                "llvm-dwp",
                "llvm-elfabi",
                "llvm-exegesis",
                "llvm-extract",
                "llvm-ifs",
                "llvm-jitlink",
                "llvm-link",
                "llvm-lipo",
                "llvm-lto",
                "llvm-lto2",
                "llvm-mc",
                "llvm-mca",
                "llvm-modextract",
                "llvm-mt",
                "llvm-nm",
                "llvm-objcopy",
                "llvm-objdump",
                "llvm-opt-report",
                "llvm-pdbutil",
                "llvm-profdata",
                "llvm-rc",
                "llvm-readobj",
                "llvm-reduce",
                "llvm-rtdyld",
                "llvm-size",
                "llvm-split",
                "llvm-stress",
                "llvm-strings",
                "llvm-symbolizer",
                "llvm-undname",
                "llvm-xray",
                "obj2yaml",
                "sancov",
                "sanstats",
                "verify-uselistorder",
                "yaml2obj"
            }
        ),
    }
)
public class llvm {
    static {
        try {
            org.bytedeco.llvm.presets.LLVM.cachePackage();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
