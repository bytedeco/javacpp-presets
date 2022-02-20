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

package org.bytedeco.tvm.presets;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.dnnl.presets.*;
import org.bytedeco.llvm.presets.*;
import org.bytedeco.mkl.presets.*;
import org.bytedeco.scipy.presets.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = {dnnl.class, LLVM.class, mkl_rt.class, scipy.class},
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            exclude = {"<polly/LinkAllPasses.h>", "<FullOptimization.h>", "<NamedMetadataOperations.h>",
                       "openblas_config.h", "cblas.h", "lapacke_config.h",
                       "lapacke_mangling.h", "lapack.h", "lapacke.h", "lapacke_utils.h"},
            resource = "python"
        ),
        @Platform(
            value = {"linux", "macosx", "windows"},
            extension = "-gpu"
        ),
    }
)
@NoException
public class tvm {
    static { Loader.checkVersion("org.bytedeco", "tvm"); }

    private static File packageFile = null;

    /** Returns {@code Loader.cacheResource("/org/bytedeco/tvm/" + Loader.getPlatform() + extension + "/python/")}. */
    public static synchronized File cachePackage() throws IOException {
        if (packageFile != null) {
            return packageFile;
        }
        Loader.load(org.bytedeco.cpython.global.python.class);
        String path = Loader.load(tvm.class);
        if (path != null) {
            path = path.replace(File.separatorChar, '/');
            int i = path.indexOf("/org/bytedeco/tvm/" + Loader.getPlatform());
            int j = path.lastIndexOf("/");
            File f = Loader.cacheResource(path.substring(i, j) + "/python/");
            Loader.load(tvm_runtime.class);
            packageFile = f;
        }
        return packageFile;
    }

    /** Returns {@code {scipy.cachePackages(), tvm.cachePackage()}}. */
    public static File[] cachePackages() throws IOException {
        File[] path = org.bytedeco.scipy.presets.scipy.cachePackages();
        path = Arrays.copyOf(path, path.length + 1);
        path[path.length - 1] = cachePackage();
        return path;
    }
}
