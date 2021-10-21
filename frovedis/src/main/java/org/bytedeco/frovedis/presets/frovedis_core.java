/*
 * Copyright (C) 2021 Samuel Audet
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

package org.bytedeco.frovedis.presets;

import java.io.File;
import java.io.IOException;
import org.bytedeco.javacpp.Loader;
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
@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
            value = "linux-x86_64",
            include = {
                "frovedis/core/frovedis_init.hpp",
            },
            link = {"open-pal@.40", "open-rte@.40", "mpi@.40", "mpi_mpifh@.40", "mpi_usempi@.40", "frovedis_core"},
            resource = {"include", "lib", "bin", "opt/openmpi"},
            includepath = {"/opt/nec/frovedis/x86/include/"},
            linkpath = {"/opt/nec/frovedis/x86/lib/", "/opt/nec/frovedis/x86/opt/openmpi/lib/"},
            resourcepath = {"/opt/nec/frovedis/x86/"}
        ),
        @Platform(
            value = "linux-x86_64",
            extension = "-ve",
            includepath = {"/opt/nec/frovedis/ve/include/", "/opt/nec/frovedis/x86/include/"},
            linkpath = {"/opt/nec/frovedis/ve/lib/", "/opt/nec/frovedis/x86/lib/", "/opt/nec/frovedis/x86/opt/openmpi/lib/"},
            resourcepath = {"/opt/nec/frovedis/ve/", "/opt/nec/frovedis/x86/"}
        )
    },
    target = "org.bytedeco.frovedis",
    global = "org.bytedeco.frovedis.global.frovedis_core"
)
public class frovedis_core implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "frovedis"); }

    private static File packageFile = null;

    /** Returns {@code Loader.cacheResource("/org/bytedeco/frovedis/" + Loader.getPlatform() + extension)}. */
    public static synchronized File cachePackage() throws IOException {
        if (packageFile != null) {
            return packageFile;
        }
        String path = Loader.load(frovedis_core.class);
        if (path != null) {
            path = path.replace(File.separatorChar, '/');
            int i = path.indexOf("/org/bytedeco/frovedis/" + Loader.getPlatform());
            int j = path.lastIndexOf("/");
            packageFile = Loader.cacheResource(path.substring(i, j));
        }
        return packageFile;
    }

    public void map(InfoMap infoMap) {
    }

    public static native String getenv(String name);
    public static int setenv(String name, String value) { return setenv(name, value, 1); }
    public static native int setenv(String name, String value, int overwrite);
    public static native int unsetenv(String name);
}
