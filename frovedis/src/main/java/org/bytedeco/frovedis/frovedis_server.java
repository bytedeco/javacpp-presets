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

package org.bytedeco.frovedis;

import com.nec.frovedis.Jexrpc.FrovedisServer;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Comparator;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.frovedis.presets.frovedis_core.*;

/**
 * With this class, we can extract easily the {@code frovedis_server} program ready for execution.
 * For example, we can start the server on 4 MPI processes from Java in a portable fashion this way:
 * <pre>{@code
 *     frovedis_server.initialize("-np 4");
 * }</pre>
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = org.bytedeco.frovedis.presets.frovedis_client_spark.class,
    value = {
        @Platform(
            value = "linux-x86_64",
            executable = {"frovedis_server", "orterun"},
            executablepath = {"/opt/nec/frovedis/x86/bin/", "/opt/nec/frovedis/x86/opt/openmpi/bin/"}
        ),
        @Platform(
            value = "linux-x86_64",
            extension = "-ve",
            executablepath = {"/opt/nec/frovedis/ve/bin/"}
        ),
    }
)
public class frovedis_server {
    static {
        try {
            org.bytedeco.frovedis.presets.frovedis_core.cachePackage();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static class VersionComparator implements Comparator<String> {
        @Override public int compare(String version1, String version2) {
            String[] versions1 = version1.split("\\.");
            String[] versions2 = version2.split("\\.");
            int n = Math.max(versions1.length, versions2.length);
            for (int i = 0; i < n; i++) {
                int v1 = 0, v2 = 0;
                try {
                    v1 = Integer.parseInt(versions1[i]);
                } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                    // ignore, leave it at 0
                }
                try {
                    v2 = Integer.parseInt(versions2[i]);
                } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                    // ignore, leave it at 0
                }
                if (v1 < v2) {
                    return -1;
                } else if (v1 > v2) {
                    return 1;
                }
            }
            return 0;
        }
    }

    public static String joinPaths(String path1, String path2) {
        if (path1 == null) {
            path1 = "";
        }
        if (path1.length() > 0 && !path1.endsWith(File.pathSeparator)) {
            path1 += File.pathSeparator;
        }
        return path1 + path2;
    }

    /** Returns {@code initialize("-np 8")}. */
    public static FrovedisServer initialize() throws IOException {
        return initialize("-np 8");
    }

    /** Returns {@code FrovedisServer.initialize(mpirun + " " + mpiargs + " " + frovedis_server)},
     * after extracting everything for "mpirun" and "frovedis_server", setting environment variables as necessary. */
    public static FrovedisServer initialize(String mpiargs) throws IOException {
        String frovedis_server = Loader.load(frovedis_server.class);
        String frovedis_dir = frovedis_server.substring(0, frovedis_server.lastIndexOf(File.separatorChar));
        setenv("LD_LIBRARY_PATH", joinPaths(getenv("LD_LIBRARY_PATH"), frovedis_dir));
        System.setProperty("java.library.path", joinPaths(System.getProperty("java.library.path"), frovedis_dir));

        // Force reload of java.library.path
        try {
            Field f = ClassLoader.class.getDeclaredField("sys_paths");
            f.setAccessible(true);
            f.set(null, null);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }

        String mpirun = "mpirun";
        String orterun = frovedis_dir + "/orterun";
        if (new File(orterun).exists()) {
            mpirun = orterun;
            setenv("OPAL_PREFIX", frovedis_dir + "/opt/openmpi/");
        } else {
            File mpidir = new File("/opt/nec/ve/mpi/");
            if (mpidir.isDirectory()) {
                mpirun = "/opt/nec/ve/bin/mpirun";
                String[] mpiversions = mpidir.list();
                Arrays.sort(mpiversions, new VersionComparator());
                String mpiversion = mpiversions[mpiversions.length - 1];
                setenv("VE_LD_LIBRARY_PATH", "/opt/nec/ve/mpi/" + mpiversion + "/lib64/ve/");
            }
        }

        setenv("FROVEDIS_SEQUENTIAL_SAVE", "true"); // for NFS that doesn't support multi writer
        setenv("VE_LD_PRELOAD", "libveaccio.so.1");
        setenv("VE_OMP_NUM_THREADS", "1"); // needed for frovedis_server; otherwise it stucks!

        return FrovedisServer.initialize(mpirun + " " + mpiargs + " " + frovedis_server);
    }

    /** Calls {@code FrovedisServer.shut_down}. */
    public static void shut_down() throws IOException {
        FrovedisServer.shut_down();
    }
}
