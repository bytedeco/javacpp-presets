/*
 * Copyright (C) 2026 Mark Kittisopikul
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

import java.io.File;
import hdf.hdf5lib.H5;
import hdf.hdf5lib.HDF5Constants;
import org.bytedeco.javacpp.Loader;
import org.junit.Test;

import static org.bytedeco.hdf5.global.hdf5.H5F_ACC_TRUNC;
import static org.junit.Assert.assertTrue;

/**
 * Confirms both HDF5 Java binding sets shipped in this artifact are usable, on every
 * platform this module is built for: the official HDF Group API ({@code hdf.hdf5lib.H5},
 * bundled via {@link hdf5_java}) and the JavaCPP-generated C++ bindings
 * ({@code org.bytedeco.hdf5.*}). See issue #1405.
 *
 * @author Mark Kittisopikul
 */
public class HDF5BindingsTest {

    @Test
    public void officialHdfGroupBindingsAreUsable() throws java.lang.Exception {
        Loader.load(hdf5_java.class);

        int[] version = new int[3];
        H5.H5get_libversion(version);
        assertTrue("H5get_libversion returned a bogus version", version[0] > 0);

        File file = File.createTempFile("hdf5-official-", ".h5");
        assertTrue(file.delete());
        long fileId = H5.H5Fcreate(file.getAbsolutePath(), HDF5Constants.H5F_ACC_TRUNC,
                HDF5Constants.H5P_DEFAULT, HDF5Constants.H5P_DEFAULT);
        assertTrue("H5Fcreate via the official hdf.hdf5lib.H5 API failed", fileId >= 0);
        H5.H5Fclose(fileId);
        assertTrue("H5Fcreate did not produce a file on disk", file.exists());
        file.delete();
    }

    @Test
    public void javaCppBindingsAreUsable() throws java.lang.Exception {
        File file = File.createTempFile("hdf5-javacpp-", ".h5");
        assertTrue(file.delete());
        H5File h5file = new H5File(file.getAbsolutePath(), H5F_ACC_TRUNC);
        h5file.close();
        assertTrue("H5File did not produce a file on disk", file.exists());
        file.delete();
    }

    @Test
    public void bothBindingSetsWorkTogetherInTheSameProcess() throws java.lang.Exception {
        officialHdfGroupBindingsAreUsable();
        javaCppBindingsAreUsable();
        officialHdfGroupBindingsAreUsable();
    }
}
