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
import org.bytedeco.hdf5.global.hdf5;
import org.bytedeco.javacpp.Loader;
import org.junit.Test;

import static org.bytedeco.hdf5.global.hdf5.H5F_ACC_TRUNC;
import static org.junit.Assert.assertEquals;
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

    /**
     * Loading hdf5_java points hdf.hdf5lib.H5.loadH5Lib() at the library JavaCPP loaded, so that
     * it doesn't fail (and print an UnsatisfiedLinkError) looking for it on java.library.path.
     */
    @Test
    public void loadingSetsTheHdf5libPathProperty() throws java.lang.Exception {
        Loader.load(hdf5_java.class);
        String path = System.getProperty("hdf.hdf5lib.H5.hdf5lib");
        assertTrue("hdf.hdf5lib.H5.hdf5lib is not set", path != null && path.length() > 0);
        assertTrue("hdf.hdf5lib.H5.hdf5lib does not name a file: " + path, new File(path).isFile());
    }

    /**
     * Both binding sets must share a single HDF5 library, so that an ID created through one is
     * valid in the other. On Windows, jnihdf5.dll used to link HDF5 statically while
     * hdf5_java.dll used hdf5.dll, giving two separate libraries. See issue #1813.
     */
    @Test
    public void idsAreSharedBetweenBindingSets() throws java.lang.Exception {
        Loader.load(hdf5_java.class);

        long officialPlist = H5.H5Pcreate(HDF5Constants.H5P_DATASET_XFER);
        try {
            assertTrue("An ID created by hdf.hdf5lib.H5 is not valid in org.bytedeco.hdf5",
                    hdf5.H5Iis_valid(officialPlist) > 0);
        } finally {
            H5.H5Pclose(officialPlist);
        }

        long javaCppPlist = hdf5.H5Pcreate(hdf5.H5P_FILE_ACCESS);
        assertTrue("H5Pcreate via org.bytedeco.hdf5 failed", javaCppPlist >= 0);
        try {
            assertTrue("An ID created by org.bytedeco.hdf5 is not valid in hdf.hdf5lib.H5",
                    H5.H5Iis_valid(javaCppPlist));
        } finally {
            hdf5.H5Pclose(javaCppPlist);
        }
    }

    /**
     * Constants defined as "(H5OPEN X_g)" macros are 64-bit hid_t values and used to be bound as
     * int, truncating them to garbage. See issue #1812.
     */
    @Test
    public void hidMacroConstantsMatchOfficialBindings() throws java.lang.Exception {
        Loader.load(hdf5_java.class);

        assertEquals(HDF5Constants.H5P_DATASET_XFER, hdf5.H5P_DATASET_XFER);
        assertEquals(HDF5Constants.H5P_FILE_ACCESS, hdf5.H5P_FILE_ACCESS);
        assertEquals(HDF5Constants.H5T_NATIVE_INT, hdf5.H5T_NATIVE_INT);
        assertEquals(HDF5Constants.H5T_NATIVE_CHAR, hdf5.H5T_NATIVE_CHAR);
        assertEquals(HDF5Constants.H5T_STD_I32LE, hdf5.H5T_INTEL_I32); // alias of H5T_STD_I32LE
        assertEquals(HDF5Constants.H5FD_SEC2, hdf5.H5FD_SEC2);

        long plist = hdf5.H5Pcreate(hdf5.H5P_DATASET_XFER);
        assertTrue("H5Pcreate(H5P_DATASET_XFER) via org.bytedeco.hdf5 failed", plist >= 0);
        hdf5.H5Pclose(plist);
        assertEquals(4, hdf5.H5Tget_size(hdf5.H5T_NATIVE_INT));
    }
}
