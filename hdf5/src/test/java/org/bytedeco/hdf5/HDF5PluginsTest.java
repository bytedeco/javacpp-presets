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

package org.bytedeco.hdf5;

import java.io.File;
import java.lang.Exception; // not org.bytedeco.hdf5.Exception
import org.bytedeco.javacpp.*;
import org.bytedeco.hdf5.global.hdf5;
import org.junit.Test;

import static org.bytedeco.hdf5.global.hdf5.*;
import static org.bytedeco.hdf5.HDF5Plugins.*;
import static org.junit.Assert.*;

/**
 * Round-trips a chunked dataset through each of the bundled filters.
 */
public class HDF5PluginsTest {
    static final int N = 4096;

    /** Plugins linked against HDF5 only load if the same shared library is already in the process. */
    @Test public void testHDF5LoadedFirst() throws Exception {
        Loader.load(hdf5.class);
        HDF5Plugins.load();
        File dir = HDF5Plugins.getPluginPath();
        assertTrue(dir.isDirectory());
        assertTrue(H5Zfilter_avail(H5Z_FILTER_BLOSC) > 0);
        assertTrue(H5Zfilter_avail(H5Z_FILTER_ZSTD) > 0);
    }

    @Test public void testPlugins() throws Exception {
        HDF5Plugins.load();
        boolean windows = Loader.getPlatform().startsWith("windows");
        int[] ids = windows ? new int[] {H5Z_FILTER_BZIP2, H5Z_FILTER_BLOSC, H5Z_FILTER_LZ4, H5Z_FILTER_BSHUF, H5Z_FILTER_ZSTD, H5Z_FILTER_BLOSC2}
                            : new int[] {H5Z_FILTER_BZIP2, H5Z_FILTER_BLOSC, H5Z_FILTER_LZ4, H5Z_FILTER_BSHUF, H5Z_FILTER_ZSTD, H5Z_FILTER_JPEG, H5Z_FILTER_BLOSC2};
        for (int id : ids) {
            roundTrip(id);
        }
    }

    /** Quantizing filters keep only a number of significant digits or bits, and ZFP is run in its lossless mode. */
    @Test public void testFloatPlugins() throws Exception {
        HDF5Plugins.load();
        floatRoundTrip(H5Z_FILTER_BITGROOM, new int[] {3}, 1e-2);
        floatRoundTrip(H5Z_FILTER_GRANULAR_BITROUND, new int[] {3}, 1e-2);
        floatRoundTrip(H5Z_FILTER_BITROUND, new int[] {10}, 1e-2);
        floatRoundTrip(H5Z_FILTER_ZFP, new int[] {5}, 0); // H5Z_ZFP_MODE_REVERSIBLE
    }

    /** Registers a plugin by itself with HDF5 with plugin loading from the search path turned off. */
    @Test public void testRegister() throws Exception {
        Loader.load(hdf5.class);
        assertTrue(H5PLset_loading_state(0) >= 0);
        // other tests in this JVM may have loaded these through the search path already
        H5Zunregister(H5Z_FILTER_ZSTD);
        H5Zunregister(H5Z_FILTER_LZ4);
        try {
            assertEquals(0, H5Zfilter_avail(H5Z_FILTER_ZSTD));
            assertTrue(HDF5Plugins.register(H5Z_FILTER_ZSTD));
            assertTrue(H5Zfilter_avail(H5Z_FILTER_ZSTD) > 0);
            assertEquals(0, H5Zfilter_avail(H5Z_FILTER_LZ4)); // others stay unregistered
            roundTrip(H5Z_FILTER_ZSTD);
        } finally {
            H5Zunregister(H5Z_FILTER_ZSTD);
            H5PLset_loading_state(H5PL_ALL_PLUGIN);
        }
    }

    /** All the bundled plugins are filters. */
    @Test public void testPluginType() throws Exception {
        Loader.load(hdf5.class);
        int[] ids = {H5Z_FILTER_BZIP2, H5Z_FILTER_LZF, H5Z_FILTER_BLOSC, H5Z_FILTER_LZ4, H5Z_FILTER_BSHUF, H5Z_FILTER_ZFP, H5Z_FILTER_ZSTD,
                     H5Z_FILTER_JPEG, H5Z_FILTER_BITGROOM, H5Z_FILTER_GRANULAR_BITROUND, H5Z_FILTER_BLOSC2, H5Z_FILTER_BITROUND};
        for (int id : ids) {
            if (id == H5Z_FILTER_JPEG && Loader.getPlatform().startsWith("windows")) {
                assertEquals(-1, HDF5Plugins.getPluginType(id));
            } else {
                assertEquals("filter " + id, H5PL_TYPE_FILTER, HDF5Plugins.getPluginType(id));
            }
        }
    }

    /** A filter written in Java, which XORs every byte with a constant, registered without any plugin library. */
    @Test public void testJavaFilter() throws Exception {
        final int id = 256; // first ID of the range for testing
        Loader.load(hdf5.class);
        final byte key = 0x5A;
        HDF5Plugins.registerJavaFilter(id, "xor", new H5Z_func_t() {
            @Override public long call(int flags, long cd_nelmts, IntPointer cd_values, long nbytes, SizeTPointer buf_size, PointerPointer buf) {
                // XOR is its own inverse, so encoding and decoding (H5Z_FLAG_REVERSE) are the same, in place
                BytePointer data = new BytePointer(buf.get(0)).capacity(nbytes);
                for (long i = 0; i < nbytes; i++) {
                    data.put(i, (byte)(data.get(i) ^ key));
                }
                return nbytes;
            }
        });
        try {
            assertTrue(H5Zfilter_avail(id) > 0);
            File file = File.createTempFile("hdf5_javafilter", ".h5");
            file.deleteOnExit();
            int[] ints = new int[N];
            for (int i = 0; i < N; i++) {
                ints[i] = i * 31;
            }
            long fid = H5Fcreate(file.getAbsolutePath(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            long space = H5Screate_simple(1, new long[] {N}, null);
            long dcpl = H5Pcreate(H5P_DATASET_CREATE());
            H5Pset_chunk(dcpl, 1, new long[] {N});
            assertTrue(H5Pset_filter(dcpl, id, H5Z_FLAG_MANDATORY, 0, new int[0]) >= 0);
            long dset = H5Dcreate2(fid, "data", H5T_NATIVE_INT(), space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            assertTrue(H5Dwrite(dset, H5T_NATIVE_INT(), H5S_ALL, H5S_ALL, H5P_DEFAULT, new IntPointer(ints)) >= 0);
            H5Dclose(dset);
            H5Pclose(dcpl);
            H5Sclose(space);
            H5Fclose(fid);

            fid = H5Fopen(file.getAbsolutePath(), H5F_ACC_RDONLY, H5P_DEFAULT);
            dset = H5Dopen2(fid, "data", H5P_DEFAULT);
            IntPointer out = new IntPointer(N);
            assertTrue(H5Dread(dset, H5T_NATIVE_INT(), H5S_ALL, H5S_ALL, H5P_DEFAULT, out) >= 0);
            for (int i = 0; i < N; i++) {
                assertEquals(ints[i], out.get(i));
            }
            H5Dclose(dset);
            H5Fclose(fid);
        } finally {
            H5Zunregister(id);
        }
    }

    @Test public void testLZF() throws Exception {
        HDF5Plugins.load();
        roundTrip(H5Z_FILTER_LZF);
    }

    static void floatRoundTrip(int filter, int[] cd, double relativeTolerance) throws Exception {
        File file = File.createTempFile("hdf5_plugins" + filter, ".h5");
        file.deleteOnExit();
        float[] values = new float[N];
        for (int i = 0; i < N; i++) {
            values[i] = 1 + i * 0.37f;
        }

        long fid = H5Fcreate(file.getAbsolutePath(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        long space = H5Screate_simple(1, new long[] {N}, null);
        long dcpl = H5Pcreate(H5P_DATASET_CREATE());
        H5Pset_chunk(dcpl, 1, new long[] {N});
        assertTrue("filter " + filter, H5Pset_filter(dcpl, filter, H5Z_FLAG_MANDATORY, cd.length, cd) >= 0);
        long dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT(), space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assertTrue("create " + filter, dset >= 0);
        assertTrue("write " + filter, H5Dwrite(dset, H5T_NATIVE_FLOAT(), H5S_ALL, H5S_ALL, H5P_DEFAULT, new FloatPointer(values)) >= 0);
        H5Dclose(dset);
        H5Pclose(dcpl);
        H5Sclose(space);
        H5Fclose(fid);

        fid = H5Fopen(file.getAbsolutePath(), H5F_ACC_RDONLY, H5P_DEFAULT);
        dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        FloatPointer out = new FloatPointer(N);
        assertTrue("read " + filter, H5Dread(dset, H5T_NATIVE_FLOAT(), H5S_ALL, H5S_ALL, H5P_DEFAULT, out) >= 0);
        boolean changed = false;
        for (int i = 0; i < N; i++) {
            assertEquals("filter " + filter + " at " + i, values[i], out.get(i), Math.abs(values[i]) * relativeTolerance);
            changed |= values[i] != out.get(i);
        }
        // a lossy filter that was actually applied has to alter some values
        assertEquals("filter " + filter, relativeTolerance > 0, changed);
        H5Dclose(dset);
        H5Fclose(fid);
    }

    static void roundTrip(int filter) throws Exception {
        File file = File.createTempFile("hdf5_plugins" + filter, ".h5");
        file.deleteOnExit();
        // JPEG operates on 8-bit images, the others on any element type
        boolean jpeg = filter == H5Z_FILTER_JPEG;
        long type = jpeg ? H5T_NATIVE_UCHAR() : H5T_NATIVE_INT();
        byte[] bytes = new byte[N];
        int[] ints = new int[N];
        for (int i = 0; i < N; i++) {
            ints[i] = i / 8;
            bytes[i] = (byte)(i / 64);
        }

        long fid = H5Fcreate(file.getAbsolutePath(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        long space = H5Screate_simple(jpeg ? 2 : 1, jpeg ? new long[] {64, 64} : new long[] {N}, null);
        long dcpl = H5Pcreate(H5P_DATASET_CREATE());
        H5Pset_chunk(dcpl, jpeg ? 2 : 1, jpeg ? new long[] {64, 64} : new long[] {N});
        // Per HDF5 filter registry: JPEG takes (quality, columns, rows, color mode), Blosc2 takes
        // (4 reserved values, compression level, shuffle, compressor), and the others accept defaults
        int[] cd = jpeg ? new int[] {100, 64, 64, 0}
                 : filter == H5Z_FILTER_BLOSC2 ? new int[] {0, 0, 0, 0, 5, 1, 0}
                 : new int[0];
        assertTrue("filter " + filter, H5Pset_filter(dcpl, filter, H5Z_FLAG_MANDATORY, cd.length, cd) >= 0);
        long dset = H5Dcreate2(fid, "data", type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        assertTrue(dset >= 0);
        Pointer data = jpeg ? new BytePointer(bytes) : new IntPointer(ints);
        assertTrue("write " + filter, H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0);
        H5Dclose(dset);
        H5Pclose(dcpl);
        H5Sclose(space);
        H5Fclose(fid);

        fid = H5Fopen(file.getAbsolutePath(), H5F_ACC_RDONLY, H5P_DEFAULT);
        dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        Pointer out = jpeg ? new BytePointer(N) : new IntPointer(N);
        assertTrue("read " + filter, H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) >= 0);
        if (jpeg) {
            // JPEG is lossy
            BytePointer b = (BytePointer)out;
            for (int i = 0; i < N; i++) {
                assertEquals(bytes[i], b.get(i), 8);
            }
        } else {
            IntPointer p = (IntPointer)out;
            for (int i = 0; i < N; i++) {
                assertEquals("filter " + filter + " at " + i, ints[i], p.get(i));
            }
        }
        H5Dclose(dset);
        H5Fclose(fid);
    }
}
