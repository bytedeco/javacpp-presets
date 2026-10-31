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
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.bytedeco.hdf5.global.hdf5;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;

/**
 * Makes the bundled HDF5 filter plugins available to the HDF5 library.
 *
 * @author Samuel Audet
 */
public class HDF5Plugins {
    /** Registered filter IDs, see https://github.com/HDFGroup/hdf5_plugins/blob/master/docs/RegisteredFilterPlugins.md */
    public static final int
            H5Z_FILTER_BZIP2 = 307,
            H5Z_FILTER_LZF = 32000,
            H5Z_FILTER_BLOSC = 32001,
            H5Z_FILTER_LZ4 = 32004,
            H5Z_FILTER_BSHUF = 32008,
            H5Z_FILTER_ZFP = 32013,
            H5Z_FILTER_ZSTD = 32015,
            H5Z_FILTER_JPEG = 32019,
            H5Z_FILTER_BITGROOM = 32022,
            H5Z_FILTER_GRANULAR_BITROUND = 32023,
            H5Z_FILTER_BLOSC2 = 32026,
            H5Z_FILTER_BITROUND = 32032;

    private static final String[] NAMES = {"h5bz2", "h5blosc", "h5lz4", "h5bshuf", "h5zstd", "h5jpeg", "h5blosc2",
                                       "h5lzf", "h5zzfp", "h5bitgroom", "h5granular_bitround", "h5bitround"};

    private static File pluginPath = null;

    /**
     * Extracts the plugin library with the given name (without prefix and extension) from the JAR files if needed,
     * and returns it, or null if it is not available on this platform. Most plugins are modules (.so on macOS), but
     * some, like ZFP, are regular shared libraries (.dylib on macOS).
     */
    private static File cachePlugin(String name) throws IOException {
        String platform = Loader.getPlatform();
        String[] extensions = platform.startsWith("windows") ? new String[] {".dll"}
                            : platform.startsWith("macosx") ? new String[] {".so", ".dylib"} : new String[] {".so"};
        // CMake leaves out the "lib" prefix for some plugins on Windows, like ZFP
        for (String prefix : new String[] {"lib", ""}) {
            for (String extension : extensions) {
                File f = Loader.cacheResource("/org/bytedeco/hdf5/" + platform + "/lib/plugin/" + prefix + name + extension);
                if (f != null) {
                    return f;
                }
            }
        }
        return null;
    }

    /** Extracts the plugins from the JAR files if needed and returns the directory containing them. */
    public static synchronized File getPluginPath() throws IOException {
        if (pluginPath == null) {
            File dir = null;
            for (String name : NAMES) {
                // Loader cannot extract a whole directory, so go through the individual files,
                // which all end up side by side in the cache. Some plugins are not built on all platforms.
                File f = cachePlugin(name);
                if (f != null) {
                    dir = f.getParentFile();
                }
            }
            if (dir == null) {
                throw new IOException("Could not find the HDF5 plugins for " + Loader.getPlatform());
            }
            pluginPath = dir;
        }
        return pluginPath;
    }

    /** Plugin library names (without prefix and extension) by filter ID, for the ones that are built on this platform. */
    private static String pluginName(int filter) {
        switch (filter) {
            case H5Z_FILTER_BZIP2: return "h5bz2";
            case H5Z_FILTER_LZF: return "h5lzf";
            case H5Z_FILTER_BLOSC: return "h5blosc";
            case H5Z_FILTER_LZ4: return "h5lz4";
            case H5Z_FILTER_BSHUF: return "h5bshuf";
            case H5Z_FILTER_ZFP: return "h5zzfp";
            case H5Z_FILTER_ZSTD: return "h5zstd";
            case H5Z_FILTER_JPEG: return "h5jpeg";
            case H5Z_FILTER_BITGROOM: return "h5bitgroom";
            case H5Z_FILTER_GRANULAR_BITROUND: return "h5granular_bitround";
            case H5Z_FILTER_BLOSC2: return "h5blosc2";
            case H5Z_FILTER_BITROUND: return "h5bitround";
            default: throw new IllegalArgumentException("Unknown filter ID " + filter);
        }
    }

    /**
     * Returns the filter class given by H5PLget_plugin_info() of the plugin for the given filter ID,
     * as expected by H5Zregister(), or null if the plugin is not available on this platform.
     */
    public static synchronized Pointer getPluginInfo(int filter) throws IOException {
        File f = cachePlugin(pluginName(filter));
        if (f == null) {
            return null;
        }
        Loader.load(hdf5.class); // the plugin may link against the shared HDF5 library
        return hdf5.h5pl_get_plugin_info(f.getAbsolutePath());
    }

    /**
     * Returns the plugin type given by H5PLget_plugin_type() of the plugin for the given filter ID,
     * an {@code H5PL_type_t} such as {@code H5PL_TYPE_FILTER}, or -1 if the plugin is not available on this platform.
     */
    public static synchronized int getPluginType(int filter) throws IOException {
        File f = cachePlugin(pluginName(filter));
        if (f == null) {
            return -1;
        }
        Loader.load(hdf5.class); // the plugin may link against the shared HDF5 library
        return hdf5.h5pl_get_plugin_type(f.getAbsolutePath());
    }

    /**
     * Registers the filter with the given ID directly with HDF5 using H5Zregister(), without going through
     * the plugin search path, so it also works with {@code H5PLset_loading_state(0)}.
     *
     * @return false if the plugin is not available on this platform
     */
    public static synchronized boolean register(int filter) throws IOException {
        Pointer info = getPluginInfo(filter);
        if (info == null) {
            return false;
        }
        if (hdf5.H5Zregister(info) < 0) {
            throw new IOException("H5Zregister() failed for filter " + filter);
        }
        return true;
    }

    /** Appends the directory of the bundled plugins to the HDF5 plugin search path, if not already done. */
    public static synchronized void load() throws IOException {
        // Some plugins link against the shared HDF5 library, which therefore needs to be loaded first
        Loader.load(hdf5.class);
        String path = getPluginPath().getAbsolutePath();
        if (hdf5.H5PLappend(path) < 0) {
            throw new IOException("H5PLappend() failed for " + path);
        }
    }

    private static final List<Object> javaFilters = new ArrayList<Object>(); // keeps them from being deallocated

    /**
     * Registers a filter implemented in Java with HDF5 using H5Zregister(), without any plugin library. The callback
     * must follow the contract of {@code H5Z_func_t}: on input {@code buf} holds {@code nbytes} valid bytes of the
     * {@code *buf_size} allocated ones, and on output it returns the number of valid bytes, or 0 on failure, after
     * possibly replacing the buffer with one obtained from {@code H5allocate_memory()} or {@code H5resize_memory()},
     * and releasing the old one with {@code H5free_memory()}. {@code H5Z_FLAG_REVERSE} is set in {@code flags} when
     * decoding. The filter has to be registered the same way to read back the data it wrote, since it cannot be found
     * through the plugin search path, and stays registered, along with the callback, until H5Zunregister().
     *
     * @param id a filter ID, see https://github.com/HDFGroup/hdf5_plugins/blob/master/docs/RegisteredFilterPlugins.md
     * @param name a name for the filter
     * @param filter the callback doing the work, or encoding and decoding
     */
    public static synchronized void registerJavaFilter(int id, String name, H5Z_func_t filter) throws IOException {
        Loader.load(hdf5.class);
        H5Z_class2_t cls = new H5Z_class2_t();
        cls.version(hdf5.H5Z_CLASS_T_VERS)
           .id(id)
           .encoder_present(1)
           .decoder_present(1)
           .name(new BytePointer(name))
           .filter(filter);
        if (hdf5.H5Zregister(cls) < 0) {
            throw new IOException("H5Zregister() failed for filter " + id);
        }
        // HDF5 keeps pointing to the struct, the name, and the callback
        javaFilters.add(cls);
        javaFilters.add(filter);
    }
}
