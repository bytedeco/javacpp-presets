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

package org.bytedeco.hdf5_plugins;

import java.io.File;
import java.io.IOException;
import org.bytedeco.hdf5.global.hdf5;
import org.bytedeco.hdf5_plugins.global.hdf5_plugins;
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

    /** Extracts the plugins from the JAR files if needed and returns the directory containing them. */
    public static synchronized File getPluginPath() throws IOException {
        if (pluginPath == null) {
            String platform = Loader.getPlatform();
            boolean windows = platform.startsWith("windows");
            File dir = null;
            for (String name : NAMES) {
                // Loader cannot extract a whole directory, so go through the individual files,
                // which all end up side by side in the cache. Some plugins are not built on all platforms.
                String file = (windows ? "" : "lib") + name + (windows ? ".dll" : ".so");
                File f = Loader.cacheResource("/org/bytedeco/hdf5_plugins/" + platform + "/lib/plugin/" + file);
                if (f != null) {
                    dir = f.getParentFile();
                }
            }
            if (dir == null) {
                throw new IOException("Could not find the HDF5 plugins for " + platform);
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
        String platform = Loader.getPlatform();
        boolean windows = platform.startsWith("windows");
        String file = (windows ? "" : "lib") + pluginName(filter) + (windows ? ".dll" : ".so");
        File f = Loader.cacheResource("/org/bytedeco/hdf5_plugins/" + platform + "/lib/plugin/" + file);
        if (f == null) {
            return null;
        }
        Loader.load(hdf5.class); // the plugin may link against the shared HDF5 library
        return hdf5_plugins.h5pl_get_plugin_info(f.getAbsolutePath());
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
}
