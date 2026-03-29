/*
 * Copyright (C) 2019-2024 Samuel Audet
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

package org.bytedeco.cpython.helper;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import static org.bytedeco.cpython.global.python.*;

public class python extends org.bytedeco.cpython.presets.python {

    /** Effectively returns {@code PyStatus_Exception(Py_InitializeFromConfig(config)) == 0}
     *  after setting the {@code home} and {@code pythonpath_env} values, for convenience. */
    public static boolean Py_Initialize(File... path) throws IOException {
        String[] strings = new String[path.length];
        for (int i = 0; i < path.length; i++) {
            strings[i] = path[i].getCanonicalPath();
        }
        return Py_Initialize(strings);
    }

    /** Effectively returns {@code PyStatus_Exception(Py_InitializeFromConfig(config)) == 0}
     *  after setting the {@code home} and {@code pythonpath_env} values, for convenience. */
    public static boolean Py_Initialize(String... path) throws IOException {
        PyConfig config = new PyConfig();
        PyConfig_InitPythonConfig(config);
        PointerPointer home = new PointerPointer(config.getPointer(BytePointer.class, config.offsetof("home")));
        Pointer p = Py_DecodeLocale(cachePackage().getCanonicalPath(), null);
        PyStatus status = PyConfig_SetString(config, home, p);
        PyMem_RawFree(p);
        if (PyStatus_Exception(status) != 0) {
            Py_ExitStatusException(status);
            return false;
        }

        PointerPointer pythonpath_env = new PointerPointer(config.getPointer(BytePointer.class, config.offsetof("pythonpath_env")));
        String separator = "";
        String string = "";
        for (String s : path) {
            string += separator + s;
            separator = File.pathSeparator;
        }
        p = Py_DecodeLocale(string, null);
        status = PyConfig_SetString(config, pythonpath_env, p);
        PyMem_RawFree(p);
        if (PyStatus_Exception(status) != 0) {
            Py_ExitStatusException(status);
            return false;
        }

        status = Py_InitializeFromConfig(config);
        if (PyStatus_Exception(status) != 0) {
            Py_ExitStatusException(status);
            return false;
        }
       return true;
    }
}
