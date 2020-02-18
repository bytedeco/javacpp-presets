/*
 * Copyright (C) 2019-2020 Samuel Audet
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

    public static String Py_GetPathString() {
        Py_FrozenFlag(1); // prevent Python from printing useless warnings
        BytePointer p = Py_EncodeLocale(org.bytedeco.cpython.global.python.Py_GetPath(), null);
        String string = p.getString();
        PyMem_Free(p);
        return string;
    }

    public static void Py_SetPath(File... path) throws IOException {
        String[] strings = new String[path.length];
        for (int i = 0; i < path.length; i++) {
            strings[i] = path[i].getCanonicalPath();
        }
        Py_SetPath(strings);
    }

    public static void Py_SetPath(String... path) throws IOException {
        Py_FrozenFlag(1); // prevent Python from printing useless warnings
        String separator = "";
        String string = "";
        for (String s : path) {
            string += separator + s;
            separator = File.pathSeparator;
        }
        Pointer p = Py_DecodeLocale(string, null);
        org.bytedeco.cpython.global.python.Py_SetPath(p);
        PyMem_RawFree(p);
    }

    /** Effectively calls {@code Py_SetPath(path, Py_GetPath())}, for convenience. */
    public static void Py_AddPath(File... path) throws IOException {
        String[] strings = new String[path.length];
        for (int i = 0; i < path.length; i++) {
            strings[i] = path[i].getCanonicalPath();
        }
        Py_AddPath(strings);
    }

    /** Effectively calls {@code Py_SetPath(path, Py_GetPath())}, for convenience. */
    public static void Py_AddPath(String... path) throws IOException {
        path = Arrays.copyOf(path, path.length + 1);
        path[path.length - 1] = Py_GetPathString();
        Py_SetPath(path);
    }
}
