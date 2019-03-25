/*
 * Copyright (C) 2015-2019 Samuel Audet
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

package org.bytedeco.tensorflow;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import org.bytedeco.tensorflow.*;
import static org.bytedeco.tensorflow.global.tensorflow.*;

@Properties(inherit = org.bytedeco.tensorflow.presets.tensorflow.class)
@Name("std::string") public class StringArray extends Pointer {
    static { Loader.load(); }
    public StringArray(Pointer p) { super(p); }
    public StringArray() { allocate(); }
    private native void allocate();
    public StringArray(StringArray p) { allocate(p); }
    private native void allocate(@ByRef StringArray p);
    public StringArray(BytePointer s, long count) { allocate(s, count); }
    private native void allocate(@Cast("char*") BytePointer s, long count);
    public StringArray(String s) { allocate(s); }
    private native void allocate(String s);
    public native @Name("operator=") @ByRef StringArray put(@ByRef StringArray str);
    public native @Name("operator=") @ByRef StringArray put(String str);
    public native @Name("operator=") @ByRef StringArray put(@Cast("const char*") BytePointer str);
    @Override public StringArray position(long position) {
        return (StringArray)super.position(position);
    }

    public native @Cast("size_t") long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @Cast("char") int get(@Cast("size_t") long pos);
    public native StringArray put(@Cast("size_t") long pos, int c);
    public native @Cast("const char*") BytePointer data();

    @Override public String toString() {
        long length = size();
        byte[] bytes = new byte[length < Integer.MAX_VALUE ? (int)length : Integer.MAX_VALUE];
        data().get(bytes);
        return new String(bytes);
    }
}
