/*
 * Copyright (C) 2015-2016 Samuel Audet
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

package org.bytedeco.javacpp.helper;

import java.nio.ByteBuffer;
import java.nio.Buffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Index;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.indexer.ByteIndexer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexable;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.bytedeco.javacpp.indexer.ShortIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UShortIndexer;

// required by javac to resolve circular dependencies
import org.bytedeco.javacpp.tensorflow.*;
import static org.bytedeco.javacpp.tensorflow.DT_BFLOAT16;
import static org.bytedeco.javacpp.tensorflow.DT_BOOL;
import static org.bytedeco.javacpp.tensorflow.DT_COMPLEX64;
import static org.bytedeco.javacpp.tensorflow.DT_DOUBLE;
import static org.bytedeco.javacpp.tensorflow.DT_FLOAT;
import static org.bytedeco.javacpp.tensorflow.DT_INT16;
import static org.bytedeco.javacpp.tensorflow.DT_INT32;
import static org.bytedeco.javacpp.tensorflow.DT_INT64;
import static org.bytedeco.javacpp.tensorflow.DT_INT8;
import static org.bytedeco.javacpp.tensorflow.DT_QINT32;
import static org.bytedeco.javacpp.tensorflow.DT_QINT8;
import static org.bytedeco.javacpp.tensorflow.DT_QUINT8;
import static org.bytedeco.javacpp.tensorflow.DT_STRING;
import static org.bytedeco.javacpp.tensorflow.DT_UINT8;
import static org.bytedeco.javacpp.tensorflow.NewSession;

/**
 *
 * @author Samuel Audet
 */
public class tensorflow extends org.bytedeco.javacpp.presets.tensorflow {

    @Name("std::string") public static class StringArray extends Pointer {
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

    public static abstract class AbstractTensor extends Pointer implements Indexable {
        static { Loader.load(); }
        public AbstractTensor(Pointer p) { super(p); }

        public static Tensor create(float[]  data, TensorShape shape) { Tensor t = new Tensor(DT_FLOAT,  shape); FloatBuffer  b = t.createBuffer(); b.put(data); return t; }
        public static Tensor create(double[] data, TensorShape shape) { Tensor t = new Tensor(DT_DOUBLE, shape); DoubleBuffer b = t.createBuffer(); b.put(data); return t; }
        public static Tensor create(int[]    data, TensorShape shape) { Tensor t = new Tensor(DT_INT32,  shape); IntBuffer    b = t.createBuffer(); b.put(data); return t; }
        public static Tensor create(short[]  data, TensorShape shape) { Tensor t = new Tensor(DT_INT16,  shape); ShortBuffer  b = t.createBuffer(); b.put(data); return t; }
        public static Tensor create(byte[]   data, TensorShape shape) { Tensor t = new Tensor(DT_INT8,   shape); ByteBuffer   b = t.createBuffer(); b.put(data); return t; }
        public static Tensor create(long[]   data, TensorShape shape) { Tensor t = new Tensor(DT_INT64,  shape); LongBuffer   b = t.createBuffer(); b.put(data); return t; }
        public static Tensor create(String[] data, TensorShape shape) {
            Tensor t = new Tensor(DT_STRING, shape);
            StringArray a = t.createStringArray();
            for (int i = 0; i < a.capacity(); i++) {
                a.position(i).put(data[i]);
            }
            return t;
        }

        public abstract int dtype();
        public abstract int dims();
        public abstract long dim_size(int d);
        public abstract long NumElements();
        public abstract long TotalBytes();
        public abstract BytePointer tensor_data();

        /** Returns {@code createBuffer(0)}. */
        public <B extends Buffer> B createBuffer() {
            return (B)createBuffer(0);
        }
        /** Returns {@link #tensor_data()} wrapped in a {@link Buffer} of appropriate type starting at given index. */
        public <B extends Buffer> B createBuffer(int index) {
            BytePointer ptr = tensor_data();
            int size = (int)TotalBytes();
            switch (dtype()) {
                case DT_COMPLEX64:
                case DT_FLOAT:    return (B)new FloatPointer(ptr).position(index).capacity(size/4).asBuffer();
                case DT_DOUBLE:   return (B)new DoublePointer(ptr).position(index).capacity(size/8).asBuffer();
                case DT_QINT32:
                case DT_INT32:    return (B)new IntPointer(ptr).position(index).capacity(size/4).asBuffer();
                case DT_BOOL:
                case DT_QUINT8:
                case DT_UINT8:
                case DT_QINT8:
                case DT_INT8:     return (B)ptr.position(index).capacity(size).asBuffer();
                case DT_BFLOAT16:
                case DT_INT16:    return (B)new ShortPointer(ptr).position(index).capacity(size/2).asBuffer();
                case DT_INT64:    return (B)new LongPointer(ptr).position(index).capacity(size/8).asBuffer();
                case DT_STRING:
                default: assert false;
            }
            return null;
        }

        /** Returns {@code createIndexer(true)}. */
        public <I extends Indexer> I createIndexer() {
            return (I)createIndexer(true);
        }
        @Override public <I extends Indexer> I createIndexer(boolean direct) {
            BytePointer ptr = tensor_data();
            int size = (int)TotalBytes();
            boolean complex = dtype() == DT_COMPLEX64;
            int dims = complex ? dims() + 1 : dims();
            long[] sizes = new long[dims];
            long[] strides = new long[dims];
            sizes[dims - 1] = complex ? 2 : (int)dim_size(dims - 1);
            strides[dims - 1] = 1;
            for (int i = dims - 2; i >= 0; i--) {
                sizes[i] = (int)dim_size(i);
                strides[i] = sizes[i + 1] * strides[i + 1];
            }
            switch (dtype()) {
                case DT_COMPLEX64:
                case DT_FLOAT:    return (I)FloatIndexer.create(new FloatPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
                case DT_DOUBLE:   return (I)DoubleIndexer.create(new DoublePointer(ptr).capacity(size/8), sizes, strides, direct).indexable(this);
                case DT_QINT32:
                case DT_INT32:    return (I)IntIndexer.create(new IntPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
                case DT_BOOL:
                case DT_QUINT8:
                case DT_UINT8:    return (I)UByteIndexer.create(ptr.capacity(size), sizes, strides, direct).indexable(this);
                case DT_QINT8:
                case DT_INT8:     return (I)ByteIndexer.create(ptr.capacity(size), sizes, strides, direct).indexable(this);
                case DT_BFLOAT16: return (I)UShortIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
                case DT_INT16:    return (I)ShortIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
                case DT_INT64:    return (I)LongIndexer.create(new LongPointer(ptr).capacity(size/8), sizes, strides, direct).indexable(this);
                case DT_STRING:
                default: assert false;
            }
            return null;
        }

        /** Returns {@code new StringArray(tensor_data()).capacity(NumElements()).limit(NumElements())} when {@code dtype() == DT_STRING}. */
        public StringArray createStringArray() {
            if (dtype() != DT_STRING) {
                return null;
            }
            long size = NumElements();
            return new StringArray(tensor_data()).capacity(size).limit(size);
        }
    }

    public static abstract class AbstractSession extends Pointer {
        static { Loader.load(); }

        SessionOptions options; // a reference to prevent deallocation

        public AbstractSession(Pointer p) { super(p); }
        /** Calls {@link org.bytedeco.javacpp.tensorflow#NewSession(SessionOptions)} and registers a deallocator. */
        public AbstractSession(SessionOptions options) {
            this.options = options;
            if (NewSession(options, (Session)this).ok() && !isNull()) {
                deallocator(new DeleteDeallocator((Session)this));
            }
        }

        @Namespace public static native void delete(Session session);

        protected static class DeleteDeallocator extends Session implements Pointer.Deallocator {
            DeleteDeallocator(Session p) { super(p); }
            @Override public void deallocate() { Session.delete(this); setNull(); }
        }
    }
}
