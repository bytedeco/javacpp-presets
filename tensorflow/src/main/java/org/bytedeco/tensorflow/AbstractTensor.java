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

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.tensorflow.global.tensorflow.*;

@Properties(inherit = org.bytedeco.tensorflow.presets.tensorflow.class)
public abstract class AbstractTensor extends Pointer implements Indexable {
    static { Loader.load(); }
    public AbstractTensor(Pointer p) { super(p); }

    public static Tensor create(boolean[] data, long... shape) { return create(data, new TensorShape(shape)); }
    public static Tensor create(float[]  data, long... shape) { return create(data, new TensorShape(shape)); }
    public static Tensor create(double[] data, long... shape) { return create(data, new TensorShape(shape)); }
    public static Tensor create(int[]    data, long... shape) { return create(data, new TensorShape(shape)); }
    public static Tensor create(short[]  data, long... shape) { return create(data, new TensorShape(shape)); }
    public static Tensor create(byte[]   data, long... shape) { return create(data, new TensorShape(shape)); }
    public static Tensor create(long[]   data, long... shape) { return create(data, new TensorShape(shape)); }
    public static Tensor create(String[] data, long... shape) { return create(data, new TensorShape(shape)); }

    public static Tensor create(boolean[] data, TensorShape shape) {
        Tensor t = new Tensor(DT_BOOL, shape);
        ByteBuffer b = t.createBuffer();
        for (boolean bool : data) {
            b.put(bool ? (byte)1 : (byte)0);
        }
        return t;
    }
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
    public <B extends Buffer> B createBuffer(long index) {
        BytePointer ptr = tensor_data();
        long size = TotalBytes();
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
        int dims = dims();
        long size = TotalBytes();
        boolean complex = dtype() == DT_COMPLEX64;
        boolean scalar = dims == 0;
        dims = (complex ? 1 : 0) + (scalar ? 1 : dims);
        long[] sizes = new long[dims];
        long[] strides = new long[dims];
        sizes[dims - 1] = complex ? 2 : (scalar ? 1 : dim_size(dims - 1));
        strides[dims - 1] = 1;
        for (int i = dims - 2; i >= 0; i--) {
            sizes[i] = scalar ? 1 : dim_size(i);
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
