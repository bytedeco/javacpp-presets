/*
 * Copyright (C) 2022 Samuel Audet
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

package org.bytedeco.pytorch;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.pytorch.global.torch.*;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public abstract class AbstractTensor extends Pointer implements Indexable {
    static { Loader.load(); }
    public AbstractTensor(Pointer p) { super(p); }

    public static Tensor create(byte[] data, boolean signed) { return create(data, signed, data.length); }
    public static Tensor create(byte...    data) { return create(data, false, data.length); }
    public static Tensor create(short...   data) { return create(data, data.length); }
    public static Tensor create(int...     data) { return create(data, data.length); }
    public static Tensor create(long...    data) { return create(data, data.length); }
    public static Tensor create(float...   data) { return create(data, data.length); }
    public static Tensor create(double...  data) { return create(data, data.length); }
    public static Tensor create(boolean... data) { return create(data, data.length); }

    public static Tensor create(byte[] data, boolean signed, long... shape) {
        Tensor t = empty(shape, new TensorOptions(signed ? ScalarType.Char : ScalarType.Byte), null);
        ByteBuffer b = t.createBuffer();
        b.put(data);
        return t;
    }
    public static Tensor create(byte[]    data, long... shape) { return create(data, false, shape); }
    public static Tensor create(short[]   data, long... shape) { Tensor t = empty(shape, new TensorOptions(ScalarType.Short), null);   ShortIndexer i = t.createIndexer(); i.put(0, data); return t; }
    public static Tensor create(int[]     data, long... shape) { Tensor t = empty(shape, new TensorOptions(ScalarType.Int), null);       IntIndexer i = t.createIndexer(); i.put(0, data); return t; }
    public static Tensor create(long[]    data, long... shape) { Tensor t = empty(shape, new TensorOptions(ScalarType.Long), null);     LongIndexer i = t.createIndexer(); i.put(0, data); return t; }
    public static Tensor create(float[]   data, long... shape) { Tensor t = empty(shape, new TensorOptions(ScalarType.Float), null);   FloatIndexer i = t.createIndexer(); i.put(0, data); return t; }
    public static Tensor create(double[]  data, long... shape) { Tensor t = empty(shape, new TensorOptions(ScalarType.Double), null); DoubleIndexer i = t.createIndexer(); i.put(0, data); return t; }
    public static Tensor create(boolean[] data, long... shape) { Tensor t = empty(shape, new TensorOptions(ScalarType.Bool), null);  BooleanIndexer i = t.createIndexer(); i.put(0, data); return t; }

    public abstract TensorOptions options();
    public abstract long ndimension();
    public abstract long size(long dim);
    public abstract long stride(long dim);
    public abstract long numel();
    public abstract long nbytes();
    public abstract Pointer data_ptr();

    /** Returns {@code createBuffer(0)}. */
    public <B extends Buffer> B createBuffer() {
        return (B)createBuffer(0);
    }
    /** Returns {@link #data_ptr()} wrapped in a {@link Buffer} of appropriate type starting at given index. */
    public <B extends Buffer> B createBuffer(long index) {
        TensorOptions options = options();
        if (options.layout().intern() != Layout.Strided) {
            throw new UnsupportedOperationException("Layout not supported: " + options.layout().intern());
        }
        if (options.device().type().intern() != DeviceType.CPU) {
            throw new UnsupportedOperationException("Device type not supported: " + options.device().type().intern());
        }
        ScalarType dtype = options.dtype().toScalarType().intern();
        Pointer ptr = data_ptr();
        long size = nbytes();
        switch (dtype) {
            case Byte:   return (B)new BytePointer(ptr).position(index).capacity(size).asBuffer();
            case Char:   return (B)new BytePointer(ptr).position(index).capacity(size).asBuffer();
            case Short:  return (B)new ShortPointer(ptr).position(index).capacity(size/2).asBuffer();
            case Int:    return (B)new IntPointer(ptr).position(index).capacity(size/4).asBuffer();
            case Long:   return (B)new LongPointer(ptr).position(index).capacity(size/8).asBuffer();
            case Half:   return (B)new ShortPointer(ptr).position(index).capacity(size/2).asBuffer();
            case Float:  return (B)new FloatPointer(ptr).position(index).capacity(size/4).asBuffer();
            case Double: return (B)new DoublePointer(ptr).position(index).capacity(size/8).asBuffer();
            case ComplexHalf:   return (B)new ShortPointer(ptr).position(index*2).capacity(size/2).asBuffer();
            case ComplexFloat:  return (B)new FloatPointer(ptr).position(index*2).capacity(size/4).asBuffer();
            case ComplexDouble: return (B)new DoublePointer(ptr).position(index*2).capacity(size/8).asBuffer();
            case Bool:   return (B)new BytePointer(ptr).position(index).capacity(size).asBuffer();
            case QInt8:  return (B)new BytePointer(ptr).position(index).capacity(size).asBuffer();
            case QUInt8: return (B)new BytePointer(ptr).position(index).capacity(size).asBuffer();
            case QInt32: return (B)new IntPointer(ptr).position(index).capacity(size/4).asBuffer();
            case BFloat16: return (B)new ShortPointer(ptr).position(index).capacity(size/2).asBuffer();
            case QUInt4x2: return (B)new BytePointer(ptr).position(index/2).capacity(size).asBuffer();
            default: throw new UnsupportedOperationException("Data type not supported: " + dtype);
        }
    }

    /** Returns {@code createIndexer(true)}. */
    public <I extends Indexer> I createIndexer() {
        return (I)createIndexer(true);
    }
    @Override public <I extends Indexer> I createIndexer(boolean direct) {
        TensorOptions options = options();
        if (options.layout().intern() != Layout.Strided) {
            throw new UnsupportedOperationException("Layout not supported: " + options.layout().intern());
        }
        if (options.device().type().intern() != DeviceType.CPU) {
            throw new UnsupportedOperationException("Device type not supported: " + options.device().type().intern());
        }
        ScalarType dtype = options.dtype().toScalarType().intern();
        Pointer ptr = data_ptr();
        long size = nbytes();
        int dims = (int)ndimension();
        boolean complex = dtype == ScalarType.ComplexHalf
                       || dtype == ScalarType.ComplexFloat
                       || dtype == ScalarType.ComplexDouble;
        boolean scalar = dims == 0;
        dims = (complex ? 1 : 0) + (scalar ? 1 : dims);
        long[] sizes = new long[dims];
        long[] strides = new long[dims];
        sizes[dims - 1] = complex ? 2 : (scalar ? 1 : size(dims - 1));
        strides[dims - 1] = complex ? 1 : (scalar ? 1 : stride(dims - 1));
        for (int i = dims - 2; i >= 0; i--) {
            sizes[i] = scalar ? 1 : size(i);
            strides[i] = scalar ? 1 : stride(i);
        }
        switch (dtype) {
            case Byte:   return (I)UByteIndexer.create(new BytePointer(ptr).capacity(size), sizes, strides, direct).indexable(this);
            case Char:   return (I)ByteIndexer.create(new BytePointer(ptr).capacity(size), sizes, strides, direct).indexable(this);
            case Short:  return (I)ShortIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
            case Int:    return (I)IntIndexer.create(new IntPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
            case Long:   return (I)LongIndexer.create(new LongPointer(ptr).capacity(size/8), sizes, strides, direct).indexable(this);
            case Half:   return (I)HalfIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
            case Float:  return (I)FloatIndexer.create(new FloatPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
            case Double: return (I)DoubleIndexer.create(new DoublePointer(ptr).capacity(size/8), sizes, strides, direct).indexable(this);
            case ComplexHalf:   return (I)HalfIndexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
            case ComplexFloat:  return (I)FloatIndexer.create(new FloatPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
            case ComplexDouble: return (I)DoubleIndexer.create(new DoublePointer(ptr).capacity(size/8), sizes, strides, direct).indexable(this);
            case Bool:   return (I)BooleanIndexer.create(new BooleanPointer(ptr).capacity(size), sizes, strides, direct).indexable(this);
            case QInt8:  return (I)ByteIndexer.create(new BytePointer(ptr).capacity(size), sizes, strides, direct).indexable(this);
            case QUInt8: return (I)UByteIndexer.create(new BytePointer(ptr).capacity(size), sizes, strides, direct).indexable(this);
            case QInt32: return (I)IntIndexer.create(new IntPointer(ptr).capacity(size/4), sizes, strides, direct).indexable(this);
            case BFloat16: return (I)Bfloat16Indexer.create(new ShortPointer(ptr).capacity(size/2), sizes, strides, direct).indexable(this);
            case QUInt4x2: return (I)UByteIndexer.create(new BytePointer(ptr).capacity(size), sizes, strides, direct).indexable(this);
            default: throw new UnsupportedOperationException("Data type not supported: " + dtype);
        }
    }
}
