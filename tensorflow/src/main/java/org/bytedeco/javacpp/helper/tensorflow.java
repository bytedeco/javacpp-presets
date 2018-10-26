/*
 * Copyright (C) 2015-2018 Samuel Audet
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
import org.bytedeco.javacpp.PointerScope;
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
import static org.bytedeco.javacpp.tensorflow.TF_AllocateTensor;
import static org.bytedeco.javacpp.tensorflow.TF_Buffer;
import static org.bytedeco.javacpp.tensorflow.TF_Graph;
import static org.bytedeco.javacpp.tensorflow.TF_ImportGraphDefOptions;
import static org.bytedeco.javacpp.tensorflow.TF_Session;
import static org.bytedeco.javacpp.tensorflow.TF_SessionOptions;
import static org.bytedeco.javacpp.tensorflow.TF_Status;
import static org.bytedeco.javacpp.tensorflow.TF_Tensor;
import static org.bytedeco.javacpp.tensorflow.TF_DeleteBuffer;
import static org.bytedeco.javacpp.tensorflow.TF_DeleteGraph;
import static org.bytedeco.javacpp.tensorflow.TF_DeleteImportGraphDefOptions;
import static org.bytedeco.javacpp.tensorflow.TF_DeleteSession;
import static org.bytedeco.javacpp.tensorflow.TF_DeleteSessionOptions;
import static org.bytedeco.javacpp.tensorflow.TF_DeleteStatus;
import static org.bytedeco.javacpp.tensorflow.TF_DeleteTensor;
import static org.bytedeco.javacpp.tensorflow.TF_NewBuffer;
import static org.bytedeco.javacpp.tensorflow.TF_NewGraph;
import static org.bytedeco.javacpp.tensorflow.TF_NewImportGraphDefOptions;
import static org.bytedeco.javacpp.tensorflow.TF_NewBufferFromString;
import static org.bytedeco.javacpp.tensorflow.TF_NewSession;
import static org.bytedeco.javacpp.tensorflow.TF_NewSessionOptions;
import static org.bytedeco.javacpp.tensorflow.TF_NewStatus;
import static org.bytedeco.javacpp.tensorflow.TF_NewTensor;

/**
 *
 * @author Samuel Audet
 */
public class tensorflow extends org.bytedeco.javacpp.presets.tensorflow {

    public static abstract class AbstractTF_Status extends Pointer {
        protected static class DeleteDeallocator extends TF_Status implements Pointer.Deallocator {
            DeleteDeallocator(TF_Status s) { super(s); }
            @Override public void deallocate() { if (!isNull()) TF_DeleteStatus(this); setNull(); }
        }

        public AbstractTF_Status(Pointer p) { super(p); }

        /**
         * Calls TF_NewStatus(), and registers a deallocator.
         * @return TF_Status created. Do not call TF_DeleteStatus() on it.
         */
        public static TF_Status newStatus() {
            TF_Status s = TF_NewStatus();
            if (s != null) {
                s.deallocator(new DeleteDeallocator(s));
            }
            return s;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void delete() {
            deallocate();
        }
    }

    public static abstract class AbstractTF_Buffer extends Pointer {
        protected static class DeleteDeallocator extends TF_Buffer implements Pointer.Deallocator {
            DeleteDeallocator(TF_Buffer s) { super(s); }
            @Override public void deallocate() { if (!isNull()) TF_DeleteBuffer(this); setNull(); }
        }

        public AbstractTF_Buffer(Pointer p) { super(p); }

        /**
         * Calls TF_NewBuffer(), and registers a deallocator.
         * @return TF_Buffer created. Do not call TF_DeleteBuffer() on it.
         */
        public static TF_Buffer newBuffer() {
            TF_Buffer b = TF_NewBuffer();
            if (b != null) {
                b.deallocator(new DeleteDeallocator(b));
            }
            return b;
        }

        /** Returns {@code newBufferFromString(new BytePointer(proto)). */
        public static TF_Buffer newBufferFromString(byte[] proto) {
            return newBufferFromString(new BytePointer(proto));
        }

        /**
         * Calls TF_NewBufferFromString(), and registers a deallocator.
         * @return TF_Buffer created. Do not call TF_DeleteBuffer() on it.
         */
        public static TF_Buffer newBufferFromString(Pointer proto) {
            TF_Buffer b = TF_NewBufferFromString(proto, proto.limit());
            if (b != null) {
                b.deallocator(new DeleteDeallocator(b));
            }
            return b;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void delete() {
            deallocate();
        }
    }

    public static abstract class AbstractTF_Tensor extends Pointer {
        protected static class DeleteDeallocator extends TF_Tensor implements Pointer.Deallocator {
            DeleteDeallocator(TF_Tensor s) { super(s); }
            @Override public void deallocate() { if (!isNull()) TF_DeleteTensor(this); setNull(); }
        }

        /** TensorFlow crashes if we don't pass it a deallocator, so... */
        protected static Deallocator_Pointer_long_Pointer dummyDeallocator = new Deallocator_Pointer_long_Pointer() {
            @Override public void call(Pointer data, long len, Pointer arg) { }
        };

        static {
            PointerScope s = PointerScope.getInnerScope();
            if (s != null) {
                s.detach(dummyDeallocator);
            }
        }

        /** A reference to prevent deallocation. */
        protected Pointer pointer;

        public AbstractTF_Tensor(Pointer p) { super(p); }

        /**
         * Calls TF_NewTensor(), and registers a deallocator.
         * @return TF_Tensor created. Do not call TF_DeleteTensor() on it.
         */
        public static TF_Tensor newTensor(int dtype, long[] dims, Pointer data) {
            TF_Tensor t = TF_NewTensor(dtype, dims, dims.length, data, data.limit(), dummyDeallocator, null);
            if (t != null) {
                t.pointer = data;
                t.deallocator(new DeleteDeallocator(t));
            }
            return t;
        }

        /**
         * Calls TF_AllocateTensor(), and registers a deallocator.
         * @return TF_Tensor created. Do not call TF_DeleteTensor() on it.
         */
        public static TF_Tensor allocateTensor(int dtype, long[] dims, long length) {
            TF_Tensor t = TF_AllocateTensor(dtype, dims, dims.length, length);
            if (t != null) {
                t.deallocator(new DeleteDeallocator(t));
            }
            return t;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void delete() {
            deallocate();
        }
    }

    public static abstract class AbstractTF_SessionOptions extends Pointer {
        protected static class DeleteDeallocator extends TF_SessionOptions implements Pointer.Deallocator {
            DeleteDeallocator(TF_SessionOptions s) { super(s); }
            @Override public void deallocate() { if (!isNull()) TF_DeleteSessionOptions(this); setNull(); }
        }

        public AbstractTF_SessionOptions(Pointer p) { super(p); }

        /**
         * Calls TF_NewSessionOptions(), and registers a deallocator.
         * @return TF_SessionOptions created. Do not call TF_DeleteSessionOptions() on it.
         */
        public static TF_SessionOptions newSessionOptions() {
            TF_SessionOptions o = TF_NewSessionOptions();
            if (o != null) {
                o.deallocator(new DeleteDeallocator(o));
            }
            return o;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void delete() {
            deallocate();
        }
    }

    public static abstract class AbstractTF_Graph extends Pointer {
        protected static class DeleteDeallocator extends TF_Graph implements Pointer.Deallocator {
            DeleteDeallocator(TF_Graph s) { super(s); }
            @Override public void deallocate() { if (!isNull()) TF_DeleteGraph(this); setNull(); }
        }

        public AbstractTF_Graph(Pointer p) { super(p); }

        /**
         * Calls TF_NewGraph(), and registers a deallocator.
         * @return TF_Graph created. Do not call TF_DeleteGraph() on it.
         */
        public static TF_Graph newGraph() {
            TF_Graph g = TF_NewGraph();
            if (g != null) {
                g.deallocator(new DeleteDeallocator(g));
            }
            return g;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void delete() {
            deallocate();
        }
    }

    public static abstract class AbstractTF_ImportGraphDefOptions extends Pointer {
        protected static class DeleteDeallocator extends TF_ImportGraphDefOptions implements Pointer.Deallocator {
            DeleteDeallocator(TF_ImportGraphDefOptions s) { super(s); }
            @Override public void deallocate() { if (!isNull()) TF_DeleteImportGraphDefOptions(this); setNull(); }
        }

        public AbstractTF_ImportGraphDefOptions(Pointer p) { super(p); }

        /**
         * Calls TF_NewImportGraphDefOptions(), and registers a deallocator.
         * @return TF_ImportGraphDefOptions created. Do not call TF_DeleteImportGraphDefOptions() on it.
         */
        public static TF_ImportGraphDefOptions newImportGraphDefOptions() {
            TF_ImportGraphDefOptions o = TF_NewImportGraphDefOptions();
            if (o != null) {
                o.deallocator(new DeleteDeallocator(o));
            }
            return o;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void delete() {
            deallocate();
        }
    }

    public static abstract class AbstractTF_Session extends Pointer {
        protected static class DeleteDeallocator extends TF_Session implements Pointer.Deallocator {
            DeleteDeallocator(TF_Session s) { super(s); }
            @Override public void deallocate() { if (!isNull()) TF_DeleteSession(this, TF_Status.newStatus()); setNull(); }
        }

        /** References to prevent deallocation. */
        protected TF_Graph graph;
        protected TF_SessionOptions opts;
        protected TF_Status status;

        public AbstractTF_Session(Pointer p) { super(p); }

        /**
         * Calls TF_NewSession(), and registers a deallocator.
         * @return TF_Session created. Do not call TF_DeleteSession() on it.
         */
        public static TF_Session newSession(TF_Graph graph, TF_SessionOptions opts, TF_Status status) {
            TF_Session s = TF_NewSession(graph, opts, status);
            if (s != null) {
                s.graph = graph;
                s.opts = opts;
                s.status = status;
                s.deallocator(new DeleteDeallocator(s));
            }
            return s;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void delete() {
            deallocate();
        }
    }

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
            @Override public void deallocate() { if (!isNull()) Session.delete(this); setNull(); }
        }
    }
}
