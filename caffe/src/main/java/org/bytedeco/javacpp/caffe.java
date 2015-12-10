// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_videoio.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

public class caffe extends org.bytedeco.javacpp.presets.caffe {
    static { Loader.load(); }

@Name("std::map<std::string,caffe::LayerRegistry<float>::Creator>") public static class FloatRegistry extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatRegistry(Pointer p) { super(p); }
    public FloatRegistry()       { allocate();  }
    private native void allocate();
    public native @Name("operator=") @ByRef FloatRegistry put(@ByRef FloatRegistry x);

    public native long size();

    @Index public native FloatLayerRegistry.Creator get(@StdString BytePointer i);
    public native FloatRegistry put(@StdString BytePointer i, FloatLayerRegistry.Creator value);

    public native @ByVal Iterator begin();
    public native @ByVal Iterator end();
    @NoOffset @Name("iterator") public static class Iterator extends Pointer {
        public Iterator(Pointer p) { super(p); }
        public Iterator() { }

        public native @Name("operator++") @ByRef Iterator increment();
        public native @Name("operator==") boolean equals(@ByRef Iterator it);
        public native @Name("operator*().first") @MemberGetter @StdString BytePointer first();
        public native @Name("operator*().second") @MemberGetter FloatLayerRegistry.Creator second();
    }
}

@Name("std::map<std::string,caffe::LayerRegistry<double>::Creator>") public static class DoubleRegistry extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleRegistry(Pointer p) { super(p); }
    public DoubleRegistry()       { allocate();  }
    private native void allocate();
    public native @Name("operator=") @ByRef DoubleRegistry put(@ByRef DoubleRegistry x);

    public native long size();

    @Index public native DoubleLayerRegistry.Creator get(@StdString BytePointer i);
    public native DoubleRegistry put(@StdString BytePointer i, DoubleLayerRegistry.Creator value);

    public native @ByVal Iterator begin();
    public native @ByVal Iterator end();
    @NoOffset @Name("iterator") public static class Iterator extends Pointer {
        public Iterator(Pointer p) { super(p); }
        public Iterator() { }

        public native @Name("operator++") @ByRef Iterator increment();
        public native @Name("operator==") boolean equals(@ByRef Iterator it);
        public native @Name("operator*().first") @MemberGetter @StdString BytePointer first();
        public native @Name("operator*().second") @MemberGetter DoubleLayerRegistry.Creator second();
    }
}

@Name("std::map<std::string,int>") public static class StringIntMap extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringIntMap(Pointer p) { super(p); }
    public StringIntMap()       { allocate();  }
    private native void allocate();
    public native @Name("operator=") @ByRef StringIntMap put(@ByRef StringIntMap x);

    public native long size();

    @Index public native int get(@StdString BytePointer i);
    public native StringIntMap put(@StdString BytePointer i, int value);

    public native @ByVal Iterator begin();
    public native @ByVal Iterator end();
    @NoOffset @Name("iterator") public static class Iterator extends Pointer {
        public Iterator(Pointer p) { super(p); }
        public Iterator() { }

        public native @Name("operator++") @ByRef Iterator increment();
        public native @Name("operator==") boolean equals(@ByRef Iterator it);
        public native @Name("operator*().first") @MemberGetter @StdString BytePointer first();
        public native @Name("operator*().second") @MemberGetter int second();
    }
}

@Name("std::vector<std::string>") public static class StringVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringVector(Pointer p) { super(p); }
    public StringVector(BytePointer ... array) { this(array.length); put(array); }
    public StringVector(String ... array) { this(array.length); put(array); }
    public StringVector()       { allocate();  }
    public StringVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef StringVector put(@ByRef StringVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @StdString BytePointer get(@Cast("size_t") long i);
    public native StringVector put(@Cast("size_t") long i, BytePointer value);
    @ValueSetter @Index public native StringVector put(@Cast("size_t") long i, @StdString String value);

    public StringVector put(BytePointer ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }

    public StringVector put(String ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<caffe::Datum>") public static class DatumVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DatumVector(Pointer p) { super(p); }
    public DatumVector(Datum ... array) { this(array.length); put(array); }
    public DatumVector()       { allocate();  }
    public DatumVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef DatumVector put(@ByRef DatumVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef Datum get(@Cast("size_t") long i);
    public native DatumVector put(@Cast("size_t") long i, Datum value);

    public DatumVector put(Datum ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<boost::shared_ptr<caffe::Blob<float> > >") public static class FloatBlobSharedVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBlobSharedVector(Pointer p) { super(p); }
    public FloatBlobSharedVector(FloatBlob ... array) { this(array.length); put(array); }
    public FloatBlobSharedVector()       { allocate();  }
    public FloatBlobSharedVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef FloatBlobSharedVector put(@ByRef FloatBlobSharedVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @SharedPtr FloatBlob get(@Cast("size_t") long i);
    public native FloatBlobSharedVector put(@Cast("size_t") long i, FloatBlob value);

    public FloatBlobSharedVector put(FloatBlob ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<boost::shared_ptr<caffe::Blob<double> > >") public static class DoubleBlobSharedVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBlobSharedVector(Pointer p) { super(p); }
    public DoubleBlobSharedVector(DoubleBlob ... array) { this(array.length); put(array); }
    public DoubleBlobSharedVector()       { allocate();  }
    public DoubleBlobSharedVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef DoubleBlobSharedVector put(@ByRef DoubleBlobSharedVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @SharedPtr DoubleBlob get(@Cast("size_t") long i);
    public native DoubleBlobSharedVector put(@Cast("size_t") long i, DoubleBlob value);

    public DoubleBlobSharedVector put(DoubleBlob ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<boost::shared_ptr<caffe::Layer<float> > >") public static class FloatLayerSharedVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatLayerSharedVector(Pointer p) { super(p); }
    public FloatLayerSharedVector(FloatLayer ... array) { this(array.length); put(array); }
    public FloatLayerSharedVector()       { allocate();  }
    public FloatLayerSharedVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef FloatLayerSharedVector put(@ByRef FloatLayerSharedVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @Cast({"", "boost::shared_ptr<caffe::Layer<float> >"}) @SharedPtr FloatLayer get(@Cast("size_t") long i);
    public native FloatLayerSharedVector put(@Cast("size_t") long i, FloatLayer value);

    public FloatLayerSharedVector put(FloatLayer ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<boost::shared_ptr<caffe::Layer<double> > >") public static class DoubleLayerSharedVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleLayerSharedVector(Pointer p) { super(p); }
    public DoubleLayerSharedVector(DoubleLayer ... array) { this(array.length); put(array); }
    public DoubleLayerSharedVector()       { allocate();  }
    public DoubleLayerSharedVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef DoubleLayerSharedVector put(@ByRef DoubleLayerSharedVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @Cast({"", "boost::shared_ptr<caffe::Layer<double> >"}) @SharedPtr DoubleLayer get(@Cast("size_t") long i);
    public native DoubleLayerSharedVector put(@Cast("size_t") long i, DoubleLayer value);

    public DoubleLayerSharedVector put(DoubleLayer ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<boost::shared_ptr<caffe::Net<float> > >") public static class FloatNetSharedVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatNetSharedVector(Pointer p) { super(p); }
    public FloatNetSharedVector(FloatNet ... array) { this(array.length); put(array); }
    public FloatNetSharedVector()       { allocate();  }
    public FloatNetSharedVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef FloatNetSharedVector put(@ByRef FloatNetSharedVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @SharedPtr FloatNet get(@Cast("size_t") long i);
    public native FloatNetSharedVector put(@Cast("size_t") long i, FloatNet value);

    public FloatNetSharedVector put(FloatNet ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<boost::shared_ptr<caffe::Net<double> > >") public static class DoubleNetSharedVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleNetSharedVector(Pointer p) { super(p); }
    public DoubleNetSharedVector(DoubleNet ... array) { this(array.length); put(array); }
    public DoubleNetSharedVector()       { allocate();  }
    public DoubleNetSharedVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef DoubleNetSharedVector put(@ByRef DoubleNetSharedVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @SharedPtr DoubleNet get(@Cast("size_t") long i);
    public native DoubleNetSharedVector put(@Cast("size_t") long i, DoubleNet value);

    public DoubleNetSharedVector put(DoubleNet ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<caffe::Blob<float>*>") public static class FloatBlobVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBlobVector(Pointer p) { super(p); }
    public FloatBlobVector(FloatBlob ... array) { this(array.length); put(array); }
    public FloatBlobVector()       { allocate();  }
    public FloatBlobVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef FloatBlobVector put(@ByRef FloatBlobVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native FloatBlob get(@Cast("size_t") long i);
    public native FloatBlobVector put(@Cast("size_t") long i, FloatBlob value);

    public FloatBlobVector put(FloatBlob ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<caffe::Blob<double>*>") public static class DoubleBlobVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBlobVector(Pointer p) { super(p); }
    public DoubleBlobVector(DoubleBlob ... array) { this(array.length); put(array); }
    public DoubleBlobVector()       { allocate();  }
    public DoubleBlobVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef DoubleBlobVector put(@ByRef DoubleBlobVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native DoubleBlob get(@Cast("size_t") long i);
    public native DoubleBlobVector put(@Cast("size_t") long i, DoubleBlob value);

    public DoubleBlobVector put(DoubleBlob ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<std::vector<caffe::Blob<float>*> >") public static class FloatBlobVectorVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBlobVectorVector(Pointer p) { super(p); }
    public FloatBlobVectorVector(FloatBlobVector ... array) { this(array.length); put(array); }
    public FloatBlobVectorVector()       { allocate();  }
    public FloatBlobVectorVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef FloatBlobVectorVector put(@ByRef FloatBlobVectorVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef FloatBlobVector get(@Cast("size_t") long i);
    public native FloatBlobVectorVector put(@Cast("size_t") long i, FloatBlobVector value);

    public FloatBlobVectorVector put(FloatBlobVector ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<std::vector<caffe::Blob<double>*> >") public static class DoubleBlobVectorVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBlobVectorVector(Pointer p) { super(p); }
    public DoubleBlobVectorVector(DoubleBlobVector ... array) { this(array.length); put(array); }
    public DoubleBlobVectorVector()       { allocate();  }
    public DoubleBlobVectorVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef DoubleBlobVectorVector put(@ByRef DoubleBlobVectorVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef DoubleBlobVector get(@Cast("size_t") long i);
    public native DoubleBlobVectorVector put(@Cast("size_t") long i, DoubleBlobVector value);

    public DoubleBlobVectorVector put(DoubleBlobVector ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<bool>") public static class BoolVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BoolVector(Pointer p) { super(p); }
    public BoolVector(boolean ... array) { this(array.length); put(array); }
    public BoolVector()       { allocate();  }
    public BoolVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef BoolVector put(@ByRef BoolVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @Cast("bool") boolean get(@Cast("size_t") long i);
    public native BoolVector put(@Cast("size_t") long i, boolean value);

    public BoolVector put(boolean ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<std::vector<bool> >") public static class BoolVectorVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BoolVectorVector(Pointer p) { super(p); }
    public BoolVectorVector(BoolVector ... array) { this(array.length); put(array); }
    public BoolVectorVector()       { allocate();  }
    public BoolVectorVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef BoolVectorVector put(@ByRef BoolVectorVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef BoolVector get(@Cast("size_t") long i);
    public native BoolVectorVector put(@Cast("size_t") long i, BoolVector value);

    public BoolVectorVector put(BoolVector ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<caffe::Solver<float>::Callback*>") public static class FloatCallbackVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatCallbackVector(Pointer p) { super(p); }
    public FloatCallbackVector(FloatSolver.Callback ... array) { this(array.length); put(array); }
    public FloatCallbackVector()       { allocate();  }
    public FloatCallbackVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef FloatCallbackVector put(@ByRef FloatCallbackVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native FloatSolver.Callback get(@Cast("size_t") long i);
    public native FloatCallbackVector put(@Cast("size_t") long i, FloatSolver.Callback value);

    public FloatCallbackVector put(FloatSolver.Callback ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<caffe::Solver<double>::Callback*>") public static class DoubleCallbackVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleCallbackVector(Pointer p) { super(p); }
    public DoubleCallbackVector(DoubleSolver.Callback ... array) { this(array.length); put(array); }
    public DoubleCallbackVector()       { allocate();  }
    public DoubleCallbackVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef DoubleCallbackVector put(@ByRef DoubleCallbackVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native DoubleSolver.Callback get(@Cast("size_t") long i);
    public native DoubleCallbackVector put(@Cast("size_t") long i, DoubleSolver.Callback value);

    public DoubleCallbackVector put(DoubleSolver.Callback ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

// Parsed from caffe/caffe.hpp

// caffe.hpp is the header file that you need to include in your code. It wraps
// all the internal caffe header files into one for simpler inclusion.

// #ifndef CAFFE_CAFFE_HPP_
// #define CAFFE_CAFFE_HPP_

// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/filler.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/layer_factory.hpp"
// #include "caffe/net.hpp"
// #include "caffe/parallel.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/solver.hpp"
// #include "caffe/solver_factory.hpp"
// #include "caffe/util/benchmark.hpp"
// #include "caffe/util/io.hpp"
// #include "caffe/util/upgrade_proto.hpp"

// #endif  // CAFFE_CAFFE_HPP_


// Parsed from caffe/util/device_alternate.hpp

// #ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
// #define CAFFE_UTIL_DEVICE_ALTERNATE_H_

// #ifdef CPU_ONLY  // CPU-only Caffe.

// #include <vector>

// Stub out GPU calls as unavailable.

// #define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

// #define STUB_GPU(classname)
// template <typename Dtype>
// void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) { NO_GPU; }
// template <typename Dtype>
// void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down,
//     const vector<Blob<Dtype>*>& bottom) { NO_GPU; } 

// #define STUB_GPU_FORWARD(classname, funcname)
// template <typename Dtype>
// void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) { NO_GPU; } 

// #define STUB_GPU_BACKWARD(classname, funcname)
// template <typename Dtype>
// void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down,
//     const vector<Blob<Dtype>*>& bottom) { NO_GPU; } 

// #else  // Normal GPU + CPU Caffe.

// #endif  // CPU_ONLY

// #endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_


// Parsed from caffe/common.hpp

// #ifndef CAFFE_COMMON_HPP_
// #define CAFFE_COMMON_HPP_

// #include <boost/shared_ptr.hpp>
// #include <gflags/gflags.h>
// #include <glog/logging.h>

// #include <climits>
// #include <cmath>
// #include <fstream>  // NOLINT(readability/streams)
// #include <iostream>  // NOLINT(readability/streams)
// #include <map>
// #include <set>
// #include <sstream>
// #include <string>
// #include <utility>  // pair
// #include <vector>

// #include "caffe/util/device_alternate.hpp"

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
// #ifndef GFLAGS_GFLAGS_H_
// #endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
// #define DISABLE_COPY_AND_ASSIGN(classname)
// private:
//   classname(const classname&);
//   classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
// #define INSTANTIATE_CLASS(classname)
//   char gInstantiationGuard##classname;
//   template class classname<float>;
//   template class classname<double>

// #define INSTANTIATE_LAYER_GPU_FORWARD(classname)
//   template void classname<float>::Forward_gpu(
//       const std::vector<Blob<float>*>& bottom,
//       const std::vector<Blob<float>*>& top);
//   template void classname<double>::Forward_gpu(
//       const std::vector<Blob<double>*>& bottom,
//       const std::vector<Blob<double>*>& top);

// #define INSTANTIATE_LAYER_GPU_BACKWARD(classname)
//   template void classname<float>::Backward_gpu(
//       const std::vector<Blob<float>*>& top,
//       const std::vector<bool>& propagate_down,
//       const std::vector<Blob<float>*>& bottom);
//   template void classname<double>::Backward_gpu(
//       const std::vector<Blob<double>*>& top,
//       const std::vector<bool>& propagate_down,
//       const std::vector<Blob<double>*>& bottom)

// #define INSTANTIATE_LAYER_GPU_FUNCS(classname)
//   INSTANTIATE_LAYER_GPU_FORWARD(classname);
//   INSTANTIATE_LAYER_GPU_BACKWARD(classname)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
// #define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236 

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.

// Common functions and classes from std that caffe often uses.

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
@Namespace("caffe") public static native void GlobalInit(IntPointer pargc, @Cast("char***") PointerPointer pargv);
@Namespace("caffe") public static native void GlobalInit(IntBuffer pargc, @Cast("char***") PointerPointer pargv);
@Namespace("caffe") public static native void GlobalInit(int[] pargc, @Cast("char***") PointerPointer pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
@Namespace("caffe") @NoOffset public static class Caffe extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Caffe(Pointer p) { super(p); }


  // Thread local context for Caffe. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  public static native @ByRef Caffe Get();

  /** enum caffe::Caffe::Brew */
  public static final int CPU = 0, GPU = 1;

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  @NoOffset public static class RNG extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public RNG(Pointer p) { super(p); }
  
    public RNG() { super((Pointer)null); allocate(); }
    private native void allocate();
    public RNG(@Cast("unsigned int") int seed) { super((Pointer)null); allocate(seed); }
    private native void allocate(@Cast("unsigned int") int seed);
    public RNG(@Const @ByRef RNG arg0) { super((Pointer)null); allocate(arg0); }
    private native void allocate(@Const @ByRef RNG arg0);
    public native @ByRef @Name("operator =") RNG put(@Const @ByRef RNG arg0);
    public native Pointer generator();
  }

  // Getters for boost rng, curand, and cublas handles
  public static native @ByRef RNG rng_stream();
// #ifndef CPU_ONLY
// #endif

  // Returns the mode: running on CPU or GPU.
  public static native @Cast("caffe::Caffe::Brew") int mode();
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  public static native void set_mode(@Cast("caffe::Caffe::Brew") int mode);
  // Sets the random seed of both boost and curand
  public static native void set_random_seed(@Cast("const unsigned int") int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  public static native void SetDevice(int device_id);
  // Prints the current GPU status.
  public static native void DeviceQuery();
  // Parallel training info
  public static native int solver_count();
  public static native void set_solver_count(int val);
  public static native @Cast("bool") boolean root_solver();
  public static native void set_root_solver(@Cast("bool") boolean val);
}

  // namespace caffe

// #endif  // CAFFE_COMMON_HPP_


// Parsed from caffe/proto/caffe.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: caffe.proto

// #ifndef PROTOBUF_caffe_2eproto__INCLUDED
// #define PROTOBUF_caffe_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 2006000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 2006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/generated_enum_reflection.h>
// #include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("caffe") public static native void protobuf_AddDesc_caffe_2eproto();
@Namespace("caffe") public static native void protobuf_AssignDesc_caffe_2eproto();
@Namespace("caffe") public static native void protobuf_ShutdownFile_caffe_2eproto();

/** enum caffe::FillerParameter_VarianceNorm */
public static final int
  FillerParameter_VarianceNorm_FAN_IN = 0,
  FillerParameter_VarianceNorm_FAN_OUT = 1,
  FillerParameter_VarianceNorm_AVERAGE = 2;
@Namespace("caffe") public static native @Cast("bool") boolean FillerParameter_VarianceNorm_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::FillerParameter_VarianceNorm") int FillerParameter_VarianceNorm_VarianceNorm_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::FillerParameter_VarianceNorm") int FillerParameter_VarianceNorm_VarianceNorm_MAX();
@Namespace("caffe") @MemberGetter public static native int FillerParameter_VarianceNorm_VarianceNorm_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer FillerParameter_VarianceNorm_descriptor();
@Namespace("caffe") public static native @StdString BytePointer FillerParameter_VarianceNorm_Name(@Cast("caffe::FillerParameter_VarianceNorm") int value);
@Namespace("caffe") public static native @Cast("bool") boolean FillerParameter_VarianceNorm_Parse(
    @StdString BytePointer name, @Cast("caffe::FillerParameter_VarianceNorm*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean FillerParameter_VarianceNorm_Parse(
    @StdString String name, @Cast("caffe::FillerParameter_VarianceNorm*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean FillerParameter_VarianceNorm_Parse(
    @StdString BytePointer name, @Cast("caffe::FillerParameter_VarianceNorm*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean FillerParameter_VarianceNorm_Parse(
    @StdString String name, @Cast("caffe::FillerParameter_VarianceNorm*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean FillerParameter_VarianceNorm_Parse(
    @StdString BytePointer name, @Cast("caffe::FillerParameter_VarianceNorm*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean FillerParameter_VarianceNorm_Parse(
    @StdString String name, @Cast("caffe::FillerParameter_VarianceNorm*") int[] value);
/** enum caffe::SolverParameter_SnapshotFormat */
public static final int
  SolverParameter_SnapshotFormat_HDF5 = 0,
  SolverParameter_SnapshotFormat_BINARYPROTO = 1;
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SnapshotFormat_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SolverParameter_SnapshotFormat") int SolverParameter_SnapshotFormat_SnapshotFormat_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SolverParameter_SnapshotFormat") int SolverParameter_SnapshotFormat_SnapshotFormat_MAX();
@Namespace("caffe") @MemberGetter public static native int SolverParameter_SnapshotFormat_SnapshotFormat_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SolverParameter_SnapshotFormat_descriptor();
@Namespace("caffe") public static native @StdString BytePointer SolverParameter_SnapshotFormat_Name(@Cast("caffe::SolverParameter_SnapshotFormat") int value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SnapshotFormat_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SnapshotFormat*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SnapshotFormat_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SnapshotFormat*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SnapshotFormat_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SnapshotFormat*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SnapshotFormat_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SnapshotFormat*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SnapshotFormat_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SnapshotFormat*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SnapshotFormat_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SnapshotFormat*") int[] value);
/** enum caffe::SolverParameter_SolverMode */
public static final int
  SolverParameter_SolverMode_CPU = 0,
  SolverParameter_SolverMode_GPU = 1;
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverMode_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SolverParameter_SolverMode") int SolverParameter_SolverMode_SolverMode_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SolverParameter_SolverMode") int SolverParameter_SolverMode_SolverMode_MAX();
@Namespace("caffe") @MemberGetter public static native int SolverParameter_SolverMode_SolverMode_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SolverParameter_SolverMode_descriptor();
@Namespace("caffe") public static native @StdString BytePointer SolverParameter_SolverMode_Name(@Cast("caffe::SolverParameter_SolverMode") int value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverMode_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SolverMode*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverMode_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SolverMode*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverMode_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SolverMode*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverMode_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SolverMode*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverMode_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SolverMode*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverMode_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SolverMode*") int[] value);
/** enum caffe::SolverParameter_SolverType */
public static final int
  SolverParameter_SolverType_SGD = 0,
  SolverParameter_SolverType_NESTEROV = 1,
  SolverParameter_SolverType_ADAGRAD = 2,
  SolverParameter_SolverType_RMSPROP = 3,
  SolverParameter_SolverType_ADADELTA = 4,
  SolverParameter_SolverType_ADAM = 5;
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverType_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SolverParameter_SolverType") int SolverParameter_SolverType_SolverType_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SolverParameter_SolverType") int SolverParameter_SolverType_SolverType_MAX();
@Namespace("caffe") @MemberGetter public static native int SolverParameter_SolverType_SolverType_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SolverParameter_SolverType_descriptor();
@Namespace("caffe") public static native @StdString BytePointer SolverParameter_SolverType_Name(@Cast("caffe::SolverParameter_SolverType") int value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverType_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SolverType*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverType_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SolverType*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverType_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SolverType*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverType_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SolverType*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverType_Parse(
    @StdString BytePointer name, @Cast("caffe::SolverParameter_SolverType*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SolverParameter_SolverType_Parse(
    @StdString String name, @Cast("caffe::SolverParameter_SolverType*") int[] value);
/** enum caffe::ParamSpec_DimCheckMode */
public static final int
  ParamSpec_DimCheckMode_STRICT = 0,
  ParamSpec_DimCheckMode_PERMISSIVE = 1;
@Namespace("caffe") public static native @Cast("bool") boolean ParamSpec_DimCheckMode_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::ParamSpec_DimCheckMode") int ParamSpec_DimCheckMode_DimCheckMode_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::ParamSpec_DimCheckMode") int ParamSpec_DimCheckMode_DimCheckMode_MAX();
@Namespace("caffe") @MemberGetter public static native int ParamSpec_DimCheckMode_DimCheckMode_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer ParamSpec_DimCheckMode_descriptor();
@Namespace("caffe") public static native @StdString BytePointer ParamSpec_DimCheckMode_Name(@Cast("caffe::ParamSpec_DimCheckMode") int value);
@Namespace("caffe") public static native @Cast("bool") boolean ParamSpec_DimCheckMode_Parse(
    @StdString BytePointer name, @Cast("caffe::ParamSpec_DimCheckMode*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean ParamSpec_DimCheckMode_Parse(
    @StdString String name, @Cast("caffe::ParamSpec_DimCheckMode*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean ParamSpec_DimCheckMode_Parse(
    @StdString BytePointer name, @Cast("caffe::ParamSpec_DimCheckMode*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean ParamSpec_DimCheckMode_Parse(
    @StdString String name, @Cast("caffe::ParamSpec_DimCheckMode*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean ParamSpec_DimCheckMode_Parse(
    @StdString BytePointer name, @Cast("caffe::ParamSpec_DimCheckMode*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean ParamSpec_DimCheckMode_Parse(
    @StdString String name, @Cast("caffe::ParamSpec_DimCheckMode*") int[] value);
/** enum caffe::LossParameter_NormalizationMode */
public static final int
  LossParameter_NormalizationMode_FULL = 0,
  LossParameter_NormalizationMode_VALID = 1,
  LossParameter_NormalizationMode_BATCH_SIZE = 2,
  LossParameter_NormalizationMode_NONE = 3;
@Namespace("caffe") public static native @Cast("bool") boolean LossParameter_NormalizationMode_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::LossParameter_NormalizationMode") int LossParameter_NormalizationMode_NormalizationMode_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::LossParameter_NormalizationMode") int LossParameter_NormalizationMode_NormalizationMode_MAX();
@Namespace("caffe") @MemberGetter public static native int LossParameter_NormalizationMode_NormalizationMode_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer LossParameter_NormalizationMode_descriptor();
@Namespace("caffe") public static native @StdString BytePointer LossParameter_NormalizationMode_Name(@Cast("caffe::LossParameter_NormalizationMode") int value);
@Namespace("caffe") public static native @Cast("bool") boolean LossParameter_NormalizationMode_Parse(
    @StdString BytePointer name, @Cast("caffe::LossParameter_NormalizationMode*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean LossParameter_NormalizationMode_Parse(
    @StdString String name, @Cast("caffe::LossParameter_NormalizationMode*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean LossParameter_NormalizationMode_Parse(
    @StdString BytePointer name, @Cast("caffe::LossParameter_NormalizationMode*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean LossParameter_NormalizationMode_Parse(
    @StdString String name, @Cast("caffe::LossParameter_NormalizationMode*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean LossParameter_NormalizationMode_Parse(
    @StdString BytePointer name, @Cast("caffe::LossParameter_NormalizationMode*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean LossParameter_NormalizationMode_Parse(
    @StdString String name, @Cast("caffe::LossParameter_NormalizationMode*") int[] value);
/** enum caffe::ConvolutionParameter_Engine */
public static final int
  ConvolutionParameter_Engine_DEFAULT = 0,
  ConvolutionParameter_Engine_CAFFE = 1,
  ConvolutionParameter_Engine_CUDNN = 2;
@Namespace("caffe") public static native @Cast("bool") boolean ConvolutionParameter_Engine_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::ConvolutionParameter_Engine") int ConvolutionParameter_Engine_Engine_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::ConvolutionParameter_Engine") int ConvolutionParameter_Engine_Engine_MAX();
@Namespace("caffe") @MemberGetter public static native int ConvolutionParameter_Engine_Engine_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer ConvolutionParameter_Engine_descriptor();
@Namespace("caffe") public static native @StdString BytePointer ConvolutionParameter_Engine_Name(@Cast("caffe::ConvolutionParameter_Engine") int value);
@Namespace("caffe") public static native @Cast("bool") boolean ConvolutionParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::ConvolutionParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean ConvolutionParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::ConvolutionParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean ConvolutionParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::ConvolutionParameter_Engine*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean ConvolutionParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::ConvolutionParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean ConvolutionParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::ConvolutionParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean ConvolutionParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::ConvolutionParameter_Engine*") int[] value);
/** enum caffe::DataParameter_DB */
public static final int
  DataParameter_DB_LEVELDB = 0,
  DataParameter_DB_LMDB = 1;
@Namespace("caffe") public static native @Cast("bool") boolean DataParameter_DB_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::DataParameter_DB") int DataParameter_DB_DB_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::DataParameter_DB") int DataParameter_DB_DB_MAX();
@Namespace("caffe") @MemberGetter public static native int DataParameter_DB_DB_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer DataParameter_DB_descriptor();
@Namespace("caffe") public static native @StdString BytePointer DataParameter_DB_Name(@Cast("caffe::DataParameter_DB") int value);
@Namespace("caffe") public static native @Cast("bool") boolean DataParameter_DB_Parse(
    @StdString BytePointer name, @Cast("caffe::DataParameter_DB*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean DataParameter_DB_Parse(
    @StdString String name, @Cast("caffe::DataParameter_DB*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean DataParameter_DB_Parse(
    @StdString BytePointer name, @Cast("caffe::DataParameter_DB*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean DataParameter_DB_Parse(
    @StdString String name, @Cast("caffe::DataParameter_DB*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean DataParameter_DB_Parse(
    @StdString BytePointer name, @Cast("caffe::DataParameter_DB*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean DataParameter_DB_Parse(
    @StdString String name, @Cast("caffe::DataParameter_DB*") int[] value);
/** enum caffe::EltwiseParameter_EltwiseOp */
public static final int
  EltwiseParameter_EltwiseOp_PROD = 0,
  EltwiseParameter_EltwiseOp_SUM = 1,
  EltwiseParameter_EltwiseOp_MAX = 2;
@Namespace("caffe") public static native @Cast("bool") boolean EltwiseParameter_EltwiseOp_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::EltwiseParameter_EltwiseOp") int EltwiseParameter_EltwiseOp_EltwiseOp_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::EltwiseParameter_EltwiseOp") int EltwiseParameter_EltwiseOp_EltwiseOp_MAX();
@Namespace("caffe") @MemberGetter public static native int EltwiseParameter_EltwiseOp_EltwiseOp_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer EltwiseParameter_EltwiseOp_descriptor();
@Namespace("caffe") public static native @StdString BytePointer EltwiseParameter_EltwiseOp_Name(@Cast("caffe::EltwiseParameter_EltwiseOp") int value);
@Namespace("caffe") public static native @Cast("bool") boolean EltwiseParameter_EltwiseOp_Parse(
    @StdString BytePointer name, @Cast("caffe::EltwiseParameter_EltwiseOp*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean EltwiseParameter_EltwiseOp_Parse(
    @StdString String name, @Cast("caffe::EltwiseParameter_EltwiseOp*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean EltwiseParameter_EltwiseOp_Parse(
    @StdString BytePointer name, @Cast("caffe::EltwiseParameter_EltwiseOp*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean EltwiseParameter_EltwiseOp_Parse(
    @StdString String name, @Cast("caffe::EltwiseParameter_EltwiseOp*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean EltwiseParameter_EltwiseOp_Parse(
    @StdString BytePointer name, @Cast("caffe::EltwiseParameter_EltwiseOp*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean EltwiseParameter_EltwiseOp_Parse(
    @StdString String name, @Cast("caffe::EltwiseParameter_EltwiseOp*") int[] value);
/** enum caffe::HingeLossParameter_Norm */
public static final int
  HingeLossParameter_Norm_L1 = 1,
  HingeLossParameter_Norm_L2 = 2;
@Namespace("caffe") public static native @Cast("bool") boolean HingeLossParameter_Norm_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::HingeLossParameter_Norm") int HingeLossParameter_Norm_Norm_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::HingeLossParameter_Norm") int HingeLossParameter_Norm_Norm_MAX();
@Namespace("caffe") @MemberGetter public static native int HingeLossParameter_Norm_Norm_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer HingeLossParameter_Norm_descriptor();
@Namespace("caffe") public static native @StdString BytePointer HingeLossParameter_Norm_Name(@Cast("caffe::HingeLossParameter_Norm") int value);
@Namespace("caffe") public static native @Cast("bool") boolean HingeLossParameter_Norm_Parse(
    @StdString BytePointer name, @Cast("caffe::HingeLossParameter_Norm*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean HingeLossParameter_Norm_Parse(
    @StdString String name, @Cast("caffe::HingeLossParameter_Norm*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean HingeLossParameter_Norm_Parse(
    @StdString BytePointer name, @Cast("caffe::HingeLossParameter_Norm*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean HingeLossParameter_Norm_Parse(
    @StdString String name, @Cast("caffe::HingeLossParameter_Norm*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean HingeLossParameter_Norm_Parse(
    @StdString BytePointer name, @Cast("caffe::HingeLossParameter_Norm*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean HingeLossParameter_Norm_Parse(
    @StdString String name, @Cast("caffe::HingeLossParameter_Norm*") int[] value);
/** enum caffe::LRNParameter_NormRegion */
public static final int
  LRNParameter_NormRegion_ACROSS_CHANNELS = 0,
  LRNParameter_NormRegion_WITHIN_CHANNEL = 1;
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_NormRegion_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::LRNParameter_NormRegion") int LRNParameter_NormRegion_NormRegion_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::LRNParameter_NormRegion") int LRNParameter_NormRegion_NormRegion_MAX();
@Namespace("caffe") @MemberGetter public static native int LRNParameter_NormRegion_NormRegion_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer LRNParameter_NormRegion_descriptor();
@Namespace("caffe") public static native @StdString BytePointer LRNParameter_NormRegion_Name(@Cast("caffe::LRNParameter_NormRegion") int value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_NormRegion_Parse(
    @StdString BytePointer name, @Cast("caffe::LRNParameter_NormRegion*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_NormRegion_Parse(
    @StdString String name, @Cast("caffe::LRNParameter_NormRegion*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_NormRegion_Parse(
    @StdString BytePointer name, @Cast("caffe::LRNParameter_NormRegion*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_NormRegion_Parse(
    @StdString String name, @Cast("caffe::LRNParameter_NormRegion*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_NormRegion_Parse(
    @StdString BytePointer name, @Cast("caffe::LRNParameter_NormRegion*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_NormRegion_Parse(
    @StdString String name, @Cast("caffe::LRNParameter_NormRegion*") int[] value);
/** enum caffe::LRNParameter_Engine */
public static final int
  LRNParameter_Engine_DEFAULT = 0,
  LRNParameter_Engine_CAFFE = 1,
  LRNParameter_Engine_CUDNN = 2;
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_Engine_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::LRNParameter_Engine") int LRNParameter_Engine_Engine_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::LRNParameter_Engine") int LRNParameter_Engine_Engine_MAX();
@Namespace("caffe") @MemberGetter public static native int LRNParameter_Engine_Engine_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer LRNParameter_Engine_descriptor();
@Namespace("caffe") public static native @StdString BytePointer LRNParameter_Engine_Name(@Cast("caffe::LRNParameter_Engine") int value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::LRNParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::LRNParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::LRNParameter_Engine*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::LRNParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::LRNParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean LRNParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::LRNParameter_Engine*") int[] value);
/** enum caffe::PoolingParameter_PoolMethod */
public static final int
  PoolingParameter_PoolMethod_MAX = 0,
  PoolingParameter_PoolMethod_AVE = 1,
  PoolingParameter_PoolMethod_STOCHASTIC = 2;
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_PoolMethod_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::PoolingParameter_PoolMethod") int PoolingParameter_PoolMethod_PoolMethod_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::PoolingParameter_PoolMethod") int PoolingParameter_PoolMethod_PoolMethod_MAX();
@Namespace("caffe") @MemberGetter public static native int PoolingParameter_PoolMethod_PoolMethod_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer PoolingParameter_PoolMethod_descriptor();
@Namespace("caffe") public static native @StdString BytePointer PoolingParameter_PoolMethod_Name(@Cast("caffe::PoolingParameter_PoolMethod") int value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::PoolingParameter_PoolMethod*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::PoolingParameter_PoolMethod*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::PoolingParameter_PoolMethod*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::PoolingParameter_PoolMethod*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::PoolingParameter_PoolMethod*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::PoolingParameter_PoolMethod*") int[] value);
/** enum caffe::PoolingParameter_Engine */
public static final int
  PoolingParameter_Engine_DEFAULT = 0,
  PoolingParameter_Engine_CAFFE = 1,
  PoolingParameter_Engine_CUDNN = 2;
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_Engine_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::PoolingParameter_Engine") int PoolingParameter_Engine_Engine_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::PoolingParameter_Engine") int PoolingParameter_Engine_Engine_MAX();
@Namespace("caffe") @MemberGetter public static native int PoolingParameter_Engine_Engine_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer PoolingParameter_Engine_descriptor();
@Namespace("caffe") public static native @StdString BytePointer PoolingParameter_Engine_Name(@Cast("caffe::PoolingParameter_Engine") int value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::PoolingParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::PoolingParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::PoolingParameter_Engine*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::PoolingParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::PoolingParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean PoolingParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::PoolingParameter_Engine*") int[] value);
/** enum caffe::ReductionParameter_ReductionOp */
public static final int
  ReductionParameter_ReductionOp_SUM = 1,
  ReductionParameter_ReductionOp_ASUM = 2,
  ReductionParameter_ReductionOp_SUMSQ = 3,
  ReductionParameter_ReductionOp_MEAN = 4;
@Namespace("caffe") public static native @Cast("bool") boolean ReductionParameter_ReductionOp_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::ReductionParameter_ReductionOp") int ReductionParameter_ReductionOp_ReductionOp_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::ReductionParameter_ReductionOp") int ReductionParameter_ReductionOp_ReductionOp_MAX();
@Namespace("caffe") @MemberGetter public static native int ReductionParameter_ReductionOp_ReductionOp_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer ReductionParameter_ReductionOp_descriptor();
@Namespace("caffe") public static native @StdString BytePointer ReductionParameter_ReductionOp_Name(@Cast("caffe::ReductionParameter_ReductionOp") int value);
@Namespace("caffe") public static native @Cast("bool") boolean ReductionParameter_ReductionOp_Parse(
    @StdString BytePointer name, @Cast("caffe::ReductionParameter_ReductionOp*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean ReductionParameter_ReductionOp_Parse(
    @StdString String name, @Cast("caffe::ReductionParameter_ReductionOp*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean ReductionParameter_ReductionOp_Parse(
    @StdString BytePointer name, @Cast("caffe::ReductionParameter_ReductionOp*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean ReductionParameter_ReductionOp_Parse(
    @StdString String name, @Cast("caffe::ReductionParameter_ReductionOp*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean ReductionParameter_ReductionOp_Parse(
    @StdString BytePointer name, @Cast("caffe::ReductionParameter_ReductionOp*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean ReductionParameter_ReductionOp_Parse(
    @StdString String name, @Cast("caffe::ReductionParameter_ReductionOp*") int[] value);
/** enum caffe::ReLUParameter_Engine */
public static final int
  ReLUParameter_Engine_DEFAULT = 0,
  ReLUParameter_Engine_CAFFE = 1,
  ReLUParameter_Engine_CUDNN = 2;
@Namespace("caffe") public static native @Cast("bool") boolean ReLUParameter_Engine_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::ReLUParameter_Engine") int ReLUParameter_Engine_Engine_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::ReLUParameter_Engine") int ReLUParameter_Engine_Engine_MAX();
@Namespace("caffe") @MemberGetter public static native int ReLUParameter_Engine_Engine_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer ReLUParameter_Engine_descriptor();
@Namespace("caffe") public static native @StdString BytePointer ReLUParameter_Engine_Name(@Cast("caffe::ReLUParameter_Engine") int value);
@Namespace("caffe") public static native @Cast("bool") boolean ReLUParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::ReLUParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean ReLUParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::ReLUParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean ReLUParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::ReLUParameter_Engine*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean ReLUParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::ReLUParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean ReLUParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::ReLUParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean ReLUParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::ReLUParameter_Engine*") int[] value);
/** enum caffe::SigmoidParameter_Engine */
public static final int
  SigmoidParameter_Engine_DEFAULT = 0,
  SigmoidParameter_Engine_CAFFE = 1,
  SigmoidParameter_Engine_CUDNN = 2;
@Namespace("caffe") public static native @Cast("bool") boolean SigmoidParameter_Engine_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SigmoidParameter_Engine") int SigmoidParameter_Engine_Engine_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SigmoidParameter_Engine") int SigmoidParameter_Engine_Engine_MAX();
@Namespace("caffe") @MemberGetter public static native int SigmoidParameter_Engine_Engine_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SigmoidParameter_Engine_descriptor();
@Namespace("caffe") public static native @StdString BytePointer SigmoidParameter_Engine_Name(@Cast("caffe::SigmoidParameter_Engine") int value);
@Namespace("caffe") public static native @Cast("bool") boolean SigmoidParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SigmoidParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SigmoidParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SigmoidParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SigmoidParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SigmoidParameter_Engine*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean SigmoidParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SigmoidParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SigmoidParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SigmoidParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SigmoidParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SigmoidParameter_Engine*") int[] value);
/** enum caffe::SoftmaxParameter_Engine */
public static final int
  SoftmaxParameter_Engine_DEFAULT = 0,
  SoftmaxParameter_Engine_CAFFE = 1,
  SoftmaxParameter_Engine_CUDNN = 2;
@Namespace("caffe") public static native @Cast("bool") boolean SoftmaxParameter_Engine_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SoftmaxParameter_Engine") int SoftmaxParameter_Engine_Engine_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SoftmaxParameter_Engine") int SoftmaxParameter_Engine_Engine_MAX();
@Namespace("caffe") @MemberGetter public static native int SoftmaxParameter_Engine_Engine_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SoftmaxParameter_Engine_descriptor();
@Namespace("caffe") public static native @StdString BytePointer SoftmaxParameter_Engine_Name(@Cast("caffe::SoftmaxParameter_Engine") int value);
@Namespace("caffe") public static native @Cast("bool") boolean SoftmaxParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SoftmaxParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SoftmaxParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SoftmaxParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SoftmaxParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SoftmaxParameter_Engine*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean SoftmaxParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SoftmaxParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SoftmaxParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SoftmaxParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SoftmaxParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SoftmaxParameter_Engine*") int[] value);
/** enum caffe::TanHParameter_Engine */
public static final int
  TanHParameter_Engine_DEFAULT = 0,
  TanHParameter_Engine_CAFFE = 1,
  TanHParameter_Engine_CUDNN = 2;
@Namespace("caffe") public static native @Cast("bool") boolean TanHParameter_Engine_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::TanHParameter_Engine") int TanHParameter_Engine_Engine_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::TanHParameter_Engine") int TanHParameter_Engine_Engine_MAX();
@Namespace("caffe") @MemberGetter public static native int TanHParameter_Engine_Engine_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer TanHParameter_Engine_descriptor();
@Namespace("caffe") public static native @StdString BytePointer TanHParameter_Engine_Name(@Cast("caffe::TanHParameter_Engine") int value);
@Namespace("caffe") public static native @Cast("bool") boolean TanHParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::TanHParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean TanHParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::TanHParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean TanHParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::TanHParameter_Engine*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean TanHParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::TanHParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean TanHParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::TanHParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean TanHParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::TanHParameter_Engine*") int[] value);
/** enum caffe::SPPParameter_PoolMethod */
public static final int
  SPPParameter_PoolMethod_MAX = 0,
  SPPParameter_PoolMethod_AVE = 1,
  SPPParameter_PoolMethod_STOCHASTIC = 2;
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_PoolMethod_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SPPParameter_PoolMethod") int SPPParameter_PoolMethod_PoolMethod_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SPPParameter_PoolMethod") int SPPParameter_PoolMethod_PoolMethod_MAX();
@Namespace("caffe") @MemberGetter public static native int SPPParameter_PoolMethod_PoolMethod_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SPPParameter_PoolMethod_descriptor();
@Namespace("caffe") public static native @StdString BytePointer SPPParameter_PoolMethod_Name(@Cast("caffe::SPPParameter_PoolMethod") int value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::SPPParameter_PoolMethod*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::SPPParameter_PoolMethod*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::SPPParameter_PoolMethod*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::SPPParameter_PoolMethod*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::SPPParameter_PoolMethod*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::SPPParameter_PoolMethod*") int[] value);
/** enum caffe::SPPParameter_Engine */
public static final int
  SPPParameter_Engine_DEFAULT = 0,
  SPPParameter_Engine_CAFFE = 1,
  SPPParameter_Engine_CUDNN = 2;
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_Engine_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SPPParameter_Engine") int SPPParameter_Engine_Engine_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::SPPParameter_Engine") int SPPParameter_Engine_Engine_MAX();
@Namespace("caffe") @MemberGetter public static native int SPPParameter_Engine_Engine_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SPPParameter_Engine_descriptor();
@Namespace("caffe") public static native @StdString BytePointer SPPParameter_Engine_Name(@Cast("caffe::SPPParameter_Engine") int value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SPPParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SPPParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SPPParameter_Engine*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SPPParameter_Engine*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_Engine_Parse(
    @StdString BytePointer name, @Cast("caffe::SPPParameter_Engine*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean SPPParameter_Engine_Parse(
    @StdString String name, @Cast("caffe::SPPParameter_Engine*") int[] value);
/** enum caffe::V1LayerParameter_LayerType */
public static final int
  V1LayerParameter_LayerType_NONE = 0,
  V1LayerParameter_LayerType_ABSVAL = 35,
  V1LayerParameter_LayerType_ACCURACY = 1,
  V1LayerParameter_LayerType_ARGMAX = 30,
  V1LayerParameter_LayerType_BNLL = 2,
  V1LayerParameter_LayerType_CONCAT = 3,
  V1LayerParameter_LayerType_CONTRASTIVE_LOSS = 37,
  V1LayerParameter_LayerType_CONVOLUTION = 4,
  V1LayerParameter_LayerType_DATA = 5,
  V1LayerParameter_LayerType_DECONVOLUTION = 39,
  V1LayerParameter_LayerType_DROPOUT = 6,
  V1LayerParameter_LayerType_DUMMY_DATA = 32,
  V1LayerParameter_LayerType_EUCLIDEAN_LOSS = 7,
  V1LayerParameter_LayerType_ELTWISE = 25,
  V1LayerParameter_LayerType_EXP = 38,
  V1LayerParameter_LayerType_FLATTEN = 8,
  V1LayerParameter_LayerType_HDF5_DATA = 9,
  V1LayerParameter_LayerType_HDF5_OUTPUT = 10,
  V1LayerParameter_LayerType_HINGE_LOSS = 28,
  V1LayerParameter_LayerType_IM2COL = 11,
  V1LayerParameter_LayerType_IMAGE_DATA = 12,
  V1LayerParameter_LayerType_INFOGAIN_LOSS = 13,
  V1LayerParameter_LayerType_INNER_PRODUCT = 14,
  V1LayerParameter_LayerType_LRN = 15,
  V1LayerParameter_LayerType_MEMORY_DATA = 29,
  V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS = 16,
  V1LayerParameter_LayerType_MVN = 34,
  V1LayerParameter_LayerType_POOLING = 17,
  V1LayerParameter_LayerType_POWER = 26,
  V1LayerParameter_LayerType_RELU = 18,
  V1LayerParameter_LayerType_SIGMOID = 19,
  V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS = 27,
  V1LayerParameter_LayerType_SILENCE = 36,
  V1LayerParameter_LayerType_SOFTMAX = 20,
  V1LayerParameter_LayerType_SOFTMAX_LOSS = 21,
  V1LayerParameter_LayerType_SPLIT = 22,
  V1LayerParameter_LayerType_SLICE = 33,
  V1LayerParameter_LayerType_TANH = 23,
  V1LayerParameter_LayerType_WINDOW_DATA = 24,
  V1LayerParameter_LayerType_THRESHOLD = 31;
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_LayerType_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::V1LayerParameter_LayerType") int V1LayerParameter_LayerType_LayerType_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::V1LayerParameter_LayerType") int V1LayerParameter_LayerType_LayerType_MAX();
@Namespace("caffe") @MemberGetter public static native int V1LayerParameter_LayerType_LayerType_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer V1LayerParameter_LayerType_descriptor();
@Namespace("caffe") public static native @StdString BytePointer V1LayerParameter_LayerType_Name(@Cast("caffe::V1LayerParameter_LayerType") int value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_LayerType_Parse(
    @StdString BytePointer name, @Cast("caffe::V1LayerParameter_LayerType*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_LayerType_Parse(
    @StdString String name, @Cast("caffe::V1LayerParameter_LayerType*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_LayerType_Parse(
    @StdString BytePointer name, @Cast("caffe::V1LayerParameter_LayerType*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_LayerType_Parse(
    @StdString String name, @Cast("caffe::V1LayerParameter_LayerType*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_LayerType_Parse(
    @StdString BytePointer name, @Cast("caffe::V1LayerParameter_LayerType*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_LayerType_Parse(
    @StdString String name, @Cast("caffe::V1LayerParameter_LayerType*") int[] value);
/** enum caffe::V1LayerParameter_DimCheckMode */
public static final int
  V1LayerParameter_DimCheckMode_STRICT = 0,
  V1LayerParameter_DimCheckMode_PERMISSIVE = 1;
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_DimCheckMode_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::V1LayerParameter_DimCheckMode") int V1LayerParameter_DimCheckMode_DimCheckMode_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::V1LayerParameter_DimCheckMode") int V1LayerParameter_DimCheckMode_DimCheckMode_MAX();
@Namespace("caffe") @MemberGetter public static native int V1LayerParameter_DimCheckMode_DimCheckMode_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer V1LayerParameter_DimCheckMode_descriptor();
@Namespace("caffe") public static native @StdString BytePointer V1LayerParameter_DimCheckMode_Name(@Cast("caffe::V1LayerParameter_DimCheckMode") int value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_DimCheckMode_Parse(
    @StdString BytePointer name, @Cast("caffe::V1LayerParameter_DimCheckMode*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_DimCheckMode_Parse(
    @StdString String name, @Cast("caffe::V1LayerParameter_DimCheckMode*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_DimCheckMode_Parse(
    @StdString BytePointer name, @Cast("caffe::V1LayerParameter_DimCheckMode*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_DimCheckMode_Parse(
    @StdString String name, @Cast("caffe::V1LayerParameter_DimCheckMode*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_DimCheckMode_Parse(
    @StdString BytePointer name, @Cast("caffe::V1LayerParameter_DimCheckMode*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean V1LayerParameter_DimCheckMode_Parse(
    @StdString String name, @Cast("caffe::V1LayerParameter_DimCheckMode*") int[] value);
/** enum caffe::V0LayerParameter_PoolMethod */
public static final int
  V0LayerParameter_PoolMethod_MAX = 0,
  V0LayerParameter_PoolMethod_AVE = 1,
  V0LayerParameter_PoolMethod_STOCHASTIC = 2;
@Namespace("caffe") public static native @Cast("bool") boolean V0LayerParameter_PoolMethod_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::V0LayerParameter_PoolMethod") int V0LayerParameter_PoolMethod_PoolMethod_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::V0LayerParameter_PoolMethod") int V0LayerParameter_PoolMethod_PoolMethod_MAX();
@Namespace("caffe") @MemberGetter public static native int V0LayerParameter_PoolMethod_PoolMethod_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer V0LayerParameter_PoolMethod_descriptor();
@Namespace("caffe") public static native @StdString BytePointer V0LayerParameter_PoolMethod_Name(@Cast("caffe::V0LayerParameter_PoolMethod") int value);
@Namespace("caffe") public static native @Cast("bool") boolean V0LayerParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::V0LayerParameter_PoolMethod*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean V0LayerParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::V0LayerParameter_PoolMethod*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean V0LayerParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::V0LayerParameter_PoolMethod*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean V0LayerParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::V0LayerParameter_PoolMethod*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean V0LayerParameter_PoolMethod_Parse(
    @StdString BytePointer name, @Cast("caffe::V0LayerParameter_PoolMethod*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean V0LayerParameter_PoolMethod_Parse(
    @StdString String name, @Cast("caffe::V0LayerParameter_PoolMethod*") int[] value);
/** enum caffe::Phase */
public static final int
  TRAIN = 0,
  TEST = 1;
@Namespace("caffe") public static native @Cast("bool") boolean Phase_IsValid(int value);
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::Phase") int Phase_MIN();
@Namespace("caffe") @MemberGetter public static native @Cast("const caffe::Phase") int Phase_MAX();
@Namespace("caffe") @MemberGetter public static native int Phase_ARRAYSIZE();

@Namespace("caffe") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Phase_descriptor();
@Namespace("caffe") public static native @StdString BytePointer Phase_Name(@Cast("caffe::Phase") int value);
@Namespace("caffe") public static native @Cast("bool") boolean Phase_Parse(
    @StdString BytePointer name, @Cast("caffe::Phase*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean Phase_Parse(
    @StdString String name, @Cast("caffe::Phase*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean Phase_Parse(
    @StdString BytePointer name, @Cast("caffe::Phase*") int[] value);
@Namespace("caffe") public static native @Cast("bool") boolean Phase_Parse(
    @StdString String name, @Cast("caffe::Phase*") IntPointer value);
@Namespace("caffe") public static native @Cast("bool") boolean Phase_Parse(
    @StdString BytePointer name, @Cast("caffe::Phase*") IntBuffer value);
@Namespace("caffe") public static native @Cast("bool") boolean Phase_Parse(
    @StdString String name, @Cast("caffe::Phase*") int[] value);
// ===================================================================

@Namespace("caffe") @NoOffset public static class BlobShape extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BlobShape(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BlobShape(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BlobShape position(int position) {
        return (BlobShape)super.position(position);
    }

  public BlobShape() { super((Pointer)null); allocate(); }
  private native void allocate();

  public BlobShape(@Const @ByRef BlobShape from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef BlobShape from);

  public native @ByRef @Name("operator =") BlobShape put(@Const @ByRef BlobShape from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef BlobShape default_instance();

  public native void Swap(BlobShape other);

  // implements Message ----------------------------------------------

  public native BlobShape New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef BlobShape from);
  public native void MergeFrom(@Const @ByRef BlobShape from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated int64 dim = 1 [packed = true];
  public native int dim_size();
  public native void clear_dim();
  @MemberGetter public static native int kDimFieldNumber();
  public static final int kDimFieldNumber = kDimFieldNumber();
  public native @Cast("google::protobuf::int64") long dim(int index);
  public native void set_dim(int index, @Cast("google::protobuf::int64") long value);
  public native void add_dim(@Cast("google::protobuf::int64") long value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class BlobProto extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BlobProto(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BlobProto(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BlobProto position(int position) {
        return (BlobProto)super.position(position);
    }

  public BlobProto() { super((Pointer)null); allocate(); }
  private native void allocate();

  public BlobProto(@Const @ByRef BlobProto from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef BlobProto from);

  public native @ByRef @Name("operator =") BlobProto put(@Const @ByRef BlobProto from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef BlobProto default_instance();

  public native void Swap(BlobProto other);

  // implements Message ----------------------------------------------

  public native BlobProto New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef BlobProto from);
  public native void MergeFrom(@Const @ByRef BlobProto from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional .caffe.BlobShape shape = 7;
  public native @Cast("bool") boolean has_shape();
  public native void clear_shape();
  @MemberGetter public static native int kShapeFieldNumber();
  public static final int kShapeFieldNumber = kShapeFieldNumber();
  public native @Const @ByRef BlobShape shape();
  public native BlobShape mutable_shape();
  public native BlobShape release_shape();
  public native void set_allocated_shape(BlobShape shape);

  // repeated float data = 5 [packed = true];
  public native int data_size();
  public native void clear_data();
  @MemberGetter public static native int kDataFieldNumber();
  public static final int kDataFieldNumber = kDataFieldNumber();
  public native float data(int index);
  public native void set_data(int index, float value);
  public native void add_data(float value);

  // repeated float diff = 6 [packed = true];
  public native int diff_size();
  public native void clear_diff();
  @MemberGetter public static native int kDiffFieldNumber();
  public static final int kDiffFieldNumber = kDiffFieldNumber();
  public native float diff(int index);
  public native void set_diff(int index, float value);
  public native void add_diff(float value);

  // repeated double double_data = 8 [packed = true];
  public native int double_data_size();
  public native void clear_double_data();
  @MemberGetter public static native int kDoubleDataFieldNumber();
  public static final int kDoubleDataFieldNumber = kDoubleDataFieldNumber();
  public native double double_data(int index);
  public native void set_double_data(int index, double value);
  public native void add_double_data(double value);

  // repeated double double_diff = 9 [packed = true];
  public native int double_diff_size();
  public native void clear_double_diff();
  @MemberGetter public static native int kDoubleDiffFieldNumber();
  public static final int kDoubleDiffFieldNumber = kDoubleDiffFieldNumber();
  public native double double_diff(int index);
  public native void set_double_diff(int index, double value);
  public native void add_double_diff(double value);

  // optional int32 num = 1 [default = 0];
  public native @Cast("bool") boolean has_num();
  public native void clear_num();
  @MemberGetter public static native int kNumFieldNumber();
  public static final int kNumFieldNumber = kNumFieldNumber();
  public native @Cast("google::protobuf::int32") int num();
  public native void set_num(@Cast("google::protobuf::int32") int value);

  // optional int32 channels = 2 [default = 0];
  public native @Cast("bool") boolean has_channels();
  public native void clear_channels();
  @MemberGetter public static native int kChannelsFieldNumber();
  public static final int kChannelsFieldNumber = kChannelsFieldNumber();
  public native @Cast("google::protobuf::int32") int channels();
  public native void set_channels(@Cast("google::protobuf::int32") int value);

  // optional int32 height = 3 [default = 0];
  public native @Cast("bool") boolean has_height();
  public native void clear_height();
  @MemberGetter public static native int kHeightFieldNumber();
  public static final int kHeightFieldNumber = kHeightFieldNumber();
  public native @Cast("google::protobuf::int32") int height();
  public native void set_height(@Cast("google::protobuf::int32") int value);

  // optional int32 width = 4 [default = 0];
  public native @Cast("bool") boolean has_width();
  public native void clear_width();
  @MemberGetter public static native int kWidthFieldNumber();
  public static final int kWidthFieldNumber = kWidthFieldNumber();
  public native @Cast("google::protobuf::int32") int width();
  public native void set_width(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class BlobProtoVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BlobProtoVector(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BlobProtoVector(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BlobProtoVector position(int position) {
        return (BlobProtoVector)super.position(position);
    }

  public BlobProtoVector() { super((Pointer)null); allocate(); }
  private native void allocate();

  public BlobProtoVector(@Const @ByRef BlobProtoVector from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef BlobProtoVector from);

  public native @ByRef @Name("operator =") BlobProtoVector put(@Const @ByRef BlobProtoVector from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef BlobProtoVector default_instance();

  public native void Swap(BlobProtoVector other);

  // implements Message ----------------------------------------------

  public native BlobProtoVector New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef BlobProtoVector from);
  public native void MergeFrom(@Const @ByRef BlobProtoVector from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .caffe.BlobProto blobs = 1;
  public native int blobs_size();
  public native void clear_blobs();
  @MemberGetter public static native int kBlobsFieldNumber();
  public static final int kBlobsFieldNumber = kBlobsFieldNumber();
  public native @Const @ByRef BlobProto blobs(int index);
  public native BlobProto mutable_blobs(int index);
  public native BlobProto add_blobs();
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class Datum extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Datum(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Datum(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public Datum position(int position) {
        return (Datum)super.position(position);
    }

  public Datum() { super((Pointer)null); allocate(); }
  private native void allocate();

  public Datum(@Const @ByRef Datum from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef Datum from);

  public native @ByRef @Name("operator =") Datum put(@Const @ByRef Datum from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef Datum default_instance();

  public native void Swap(Datum other);

  // implements Message ----------------------------------------------

  public native Datum New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef Datum from);
  public native void MergeFrom(@Const @ByRef Datum from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional int32 channels = 1;
  public native @Cast("bool") boolean has_channels();
  public native void clear_channels();
  @MemberGetter public static native int kChannelsFieldNumber();
  public static final int kChannelsFieldNumber = kChannelsFieldNumber();
  public native @Cast("google::protobuf::int32") int channels();
  public native void set_channels(@Cast("google::protobuf::int32") int value);

  // optional int32 height = 2;
  public native @Cast("bool") boolean has_height();
  public native void clear_height();
  @MemberGetter public static native int kHeightFieldNumber();
  public static final int kHeightFieldNumber = kHeightFieldNumber();
  public native @Cast("google::protobuf::int32") int height();
  public native void set_height(@Cast("google::protobuf::int32") int value);

  // optional int32 width = 3;
  public native @Cast("bool") boolean has_width();
  public native void clear_width();
  @MemberGetter public static native int kWidthFieldNumber();
  public static final int kWidthFieldNumber = kWidthFieldNumber();
  public native @Cast("google::protobuf::int32") int width();
  public native void set_width(@Cast("google::protobuf::int32") int value);

  // optional bytes data = 4;
  public native @Cast("bool") boolean has_data();
  public native void clear_data();
  @MemberGetter public static native int kDataFieldNumber();
  public static final int kDataFieldNumber = kDataFieldNumber();
  public native @StdString BytePointer data();
  public native void set_data(@StdString BytePointer value);
  public native void set_data(@StdString String value);
  public native void set_data(@Const Pointer value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_data();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_data();
  public native void set_allocated_data(@StdString @Cast({"char*", "std::string*"}) BytePointer data);

  // optional int32 label = 5;
  public native @Cast("bool") boolean has_label();
  public native void clear_label();
  @MemberGetter public static native int kLabelFieldNumber();
  public static final int kLabelFieldNumber = kLabelFieldNumber();
  public native @Cast("google::protobuf::int32") int label();
  public native void set_label(@Cast("google::protobuf::int32") int value);

  // repeated float float_data = 6;
  public native int float_data_size();
  public native void clear_float_data();
  @MemberGetter public static native int kFloatDataFieldNumber();
  public static final int kFloatDataFieldNumber = kFloatDataFieldNumber();
  public native float float_data(int index);
  public native void set_float_data(int index, float value);
  public native void add_float_data(float value);

  // optional bool encoded = 7 [default = false];
  public native @Cast("bool") boolean has_encoded();
  public native void clear_encoded();
  @MemberGetter public static native int kEncodedFieldNumber();
  public static final int kEncodedFieldNumber = kEncodedFieldNumber();
  public native @Cast("bool") boolean encoded();
  public native void set_encoded(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class FillerParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FillerParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FillerParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FillerParameter position(int position) {
        return (FillerParameter)super.position(position);
    }

  public FillerParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public FillerParameter(@Const @ByRef FillerParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef FillerParameter from);

  public native @ByRef @Name("operator =") FillerParameter put(@Const @ByRef FillerParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef FillerParameter default_instance();

  public native void Swap(FillerParameter other);

  // implements Message ----------------------------------------------

  public native FillerParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef FillerParameter from);
  public native void MergeFrom(@Const @ByRef FillerParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::FillerParameter::VarianceNorm") int FAN_IN();
  public static final int FAN_IN = FAN_IN();
  @MemberGetter public static native @Cast("const caffe::FillerParameter::VarianceNorm") int FAN_OUT();
  public static final int FAN_OUT = FAN_OUT();
  @MemberGetter public static native @Cast("const caffe::FillerParameter::VarianceNorm") int AVERAGE();
  public static final int AVERAGE = AVERAGE();
  public static native @Cast("bool") boolean VarianceNorm_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::FillerParameter::VarianceNorm") int VarianceNorm_MIN();
  public static final int VarianceNorm_MIN = VarianceNorm_MIN();
  @MemberGetter public static native @Cast("const caffe::FillerParameter::VarianceNorm") int VarianceNorm_MAX();
  public static final int VarianceNorm_MAX = VarianceNorm_MAX();
  @MemberGetter public static native int VarianceNorm_ARRAYSIZE();
  public static final int VarianceNorm_ARRAYSIZE = VarianceNorm_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer VarianceNorm_descriptor();
  public static native @StdString BytePointer VarianceNorm_Name(@Cast("caffe::FillerParameter::VarianceNorm") int value);
  public static native @Cast("bool") boolean VarianceNorm_Parse(@StdString BytePointer name,
        @Cast("caffe::FillerParameter::VarianceNorm*") IntPointer value);
  public static native @Cast("bool") boolean VarianceNorm_Parse(@StdString String name,
        @Cast("caffe::FillerParameter::VarianceNorm*") IntBuffer value);
  public static native @Cast("bool") boolean VarianceNorm_Parse(@StdString BytePointer name,
        @Cast("caffe::FillerParameter::VarianceNorm*") int[] value);
  public static native @Cast("bool") boolean VarianceNorm_Parse(@StdString String name,
        @Cast("caffe::FillerParameter::VarianceNorm*") IntPointer value);
  public static native @Cast("bool") boolean VarianceNorm_Parse(@StdString BytePointer name,
        @Cast("caffe::FillerParameter::VarianceNorm*") IntBuffer value);
  public static native @Cast("bool") boolean VarianceNorm_Parse(@StdString String name,
        @Cast("caffe::FillerParameter::VarianceNorm*") int[] value);

  // accessors -------------------------------------------------------

  // optional string type = 1 [default = "constant"];
  public native @Cast("bool") boolean has_type();
  public native void clear_type();
  @MemberGetter public static native int kTypeFieldNumber();
  public static final int kTypeFieldNumber = kTypeFieldNumber();
  public native @StdString BytePointer type();
  public native void set_type(@StdString BytePointer value);
  public native void set_type(@StdString String value);
  public native void set_type(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_type(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_type();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_type();
  public native void set_allocated_type(@StdString @Cast({"char*", "std::string*"}) BytePointer type);

  // optional float value = 2 [default = 0];
  public native @Cast("bool") boolean has_value();
  public native void clear_value();
  @MemberGetter public static native int kValueFieldNumber();
  public static final int kValueFieldNumber = kValueFieldNumber();
  public native float value();
  public native void set_value(float value);

  // optional float min = 3 [default = 0];
  public native @Cast("bool") boolean has_min();
  public native void clear_min();
  @MemberGetter public static native int kMinFieldNumber();
  public static final int kMinFieldNumber = kMinFieldNumber();
  public native float min();
  public native void set_min(float value);

  // optional float max = 4 [default = 1];
  public native @Cast("bool") boolean has_max();
  public native void clear_max();
  @MemberGetter public static native int kMaxFieldNumber();
  public static final int kMaxFieldNumber = kMaxFieldNumber();
  public native float max();
  public native void set_max(float value);

  // optional float mean = 5 [default = 0];
  public native @Cast("bool") boolean has_mean();
  public native void clear_mean();
  @MemberGetter public static native int kMeanFieldNumber();
  public static final int kMeanFieldNumber = kMeanFieldNumber();
  public native float mean();
  public native void set_mean(float value);

  // optional float std = 6 [default = 1];
  public native @Cast("bool") boolean has_std();
  public native void clear_std();
  @MemberGetter public static native int kStdFieldNumber();
  public static final int kStdFieldNumber = kStdFieldNumber();
  public native float std();
  public native void set_std(float value);

  // optional int32 sparse = 7 [default = -1];
  public native @Cast("bool") boolean has_sparse();
  public native void clear_sparse();
  @MemberGetter public static native int kSparseFieldNumber();
  public static final int kSparseFieldNumber = kSparseFieldNumber();
  public native @Cast("google::protobuf::int32") int sparse();
  public native void set_sparse(@Cast("google::protobuf::int32") int value);

  // optional .caffe.FillerParameter.VarianceNorm variance_norm = 8 [default = FAN_IN];
  public native @Cast("bool") boolean has_variance_norm();
  public native void clear_variance_norm();
  @MemberGetter public static native int kVarianceNormFieldNumber();
  public static final int kVarianceNormFieldNumber = kVarianceNormFieldNumber();
  public native @Cast("caffe::FillerParameter_VarianceNorm") int variance_norm();
  public native void set_variance_norm(@Cast("caffe::FillerParameter_VarianceNorm") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class NetParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NetParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NetParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public NetParameter position(int position) {
        return (NetParameter)super.position(position);
    }

  public NetParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public NetParameter(@Const @ByRef NetParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef NetParameter from);

  public native @ByRef @Name("operator =") NetParameter put(@Const @ByRef NetParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef NetParameter default_instance();

  public native void Swap(NetParameter other);

  // implements Message ----------------------------------------------

  public native NetParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef NetParameter from);
  public native void MergeFrom(@Const @ByRef NetParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string name = 1;
  public native @Cast("bool") boolean has_name();
  public native void clear_name();
  @MemberGetter public static native int kNameFieldNumber();
  public static final int kNameFieldNumber = kNameFieldNumber();
  public native @StdString BytePointer name();
  public native void set_name(@StdString BytePointer value);
  public native void set_name(@StdString String value);
  public native void set_name(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_name(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_name();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_name();
  public native void set_allocated_name(@StdString @Cast({"char*", "std::string*"}) BytePointer name);

  // repeated string input = 3;
  public native int input_size();
  public native void clear_input();
  @MemberGetter public static native int kInputFieldNumber();
  public static final int kInputFieldNumber = kInputFieldNumber();
  public native @StdString BytePointer input(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_input(int index);
  public native void set_input(int index, @StdString BytePointer value);
  public native void set_input(int index, @StdString String value);
  public native void set_input(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_input(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_input();
  public native void add_input(@StdString BytePointer value);
  public native void add_input(@StdString String value);
  public native void add_input(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_input(String value, @Cast("size_t") long size);

  // repeated .caffe.BlobShape input_shape = 8;
  public native int input_shape_size();
  public native void clear_input_shape();
  @MemberGetter public static native int kInputShapeFieldNumber();
  public static final int kInputShapeFieldNumber = kInputShapeFieldNumber();
  public native @Const @ByRef BlobShape input_shape(int index);
  public native BlobShape mutable_input_shape(int index);
  public native BlobShape add_input_shape();

  // repeated int32 input_dim = 4;
  public native int input_dim_size();
  public native void clear_input_dim();
  @MemberGetter public static native int kInputDimFieldNumber();
  public static final int kInputDimFieldNumber = kInputDimFieldNumber();
  public native @Cast("google::protobuf::int32") int input_dim(int index);
  public native void set_input_dim(int index, @Cast("google::protobuf::int32") int value);
  public native void add_input_dim(@Cast("google::protobuf::int32") int value);

  // optional bool force_backward = 5 [default = false];
  public native @Cast("bool") boolean has_force_backward();
  public native void clear_force_backward();
  @MemberGetter public static native int kForceBackwardFieldNumber();
  public static final int kForceBackwardFieldNumber = kForceBackwardFieldNumber();
  public native @Cast("bool") boolean force_backward();
  public native void set_force_backward(@Cast("bool") boolean value);

  // optional .caffe.NetState state = 6;
  public native @Cast("bool") boolean has_state();
  public native void clear_state();
  @MemberGetter public static native int kStateFieldNumber();
  public static final int kStateFieldNumber = kStateFieldNumber();
  public native @Const @ByRef NetState state();
  public native NetState mutable_state();
  public native NetState release_state();
  public native void set_allocated_state(NetState state);

  // optional bool debug_info = 7 [default = false];
  public native @Cast("bool") boolean has_debug_info();
  public native void clear_debug_info();
  @MemberGetter public static native int kDebugInfoFieldNumber();
  public static final int kDebugInfoFieldNumber = kDebugInfoFieldNumber();
  public native @Cast("bool") boolean debug_info();
  public native void set_debug_info(@Cast("bool") boolean value);

  // repeated .caffe.LayerParameter layer = 100;
  public native int layer_size();
  public native void clear_layer();
  @MemberGetter public static native int kLayerFieldNumber();
  public static final int kLayerFieldNumber = kLayerFieldNumber();
  public native @Const @ByRef LayerParameter layer(int index);
  public native LayerParameter mutable_layer(int index);
  public native LayerParameter add_layer();

  // repeated .caffe.V1LayerParameter layers = 2;
  public native int layers_size();
  public native void clear_layers();
  @MemberGetter public static native int kLayersFieldNumber();
  public static final int kLayersFieldNumber = kLayersFieldNumber();
  public native @Const @ByRef V1LayerParameter layers(int index);
  public native V1LayerParameter mutable_layers(int index);
  public native V1LayerParameter add_layers();
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class SolverParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SolverParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SolverParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SolverParameter position(int position) {
        return (SolverParameter)super.position(position);
    }

  public SolverParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public SolverParameter(@Const @ByRef SolverParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef SolverParameter from);

  public native @ByRef @Name("operator =") SolverParameter put(@Const @ByRef SolverParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef SolverParameter default_instance();

  public native void Swap(SolverParameter other);

  // implements Message ----------------------------------------------

  public native SolverParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef SolverParameter from);
  public native void MergeFrom(@Const @ByRef SolverParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SnapshotFormat") int HDF5();
  public static final int HDF5 = HDF5();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SnapshotFormat") int BINARYPROTO();
  public static final int BINARYPROTO = BINARYPROTO();
  public static native @Cast("bool") boolean SnapshotFormat_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SnapshotFormat") int SnapshotFormat_MIN();
  public static final int SnapshotFormat_MIN = SnapshotFormat_MIN();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SnapshotFormat") int SnapshotFormat_MAX();
  public static final int SnapshotFormat_MAX = SnapshotFormat_MAX();
  @MemberGetter public static native int SnapshotFormat_ARRAYSIZE();
  public static final int SnapshotFormat_ARRAYSIZE = SnapshotFormat_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SnapshotFormat_descriptor();
  public static native @StdString BytePointer SnapshotFormat_Name(@Cast("caffe::SolverParameter::SnapshotFormat") int value);
  public static native @Cast("bool") boolean SnapshotFormat_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SnapshotFormat*") IntPointer value);
  public static native @Cast("bool") boolean SnapshotFormat_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SnapshotFormat*") IntBuffer value);
  public static native @Cast("bool") boolean SnapshotFormat_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SnapshotFormat*") int[] value);
  public static native @Cast("bool") boolean SnapshotFormat_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SnapshotFormat*") IntPointer value);
  public static native @Cast("bool") boolean SnapshotFormat_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SnapshotFormat*") IntBuffer value);
  public static native @Cast("bool") boolean SnapshotFormat_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SnapshotFormat*") int[] value);
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverMode") int CPU();
  public static final int CPU = CPU();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverMode") int GPU();
  public static final int GPU = GPU();
  public static native @Cast("bool") boolean SolverMode_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverMode") int SolverMode_MIN();
  public static final int SolverMode_MIN = SolverMode_MIN();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverMode") int SolverMode_MAX();
  public static final int SolverMode_MAX = SolverMode_MAX();
  @MemberGetter public static native int SolverMode_ARRAYSIZE();
  public static final int SolverMode_ARRAYSIZE = SolverMode_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SolverMode_descriptor();
  public static native @StdString BytePointer SolverMode_Name(@Cast("caffe::SolverParameter::SolverMode") int value);
  public static native @Cast("bool") boolean SolverMode_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SolverMode*") IntPointer value);
  public static native @Cast("bool") boolean SolverMode_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SolverMode*") IntBuffer value);
  public static native @Cast("bool") boolean SolverMode_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SolverMode*") int[] value);
  public static native @Cast("bool") boolean SolverMode_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SolverMode*") IntPointer value);
  public static native @Cast("bool") boolean SolverMode_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SolverMode*") IntBuffer value);
  public static native @Cast("bool") boolean SolverMode_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SolverMode*") int[] value);
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverType") int SGD();
  public static final int SGD = SGD();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverType") int NESTEROV();
  public static final int NESTEROV = NESTEROV();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverType") int ADAGRAD();
  public static final int ADAGRAD = ADAGRAD();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverType") int RMSPROP();
  public static final int RMSPROP = RMSPROP();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverType") int ADADELTA();
  public static final int ADADELTA = ADADELTA();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverType") int ADAM();
  public static final int ADAM = ADAM();
  public static native @Cast("bool") boolean SolverType_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverType") int SolverType_MIN();
  public static final int SolverType_MIN = SolverType_MIN();
  @MemberGetter public static native @Cast("const caffe::SolverParameter::SolverType") int SolverType_MAX();
  public static final int SolverType_MAX = SolverType_MAX();
  @MemberGetter public static native int SolverType_ARRAYSIZE();
  public static final int SolverType_ARRAYSIZE = SolverType_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer SolverType_descriptor();
  public static native @StdString BytePointer SolverType_Name(@Cast("caffe::SolverParameter::SolverType") int value);
  public static native @Cast("bool") boolean SolverType_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SolverType*") IntPointer value);
  public static native @Cast("bool") boolean SolverType_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SolverType*") IntBuffer value);
  public static native @Cast("bool") boolean SolverType_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SolverType*") int[] value);
  public static native @Cast("bool") boolean SolverType_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SolverType*") IntPointer value);
  public static native @Cast("bool") boolean SolverType_Parse(@StdString BytePointer name,
        @Cast("caffe::SolverParameter::SolverType*") IntBuffer value);
  public static native @Cast("bool") boolean SolverType_Parse(@StdString String name,
        @Cast("caffe::SolverParameter::SolverType*") int[] value);

  // accessors -------------------------------------------------------

  // optional string net = 24;
  public native @Cast("bool") boolean has_net();
  public native void clear_net();
  @MemberGetter public static native int kNetFieldNumber();
  public static final int kNetFieldNumber = kNetFieldNumber();
  public native @StdString BytePointer net();
  public native void set_net(@StdString BytePointer value);
  public native void set_net(@StdString String value);
  public native void set_net(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_net(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_net();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_net();
  public native void set_allocated_net(@StdString @Cast({"char*", "std::string*"}) BytePointer net);

  // optional .caffe.NetParameter net_param = 25;
  public native @Cast("bool") boolean has_net_param();
  public native void clear_net_param();
  @MemberGetter public static native int kNetParamFieldNumber();
  public static final int kNetParamFieldNumber = kNetParamFieldNumber();
  public native @Const @ByRef NetParameter net_param();
  public native NetParameter mutable_net_param();
  public native NetParameter release_net_param();
  public native void set_allocated_net_param(NetParameter net_param);

  // optional string train_net = 1;
  public native @Cast("bool") boolean has_train_net();
  public native void clear_train_net();
  @MemberGetter public static native int kTrainNetFieldNumber();
  public static final int kTrainNetFieldNumber = kTrainNetFieldNumber();
  public native @StdString BytePointer train_net();
  public native void set_train_net(@StdString BytePointer value);
  public native void set_train_net(@StdString String value);
  public native void set_train_net(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_train_net(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_train_net();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_train_net();
  public native void set_allocated_train_net(@StdString @Cast({"char*", "std::string*"}) BytePointer train_net);

  // repeated string test_net = 2;
  public native int test_net_size();
  public native void clear_test_net();
  @MemberGetter public static native int kTestNetFieldNumber();
  public static final int kTestNetFieldNumber = kTestNetFieldNumber();
  public native @StdString BytePointer test_net(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_test_net(int index);
  public native void set_test_net(int index, @StdString BytePointer value);
  public native void set_test_net(int index, @StdString String value);
  public native void set_test_net(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_test_net(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_test_net();
  public native void add_test_net(@StdString BytePointer value);
  public native void add_test_net(@StdString String value);
  public native void add_test_net(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_test_net(String value, @Cast("size_t") long size);

  // optional .caffe.NetParameter train_net_param = 21;
  public native @Cast("bool") boolean has_train_net_param();
  public native void clear_train_net_param();
  @MemberGetter public static native int kTrainNetParamFieldNumber();
  public static final int kTrainNetParamFieldNumber = kTrainNetParamFieldNumber();
  public native @Const @ByRef NetParameter train_net_param();
  public native NetParameter mutable_train_net_param();
  public native NetParameter release_train_net_param();
  public native void set_allocated_train_net_param(NetParameter train_net_param);

  // repeated .caffe.NetParameter test_net_param = 22;
  public native int test_net_param_size();
  public native void clear_test_net_param();
  @MemberGetter public static native int kTestNetParamFieldNumber();
  public static final int kTestNetParamFieldNumber = kTestNetParamFieldNumber();
  public native @Const @ByRef NetParameter test_net_param(int index);
  public native NetParameter mutable_test_net_param(int index);
  public native NetParameter add_test_net_param();

  // optional .caffe.NetState train_state = 26;
  public native @Cast("bool") boolean has_train_state();
  public native void clear_train_state();
  @MemberGetter public static native int kTrainStateFieldNumber();
  public static final int kTrainStateFieldNumber = kTrainStateFieldNumber();
  public native @Const @ByRef NetState train_state();
  public native NetState mutable_train_state();
  public native NetState release_train_state();
  public native void set_allocated_train_state(NetState train_state);

  // repeated .caffe.NetState test_state = 27;
  public native int test_state_size();
  public native void clear_test_state();
  @MemberGetter public static native int kTestStateFieldNumber();
  public static final int kTestStateFieldNumber = kTestStateFieldNumber();
  public native @Const @ByRef NetState test_state(int index);
  public native NetState mutable_test_state(int index);
  public native NetState add_test_state();

  // repeated int32 test_iter = 3;
  public native int test_iter_size();
  public native void clear_test_iter();
  @MemberGetter public static native int kTestIterFieldNumber();
  public static final int kTestIterFieldNumber = kTestIterFieldNumber();
  public native @Cast("google::protobuf::int32") int test_iter(int index);
  public native void set_test_iter(int index, @Cast("google::protobuf::int32") int value);
  public native void add_test_iter(@Cast("google::protobuf::int32") int value);

  // optional int32 test_interval = 4 [default = 0];
  public native @Cast("bool") boolean has_test_interval();
  public native void clear_test_interval();
  @MemberGetter public static native int kTestIntervalFieldNumber();
  public static final int kTestIntervalFieldNumber = kTestIntervalFieldNumber();
  public native @Cast("google::protobuf::int32") int test_interval();
  public native void set_test_interval(@Cast("google::protobuf::int32") int value);

  // optional bool test_compute_loss = 19 [default = false];
  public native @Cast("bool") boolean has_test_compute_loss();
  public native void clear_test_compute_loss();
  @MemberGetter public static native int kTestComputeLossFieldNumber();
  public static final int kTestComputeLossFieldNumber = kTestComputeLossFieldNumber();
  public native @Cast("bool") boolean test_compute_loss();
  public native void set_test_compute_loss(@Cast("bool") boolean value);

  // optional bool test_initialization = 32 [default = true];
  public native @Cast("bool") boolean has_test_initialization();
  public native void clear_test_initialization();
  @MemberGetter public static native int kTestInitializationFieldNumber();
  public static final int kTestInitializationFieldNumber = kTestInitializationFieldNumber();
  public native @Cast("bool") boolean test_initialization();
  public native void set_test_initialization(@Cast("bool") boolean value);

  // optional float base_lr = 5;
  public native @Cast("bool") boolean has_base_lr();
  public native void clear_base_lr();
  @MemberGetter public static native int kBaseLrFieldNumber();
  public static final int kBaseLrFieldNumber = kBaseLrFieldNumber();
  public native float base_lr();
  public native void set_base_lr(float value);

  // optional int32 display = 6;
  public native @Cast("bool") boolean has_display();
  public native void clear_display();
  @MemberGetter public static native int kDisplayFieldNumber();
  public static final int kDisplayFieldNumber = kDisplayFieldNumber();
  public native @Cast("google::protobuf::int32") int display();
  public native void set_display(@Cast("google::protobuf::int32") int value);

  // optional int32 average_loss = 33 [default = 1];
  public native @Cast("bool") boolean has_average_loss();
  public native void clear_average_loss();
  @MemberGetter public static native int kAverageLossFieldNumber();
  public static final int kAverageLossFieldNumber = kAverageLossFieldNumber();
  public native @Cast("google::protobuf::int32") int average_loss();
  public native void set_average_loss(@Cast("google::protobuf::int32") int value);

  // optional int32 max_iter = 7;
  public native @Cast("bool") boolean has_max_iter();
  public native void clear_max_iter();
  @MemberGetter public static native int kMaxIterFieldNumber();
  public static final int kMaxIterFieldNumber = kMaxIterFieldNumber();
  public native @Cast("google::protobuf::int32") int max_iter();
  public native void set_max_iter(@Cast("google::protobuf::int32") int value);

  // optional int32 iter_size = 36 [default = 1];
  public native @Cast("bool") boolean has_iter_size();
  public native void clear_iter_size();
  @MemberGetter public static native int kIterSizeFieldNumber();
  public static final int kIterSizeFieldNumber = kIterSizeFieldNumber();
  public native @Cast("google::protobuf::int32") int iter_size();
  public native void set_iter_size(@Cast("google::protobuf::int32") int value);

  // optional string lr_policy = 8;
  public native @Cast("bool") boolean has_lr_policy();
  public native void clear_lr_policy();
  @MemberGetter public static native int kLrPolicyFieldNumber();
  public static final int kLrPolicyFieldNumber = kLrPolicyFieldNumber();
  public native @StdString BytePointer lr_policy();
  public native void set_lr_policy(@StdString BytePointer value);
  public native void set_lr_policy(@StdString String value);
  public native void set_lr_policy(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_lr_policy(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_lr_policy();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_lr_policy();
  public native void set_allocated_lr_policy(@StdString @Cast({"char*", "std::string*"}) BytePointer lr_policy);

  // optional float gamma = 9;
  public native @Cast("bool") boolean has_gamma();
  public native void clear_gamma();
  @MemberGetter public static native int kGammaFieldNumber();
  public static final int kGammaFieldNumber = kGammaFieldNumber();
  public native float gamma();
  public native void set_gamma(float value);

  // optional float power = 10;
  public native @Cast("bool") boolean has_power();
  public native void clear_power();
  @MemberGetter public static native int kPowerFieldNumber();
  public static final int kPowerFieldNumber = kPowerFieldNumber();
  public native float power();
  public native void set_power(float value);

  // optional float momentum = 11;
  public native @Cast("bool") boolean has_momentum();
  public native void clear_momentum();
  @MemberGetter public static native int kMomentumFieldNumber();
  public static final int kMomentumFieldNumber = kMomentumFieldNumber();
  public native float momentum();
  public native void set_momentum(float value);

  // optional float weight_decay = 12;
  public native @Cast("bool") boolean has_weight_decay();
  public native void clear_weight_decay();
  @MemberGetter public static native int kWeightDecayFieldNumber();
  public static final int kWeightDecayFieldNumber = kWeightDecayFieldNumber();
  public native float weight_decay();
  public native void set_weight_decay(float value);

  // optional string regularization_type = 29 [default = "L2"];
  public native @Cast("bool") boolean has_regularization_type();
  public native void clear_regularization_type();
  @MemberGetter public static native int kRegularizationTypeFieldNumber();
  public static final int kRegularizationTypeFieldNumber = kRegularizationTypeFieldNumber();
  public native @StdString BytePointer regularization_type();
  public native void set_regularization_type(@StdString BytePointer value);
  public native void set_regularization_type(@StdString String value);
  public native void set_regularization_type(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_regularization_type(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_regularization_type();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_regularization_type();
  public native void set_allocated_regularization_type(@StdString @Cast({"char*", "std::string*"}) BytePointer regularization_type);

  // optional int32 stepsize = 13;
  public native @Cast("bool") boolean has_stepsize();
  public native void clear_stepsize();
  @MemberGetter public static native int kStepsizeFieldNumber();
  public static final int kStepsizeFieldNumber = kStepsizeFieldNumber();
  public native @Cast("google::protobuf::int32") int stepsize();
  public native void set_stepsize(@Cast("google::protobuf::int32") int value);

  // repeated int32 stepvalue = 34;
  public native int stepvalue_size();
  public native void clear_stepvalue();
  @MemberGetter public static native int kStepvalueFieldNumber();
  public static final int kStepvalueFieldNumber = kStepvalueFieldNumber();
  public native @Cast("google::protobuf::int32") int stepvalue(int index);
  public native void set_stepvalue(int index, @Cast("google::protobuf::int32") int value);
  public native void add_stepvalue(@Cast("google::protobuf::int32") int value);

  // optional float clip_gradients = 35 [default = -1];
  public native @Cast("bool") boolean has_clip_gradients();
  public native void clear_clip_gradients();
  @MemberGetter public static native int kClipGradientsFieldNumber();
  public static final int kClipGradientsFieldNumber = kClipGradientsFieldNumber();
  public native float clip_gradients();
  public native void set_clip_gradients(float value);

  // optional int32 snapshot = 14 [default = 0];
  public native @Cast("bool") boolean has_snapshot();
  public native void clear_snapshot();
  @MemberGetter public static native int kSnapshotFieldNumber();
  public static final int kSnapshotFieldNumber = kSnapshotFieldNumber();
  public native @Cast("google::protobuf::int32") int snapshot();
  public native void set_snapshot(@Cast("google::protobuf::int32") int value);

  // optional string snapshot_prefix = 15;
  public native @Cast("bool") boolean has_snapshot_prefix();
  public native void clear_snapshot_prefix();
  @MemberGetter public static native int kSnapshotPrefixFieldNumber();
  public static final int kSnapshotPrefixFieldNumber = kSnapshotPrefixFieldNumber();
  public native @StdString BytePointer snapshot_prefix();
  public native void set_snapshot_prefix(@StdString BytePointer value);
  public native void set_snapshot_prefix(@StdString String value);
  public native void set_snapshot_prefix(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_snapshot_prefix(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_snapshot_prefix();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_snapshot_prefix();
  public native void set_allocated_snapshot_prefix(@StdString @Cast({"char*", "std::string*"}) BytePointer snapshot_prefix);

  // optional bool snapshot_diff = 16 [default = false];
  public native @Cast("bool") boolean has_snapshot_diff();
  public native void clear_snapshot_diff();
  @MemberGetter public static native int kSnapshotDiffFieldNumber();
  public static final int kSnapshotDiffFieldNumber = kSnapshotDiffFieldNumber();
  public native @Cast("bool") boolean snapshot_diff();
  public native void set_snapshot_diff(@Cast("bool") boolean value);

  // optional .caffe.SolverParameter.SnapshotFormat snapshot_format = 37 [default = BINARYPROTO];
  public native @Cast("bool") boolean has_snapshot_format();
  public native void clear_snapshot_format();
  @MemberGetter public static native int kSnapshotFormatFieldNumber();
  public static final int kSnapshotFormatFieldNumber = kSnapshotFormatFieldNumber();
  public native @Cast("caffe::SolverParameter_SnapshotFormat") int snapshot_format();
  public native void set_snapshot_format(@Cast("caffe::SolverParameter_SnapshotFormat") int value);

  // optional .caffe.SolverParameter.SolverMode solver_mode = 17 [default = GPU];
  public native @Cast("bool") boolean has_solver_mode();
  public native void clear_solver_mode();
  @MemberGetter public static native int kSolverModeFieldNumber();
  public static final int kSolverModeFieldNumber = kSolverModeFieldNumber();
  public native @Cast("caffe::SolverParameter_SolverMode") int solver_mode();
  public native void set_solver_mode(@Cast("caffe::SolverParameter_SolverMode") int value);

  // optional int32 device_id = 18 [default = 0];
  public native @Cast("bool") boolean has_device_id();
  public native void clear_device_id();
  @MemberGetter public static native int kDeviceIdFieldNumber();
  public static final int kDeviceIdFieldNumber = kDeviceIdFieldNumber();
  public native @Cast("google::protobuf::int32") int device_id();
  public native void set_device_id(@Cast("google::protobuf::int32") int value);

  // optional int64 random_seed = 20 [default = -1];
  public native @Cast("bool") boolean has_random_seed();
  public native void clear_random_seed();
  @MemberGetter public static native int kRandomSeedFieldNumber();
  public static final int kRandomSeedFieldNumber = kRandomSeedFieldNumber();
  public native @Cast("google::protobuf::int64") long random_seed();
  public native void set_random_seed(@Cast("google::protobuf::int64") long value);

  // optional string type = 40 [default = "SGD"];
  public native @Cast("bool") boolean has_type();
  public native void clear_type();
  @MemberGetter public static native int kTypeFieldNumber();
  public static final int kTypeFieldNumber = kTypeFieldNumber();
  public native @StdString BytePointer type();
  public native void set_type(@StdString BytePointer value);
  public native void set_type(@StdString String value);
  public native void set_type(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_type(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_type();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_type();
  public native void set_allocated_type(@StdString @Cast({"char*", "std::string*"}) BytePointer type);

  // optional float delta = 31 [default = 1e-08];
  public native @Cast("bool") boolean has_delta();
  public native void clear_delta();
  @MemberGetter public static native int kDeltaFieldNumber();
  public static final int kDeltaFieldNumber = kDeltaFieldNumber();
  public native float delta();
  public native void set_delta(float value);

  // optional float momentum2 = 39 [default = 0.999];
  public native @Cast("bool") boolean has_momentum2();
  public native void clear_momentum2();
  @MemberGetter public static native int kMomentum2FieldNumber();
  public static final int kMomentum2FieldNumber = kMomentum2FieldNumber();
  public native float momentum2();
  public native void set_momentum2(float value);

  // optional float rms_decay = 38;
  public native @Cast("bool") boolean has_rms_decay();
  public native void clear_rms_decay();
  @MemberGetter public static native int kRmsDecayFieldNumber();
  public static final int kRmsDecayFieldNumber = kRmsDecayFieldNumber();
  public native float rms_decay();
  public native void set_rms_decay(float value);

  // optional bool debug_info = 23 [default = false];
  public native @Cast("bool") boolean has_debug_info();
  public native void clear_debug_info();
  @MemberGetter public static native int kDebugInfoFieldNumber();
  public static final int kDebugInfoFieldNumber = kDebugInfoFieldNumber();
  public native @Cast("bool") boolean debug_info();
  public native void set_debug_info(@Cast("bool") boolean value);

  // optional bool snapshot_after_train = 28 [default = true];
  public native @Cast("bool") boolean has_snapshot_after_train();
  public native void clear_snapshot_after_train();
  @MemberGetter public static native int kSnapshotAfterTrainFieldNumber();
  public static final int kSnapshotAfterTrainFieldNumber = kSnapshotAfterTrainFieldNumber();
  public native @Cast("bool") boolean snapshot_after_train();
  public native void set_snapshot_after_train(@Cast("bool") boolean value);

  // optional .caffe.SolverParameter.SolverType solver_type = 30 [default = SGD];
  public native @Cast("bool") boolean has_solver_type();
  public native void clear_solver_type();
  @MemberGetter public static native int kSolverTypeFieldNumber();
  public static final int kSolverTypeFieldNumber = kSolverTypeFieldNumber();
  public native @Cast("caffe::SolverParameter_SolverType") int solver_type();
  public native void set_solver_type(@Cast("caffe::SolverParameter_SolverType") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class SolverState extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SolverState(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SolverState(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SolverState position(int position) {
        return (SolverState)super.position(position);
    }

  public SolverState() { super((Pointer)null); allocate(); }
  private native void allocate();

  public SolverState(@Const @ByRef SolverState from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef SolverState from);

  public native @ByRef @Name("operator =") SolverState put(@Const @ByRef SolverState from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef SolverState default_instance();

  public native void Swap(SolverState other);

  // implements Message ----------------------------------------------

  public native SolverState New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef SolverState from);
  public native void MergeFrom(@Const @ByRef SolverState from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional int32 iter = 1;
  public native @Cast("bool") boolean has_iter();
  public native void clear_iter();
  @MemberGetter public static native int kIterFieldNumber();
  public static final int kIterFieldNumber = kIterFieldNumber();
  public native @Cast("google::protobuf::int32") int iter();
  public native void set_iter(@Cast("google::protobuf::int32") int value);

  // optional string learned_net = 2;
  public native @Cast("bool") boolean has_learned_net();
  public native void clear_learned_net();
  @MemberGetter public static native int kLearnedNetFieldNumber();
  public static final int kLearnedNetFieldNumber = kLearnedNetFieldNumber();
  public native @StdString BytePointer learned_net();
  public native void set_learned_net(@StdString BytePointer value);
  public native void set_learned_net(@StdString String value);
  public native void set_learned_net(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_learned_net(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_learned_net();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_learned_net();
  public native void set_allocated_learned_net(@StdString @Cast({"char*", "std::string*"}) BytePointer learned_net);

  // repeated .caffe.BlobProto history = 3;
  public native int history_size();
  public native void clear_history();
  @MemberGetter public static native int kHistoryFieldNumber();
  public static final int kHistoryFieldNumber = kHistoryFieldNumber();
  public native @Const @ByRef BlobProto history(int index);
  public native BlobProto mutable_history(int index);
  public native BlobProto add_history();

  // optional int32 current_step = 4 [default = 0];
  public native @Cast("bool") boolean has_current_step();
  public native void clear_current_step();
  @MemberGetter public static native int kCurrentStepFieldNumber();
  public static final int kCurrentStepFieldNumber = kCurrentStepFieldNumber();
  public native @Cast("google::protobuf::int32") int current_step();
  public native void set_current_step(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class NetState extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NetState(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NetState(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public NetState position(int position) {
        return (NetState)super.position(position);
    }

  public NetState() { super((Pointer)null); allocate(); }
  private native void allocate();

  public NetState(@Const @ByRef NetState from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef NetState from);

  public native @ByRef @Name("operator =") NetState put(@Const @ByRef NetState from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef NetState default_instance();

  public native void Swap(NetState other);

  // implements Message ----------------------------------------------

  public native NetState New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef NetState from);
  public native void MergeFrom(@Const @ByRef NetState from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional .caffe.Phase phase = 1 [default = TEST];
  public native @Cast("bool") boolean has_phase();
  public native void clear_phase();
  @MemberGetter public static native int kPhaseFieldNumber();
  public static final int kPhaseFieldNumber = kPhaseFieldNumber();
  public native @Cast("caffe::Phase") int phase();
  public native void set_phase(@Cast("caffe::Phase") int value);

  // optional int32 level = 2 [default = 0];
  public native @Cast("bool") boolean has_level();
  public native void clear_level();
  @MemberGetter public static native int kLevelFieldNumber();
  public static final int kLevelFieldNumber = kLevelFieldNumber();
  public native @Cast("google::protobuf::int32") int level();
  public native void set_level(@Cast("google::protobuf::int32") int value);

  // repeated string stage = 3;
  public native int stage_size();
  public native void clear_stage();
  @MemberGetter public static native int kStageFieldNumber();
  public static final int kStageFieldNumber = kStageFieldNumber();
  public native @StdString BytePointer stage(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_stage(int index);
  public native void set_stage(int index, @StdString BytePointer value);
  public native void set_stage(int index, @StdString String value);
  public native void set_stage(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_stage(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_stage();
  public native void add_stage(@StdString BytePointer value);
  public native void add_stage(@StdString String value);
  public native void add_stage(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_stage(String value, @Cast("size_t") long size);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class NetStateRule extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NetStateRule(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NetStateRule(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public NetStateRule position(int position) {
        return (NetStateRule)super.position(position);
    }

  public NetStateRule() { super((Pointer)null); allocate(); }
  private native void allocate();

  public NetStateRule(@Const @ByRef NetStateRule from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef NetStateRule from);

  public native @ByRef @Name("operator =") NetStateRule put(@Const @ByRef NetStateRule from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef NetStateRule default_instance();

  public native void Swap(NetStateRule other);

  // implements Message ----------------------------------------------

  public native NetStateRule New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef NetStateRule from);
  public native void MergeFrom(@Const @ByRef NetStateRule from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional .caffe.Phase phase = 1;
  public native @Cast("bool") boolean has_phase();
  public native void clear_phase();
  @MemberGetter public static native int kPhaseFieldNumber();
  public static final int kPhaseFieldNumber = kPhaseFieldNumber();
  public native @Cast("caffe::Phase") int phase();
  public native void set_phase(@Cast("caffe::Phase") int value);

  // optional int32 min_level = 2;
  public native @Cast("bool") boolean has_min_level();
  public native void clear_min_level();
  @MemberGetter public static native int kMinLevelFieldNumber();
  public static final int kMinLevelFieldNumber = kMinLevelFieldNumber();
  public native @Cast("google::protobuf::int32") int min_level();
  public native void set_min_level(@Cast("google::protobuf::int32") int value);

  // optional int32 max_level = 3;
  public native @Cast("bool") boolean has_max_level();
  public native void clear_max_level();
  @MemberGetter public static native int kMaxLevelFieldNumber();
  public static final int kMaxLevelFieldNumber = kMaxLevelFieldNumber();
  public native @Cast("google::protobuf::int32") int max_level();
  public native void set_max_level(@Cast("google::protobuf::int32") int value);

  // repeated string stage = 4;
  public native int stage_size();
  public native void clear_stage();
  @MemberGetter public static native int kStageFieldNumber();
  public static final int kStageFieldNumber = kStageFieldNumber();
  public native @StdString BytePointer stage(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_stage(int index);
  public native void set_stage(int index, @StdString BytePointer value);
  public native void set_stage(int index, @StdString String value);
  public native void set_stage(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_stage(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_stage();
  public native void add_stage(@StdString BytePointer value);
  public native void add_stage(@StdString String value);
  public native void add_stage(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_stage(String value, @Cast("size_t") long size);

  // repeated string not_stage = 5;
  public native int not_stage_size();
  public native void clear_not_stage();
  @MemberGetter public static native int kNotStageFieldNumber();
  public static final int kNotStageFieldNumber = kNotStageFieldNumber();
  public native @StdString BytePointer not_stage(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_not_stage(int index);
  public native void set_not_stage(int index, @StdString BytePointer value);
  public native void set_not_stage(int index, @StdString String value);
  public native void set_not_stage(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_not_stage(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_not_stage();
  public native void add_not_stage(@StdString BytePointer value);
  public native void add_not_stage(@StdString String value);
  public native void add_not_stage(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_not_stage(String value, @Cast("size_t") long size);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ParamSpec extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ParamSpec(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ParamSpec(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ParamSpec position(int position) {
        return (ParamSpec)super.position(position);
    }

  public ParamSpec() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ParamSpec(@Const @ByRef ParamSpec from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ParamSpec from);

  public native @ByRef @Name("operator =") ParamSpec put(@Const @ByRef ParamSpec from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ParamSpec default_instance();

  public native void Swap(ParamSpec other);

  // implements Message ----------------------------------------------

  public native ParamSpec New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ParamSpec from);
  public native void MergeFrom(@Const @ByRef ParamSpec from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::ParamSpec::DimCheckMode") int STRICT();
  public static final int STRICT = STRICT();
  @MemberGetter public static native @Cast("const caffe::ParamSpec::DimCheckMode") int PERMISSIVE();
  public static final int PERMISSIVE = PERMISSIVE();
  public static native @Cast("bool") boolean DimCheckMode_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::ParamSpec::DimCheckMode") int DimCheckMode_MIN();
  public static final int DimCheckMode_MIN = DimCheckMode_MIN();
  @MemberGetter public static native @Cast("const caffe::ParamSpec::DimCheckMode") int DimCheckMode_MAX();
  public static final int DimCheckMode_MAX = DimCheckMode_MAX();
  @MemberGetter public static native int DimCheckMode_ARRAYSIZE();
  public static final int DimCheckMode_ARRAYSIZE = DimCheckMode_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer DimCheckMode_descriptor();
  public static native @StdString BytePointer DimCheckMode_Name(@Cast("caffe::ParamSpec::DimCheckMode") int value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString BytePointer name,
        @Cast("caffe::ParamSpec::DimCheckMode*") IntPointer value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString String name,
        @Cast("caffe::ParamSpec::DimCheckMode*") IntBuffer value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString BytePointer name,
        @Cast("caffe::ParamSpec::DimCheckMode*") int[] value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString String name,
        @Cast("caffe::ParamSpec::DimCheckMode*") IntPointer value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString BytePointer name,
        @Cast("caffe::ParamSpec::DimCheckMode*") IntBuffer value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString String name,
        @Cast("caffe::ParamSpec::DimCheckMode*") int[] value);

  // accessors -------------------------------------------------------

  // optional string name = 1;
  public native @Cast("bool") boolean has_name();
  public native void clear_name();
  @MemberGetter public static native int kNameFieldNumber();
  public static final int kNameFieldNumber = kNameFieldNumber();
  public native @StdString BytePointer name();
  public native void set_name(@StdString BytePointer value);
  public native void set_name(@StdString String value);
  public native void set_name(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_name(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_name();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_name();
  public native void set_allocated_name(@StdString @Cast({"char*", "std::string*"}) BytePointer name);

  // optional .caffe.ParamSpec.DimCheckMode share_mode = 2;
  public native @Cast("bool") boolean has_share_mode();
  public native void clear_share_mode();
  @MemberGetter public static native int kShareModeFieldNumber();
  public static final int kShareModeFieldNumber = kShareModeFieldNumber();
  public native @Cast("caffe::ParamSpec_DimCheckMode") int share_mode();
  public native void set_share_mode(@Cast("caffe::ParamSpec_DimCheckMode") int value);

  // optional float lr_mult = 3 [default = 1];
  public native @Cast("bool") boolean has_lr_mult();
  public native void clear_lr_mult();
  @MemberGetter public static native int kLrMultFieldNumber();
  public static final int kLrMultFieldNumber = kLrMultFieldNumber();
  public native float lr_mult();
  public native void set_lr_mult(float value);

  // optional float decay_mult = 4 [default = 1];
  public native @Cast("bool") boolean has_decay_mult();
  public native void clear_decay_mult();
  @MemberGetter public static native int kDecayMultFieldNumber();
  public static final int kDecayMultFieldNumber = kDecayMultFieldNumber();
  public native float decay_mult();
  public native void set_decay_mult(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class LayerParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LayerParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LayerParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public LayerParameter position(int position) {
        return (LayerParameter)super.position(position);
    }

  public LayerParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public LayerParameter(@Const @ByRef LayerParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef LayerParameter from);

  public native @ByRef @Name("operator =") LayerParameter put(@Const @ByRef LayerParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef LayerParameter default_instance();

  public native void Swap(LayerParameter other);

  // implements Message ----------------------------------------------

  public native LayerParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef LayerParameter from);
  public native void MergeFrom(@Const @ByRef LayerParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string name = 1;
  public native @Cast("bool") boolean has_name();
  public native void clear_name();
  @MemberGetter public static native int kNameFieldNumber();
  public static final int kNameFieldNumber = kNameFieldNumber();
  public native @StdString BytePointer name();
  public native void set_name(@StdString BytePointer value);
  public native void set_name(@StdString String value);
  public native void set_name(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_name(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_name();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_name();
  public native void set_allocated_name(@StdString @Cast({"char*", "std::string*"}) BytePointer name);

  // optional string type = 2;
  public native @Cast("bool") boolean has_type();
  public native void clear_type();
  @MemberGetter public static native int kTypeFieldNumber();
  public static final int kTypeFieldNumber = kTypeFieldNumber();
  public native @StdString BytePointer type();
  public native void set_type(@StdString BytePointer value);
  public native void set_type(@StdString String value);
  public native void set_type(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_type(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_type();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_type();
  public native void set_allocated_type(@StdString @Cast({"char*", "std::string*"}) BytePointer type);

  // repeated string bottom = 3;
  public native int bottom_size();
  public native void clear_bottom();
  @MemberGetter public static native int kBottomFieldNumber();
  public static final int kBottomFieldNumber = kBottomFieldNumber();
  public native @StdString BytePointer bottom(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_bottom(int index);
  public native void set_bottom(int index, @StdString BytePointer value);
  public native void set_bottom(int index, @StdString String value);
  public native void set_bottom(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_bottom(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_bottom();
  public native void add_bottom(@StdString BytePointer value);
  public native void add_bottom(@StdString String value);
  public native void add_bottom(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_bottom(String value, @Cast("size_t") long size);

  // repeated string top = 4;
  public native int top_size();
  public native void clear_top();
  @MemberGetter public static native int kTopFieldNumber();
  public static final int kTopFieldNumber = kTopFieldNumber();
  public native @StdString BytePointer top(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_top(int index);
  public native void set_top(int index, @StdString BytePointer value);
  public native void set_top(int index, @StdString String value);
  public native void set_top(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_top(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_top();
  public native void add_top(@StdString BytePointer value);
  public native void add_top(@StdString String value);
  public native void add_top(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_top(String value, @Cast("size_t") long size);

  // optional .caffe.Phase phase = 10;
  public native @Cast("bool") boolean has_phase();
  public native void clear_phase();
  @MemberGetter public static native int kPhaseFieldNumber();
  public static final int kPhaseFieldNumber = kPhaseFieldNumber();
  public native @Cast("caffe::Phase") int phase();
  public native void set_phase(@Cast("caffe::Phase") int value);

  // repeated float loss_weight = 5;
  public native int loss_weight_size();
  public native void clear_loss_weight();
  @MemberGetter public static native int kLossWeightFieldNumber();
  public static final int kLossWeightFieldNumber = kLossWeightFieldNumber();
  public native float loss_weight(int index);
  public native void set_loss_weight(int index, float value);
  public native void add_loss_weight(float value);

  // repeated .caffe.ParamSpec param = 6;
  public native int param_size();
  public native void clear_param();
  @MemberGetter public static native int kParamFieldNumber();
  public static final int kParamFieldNumber = kParamFieldNumber();
  public native @Const @ByRef ParamSpec param(int index);
  public native ParamSpec mutable_param(int index);
  public native ParamSpec add_param();

  // repeated .caffe.BlobProto blobs = 7;
  public native int blobs_size();
  public native void clear_blobs();
  @MemberGetter public static native int kBlobsFieldNumber();
  public static final int kBlobsFieldNumber = kBlobsFieldNumber();
  public native @Const @ByRef BlobProto blobs(int index);
  public native BlobProto mutable_blobs(int index);
  public native BlobProto add_blobs();

  // repeated bool propagate_down = 11;
  public native int propagate_down_size();
  public native void clear_propagate_down();
  @MemberGetter public static native int kPropagateDownFieldNumber();
  public static final int kPropagateDownFieldNumber = kPropagateDownFieldNumber();
  public native @Cast("bool") boolean propagate_down(int index);
  public native void set_propagate_down(int index, @Cast("bool") boolean value);
  public native void add_propagate_down(@Cast("bool") boolean value);

  // repeated .caffe.NetStateRule include = 8;
  public native int include_size();
  public native void clear_include();
  @MemberGetter public static native int kIncludeFieldNumber();
  public static final int kIncludeFieldNumber = kIncludeFieldNumber();
  public native @Const @ByRef NetStateRule include(int index);
  public native NetStateRule mutable_include(int index);
  public native NetStateRule add_include();

  // repeated .caffe.NetStateRule exclude = 9;
  public native int exclude_size();
  public native void clear_exclude();
  @MemberGetter public static native int kExcludeFieldNumber();
  public static final int kExcludeFieldNumber = kExcludeFieldNumber();
  public native @Const @ByRef NetStateRule exclude(int index);
  public native NetStateRule mutable_exclude(int index);
  public native NetStateRule add_exclude();

  // optional .caffe.TransformationParameter transform_param = 100;
  public native @Cast("bool") boolean has_transform_param();
  public native void clear_transform_param();
  @MemberGetter public static native int kTransformParamFieldNumber();
  public static final int kTransformParamFieldNumber = kTransformParamFieldNumber();
  public native @Const @ByRef TransformationParameter transform_param();
  public native TransformationParameter mutable_transform_param();
  public native TransformationParameter release_transform_param();
  public native void set_allocated_transform_param(TransformationParameter transform_param);

  // optional .caffe.LossParameter loss_param = 101;
  public native @Cast("bool") boolean has_loss_param();
  public native void clear_loss_param();
  @MemberGetter public static native int kLossParamFieldNumber();
  public static final int kLossParamFieldNumber = kLossParamFieldNumber();
  public native @Const @ByRef LossParameter loss_param();
  public native LossParameter mutable_loss_param();
  public native LossParameter release_loss_param();
  public native void set_allocated_loss_param(LossParameter loss_param);

  // optional .caffe.AccuracyParameter accuracy_param = 102;
  public native @Cast("bool") boolean has_accuracy_param();
  public native void clear_accuracy_param();
  @MemberGetter public static native int kAccuracyParamFieldNumber();
  public static final int kAccuracyParamFieldNumber = kAccuracyParamFieldNumber();
  public native @Const @ByRef AccuracyParameter accuracy_param();
  public native AccuracyParameter mutable_accuracy_param();
  public native AccuracyParameter release_accuracy_param();
  public native void set_allocated_accuracy_param(AccuracyParameter accuracy_param);

  // optional .caffe.ArgMaxParameter argmax_param = 103;
  public native @Cast("bool") boolean has_argmax_param();
  public native void clear_argmax_param();
  @MemberGetter public static native int kArgmaxParamFieldNumber();
  public static final int kArgmaxParamFieldNumber = kArgmaxParamFieldNumber();
  public native @Const @ByRef ArgMaxParameter argmax_param();
  public native ArgMaxParameter mutable_argmax_param();
  public native ArgMaxParameter release_argmax_param();
  public native void set_allocated_argmax_param(ArgMaxParameter argmax_param);

  // optional .caffe.BatchNormParameter batch_norm_param = 139;
  public native @Cast("bool") boolean has_batch_norm_param();
  public native void clear_batch_norm_param();
  @MemberGetter public static native int kBatchNormParamFieldNumber();
  public static final int kBatchNormParamFieldNumber = kBatchNormParamFieldNumber();
  public native @Const @ByRef BatchNormParameter batch_norm_param();
  public native BatchNormParameter mutable_batch_norm_param();
  public native BatchNormParameter release_batch_norm_param();
  public native void set_allocated_batch_norm_param(BatchNormParameter batch_norm_param);

  // optional .caffe.ConcatParameter concat_param = 104;
  public native @Cast("bool") boolean has_concat_param();
  public native void clear_concat_param();
  @MemberGetter public static native int kConcatParamFieldNumber();
  public static final int kConcatParamFieldNumber = kConcatParamFieldNumber();
  public native @Const @ByRef ConcatParameter concat_param();
  public native ConcatParameter mutable_concat_param();
  public native ConcatParameter release_concat_param();
  public native void set_allocated_concat_param(ConcatParameter concat_param);

  // optional .caffe.ContrastiveLossParameter contrastive_loss_param = 105;
  public native @Cast("bool") boolean has_contrastive_loss_param();
  public native void clear_contrastive_loss_param();
  @MemberGetter public static native int kContrastiveLossParamFieldNumber();
  public static final int kContrastiveLossParamFieldNumber = kContrastiveLossParamFieldNumber();
  public native @Const @ByRef ContrastiveLossParameter contrastive_loss_param();
  public native ContrastiveLossParameter mutable_contrastive_loss_param();
  public native ContrastiveLossParameter release_contrastive_loss_param();
  public native void set_allocated_contrastive_loss_param(ContrastiveLossParameter contrastive_loss_param);

  // optional .caffe.ConvolutionParameter convolution_param = 106;
  public native @Cast("bool") boolean has_convolution_param();
  public native void clear_convolution_param();
  @MemberGetter public static native int kConvolutionParamFieldNumber();
  public static final int kConvolutionParamFieldNumber = kConvolutionParamFieldNumber();
  public native @Const @ByRef ConvolutionParameter convolution_param();
  public native ConvolutionParameter mutable_convolution_param();
  public native ConvolutionParameter release_convolution_param();
  public native void set_allocated_convolution_param(ConvolutionParameter convolution_param);

  // optional .caffe.DataParameter data_param = 107;
  public native @Cast("bool") boolean has_data_param();
  public native void clear_data_param();
  @MemberGetter public static native int kDataParamFieldNumber();
  public static final int kDataParamFieldNumber = kDataParamFieldNumber();
  public native @Const @ByRef DataParameter data_param();
  public native DataParameter mutable_data_param();
  public native DataParameter release_data_param();
  public native void set_allocated_data_param(DataParameter data_param);

  // optional .caffe.DropoutParameter dropout_param = 108;
  public native @Cast("bool") boolean has_dropout_param();
  public native void clear_dropout_param();
  @MemberGetter public static native int kDropoutParamFieldNumber();
  public static final int kDropoutParamFieldNumber = kDropoutParamFieldNumber();
  public native @Const @ByRef DropoutParameter dropout_param();
  public native DropoutParameter mutable_dropout_param();
  public native DropoutParameter release_dropout_param();
  public native void set_allocated_dropout_param(DropoutParameter dropout_param);

  // optional .caffe.DummyDataParameter dummy_data_param = 109;
  public native @Cast("bool") boolean has_dummy_data_param();
  public native void clear_dummy_data_param();
  @MemberGetter public static native int kDummyDataParamFieldNumber();
  public static final int kDummyDataParamFieldNumber = kDummyDataParamFieldNumber();
  public native @Const @ByRef DummyDataParameter dummy_data_param();
  public native DummyDataParameter mutable_dummy_data_param();
  public native DummyDataParameter release_dummy_data_param();
  public native void set_allocated_dummy_data_param(DummyDataParameter dummy_data_param);

  // optional .caffe.EltwiseParameter eltwise_param = 110;
  public native @Cast("bool") boolean has_eltwise_param();
  public native void clear_eltwise_param();
  @MemberGetter public static native int kEltwiseParamFieldNumber();
  public static final int kEltwiseParamFieldNumber = kEltwiseParamFieldNumber();
  public native @Const @ByRef EltwiseParameter eltwise_param();
  public native EltwiseParameter mutable_eltwise_param();
  public native EltwiseParameter release_eltwise_param();
  public native void set_allocated_eltwise_param(EltwiseParameter eltwise_param);

  // optional .caffe.EmbedParameter embed_param = 137;
  public native @Cast("bool") boolean has_embed_param();
  public native void clear_embed_param();
  @MemberGetter public static native int kEmbedParamFieldNumber();
  public static final int kEmbedParamFieldNumber = kEmbedParamFieldNumber();
  public native @Const @ByRef EmbedParameter embed_param();
  public native EmbedParameter mutable_embed_param();
  public native EmbedParameter release_embed_param();
  public native void set_allocated_embed_param(EmbedParameter embed_param);

  // optional .caffe.ExpParameter exp_param = 111;
  public native @Cast("bool") boolean has_exp_param();
  public native void clear_exp_param();
  @MemberGetter public static native int kExpParamFieldNumber();
  public static final int kExpParamFieldNumber = kExpParamFieldNumber();
  public native @Const @ByRef ExpParameter exp_param();
  public native ExpParameter mutable_exp_param();
  public native ExpParameter release_exp_param();
  public native void set_allocated_exp_param(ExpParameter exp_param);

  // optional .caffe.FlattenParameter flatten_param = 135;
  public native @Cast("bool") boolean has_flatten_param();
  public native void clear_flatten_param();
  @MemberGetter public static native int kFlattenParamFieldNumber();
  public static final int kFlattenParamFieldNumber = kFlattenParamFieldNumber();
  public native @Const @ByRef FlattenParameter flatten_param();
  public native FlattenParameter mutable_flatten_param();
  public native FlattenParameter release_flatten_param();
  public native void set_allocated_flatten_param(FlattenParameter flatten_param);

  // optional .caffe.HDF5DataParameter hdf5_data_param = 112;
  public native @Cast("bool") boolean has_hdf5_data_param();
  public native void clear_hdf5_data_param();
  @MemberGetter public static native int kHdf5DataParamFieldNumber();
  public static final int kHdf5DataParamFieldNumber = kHdf5DataParamFieldNumber();
  public native @Const @ByRef HDF5DataParameter hdf5_data_param();
  public native HDF5DataParameter mutable_hdf5_data_param();
  public native HDF5DataParameter release_hdf5_data_param();
  public native void set_allocated_hdf5_data_param(HDF5DataParameter hdf5_data_param);

  // optional .caffe.HDF5OutputParameter hdf5_output_param = 113;
  public native @Cast("bool") boolean has_hdf5_output_param();
  public native void clear_hdf5_output_param();
  @MemberGetter public static native int kHdf5OutputParamFieldNumber();
  public static final int kHdf5OutputParamFieldNumber = kHdf5OutputParamFieldNumber();
  public native @Const @ByRef HDF5OutputParameter hdf5_output_param();
  public native HDF5OutputParameter mutable_hdf5_output_param();
  public native HDF5OutputParameter release_hdf5_output_param();
  public native void set_allocated_hdf5_output_param(HDF5OutputParameter hdf5_output_param);

  // optional .caffe.HingeLossParameter hinge_loss_param = 114;
  public native @Cast("bool") boolean has_hinge_loss_param();
  public native void clear_hinge_loss_param();
  @MemberGetter public static native int kHingeLossParamFieldNumber();
  public static final int kHingeLossParamFieldNumber = kHingeLossParamFieldNumber();
  public native @Const @ByRef HingeLossParameter hinge_loss_param();
  public native HingeLossParameter mutable_hinge_loss_param();
  public native HingeLossParameter release_hinge_loss_param();
  public native void set_allocated_hinge_loss_param(HingeLossParameter hinge_loss_param);

  // optional .caffe.ImageDataParameter image_data_param = 115;
  public native @Cast("bool") boolean has_image_data_param();
  public native void clear_image_data_param();
  @MemberGetter public static native int kImageDataParamFieldNumber();
  public static final int kImageDataParamFieldNumber = kImageDataParamFieldNumber();
  public native @Const @ByRef ImageDataParameter image_data_param();
  public native ImageDataParameter mutable_image_data_param();
  public native ImageDataParameter release_image_data_param();
  public native void set_allocated_image_data_param(ImageDataParameter image_data_param);

  // optional .caffe.InfogainLossParameter infogain_loss_param = 116;
  public native @Cast("bool") boolean has_infogain_loss_param();
  public native void clear_infogain_loss_param();
  @MemberGetter public static native int kInfogainLossParamFieldNumber();
  public static final int kInfogainLossParamFieldNumber = kInfogainLossParamFieldNumber();
  public native @Const @ByRef InfogainLossParameter infogain_loss_param();
  public native InfogainLossParameter mutable_infogain_loss_param();
  public native InfogainLossParameter release_infogain_loss_param();
  public native void set_allocated_infogain_loss_param(InfogainLossParameter infogain_loss_param);

  // optional .caffe.InnerProductParameter inner_product_param = 117;
  public native @Cast("bool") boolean has_inner_product_param();
  public native void clear_inner_product_param();
  @MemberGetter public static native int kInnerProductParamFieldNumber();
  public static final int kInnerProductParamFieldNumber = kInnerProductParamFieldNumber();
  public native @Const @ByRef InnerProductParameter inner_product_param();
  public native InnerProductParameter mutable_inner_product_param();
  public native InnerProductParameter release_inner_product_param();
  public native void set_allocated_inner_product_param(InnerProductParameter inner_product_param);

  // optional .caffe.LogParameter log_param = 134;
  public native @Cast("bool") boolean has_log_param();
  public native void clear_log_param();
  @MemberGetter public static native int kLogParamFieldNumber();
  public static final int kLogParamFieldNumber = kLogParamFieldNumber();
  public native @Const @ByRef LogParameter log_param();
  public native LogParameter mutable_log_param();
  public native LogParameter release_log_param();
  public native void set_allocated_log_param(LogParameter log_param);

  // optional .caffe.LRNParameter lrn_param = 118;
  public native @Cast("bool") boolean has_lrn_param();
  public native void clear_lrn_param();
  @MemberGetter public static native int kLrnParamFieldNumber();
  public static final int kLrnParamFieldNumber = kLrnParamFieldNumber();
  public native @Const @ByRef LRNParameter lrn_param();
  public native LRNParameter mutable_lrn_param();
  public native LRNParameter release_lrn_param();
  public native void set_allocated_lrn_param(LRNParameter lrn_param);

  // optional .caffe.MemoryDataParameter memory_data_param = 119;
  public native @Cast("bool") boolean has_memory_data_param();
  public native void clear_memory_data_param();
  @MemberGetter public static native int kMemoryDataParamFieldNumber();
  public static final int kMemoryDataParamFieldNumber = kMemoryDataParamFieldNumber();
  public native @Const @ByRef MemoryDataParameter memory_data_param();
  public native MemoryDataParameter mutable_memory_data_param();
  public native MemoryDataParameter release_memory_data_param();
  public native void set_allocated_memory_data_param(MemoryDataParameter memory_data_param);

  // optional .caffe.MVNParameter mvn_param = 120;
  public native @Cast("bool") boolean has_mvn_param();
  public native void clear_mvn_param();
  @MemberGetter public static native int kMvnParamFieldNumber();
  public static final int kMvnParamFieldNumber = kMvnParamFieldNumber();
  public native @Const @ByRef MVNParameter mvn_param();
  public native MVNParameter mutable_mvn_param();
  public native MVNParameter release_mvn_param();
  public native void set_allocated_mvn_param(MVNParameter mvn_param);

  // optional .caffe.PoolingParameter pooling_param = 121;
  public native @Cast("bool") boolean has_pooling_param();
  public native void clear_pooling_param();
  @MemberGetter public static native int kPoolingParamFieldNumber();
  public static final int kPoolingParamFieldNumber = kPoolingParamFieldNumber();
  public native @Const @ByRef PoolingParameter pooling_param();
  public native PoolingParameter mutable_pooling_param();
  public native PoolingParameter release_pooling_param();
  public native void set_allocated_pooling_param(PoolingParameter pooling_param);

  // optional .caffe.PowerParameter power_param = 122;
  public native @Cast("bool") boolean has_power_param();
  public native void clear_power_param();
  @MemberGetter public static native int kPowerParamFieldNumber();
  public static final int kPowerParamFieldNumber = kPowerParamFieldNumber();
  public native @Const @ByRef PowerParameter power_param();
  public native PowerParameter mutable_power_param();
  public native PowerParameter release_power_param();
  public native void set_allocated_power_param(PowerParameter power_param);

  // optional .caffe.PReLUParameter prelu_param = 131;
  public native @Cast("bool") boolean has_prelu_param();
  public native void clear_prelu_param();
  @MemberGetter public static native int kPreluParamFieldNumber();
  public static final int kPreluParamFieldNumber = kPreluParamFieldNumber();
  public native @Const @ByRef PReLUParameter prelu_param();
  public native PReLUParameter mutable_prelu_param();
  public native PReLUParameter release_prelu_param();
  public native void set_allocated_prelu_param(PReLUParameter prelu_param);

  // optional .caffe.PythonParameter python_param = 130;
  public native @Cast("bool") boolean has_python_param();
  public native void clear_python_param();
  @MemberGetter public static native int kPythonParamFieldNumber();
  public static final int kPythonParamFieldNumber = kPythonParamFieldNumber();
  public native @Const @ByRef PythonParameter python_param();
  public native PythonParameter mutable_python_param();
  public native PythonParameter release_python_param();
  public native void set_allocated_python_param(PythonParameter python_param);

  // optional .caffe.ReductionParameter reduction_param = 136;
  public native @Cast("bool") boolean has_reduction_param();
  public native void clear_reduction_param();
  @MemberGetter public static native int kReductionParamFieldNumber();
  public static final int kReductionParamFieldNumber = kReductionParamFieldNumber();
  public native @Const @ByRef ReductionParameter reduction_param();
  public native ReductionParameter mutable_reduction_param();
  public native ReductionParameter release_reduction_param();
  public native void set_allocated_reduction_param(ReductionParameter reduction_param);

  // optional .caffe.ReLUParameter relu_param = 123;
  public native @Cast("bool") boolean has_relu_param();
  public native void clear_relu_param();
  @MemberGetter public static native int kReluParamFieldNumber();
  public static final int kReluParamFieldNumber = kReluParamFieldNumber();
  public native @Const @ByRef ReLUParameter relu_param();
  public native ReLUParameter mutable_relu_param();
  public native ReLUParameter release_relu_param();
  public native void set_allocated_relu_param(ReLUParameter relu_param);

  // optional .caffe.ReshapeParameter reshape_param = 133;
  public native @Cast("bool") boolean has_reshape_param();
  public native void clear_reshape_param();
  @MemberGetter public static native int kReshapeParamFieldNumber();
  public static final int kReshapeParamFieldNumber = kReshapeParamFieldNumber();
  public native @Const @ByRef ReshapeParameter reshape_param();
  public native ReshapeParameter mutable_reshape_param();
  public native ReshapeParameter release_reshape_param();
  public native void set_allocated_reshape_param(ReshapeParameter reshape_param);

  // optional .caffe.SigmoidParameter sigmoid_param = 124;
  public native @Cast("bool") boolean has_sigmoid_param();
  public native void clear_sigmoid_param();
  @MemberGetter public static native int kSigmoidParamFieldNumber();
  public static final int kSigmoidParamFieldNumber = kSigmoidParamFieldNumber();
  public native @Const @ByRef SigmoidParameter sigmoid_param();
  public native SigmoidParameter mutable_sigmoid_param();
  public native SigmoidParameter release_sigmoid_param();
  public native void set_allocated_sigmoid_param(SigmoidParameter sigmoid_param);

  // optional .caffe.SoftmaxParameter softmax_param = 125;
  public native @Cast("bool") boolean has_softmax_param();
  public native void clear_softmax_param();
  @MemberGetter public static native int kSoftmaxParamFieldNumber();
  public static final int kSoftmaxParamFieldNumber = kSoftmaxParamFieldNumber();
  public native @Const @ByRef SoftmaxParameter softmax_param();
  public native SoftmaxParameter mutable_softmax_param();
  public native SoftmaxParameter release_softmax_param();
  public native void set_allocated_softmax_param(SoftmaxParameter softmax_param);

  // optional .caffe.SPPParameter spp_param = 132;
  public native @Cast("bool") boolean has_spp_param();
  public native void clear_spp_param();
  @MemberGetter public static native int kSppParamFieldNumber();
  public static final int kSppParamFieldNumber = kSppParamFieldNumber();
  public native @Const @ByRef SPPParameter spp_param();
  public native SPPParameter mutable_spp_param();
  public native SPPParameter release_spp_param();
  public native void set_allocated_spp_param(SPPParameter spp_param);

  // optional .caffe.SliceParameter slice_param = 126;
  public native @Cast("bool") boolean has_slice_param();
  public native void clear_slice_param();
  @MemberGetter public static native int kSliceParamFieldNumber();
  public static final int kSliceParamFieldNumber = kSliceParamFieldNumber();
  public native @Const @ByRef SliceParameter slice_param();
  public native SliceParameter mutable_slice_param();
  public native SliceParameter release_slice_param();
  public native void set_allocated_slice_param(SliceParameter slice_param);

  // optional .caffe.TanHParameter tanh_param = 127;
  public native @Cast("bool") boolean has_tanh_param();
  public native void clear_tanh_param();
  @MemberGetter public static native int kTanhParamFieldNumber();
  public static final int kTanhParamFieldNumber = kTanhParamFieldNumber();
  public native @Const @ByRef TanHParameter tanh_param();
  public native TanHParameter mutable_tanh_param();
  public native TanHParameter release_tanh_param();
  public native void set_allocated_tanh_param(TanHParameter tanh_param);

  // optional .caffe.ThresholdParameter threshold_param = 128;
  public native @Cast("bool") boolean has_threshold_param();
  public native void clear_threshold_param();
  @MemberGetter public static native int kThresholdParamFieldNumber();
  public static final int kThresholdParamFieldNumber = kThresholdParamFieldNumber();
  public native @Const @ByRef ThresholdParameter threshold_param();
  public native ThresholdParameter mutable_threshold_param();
  public native ThresholdParameter release_threshold_param();
  public native void set_allocated_threshold_param(ThresholdParameter threshold_param);

  // optional .caffe.TileParameter tile_param = 138;
  public native @Cast("bool") boolean has_tile_param();
  public native void clear_tile_param();
  @MemberGetter public static native int kTileParamFieldNumber();
  public static final int kTileParamFieldNumber = kTileParamFieldNumber();
  public native @Const @ByRef TileParameter tile_param();
  public native TileParameter mutable_tile_param();
  public native TileParameter release_tile_param();
  public native void set_allocated_tile_param(TileParameter tile_param);

  // optional .caffe.WindowDataParameter window_data_param = 129;
  public native @Cast("bool") boolean has_window_data_param();
  public native void clear_window_data_param();
  @MemberGetter public static native int kWindowDataParamFieldNumber();
  public static final int kWindowDataParamFieldNumber = kWindowDataParamFieldNumber();
  public native @Const @ByRef WindowDataParameter window_data_param();
  public native WindowDataParameter mutable_window_data_param();
  public native WindowDataParameter release_window_data_param();
  public native void set_allocated_window_data_param(WindowDataParameter window_data_param);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class TransformationParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TransformationParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TransformationParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TransformationParameter position(int position) {
        return (TransformationParameter)super.position(position);
    }

  public TransformationParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public TransformationParameter(@Const @ByRef TransformationParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef TransformationParameter from);

  public native @ByRef @Name("operator =") TransformationParameter put(@Const @ByRef TransformationParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef TransformationParameter default_instance();

  public native void Swap(TransformationParameter other);

  // implements Message ----------------------------------------------

  public native TransformationParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef TransformationParameter from);
  public native void MergeFrom(@Const @ByRef TransformationParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional float scale = 1 [default = 1];
  public native @Cast("bool") boolean has_scale();
  public native void clear_scale();
  @MemberGetter public static native int kScaleFieldNumber();
  public static final int kScaleFieldNumber = kScaleFieldNumber();
  public native float scale();
  public native void set_scale(float value);

  // optional bool mirror = 2 [default = false];
  public native @Cast("bool") boolean has_mirror();
  public native void clear_mirror();
  @MemberGetter public static native int kMirrorFieldNumber();
  public static final int kMirrorFieldNumber = kMirrorFieldNumber();
  public native @Cast("bool") boolean mirror();
  public native void set_mirror(@Cast("bool") boolean value);

  // optional uint32 crop_size = 3 [default = 0];
  public native @Cast("bool") boolean has_crop_size();
  public native void clear_crop_size();
  @MemberGetter public static native int kCropSizeFieldNumber();
  public static final int kCropSizeFieldNumber = kCropSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int crop_size();
  public native void set_crop_size(@Cast("google::protobuf::uint32") int value);

  // optional string mean_file = 4;
  public native @Cast("bool") boolean has_mean_file();
  public native void clear_mean_file();
  @MemberGetter public static native int kMeanFileFieldNumber();
  public static final int kMeanFileFieldNumber = kMeanFileFieldNumber();
  public native @StdString BytePointer mean_file();
  public native void set_mean_file(@StdString BytePointer value);
  public native void set_mean_file(@StdString String value);
  public native void set_mean_file(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_mean_file(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_mean_file();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_mean_file();
  public native void set_allocated_mean_file(@StdString @Cast({"char*", "std::string*"}) BytePointer mean_file);

  // repeated float mean_value = 5;
  public native int mean_value_size();
  public native void clear_mean_value();
  @MemberGetter public static native int kMeanValueFieldNumber();
  public static final int kMeanValueFieldNumber = kMeanValueFieldNumber();
  public native float mean_value(int index);
  public native void set_mean_value(int index, float value);
  public native void add_mean_value(float value);

  // optional bool force_color = 6 [default = false];
  public native @Cast("bool") boolean has_force_color();
  public native void clear_force_color();
  @MemberGetter public static native int kForceColorFieldNumber();
  public static final int kForceColorFieldNumber = kForceColorFieldNumber();
  public native @Cast("bool") boolean force_color();
  public native void set_force_color(@Cast("bool") boolean value);

  // optional bool force_gray = 7 [default = false];
  public native @Cast("bool") boolean has_force_gray();
  public native void clear_force_gray();
  @MemberGetter public static native int kForceGrayFieldNumber();
  public static final int kForceGrayFieldNumber = kForceGrayFieldNumber();
  public native @Cast("bool") boolean force_gray();
  public native void set_force_gray(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class LossParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LossParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LossParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public LossParameter position(int position) {
        return (LossParameter)super.position(position);
    }

  public LossParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public LossParameter(@Const @ByRef LossParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef LossParameter from);

  public native @ByRef @Name("operator =") LossParameter put(@Const @ByRef LossParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef LossParameter default_instance();

  public native void Swap(LossParameter other);

  // implements Message ----------------------------------------------

  public native LossParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef LossParameter from);
  public native void MergeFrom(@Const @ByRef LossParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::LossParameter::NormalizationMode") int FULL();
  public static final int FULL = FULL();
  @MemberGetter public static native @Cast("const caffe::LossParameter::NormalizationMode") int VALID();
  public static final int VALID = VALID();
  @MemberGetter public static native @Cast("const caffe::LossParameter::NormalizationMode") int BATCH_SIZE();
  public static final int BATCH_SIZE = BATCH_SIZE();
  @MemberGetter public static native @Cast("const caffe::LossParameter::NormalizationMode") int NONE();
  public static final int NONE = NONE();
  public static native @Cast("bool") boolean NormalizationMode_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::LossParameter::NormalizationMode") int NormalizationMode_MIN();
  public static final int NormalizationMode_MIN = NormalizationMode_MIN();
  @MemberGetter public static native @Cast("const caffe::LossParameter::NormalizationMode") int NormalizationMode_MAX();
  public static final int NormalizationMode_MAX = NormalizationMode_MAX();
  @MemberGetter public static native int NormalizationMode_ARRAYSIZE();
  public static final int NormalizationMode_ARRAYSIZE = NormalizationMode_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer NormalizationMode_descriptor();
  public static native @StdString BytePointer NormalizationMode_Name(@Cast("caffe::LossParameter::NormalizationMode") int value);
  public static native @Cast("bool") boolean NormalizationMode_Parse(@StdString BytePointer name,
        @Cast("caffe::LossParameter::NormalizationMode*") IntPointer value);
  public static native @Cast("bool") boolean NormalizationMode_Parse(@StdString String name,
        @Cast("caffe::LossParameter::NormalizationMode*") IntBuffer value);
  public static native @Cast("bool") boolean NormalizationMode_Parse(@StdString BytePointer name,
        @Cast("caffe::LossParameter::NormalizationMode*") int[] value);
  public static native @Cast("bool") boolean NormalizationMode_Parse(@StdString String name,
        @Cast("caffe::LossParameter::NormalizationMode*") IntPointer value);
  public static native @Cast("bool") boolean NormalizationMode_Parse(@StdString BytePointer name,
        @Cast("caffe::LossParameter::NormalizationMode*") IntBuffer value);
  public static native @Cast("bool") boolean NormalizationMode_Parse(@StdString String name,
        @Cast("caffe::LossParameter::NormalizationMode*") int[] value);

  // accessors -------------------------------------------------------

  // optional int32 ignore_label = 1;
  public native @Cast("bool") boolean has_ignore_label();
  public native void clear_ignore_label();
  @MemberGetter public static native int kIgnoreLabelFieldNumber();
  public static final int kIgnoreLabelFieldNumber = kIgnoreLabelFieldNumber();
  public native @Cast("google::protobuf::int32") int ignore_label();
  public native void set_ignore_label(@Cast("google::protobuf::int32") int value);

  // optional .caffe.LossParameter.NormalizationMode normalization = 3 [default = VALID];
  public native @Cast("bool") boolean has_normalization();
  public native void clear_normalization();
  @MemberGetter public static native int kNormalizationFieldNumber();
  public static final int kNormalizationFieldNumber = kNormalizationFieldNumber();
  public native @Cast("caffe::LossParameter_NormalizationMode") int normalization();
  public native void set_normalization(@Cast("caffe::LossParameter_NormalizationMode") int value);

  // optional bool normalize = 2;
  public native @Cast("bool") boolean has_normalize();
  public native void clear_normalize();
  @MemberGetter public static native int kNormalizeFieldNumber();
  public static final int kNormalizeFieldNumber = kNormalizeFieldNumber();
  public native @Cast("bool") boolean normalize();
  public native void set_normalize(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class AccuracyParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AccuracyParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public AccuracyParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public AccuracyParameter position(int position) {
        return (AccuracyParameter)super.position(position);
    }

  public AccuracyParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public AccuracyParameter(@Const @ByRef AccuracyParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef AccuracyParameter from);

  public native @ByRef @Name("operator =") AccuracyParameter put(@Const @ByRef AccuracyParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef AccuracyParameter default_instance();

  public native void Swap(AccuracyParameter other);

  // implements Message ----------------------------------------------

  public native AccuracyParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef AccuracyParameter from);
  public native void MergeFrom(@Const @ByRef AccuracyParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional uint32 top_k = 1 [default = 1];
  public native @Cast("bool") boolean has_top_k();
  public native void clear_top_k();
  @MemberGetter public static native int kTopKFieldNumber();
  public static final int kTopKFieldNumber = kTopKFieldNumber();
  public native @Cast("google::protobuf::uint32") int top_k();
  public native void set_top_k(@Cast("google::protobuf::uint32") int value);

  // optional int32 axis = 2 [default = 1];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);

  // optional int32 ignore_label = 3;
  public native @Cast("bool") boolean has_ignore_label();
  public native void clear_ignore_label();
  @MemberGetter public static native int kIgnoreLabelFieldNumber();
  public static final int kIgnoreLabelFieldNumber = kIgnoreLabelFieldNumber();
  public native @Cast("google::protobuf::int32") int ignore_label();
  public native void set_ignore_label(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ArgMaxParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ArgMaxParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ArgMaxParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ArgMaxParameter position(int position) {
        return (ArgMaxParameter)super.position(position);
    }

  public ArgMaxParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ArgMaxParameter(@Const @ByRef ArgMaxParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ArgMaxParameter from);

  public native @ByRef @Name("operator =") ArgMaxParameter put(@Const @ByRef ArgMaxParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ArgMaxParameter default_instance();

  public native void Swap(ArgMaxParameter other);

  // implements Message ----------------------------------------------

  public native ArgMaxParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ArgMaxParameter from);
  public native void MergeFrom(@Const @ByRef ArgMaxParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional bool out_max_val = 1 [default = false];
  public native @Cast("bool") boolean has_out_max_val();
  public native void clear_out_max_val();
  @MemberGetter public static native int kOutMaxValFieldNumber();
  public static final int kOutMaxValFieldNumber = kOutMaxValFieldNumber();
  public native @Cast("bool") boolean out_max_val();
  public native void set_out_max_val(@Cast("bool") boolean value);

  // optional uint32 top_k = 2 [default = 1];
  public native @Cast("bool") boolean has_top_k();
  public native void clear_top_k();
  @MemberGetter public static native int kTopKFieldNumber();
  public static final int kTopKFieldNumber = kTopKFieldNumber();
  public native @Cast("google::protobuf::uint32") int top_k();
  public native void set_top_k(@Cast("google::protobuf::uint32") int value);

  // optional int32 axis = 3;
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ConcatParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ConcatParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ConcatParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ConcatParameter position(int position) {
        return (ConcatParameter)super.position(position);
    }

  public ConcatParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ConcatParameter(@Const @ByRef ConcatParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ConcatParameter from);

  public native @ByRef @Name("operator =") ConcatParameter put(@Const @ByRef ConcatParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ConcatParameter default_instance();

  public native void Swap(ConcatParameter other);

  // implements Message ----------------------------------------------

  public native ConcatParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ConcatParameter from);
  public native void MergeFrom(@Const @ByRef ConcatParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional int32 axis = 2 [default = 1];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);

  // optional uint32 concat_dim = 1 [default = 1];
  public native @Cast("bool") boolean has_concat_dim();
  public native void clear_concat_dim();
  @MemberGetter public static native int kConcatDimFieldNumber();
  public static final int kConcatDimFieldNumber = kConcatDimFieldNumber();
  public native @Cast("google::protobuf::uint32") int concat_dim();
  public native void set_concat_dim(@Cast("google::protobuf::uint32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class BatchNormParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BatchNormParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public BatchNormParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public BatchNormParameter position(int position) {
        return (BatchNormParameter)super.position(position);
    }

  public BatchNormParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public BatchNormParameter(@Const @ByRef BatchNormParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef BatchNormParameter from);

  public native @ByRef @Name("operator =") BatchNormParameter put(@Const @ByRef BatchNormParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef BatchNormParameter default_instance();

  public native void Swap(BatchNormParameter other);

  // implements Message ----------------------------------------------

  public native BatchNormParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef BatchNormParameter from);
  public native void MergeFrom(@Const @ByRef BatchNormParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional bool use_global_stats = 1;
  public native @Cast("bool") boolean has_use_global_stats();
  public native void clear_use_global_stats();
  @MemberGetter public static native int kUseGlobalStatsFieldNumber();
  public static final int kUseGlobalStatsFieldNumber = kUseGlobalStatsFieldNumber();
  public native @Cast("bool") boolean use_global_stats();
  public native void set_use_global_stats(@Cast("bool") boolean value);

  // optional float moving_average_fraction = 2 [default = 0.999];
  public native @Cast("bool") boolean has_moving_average_fraction();
  public native void clear_moving_average_fraction();
  @MemberGetter public static native int kMovingAverageFractionFieldNumber();
  public static final int kMovingAverageFractionFieldNumber = kMovingAverageFractionFieldNumber();
  public native float moving_average_fraction();
  public native void set_moving_average_fraction(float value);

  // optional float eps = 3 [default = 1e-05];
  public native @Cast("bool") boolean has_eps();
  public native void clear_eps();
  @MemberGetter public static native int kEpsFieldNumber();
  public static final int kEpsFieldNumber = kEpsFieldNumber();
  public native float eps();
  public native void set_eps(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ContrastiveLossParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ContrastiveLossParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ContrastiveLossParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ContrastiveLossParameter position(int position) {
        return (ContrastiveLossParameter)super.position(position);
    }

  public ContrastiveLossParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ContrastiveLossParameter(@Const @ByRef ContrastiveLossParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ContrastiveLossParameter from);

  public native @ByRef @Name("operator =") ContrastiveLossParameter put(@Const @ByRef ContrastiveLossParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ContrastiveLossParameter default_instance();

  public native void Swap(ContrastiveLossParameter other);

  // implements Message ----------------------------------------------

  public native ContrastiveLossParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ContrastiveLossParameter from);
  public native void MergeFrom(@Const @ByRef ContrastiveLossParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional float margin = 1 [default = 1];
  public native @Cast("bool") boolean has_margin();
  public native void clear_margin();
  @MemberGetter public static native int kMarginFieldNumber();
  public static final int kMarginFieldNumber = kMarginFieldNumber();
  public native float margin();
  public native void set_margin(float value);

  // optional bool legacy_version = 2 [default = false];
  public native @Cast("bool") boolean has_legacy_version();
  public native void clear_legacy_version();
  @MemberGetter public static native int kLegacyVersionFieldNumber();
  public static final int kLegacyVersionFieldNumber = kLegacyVersionFieldNumber();
  public native @Cast("bool") boolean legacy_version();
  public native void set_legacy_version(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ConvolutionParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ConvolutionParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ConvolutionParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ConvolutionParameter position(int position) {
        return (ConvolutionParameter)super.position(position);
    }

  public ConvolutionParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ConvolutionParameter(@Const @ByRef ConvolutionParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ConvolutionParameter from);

  public native @ByRef @Name("operator =") ConvolutionParameter put(@Const @ByRef ConvolutionParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ConvolutionParameter default_instance();

  public native void Swap(ConvolutionParameter other);

  // implements Message ----------------------------------------------

  public native ConvolutionParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ConvolutionParameter from);
  public native void MergeFrom(@Const @ByRef ConvolutionParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::ConvolutionParameter::Engine") int DEFAULT();
  public static final int DEFAULT = DEFAULT();
  @MemberGetter public static native @Cast("const caffe::ConvolutionParameter::Engine") int CAFFE();
  public static final int CAFFE = CAFFE();
  @MemberGetter public static native @Cast("const caffe::ConvolutionParameter::Engine") int CUDNN();
  public static final int CUDNN = CUDNN();
  public static native @Cast("bool") boolean Engine_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::ConvolutionParameter::Engine") int Engine_MIN();
  public static final int Engine_MIN = Engine_MIN();
  @MemberGetter public static native @Cast("const caffe::ConvolutionParameter::Engine") int Engine_MAX();
  public static final int Engine_MAX = Engine_MAX();
  @MemberGetter public static native int Engine_ARRAYSIZE();
  public static final int Engine_ARRAYSIZE = Engine_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Engine_descriptor();
  public static native @StdString BytePointer Engine_Name(@Cast("caffe::ConvolutionParameter::Engine") int value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::ConvolutionParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::ConvolutionParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::ConvolutionParameter::Engine*") int[] value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::ConvolutionParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::ConvolutionParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::ConvolutionParameter::Engine*") int[] value);

  // accessors -------------------------------------------------------

  // optional uint32 num_output = 1;
  public native @Cast("bool") boolean has_num_output();
  public native void clear_num_output();
  @MemberGetter public static native int kNumOutputFieldNumber();
  public static final int kNumOutputFieldNumber = kNumOutputFieldNumber();
  public native @Cast("google::protobuf::uint32") int num_output();
  public native void set_num_output(@Cast("google::protobuf::uint32") int value);

  // optional bool bias_term = 2 [default = true];
  public native @Cast("bool") boolean has_bias_term();
  public native void clear_bias_term();
  @MemberGetter public static native int kBiasTermFieldNumber();
  public static final int kBiasTermFieldNumber = kBiasTermFieldNumber();
  public native @Cast("bool") boolean bias_term();
  public native void set_bias_term(@Cast("bool") boolean value);

  // repeated uint32 pad = 3;
  public native int pad_size();
  public native void clear_pad();
  @MemberGetter public static native int kPadFieldNumber();
  public static final int kPadFieldNumber = kPadFieldNumber();
  public native @Cast("google::protobuf::uint32") int pad(int index);
  public native void set_pad(int index, @Cast("google::protobuf::uint32") int value);
  public native void add_pad(@Cast("google::protobuf::uint32") int value);

  // repeated uint32 kernel_size = 4;
  public native int kernel_size_size();
  public native void clear_kernel_size();
  @MemberGetter public static native int kKernelSizeFieldNumber();
  public static final int kKernelSizeFieldNumber = kKernelSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int kernel_size(int index);
  public native void set_kernel_size(int index, @Cast("google::protobuf::uint32") int value);
  public native void add_kernel_size(@Cast("google::protobuf::uint32") int value);

  // repeated uint32 stride = 6;
  public native int stride_size();
  public native void clear_stride();
  @MemberGetter public static native int kStrideFieldNumber();
  public static final int kStrideFieldNumber = kStrideFieldNumber();
  public native @Cast("google::protobuf::uint32") int stride(int index);
  public native void set_stride(int index, @Cast("google::protobuf::uint32") int value);
  public native void add_stride(@Cast("google::protobuf::uint32") int value);

  // optional uint32 pad_h = 9 [default = 0];
  public native @Cast("bool") boolean has_pad_h();
  public native void clear_pad_h();
  @MemberGetter public static native int kPadHFieldNumber();
  public static final int kPadHFieldNumber = kPadHFieldNumber();
  public native @Cast("google::protobuf::uint32") int pad_h();
  public native void set_pad_h(@Cast("google::protobuf::uint32") int value);

  // optional uint32 pad_w = 10 [default = 0];
  public native @Cast("bool") boolean has_pad_w();
  public native void clear_pad_w();
  @MemberGetter public static native int kPadWFieldNumber();
  public static final int kPadWFieldNumber = kPadWFieldNumber();
  public native @Cast("google::protobuf::uint32") int pad_w();
  public native void set_pad_w(@Cast("google::protobuf::uint32") int value);

  // optional uint32 kernel_h = 11;
  public native @Cast("bool") boolean has_kernel_h();
  public native void clear_kernel_h();
  @MemberGetter public static native int kKernelHFieldNumber();
  public static final int kKernelHFieldNumber = kKernelHFieldNumber();
  public native @Cast("google::protobuf::uint32") int kernel_h();
  public native void set_kernel_h(@Cast("google::protobuf::uint32") int value);

  // optional uint32 kernel_w = 12;
  public native @Cast("bool") boolean has_kernel_w();
  public native void clear_kernel_w();
  @MemberGetter public static native int kKernelWFieldNumber();
  public static final int kKernelWFieldNumber = kKernelWFieldNumber();
  public native @Cast("google::protobuf::uint32") int kernel_w();
  public native void set_kernel_w(@Cast("google::protobuf::uint32") int value);

  // optional uint32 stride_h = 13;
  public native @Cast("bool") boolean has_stride_h();
  public native void clear_stride_h();
  @MemberGetter public static native int kStrideHFieldNumber();
  public static final int kStrideHFieldNumber = kStrideHFieldNumber();
  public native @Cast("google::protobuf::uint32") int stride_h();
  public native void set_stride_h(@Cast("google::protobuf::uint32") int value);

  // optional uint32 stride_w = 14;
  public native @Cast("bool") boolean has_stride_w();
  public native void clear_stride_w();
  @MemberGetter public static native int kStrideWFieldNumber();
  public static final int kStrideWFieldNumber = kStrideWFieldNumber();
  public native @Cast("google::protobuf::uint32") int stride_w();
  public native void set_stride_w(@Cast("google::protobuf::uint32") int value);

  // optional uint32 group = 5 [default = 1];
  public native @Cast("bool") boolean has_group();
  public native void clear_group();
  @MemberGetter public static native int kGroupFieldNumber();
  public static final int kGroupFieldNumber = kGroupFieldNumber();
  public native @Cast("google::protobuf::uint32") int group();
  public native void set_group(@Cast("google::protobuf::uint32") int value);

  // optional .caffe.FillerParameter weight_filler = 7;
  public native @Cast("bool") boolean has_weight_filler();
  public native void clear_weight_filler();
  @MemberGetter public static native int kWeightFillerFieldNumber();
  public static final int kWeightFillerFieldNumber = kWeightFillerFieldNumber();
  public native @Const @ByRef FillerParameter weight_filler();
  public native FillerParameter mutable_weight_filler();
  public native FillerParameter release_weight_filler();
  public native void set_allocated_weight_filler(FillerParameter weight_filler);

  // optional .caffe.FillerParameter bias_filler = 8;
  public native @Cast("bool") boolean has_bias_filler();
  public native void clear_bias_filler();
  @MemberGetter public static native int kBiasFillerFieldNumber();
  public static final int kBiasFillerFieldNumber = kBiasFillerFieldNumber();
  public native @Const @ByRef FillerParameter bias_filler();
  public native FillerParameter mutable_bias_filler();
  public native FillerParameter release_bias_filler();
  public native void set_allocated_bias_filler(FillerParameter bias_filler);

  // optional .caffe.ConvolutionParameter.Engine engine = 15 [default = DEFAULT];
  public native @Cast("bool") boolean has_engine();
  public native void clear_engine();
  @MemberGetter public static native int kEngineFieldNumber();
  public static final int kEngineFieldNumber = kEngineFieldNumber();
  public native @Cast("caffe::ConvolutionParameter_Engine") int engine();
  public native void set_engine(@Cast("caffe::ConvolutionParameter_Engine") int value);

  // optional int32 axis = 16 [default = 1];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);

  // optional bool force_nd_im2col = 17 [default = false];
  public native @Cast("bool") boolean has_force_nd_im2col();
  public native void clear_force_nd_im2col();
  @MemberGetter public static native int kForceNdIm2ColFieldNumber();
  public static final int kForceNdIm2ColFieldNumber = kForceNdIm2ColFieldNumber();
  public native @Cast("bool") boolean force_nd_im2col();
  public native void set_force_nd_im2col(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class DataParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DataParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DataParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DataParameter position(int position) {
        return (DataParameter)super.position(position);
    }

  public DataParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public DataParameter(@Const @ByRef DataParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef DataParameter from);

  public native @ByRef @Name("operator =") DataParameter put(@Const @ByRef DataParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef DataParameter default_instance();

  public native void Swap(DataParameter other);

  // implements Message ----------------------------------------------

  public native DataParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef DataParameter from);
  public native void MergeFrom(@Const @ByRef DataParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::DataParameter::DB") int LEVELDB();
  public static final int LEVELDB = LEVELDB();
  @MemberGetter public static native @Cast("const caffe::DataParameter::DB") int LMDB();
  public static final int LMDB = LMDB();
  public static native @Cast("bool") boolean DB_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::DataParameter::DB") int DB_MIN();
  public static final int DB_MIN = DB_MIN();
  @MemberGetter public static native @Cast("const caffe::DataParameter::DB") int DB_MAX();
  public static final int DB_MAX = DB_MAX();
  @MemberGetter public static native int DB_ARRAYSIZE();
  public static final int DB_ARRAYSIZE = DB_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer DB_descriptor();
  public static native @StdString BytePointer DB_Name(@Cast("caffe::DataParameter::DB") int value);
  public static native @Cast("bool") boolean DB_Parse(@StdString BytePointer name,
        @Cast("caffe::DataParameter::DB*") IntPointer value);
  public static native @Cast("bool") boolean DB_Parse(@StdString String name,
        @Cast("caffe::DataParameter::DB*") IntBuffer value);
  public static native @Cast("bool") boolean DB_Parse(@StdString BytePointer name,
        @Cast("caffe::DataParameter::DB*") int[] value);
  public static native @Cast("bool") boolean DB_Parse(@StdString String name,
        @Cast("caffe::DataParameter::DB*") IntPointer value);
  public static native @Cast("bool") boolean DB_Parse(@StdString BytePointer name,
        @Cast("caffe::DataParameter::DB*") IntBuffer value);
  public static native @Cast("bool") boolean DB_Parse(@StdString String name,
        @Cast("caffe::DataParameter::DB*") int[] value);

  // accessors -------------------------------------------------------

  // optional string source = 1;
  public native @Cast("bool") boolean has_source();
  public native void clear_source();
  @MemberGetter public static native int kSourceFieldNumber();
  public static final int kSourceFieldNumber = kSourceFieldNumber();
  public native @StdString BytePointer source();
  public native void set_source(@StdString BytePointer value);
  public native void set_source(@StdString String value);
  public native void set_source(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_source(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_source();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_source();
  public native void set_allocated_source(@StdString @Cast({"char*", "std::string*"}) BytePointer source);

  // optional uint32 batch_size = 4;
  public native @Cast("bool") boolean has_batch_size();
  public native void clear_batch_size();
  @MemberGetter public static native int kBatchSizeFieldNumber();
  public static final int kBatchSizeFieldNumber = kBatchSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int batch_size();
  public native void set_batch_size(@Cast("google::protobuf::uint32") int value);

  // optional uint32 rand_skip = 7 [default = 0];
  public native @Cast("bool") boolean has_rand_skip();
  public native void clear_rand_skip();
  @MemberGetter public static native int kRandSkipFieldNumber();
  public static final int kRandSkipFieldNumber = kRandSkipFieldNumber();
  public native @Cast("google::protobuf::uint32") int rand_skip();
  public native void set_rand_skip(@Cast("google::protobuf::uint32") int value);

  // optional .caffe.DataParameter.DB backend = 8 [default = LEVELDB];
  public native @Cast("bool") boolean has_backend();
  public native void clear_backend();
  @MemberGetter public static native int kBackendFieldNumber();
  public static final int kBackendFieldNumber = kBackendFieldNumber();
  public native @Cast("caffe::DataParameter_DB") int backend();
  public native void set_backend(@Cast("caffe::DataParameter_DB") int value);

  // optional float scale = 2 [default = 1];
  public native @Cast("bool") boolean has_scale();
  public native void clear_scale();
  @MemberGetter public static native int kScaleFieldNumber();
  public static final int kScaleFieldNumber = kScaleFieldNumber();
  public native float scale();
  public native void set_scale(float value);

  // optional string mean_file = 3;
  public native @Cast("bool") boolean has_mean_file();
  public native void clear_mean_file();
  @MemberGetter public static native int kMeanFileFieldNumber();
  public static final int kMeanFileFieldNumber = kMeanFileFieldNumber();
  public native @StdString BytePointer mean_file();
  public native void set_mean_file(@StdString BytePointer value);
  public native void set_mean_file(@StdString String value);
  public native void set_mean_file(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_mean_file(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_mean_file();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_mean_file();
  public native void set_allocated_mean_file(@StdString @Cast({"char*", "std::string*"}) BytePointer mean_file);

  // optional uint32 crop_size = 5 [default = 0];
  public native @Cast("bool") boolean has_crop_size();
  public native void clear_crop_size();
  @MemberGetter public static native int kCropSizeFieldNumber();
  public static final int kCropSizeFieldNumber = kCropSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int crop_size();
  public native void set_crop_size(@Cast("google::protobuf::uint32") int value);

  // optional bool mirror = 6 [default = false];
  public native @Cast("bool") boolean has_mirror();
  public native void clear_mirror();
  @MemberGetter public static native int kMirrorFieldNumber();
  public static final int kMirrorFieldNumber = kMirrorFieldNumber();
  public native @Cast("bool") boolean mirror();
  public native void set_mirror(@Cast("bool") boolean value);

  // optional bool force_encoded_color = 9 [default = false];
  public native @Cast("bool") boolean has_force_encoded_color();
  public native void clear_force_encoded_color();
  @MemberGetter public static native int kForceEncodedColorFieldNumber();
  public static final int kForceEncodedColorFieldNumber = kForceEncodedColorFieldNumber();
  public native @Cast("bool") boolean force_encoded_color();
  public native void set_force_encoded_color(@Cast("bool") boolean value);

  // optional uint32 prefetch = 10 [default = 4];
  public native @Cast("bool") boolean has_prefetch();
  public native void clear_prefetch();
  @MemberGetter public static native int kPrefetchFieldNumber();
  public static final int kPrefetchFieldNumber = kPrefetchFieldNumber();
  public native @Cast("google::protobuf::uint32") int prefetch();
  public native void set_prefetch(@Cast("google::protobuf::uint32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class DropoutParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DropoutParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DropoutParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DropoutParameter position(int position) {
        return (DropoutParameter)super.position(position);
    }

  public DropoutParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public DropoutParameter(@Const @ByRef DropoutParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef DropoutParameter from);

  public native @ByRef @Name("operator =") DropoutParameter put(@Const @ByRef DropoutParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef DropoutParameter default_instance();

  public native void Swap(DropoutParameter other);

  // implements Message ----------------------------------------------

  public native DropoutParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef DropoutParameter from);
  public native void MergeFrom(@Const @ByRef DropoutParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional float dropout_ratio = 1 [default = 0.5];
  public native @Cast("bool") boolean has_dropout_ratio();
  public native void clear_dropout_ratio();
  @MemberGetter public static native int kDropoutRatioFieldNumber();
  public static final int kDropoutRatioFieldNumber = kDropoutRatioFieldNumber();
  public native float dropout_ratio();
  public native void set_dropout_ratio(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class DummyDataParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DummyDataParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DummyDataParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DummyDataParameter position(int position) {
        return (DummyDataParameter)super.position(position);
    }

  public DummyDataParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public DummyDataParameter(@Const @ByRef DummyDataParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef DummyDataParameter from);

  public native @ByRef @Name("operator =") DummyDataParameter put(@Const @ByRef DummyDataParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef DummyDataParameter default_instance();

  public native void Swap(DummyDataParameter other);

  // implements Message ----------------------------------------------

  public native DummyDataParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef DummyDataParameter from);
  public native void MergeFrom(@Const @ByRef DummyDataParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .caffe.FillerParameter data_filler = 1;
  public native int data_filler_size();
  public native void clear_data_filler();
  @MemberGetter public static native int kDataFillerFieldNumber();
  public static final int kDataFillerFieldNumber = kDataFillerFieldNumber();
  public native @Const @ByRef FillerParameter data_filler(int index);
  public native FillerParameter mutable_data_filler(int index);
  public native FillerParameter add_data_filler();

  // repeated .caffe.BlobShape shape = 6;
  public native int shape_size();
  public native void clear_shape();
  @MemberGetter public static native int kShapeFieldNumber();
  public static final int kShapeFieldNumber = kShapeFieldNumber();
  public native @Const @ByRef BlobShape shape(int index);
  public native BlobShape mutable_shape(int index);
  public native BlobShape add_shape();

  // repeated uint32 num = 2;
  public native int num_size();
  public native void clear_num();
  @MemberGetter public static native int kNumFieldNumber();
  public static final int kNumFieldNumber = kNumFieldNumber();
  public native @Cast("google::protobuf::uint32") int num(int index);
  public native void set_num(int index, @Cast("google::protobuf::uint32") int value);
  public native void add_num(@Cast("google::protobuf::uint32") int value);

  // repeated uint32 channels = 3;
  public native int channels_size();
  public native void clear_channels();
  @MemberGetter public static native int kChannelsFieldNumber();
  public static final int kChannelsFieldNumber = kChannelsFieldNumber();
  public native @Cast("google::protobuf::uint32") int channels(int index);
  public native void set_channels(int index, @Cast("google::protobuf::uint32") int value);
  public native void add_channels(@Cast("google::protobuf::uint32") int value);

  // repeated uint32 height = 4;
  public native int height_size();
  public native void clear_height();
  @MemberGetter public static native int kHeightFieldNumber();
  public static final int kHeightFieldNumber = kHeightFieldNumber();
  public native @Cast("google::protobuf::uint32") int height(int index);
  public native void set_height(int index, @Cast("google::protobuf::uint32") int value);
  public native void add_height(@Cast("google::protobuf::uint32") int value);

  // repeated uint32 width = 5;
  public native int width_size();
  public native void clear_width();
  @MemberGetter public static native int kWidthFieldNumber();
  public static final int kWidthFieldNumber = kWidthFieldNumber();
  public native @Cast("google::protobuf::uint32") int width(int index);
  public native void set_width(int index, @Cast("google::protobuf::uint32") int value);
  public native void add_width(@Cast("google::protobuf::uint32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class EltwiseParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EltwiseParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public EltwiseParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public EltwiseParameter position(int position) {
        return (EltwiseParameter)super.position(position);
    }

  public EltwiseParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public EltwiseParameter(@Const @ByRef EltwiseParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef EltwiseParameter from);

  public native @ByRef @Name("operator =") EltwiseParameter put(@Const @ByRef EltwiseParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef EltwiseParameter default_instance();

  public native void Swap(EltwiseParameter other);

  // implements Message ----------------------------------------------

  public native EltwiseParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef EltwiseParameter from);
  public native void MergeFrom(@Const @ByRef EltwiseParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::EltwiseParameter::EltwiseOp") int PROD();
  public static final int PROD = PROD();
  @MemberGetter public static native @Cast("const caffe::EltwiseParameter::EltwiseOp") int SUM();
  public static final int SUM = SUM();
  @MemberGetter public static native @Cast("const caffe::EltwiseParameter::EltwiseOp") int MAX();
  public static final int MAX = MAX();
  public static native @Cast("bool") boolean EltwiseOp_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::EltwiseParameter::EltwiseOp") int EltwiseOp_MIN();
  public static final int EltwiseOp_MIN = EltwiseOp_MIN();
  @MemberGetter public static native @Cast("const caffe::EltwiseParameter::EltwiseOp") int EltwiseOp_MAX();
  public static final int EltwiseOp_MAX = EltwiseOp_MAX();
  @MemberGetter public static native int EltwiseOp_ARRAYSIZE();
  public static final int EltwiseOp_ARRAYSIZE = EltwiseOp_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer EltwiseOp_descriptor();
  public static native @StdString BytePointer EltwiseOp_Name(@Cast("caffe::EltwiseParameter::EltwiseOp") int value);
  public static native @Cast("bool") boolean EltwiseOp_Parse(@StdString BytePointer name,
        @Cast("caffe::EltwiseParameter::EltwiseOp*") IntPointer value);
  public static native @Cast("bool") boolean EltwiseOp_Parse(@StdString String name,
        @Cast("caffe::EltwiseParameter::EltwiseOp*") IntBuffer value);
  public static native @Cast("bool") boolean EltwiseOp_Parse(@StdString BytePointer name,
        @Cast("caffe::EltwiseParameter::EltwiseOp*") int[] value);
  public static native @Cast("bool") boolean EltwiseOp_Parse(@StdString String name,
        @Cast("caffe::EltwiseParameter::EltwiseOp*") IntPointer value);
  public static native @Cast("bool") boolean EltwiseOp_Parse(@StdString BytePointer name,
        @Cast("caffe::EltwiseParameter::EltwiseOp*") IntBuffer value);
  public static native @Cast("bool") boolean EltwiseOp_Parse(@StdString String name,
        @Cast("caffe::EltwiseParameter::EltwiseOp*") int[] value);

  // accessors -------------------------------------------------------

  // optional .caffe.EltwiseParameter.EltwiseOp operation = 1 [default = SUM];
  public native @Cast("bool") boolean has_operation();
  public native void clear_operation();
  @MemberGetter public static native int kOperationFieldNumber();
  public static final int kOperationFieldNumber = kOperationFieldNumber();
  public native @Cast("caffe::EltwiseParameter_EltwiseOp") int operation();
  public native void set_operation(@Cast("caffe::EltwiseParameter_EltwiseOp") int value);

  // repeated float coeff = 2;
  public native int coeff_size();
  public native void clear_coeff();
  @MemberGetter public static native int kCoeffFieldNumber();
  public static final int kCoeffFieldNumber = kCoeffFieldNumber();
  public native float coeff(int index);
  public native void set_coeff(int index, float value);
  public native void add_coeff(float value);

  // optional bool stable_prod_grad = 3 [default = true];
  public native @Cast("bool") boolean has_stable_prod_grad();
  public native void clear_stable_prod_grad();
  @MemberGetter public static native int kStableProdGradFieldNumber();
  public static final int kStableProdGradFieldNumber = kStableProdGradFieldNumber();
  public native @Cast("bool") boolean stable_prod_grad();
  public native void set_stable_prod_grad(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class EmbedParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EmbedParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public EmbedParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public EmbedParameter position(int position) {
        return (EmbedParameter)super.position(position);
    }

  public EmbedParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public EmbedParameter(@Const @ByRef EmbedParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef EmbedParameter from);

  public native @ByRef @Name("operator =") EmbedParameter put(@Const @ByRef EmbedParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef EmbedParameter default_instance();

  public native void Swap(EmbedParameter other);

  // implements Message ----------------------------------------------

  public native EmbedParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef EmbedParameter from);
  public native void MergeFrom(@Const @ByRef EmbedParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional uint32 num_output = 1;
  public native @Cast("bool") boolean has_num_output();
  public native void clear_num_output();
  @MemberGetter public static native int kNumOutputFieldNumber();
  public static final int kNumOutputFieldNumber = kNumOutputFieldNumber();
  public native @Cast("google::protobuf::uint32") int num_output();
  public native void set_num_output(@Cast("google::protobuf::uint32") int value);

  // optional uint32 input_dim = 2;
  public native @Cast("bool") boolean has_input_dim();
  public native void clear_input_dim();
  @MemberGetter public static native int kInputDimFieldNumber();
  public static final int kInputDimFieldNumber = kInputDimFieldNumber();
  public native @Cast("google::protobuf::uint32") int input_dim();
  public native void set_input_dim(@Cast("google::protobuf::uint32") int value);

  // optional bool bias_term = 3 [default = true];
  public native @Cast("bool") boolean has_bias_term();
  public native void clear_bias_term();
  @MemberGetter public static native int kBiasTermFieldNumber();
  public static final int kBiasTermFieldNumber = kBiasTermFieldNumber();
  public native @Cast("bool") boolean bias_term();
  public native void set_bias_term(@Cast("bool") boolean value);

  // optional .caffe.FillerParameter weight_filler = 4;
  public native @Cast("bool") boolean has_weight_filler();
  public native void clear_weight_filler();
  @MemberGetter public static native int kWeightFillerFieldNumber();
  public static final int kWeightFillerFieldNumber = kWeightFillerFieldNumber();
  public native @Const @ByRef FillerParameter weight_filler();
  public native FillerParameter mutable_weight_filler();
  public native FillerParameter release_weight_filler();
  public native void set_allocated_weight_filler(FillerParameter weight_filler);

  // optional .caffe.FillerParameter bias_filler = 5;
  public native @Cast("bool") boolean has_bias_filler();
  public native void clear_bias_filler();
  @MemberGetter public static native int kBiasFillerFieldNumber();
  public static final int kBiasFillerFieldNumber = kBiasFillerFieldNumber();
  public native @Const @ByRef FillerParameter bias_filler();
  public native FillerParameter mutable_bias_filler();
  public native FillerParameter release_bias_filler();
  public native void set_allocated_bias_filler(FillerParameter bias_filler);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ExpParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ExpParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ExpParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ExpParameter position(int position) {
        return (ExpParameter)super.position(position);
    }

  public ExpParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ExpParameter(@Const @ByRef ExpParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ExpParameter from);

  public native @ByRef @Name("operator =") ExpParameter put(@Const @ByRef ExpParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ExpParameter default_instance();

  public native void Swap(ExpParameter other);

  // implements Message ----------------------------------------------

  public native ExpParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ExpParameter from);
  public native void MergeFrom(@Const @ByRef ExpParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional float base = 1 [default = -1];
  public native @Cast("bool") boolean has_base();
  public native void clear_base();
  @MemberGetter public static native int kBaseFieldNumber();
  public static final int kBaseFieldNumber = kBaseFieldNumber();
  public native float base();
  public native void set_base(float value);

  // optional float scale = 2 [default = 1];
  public native @Cast("bool") boolean has_scale();
  public native void clear_scale();
  @MemberGetter public static native int kScaleFieldNumber();
  public static final int kScaleFieldNumber = kScaleFieldNumber();
  public native float scale();
  public native void set_scale(float value);

  // optional float shift = 3 [default = 0];
  public native @Cast("bool") boolean has_shift();
  public native void clear_shift();
  @MemberGetter public static native int kShiftFieldNumber();
  public static final int kShiftFieldNumber = kShiftFieldNumber();
  public native float shift();
  public native void set_shift(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class FlattenParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FlattenParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FlattenParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FlattenParameter position(int position) {
        return (FlattenParameter)super.position(position);
    }

  public FlattenParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public FlattenParameter(@Const @ByRef FlattenParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef FlattenParameter from);

  public native @ByRef @Name("operator =") FlattenParameter put(@Const @ByRef FlattenParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef FlattenParameter default_instance();

  public native void Swap(FlattenParameter other);

  // implements Message ----------------------------------------------

  public native FlattenParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef FlattenParameter from);
  public native void MergeFrom(@Const @ByRef FlattenParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional int32 axis = 1 [default = 1];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);

  // optional int32 end_axis = 2 [default = -1];
  public native @Cast("bool") boolean has_end_axis();
  public native void clear_end_axis();
  @MemberGetter public static native int kEndAxisFieldNumber();
  public static final int kEndAxisFieldNumber = kEndAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int end_axis();
  public native void set_end_axis(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class HDF5DataParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HDF5DataParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public HDF5DataParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public HDF5DataParameter position(int position) {
        return (HDF5DataParameter)super.position(position);
    }

  public HDF5DataParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public HDF5DataParameter(@Const @ByRef HDF5DataParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef HDF5DataParameter from);

  public native @ByRef @Name("operator =") HDF5DataParameter put(@Const @ByRef HDF5DataParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef HDF5DataParameter default_instance();

  public native void Swap(HDF5DataParameter other);

  // implements Message ----------------------------------------------

  public native HDF5DataParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef HDF5DataParameter from);
  public native void MergeFrom(@Const @ByRef HDF5DataParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string source = 1;
  public native @Cast("bool") boolean has_source();
  public native void clear_source();
  @MemberGetter public static native int kSourceFieldNumber();
  public static final int kSourceFieldNumber = kSourceFieldNumber();
  public native @StdString BytePointer source();
  public native void set_source(@StdString BytePointer value);
  public native void set_source(@StdString String value);
  public native void set_source(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_source(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_source();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_source();
  public native void set_allocated_source(@StdString @Cast({"char*", "std::string*"}) BytePointer source);

  // optional uint32 batch_size = 2;
  public native @Cast("bool") boolean has_batch_size();
  public native void clear_batch_size();
  @MemberGetter public static native int kBatchSizeFieldNumber();
  public static final int kBatchSizeFieldNumber = kBatchSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int batch_size();
  public native void set_batch_size(@Cast("google::protobuf::uint32") int value);

  // optional bool shuffle = 3 [default = false];
  public native @Cast("bool") boolean has_shuffle();
  public native void clear_shuffle();
  @MemberGetter public static native int kShuffleFieldNumber();
  public static final int kShuffleFieldNumber = kShuffleFieldNumber();
  public native @Cast("bool") boolean shuffle();
  public native void set_shuffle(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class HDF5OutputParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HDF5OutputParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public HDF5OutputParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public HDF5OutputParameter position(int position) {
        return (HDF5OutputParameter)super.position(position);
    }

  public HDF5OutputParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public HDF5OutputParameter(@Const @ByRef HDF5OutputParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef HDF5OutputParameter from);

  public native @ByRef @Name("operator =") HDF5OutputParameter put(@Const @ByRef HDF5OutputParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef HDF5OutputParameter default_instance();

  public native void Swap(HDF5OutputParameter other);

  // implements Message ----------------------------------------------

  public native HDF5OutputParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef HDF5OutputParameter from);
  public native void MergeFrom(@Const @ByRef HDF5OutputParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string file_name = 1;
  public native @Cast("bool") boolean has_file_name();
  public native void clear_file_name();
  @MemberGetter public static native int kFileNameFieldNumber();
  public static final int kFileNameFieldNumber = kFileNameFieldNumber();
  public native @StdString BytePointer file_name();
  public native void set_file_name(@StdString BytePointer value);
  public native void set_file_name(@StdString String value);
  public native void set_file_name(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_file_name(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_file_name();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_file_name();
  public native void set_allocated_file_name(@StdString @Cast({"char*", "std::string*"}) BytePointer file_name);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class HingeLossParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HingeLossParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public HingeLossParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public HingeLossParameter position(int position) {
        return (HingeLossParameter)super.position(position);
    }

  public HingeLossParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public HingeLossParameter(@Const @ByRef HingeLossParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef HingeLossParameter from);

  public native @ByRef @Name("operator =") HingeLossParameter put(@Const @ByRef HingeLossParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef HingeLossParameter default_instance();

  public native void Swap(HingeLossParameter other);

  // implements Message ----------------------------------------------

  public native HingeLossParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef HingeLossParameter from);
  public native void MergeFrom(@Const @ByRef HingeLossParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::HingeLossParameter::Norm") int L1();
  public static final int L1 = L1();
  @MemberGetter public static native @Cast("const caffe::HingeLossParameter::Norm") int L2();
  public static final int L2 = L2();
  public static native @Cast("bool") boolean Norm_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::HingeLossParameter::Norm") int Norm_MIN();
  public static final int Norm_MIN = Norm_MIN();
  @MemberGetter public static native @Cast("const caffe::HingeLossParameter::Norm") int Norm_MAX();
  public static final int Norm_MAX = Norm_MAX();
  @MemberGetter public static native int Norm_ARRAYSIZE();
  public static final int Norm_ARRAYSIZE = Norm_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Norm_descriptor();
  public static native @StdString BytePointer Norm_Name(@Cast("caffe::HingeLossParameter::Norm") int value);
  public static native @Cast("bool") boolean Norm_Parse(@StdString BytePointer name,
        @Cast("caffe::HingeLossParameter::Norm*") IntPointer value);
  public static native @Cast("bool") boolean Norm_Parse(@StdString String name,
        @Cast("caffe::HingeLossParameter::Norm*") IntBuffer value);
  public static native @Cast("bool") boolean Norm_Parse(@StdString BytePointer name,
        @Cast("caffe::HingeLossParameter::Norm*") int[] value);
  public static native @Cast("bool") boolean Norm_Parse(@StdString String name,
        @Cast("caffe::HingeLossParameter::Norm*") IntPointer value);
  public static native @Cast("bool") boolean Norm_Parse(@StdString BytePointer name,
        @Cast("caffe::HingeLossParameter::Norm*") IntBuffer value);
  public static native @Cast("bool") boolean Norm_Parse(@StdString String name,
        @Cast("caffe::HingeLossParameter::Norm*") int[] value);

  // accessors -------------------------------------------------------

  // optional .caffe.HingeLossParameter.Norm norm = 1 [default = L1];
  public native @Cast("bool") boolean has_norm();
  public native void clear_norm();
  @MemberGetter public static native int kNormFieldNumber();
  public static final int kNormFieldNumber = kNormFieldNumber();
  public native @Cast("caffe::HingeLossParameter_Norm") int norm();
  public native void set_norm(@Cast("caffe::HingeLossParameter_Norm") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ImageDataParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ImageDataParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ImageDataParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ImageDataParameter position(int position) {
        return (ImageDataParameter)super.position(position);
    }

  public ImageDataParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ImageDataParameter(@Const @ByRef ImageDataParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ImageDataParameter from);

  public native @ByRef @Name("operator =") ImageDataParameter put(@Const @ByRef ImageDataParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ImageDataParameter default_instance();

  public native void Swap(ImageDataParameter other);

  // implements Message ----------------------------------------------

  public native ImageDataParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ImageDataParameter from);
  public native void MergeFrom(@Const @ByRef ImageDataParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string source = 1;
  public native @Cast("bool") boolean has_source();
  public native void clear_source();
  @MemberGetter public static native int kSourceFieldNumber();
  public static final int kSourceFieldNumber = kSourceFieldNumber();
  public native @StdString BytePointer source();
  public native void set_source(@StdString BytePointer value);
  public native void set_source(@StdString String value);
  public native void set_source(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_source(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_source();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_source();
  public native void set_allocated_source(@StdString @Cast({"char*", "std::string*"}) BytePointer source);

  // optional uint32 batch_size = 4 [default = 1];
  public native @Cast("bool") boolean has_batch_size();
  public native void clear_batch_size();
  @MemberGetter public static native int kBatchSizeFieldNumber();
  public static final int kBatchSizeFieldNumber = kBatchSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int batch_size();
  public native void set_batch_size(@Cast("google::protobuf::uint32") int value);

  // optional uint32 rand_skip = 7 [default = 0];
  public native @Cast("bool") boolean has_rand_skip();
  public native void clear_rand_skip();
  @MemberGetter public static native int kRandSkipFieldNumber();
  public static final int kRandSkipFieldNumber = kRandSkipFieldNumber();
  public native @Cast("google::protobuf::uint32") int rand_skip();
  public native void set_rand_skip(@Cast("google::protobuf::uint32") int value);

  // optional bool shuffle = 8 [default = false];
  public native @Cast("bool") boolean has_shuffle();
  public native void clear_shuffle();
  @MemberGetter public static native int kShuffleFieldNumber();
  public static final int kShuffleFieldNumber = kShuffleFieldNumber();
  public native @Cast("bool") boolean shuffle();
  public native void set_shuffle(@Cast("bool") boolean value);

  // optional uint32 new_height = 9 [default = 0];
  public native @Cast("bool") boolean has_new_height();
  public native void clear_new_height();
  @MemberGetter public static native int kNewHeightFieldNumber();
  public static final int kNewHeightFieldNumber = kNewHeightFieldNumber();
  public native @Cast("google::protobuf::uint32") int new_height();
  public native void set_new_height(@Cast("google::protobuf::uint32") int value);

  // optional uint32 new_width = 10 [default = 0];
  public native @Cast("bool") boolean has_new_width();
  public native void clear_new_width();
  @MemberGetter public static native int kNewWidthFieldNumber();
  public static final int kNewWidthFieldNumber = kNewWidthFieldNumber();
  public native @Cast("google::protobuf::uint32") int new_width();
  public native void set_new_width(@Cast("google::protobuf::uint32") int value);

  // optional bool is_color = 11 [default = true];
  public native @Cast("bool") boolean has_is_color();
  public native void clear_is_color();
  @MemberGetter public static native int kIsColorFieldNumber();
  public static final int kIsColorFieldNumber = kIsColorFieldNumber();
  public native @Cast("bool") boolean is_color();
  public native void set_is_color(@Cast("bool") boolean value);

  // optional float scale = 2 [default = 1];
  public native @Cast("bool") boolean has_scale();
  public native void clear_scale();
  @MemberGetter public static native int kScaleFieldNumber();
  public static final int kScaleFieldNumber = kScaleFieldNumber();
  public native float scale();
  public native void set_scale(float value);

  // optional string mean_file = 3;
  public native @Cast("bool") boolean has_mean_file();
  public native void clear_mean_file();
  @MemberGetter public static native int kMeanFileFieldNumber();
  public static final int kMeanFileFieldNumber = kMeanFileFieldNumber();
  public native @StdString BytePointer mean_file();
  public native void set_mean_file(@StdString BytePointer value);
  public native void set_mean_file(@StdString String value);
  public native void set_mean_file(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_mean_file(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_mean_file();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_mean_file();
  public native void set_allocated_mean_file(@StdString @Cast({"char*", "std::string*"}) BytePointer mean_file);

  // optional uint32 crop_size = 5 [default = 0];
  public native @Cast("bool") boolean has_crop_size();
  public native void clear_crop_size();
  @MemberGetter public static native int kCropSizeFieldNumber();
  public static final int kCropSizeFieldNumber = kCropSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int crop_size();
  public native void set_crop_size(@Cast("google::protobuf::uint32") int value);

  // optional bool mirror = 6 [default = false];
  public native @Cast("bool") boolean has_mirror();
  public native void clear_mirror();
  @MemberGetter public static native int kMirrorFieldNumber();
  public static final int kMirrorFieldNumber = kMirrorFieldNumber();
  public native @Cast("bool") boolean mirror();
  public native void set_mirror(@Cast("bool") boolean value);

  // optional string root_folder = 12 [default = ""];
  public native @Cast("bool") boolean has_root_folder();
  public native void clear_root_folder();
  @MemberGetter public static native int kRootFolderFieldNumber();
  public static final int kRootFolderFieldNumber = kRootFolderFieldNumber();
  public native @StdString BytePointer root_folder();
  public native void set_root_folder(@StdString BytePointer value);
  public native void set_root_folder(@StdString String value);
  public native void set_root_folder(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_root_folder(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_root_folder();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_root_folder();
  public native void set_allocated_root_folder(@StdString @Cast({"char*", "std::string*"}) BytePointer root_folder);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class InfogainLossParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public InfogainLossParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public InfogainLossParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public InfogainLossParameter position(int position) {
        return (InfogainLossParameter)super.position(position);
    }

  public InfogainLossParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public InfogainLossParameter(@Const @ByRef InfogainLossParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef InfogainLossParameter from);

  public native @ByRef @Name("operator =") InfogainLossParameter put(@Const @ByRef InfogainLossParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef InfogainLossParameter default_instance();

  public native void Swap(InfogainLossParameter other);

  // implements Message ----------------------------------------------

  public native InfogainLossParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef InfogainLossParameter from);
  public native void MergeFrom(@Const @ByRef InfogainLossParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string source = 1;
  public native @Cast("bool") boolean has_source();
  public native void clear_source();
  @MemberGetter public static native int kSourceFieldNumber();
  public static final int kSourceFieldNumber = kSourceFieldNumber();
  public native @StdString BytePointer source();
  public native void set_source(@StdString BytePointer value);
  public native void set_source(@StdString String value);
  public native void set_source(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_source(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_source();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_source();
  public native void set_allocated_source(@StdString @Cast({"char*", "std::string*"}) BytePointer source);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class InnerProductParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public InnerProductParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public InnerProductParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public InnerProductParameter position(int position) {
        return (InnerProductParameter)super.position(position);
    }

  public InnerProductParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public InnerProductParameter(@Const @ByRef InnerProductParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef InnerProductParameter from);

  public native @ByRef @Name("operator =") InnerProductParameter put(@Const @ByRef InnerProductParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef InnerProductParameter default_instance();

  public native void Swap(InnerProductParameter other);

  // implements Message ----------------------------------------------

  public native InnerProductParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef InnerProductParameter from);
  public native void MergeFrom(@Const @ByRef InnerProductParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional uint32 num_output = 1;
  public native @Cast("bool") boolean has_num_output();
  public native void clear_num_output();
  @MemberGetter public static native int kNumOutputFieldNumber();
  public static final int kNumOutputFieldNumber = kNumOutputFieldNumber();
  public native @Cast("google::protobuf::uint32") int num_output();
  public native void set_num_output(@Cast("google::protobuf::uint32") int value);

  // optional bool bias_term = 2 [default = true];
  public native @Cast("bool") boolean has_bias_term();
  public native void clear_bias_term();
  @MemberGetter public static native int kBiasTermFieldNumber();
  public static final int kBiasTermFieldNumber = kBiasTermFieldNumber();
  public native @Cast("bool") boolean bias_term();
  public native void set_bias_term(@Cast("bool") boolean value);

  // optional .caffe.FillerParameter weight_filler = 3;
  public native @Cast("bool") boolean has_weight_filler();
  public native void clear_weight_filler();
  @MemberGetter public static native int kWeightFillerFieldNumber();
  public static final int kWeightFillerFieldNumber = kWeightFillerFieldNumber();
  public native @Const @ByRef FillerParameter weight_filler();
  public native FillerParameter mutable_weight_filler();
  public native FillerParameter release_weight_filler();
  public native void set_allocated_weight_filler(FillerParameter weight_filler);

  // optional .caffe.FillerParameter bias_filler = 4;
  public native @Cast("bool") boolean has_bias_filler();
  public native void clear_bias_filler();
  @MemberGetter public static native int kBiasFillerFieldNumber();
  public static final int kBiasFillerFieldNumber = kBiasFillerFieldNumber();
  public native @Const @ByRef FillerParameter bias_filler();
  public native FillerParameter mutable_bias_filler();
  public native FillerParameter release_bias_filler();
  public native void set_allocated_bias_filler(FillerParameter bias_filler);

  // optional int32 axis = 5 [default = 1];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class LogParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LogParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LogParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public LogParameter position(int position) {
        return (LogParameter)super.position(position);
    }

  public LogParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public LogParameter(@Const @ByRef LogParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef LogParameter from);

  public native @ByRef @Name("operator =") LogParameter put(@Const @ByRef LogParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef LogParameter default_instance();

  public native void Swap(LogParameter other);

  // implements Message ----------------------------------------------

  public native LogParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef LogParameter from);
  public native void MergeFrom(@Const @ByRef LogParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional float base = 1 [default = -1];
  public native @Cast("bool") boolean has_base();
  public native void clear_base();
  @MemberGetter public static native int kBaseFieldNumber();
  public static final int kBaseFieldNumber = kBaseFieldNumber();
  public native float base();
  public native void set_base(float value);

  // optional float scale = 2 [default = 1];
  public native @Cast("bool") boolean has_scale();
  public native void clear_scale();
  @MemberGetter public static native int kScaleFieldNumber();
  public static final int kScaleFieldNumber = kScaleFieldNumber();
  public native float scale();
  public native void set_scale(float value);

  // optional float shift = 3 [default = 0];
  public native @Cast("bool") boolean has_shift();
  public native void clear_shift();
  @MemberGetter public static native int kShiftFieldNumber();
  public static final int kShiftFieldNumber = kShiftFieldNumber();
  public native float shift();
  public native void set_shift(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class LRNParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LRNParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LRNParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public LRNParameter position(int position) {
        return (LRNParameter)super.position(position);
    }

  public LRNParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public LRNParameter(@Const @ByRef LRNParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef LRNParameter from);

  public native @ByRef @Name("operator =") LRNParameter put(@Const @ByRef LRNParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef LRNParameter default_instance();

  public native void Swap(LRNParameter other);

  // implements Message ----------------------------------------------

  public native LRNParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef LRNParameter from);
  public native void MergeFrom(@Const @ByRef LRNParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::LRNParameter::NormRegion") int ACROSS_CHANNELS();
  public static final int ACROSS_CHANNELS = ACROSS_CHANNELS();
  @MemberGetter public static native @Cast("const caffe::LRNParameter::NormRegion") int WITHIN_CHANNEL();
  public static final int WITHIN_CHANNEL = WITHIN_CHANNEL();
  public static native @Cast("bool") boolean NormRegion_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::LRNParameter::NormRegion") int NormRegion_MIN();
  public static final int NormRegion_MIN = NormRegion_MIN();
  @MemberGetter public static native @Cast("const caffe::LRNParameter::NormRegion") int NormRegion_MAX();
  public static final int NormRegion_MAX = NormRegion_MAX();
  @MemberGetter public static native int NormRegion_ARRAYSIZE();
  public static final int NormRegion_ARRAYSIZE = NormRegion_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer NormRegion_descriptor();
  public static native @StdString BytePointer NormRegion_Name(@Cast("caffe::LRNParameter::NormRegion") int value);
  public static native @Cast("bool") boolean NormRegion_Parse(@StdString BytePointer name,
        @Cast("caffe::LRNParameter::NormRegion*") IntPointer value);
  public static native @Cast("bool") boolean NormRegion_Parse(@StdString String name,
        @Cast("caffe::LRNParameter::NormRegion*") IntBuffer value);
  public static native @Cast("bool") boolean NormRegion_Parse(@StdString BytePointer name,
        @Cast("caffe::LRNParameter::NormRegion*") int[] value);
  public static native @Cast("bool") boolean NormRegion_Parse(@StdString String name,
        @Cast("caffe::LRNParameter::NormRegion*") IntPointer value);
  public static native @Cast("bool") boolean NormRegion_Parse(@StdString BytePointer name,
        @Cast("caffe::LRNParameter::NormRegion*") IntBuffer value);
  public static native @Cast("bool") boolean NormRegion_Parse(@StdString String name,
        @Cast("caffe::LRNParameter::NormRegion*") int[] value);
  @MemberGetter public static native @Cast("const caffe::LRNParameter::Engine") int DEFAULT();
  public static final int DEFAULT = DEFAULT();
  @MemberGetter public static native @Cast("const caffe::LRNParameter::Engine") int CAFFE();
  public static final int CAFFE = CAFFE();
  @MemberGetter public static native @Cast("const caffe::LRNParameter::Engine") int CUDNN();
  public static final int CUDNN = CUDNN();
  public static native @Cast("bool") boolean Engine_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::LRNParameter::Engine") int Engine_MIN();
  public static final int Engine_MIN = Engine_MIN();
  @MemberGetter public static native @Cast("const caffe::LRNParameter::Engine") int Engine_MAX();
  public static final int Engine_MAX = Engine_MAX();
  @MemberGetter public static native int Engine_ARRAYSIZE();
  public static final int Engine_ARRAYSIZE = Engine_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Engine_descriptor();
  public static native @StdString BytePointer Engine_Name(@Cast("caffe::LRNParameter::Engine") int value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::LRNParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::LRNParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::LRNParameter::Engine*") int[] value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::LRNParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::LRNParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::LRNParameter::Engine*") int[] value);

  // accessors -------------------------------------------------------

  // optional uint32 local_size = 1 [default = 5];
  public native @Cast("bool") boolean has_local_size();
  public native void clear_local_size();
  @MemberGetter public static native int kLocalSizeFieldNumber();
  public static final int kLocalSizeFieldNumber = kLocalSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int local_size();
  public native void set_local_size(@Cast("google::protobuf::uint32") int value);

  // optional float alpha = 2 [default = 1];
  public native @Cast("bool") boolean has_alpha();
  public native void clear_alpha();
  @MemberGetter public static native int kAlphaFieldNumber();
  public static final int kAlphaFieldNumber = kAlphaFieldNumber();
  public native float alpha();
  public native void set_alpha(float value);

  // optional float beta = 3 [default = 0.75];
  public native @Cast("bool") boolean has_beta();
  public native void clear_beta();
  @MemberGetter public static native int kBetaFieldNumber();
  public static final int kBetaFieldNumber = kBetaFieldNumber();
  public native float beta();
  public native void set_beta(float value);

  // optional .caffe.LRNParameter.NormRegion norm_region = 4 [default = ACROSS_CHANNELS];
  public native @Cast("bool") boolean has_norm_region();
  public native void clear_norm_region();
  @MemberGetter public static native int kNormRegionFieldNumber();
  public static final int kNormRegionFieldNumber = kNormRegionFieldNumber();
  public native @Cast("caffe::LRNParameter_NormRegion") int norm_region();
  public native void set_norm_region(@Cast("caffe::LRNParameter_NormRegion") int value);

  // optional float k = 5 [default = 1];
  public native @Cast("bool") boolean has_k();
  public native void clear_k();
  @MemberGetter public static native int kKFieldNumber();
  public static final int kKFieldNumber = kKFieldNumber();
  public native float k();
  public native void set_k(float value);

  // optional .caffe.LRNParameter.Engine engine = 6 [default = DEFAULT];
  public native @Cast("bool") boolean has_engine();
  public native void clear_engine();
  @MemberGetter public static native int kEngineFieldNumber();
  public static final int kEngineFieldNumber = kEngineFieldNumber();
  public native @Cast("caffe::LRNParameter_Engine") int engine();
  public native void set_engine(@Cast("caffe::LRNParameter_Engine") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class MemoryDataParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MemoryDataParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public MemoryDataParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public MemoryDataParameter position(int position) {
        return (MemoryDataParameter)super.position(position);
    }

  public MemoryDataParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public MemoryDataParameter(@Const @ByRef MemoryDataParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef MemoryDataParameter from);

  public native @ByRef @Name("operator =") MemoryDataParameter put(@Const @ByRef MemoryDataParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef MemoryDataParameter default_instance();

  public native void Swap(MemoryDataParameter other);

  // implements Message ----------------------------------------------

  public native MemoryDataParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef MemoryDataParameter from);
  public native void MergeFrom(@Const @ByRef MemoryDataParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional uint32 batch_size = 1;
  public native @Cast("bool") boolean has_batch_size();
  public native void clear_batch_size();
  @MemberGetter public static native int kBatchSizeFieldNumber();
  public static final int kBatchSizeFieldNumber = kBatchSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int batch_size();
  public native void set_batch_size(@Cast("google::protobuf::uint32") int value);

  // optional uint32 channels = 2;
  public native @Cast("bool") boolean has_channels();
  public native void clear_channels();
  @MemberGetter public static native int kChannelsFieldNumber();
  public static final int kChannelsFieldNumber = kChannelsFieldNumber();
  public native @Cast("google::protobuf::uint32") int channels();
  public native void set_channels(@Cast("google::protobuf::uint32") int value);

  // optional uint32 height = 3;
  public native @Cast("bool") boolean has_height();
  public native void clear_height();
  @MemberGetter public static native int kHeightFieldNumber();
  public static final int kHeightFieldNumber = kHeightFieldNumber();
  public native @Cast("google::protobuf::uint32") int height();
  public native void set_height(@Cast("google::protobuf::uint32") int value);

  // optional uint32 width = 4;
  public native @Cast("bool") boolean has_width();
  public native void clear_width();
  @MemberGetter public static native int kWidthFieldNumber();
  public static final int kWidthFieldNumber = kWidthFieldNumber();
  public native @Cast("google::protobuf::uint32") int width();
  public native void set_width(@Cast("google::protobuf::uint32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class MVNParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MVNParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public MVNParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public MVNParameter position(int position) {
        return (MVNParameter)super.position(position);
    }

  public MVNParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public MVNParameter(@Const @ByRef MVNParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef MVNParameter from);

  public native @ByRef @Name("operator =") MVNParameter put(@Const @ByRef MVNParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef MVNParameter default_instance();

  public native void Swap(MVNParameter other);

  // implements Message ----------------------------------------------

  public native MVNParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef MVNParameter from);
  public native void MergeFrom(@Const @ByRef MVNParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional bool normalize_variance = 1 [default = true];
  public native @Cast("bool") boolean has_normalize_variance();
  public native void clear_normalize_variance();
  @MemberGetter public static native int kNormalizeVarianceFieldNumber();
  public static final int kNormalizeVarianceFieldNumber = kNormalizeVarianceFieldNumber();
  public native @Cast("bool") boolean normalize_variance();
  public native void set_normalize_variance(@Cast("bool") boolean value);

  // optional bool across_channels = 2 [default = false];
  public native @Cast("bool") boolean has_across_channels();
  public native void clear_across_channels();
  @MemberGetter public static native int kAcrossChannelsFieldNumber();
  public static final int kAcrossChannelsFieldNumber = kAcrossChannelsFieldNumber();
  public native @Cast("bool") boolean across_channels();
  public native void set_across_channels(@Cast("bool") boolean value);

  // optional float eps = 3 [default = 1e-09];
  public native @Cast("bool") boolean has_eps();
  public native void clear_eps();
  @MemberGetter public static native int kEpsFieldNumber();
  public static final int kEpsFieldNumber = kEpsFieldNumber();
  public native float eps();
  public native void set_eps(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class PoolingParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PoolingParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PoolingParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PoolingParameter position(int position) {
        return (PoolingParameter)super.position(position);
    }

  public PoolingParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public PoolingParameter(@Const @ByRef PoolingParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef PoolingParameter from);

  public native @ByRef @Name("operator =") PoolingParameter put(@Const @ByRef PoolingParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef PoolingParameter default_instance();

  public native void Swap(PoolingParameter other);

  // implements Message ----------------------------------------------

  public native PoolingParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef PoolingParameter from);
  public native void MergeFrom(@Const @ByRef PoolingParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::PoolMethod") int MAX();
  public static final int MAX = MAX();
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::PoolMethod") int AVE();
  public static final int AVE = AVE();
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::PoolMethod") int STOCHASTIC();
  public static final int STOCHASTIC = STOCHASTIC();
  public static native @Cast("bool") boolean PoolMethod_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::PoolMethod") int PoolMethod_MIN();
  public static final int PoolMethod_MIN = PoolMethod_MIN();
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::PoolMethod") int PoolMethod_MAX();
  public static final int PoolMethod_MAX = PoolMethod_MAX();
  @MemberGetter public static native int PoolMethod_ARRAYSIZE();
  public static final int PoolMethod_ARRAYSIZE = PoolMethod_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer PoolMethod_descriptor();
  public static native @StdString BytePointer PoolMethod_Name(@Cast("caffe::PoolingParameter::PoolMethod") int value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::PoolingParameter::PoolMethod*") IntPointer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::PoolingParameter::PoolMethod*") IntBuffer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::PoolingParameter::PoolMethod*") int[] value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::PoolingParameter::PoolMethod*") IntPointer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::PoolingParameter::PoolMethod*") IntBuffer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::PoolingParameter::PoolMethod*") int[] value);
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::Engine") int DEFAULT();
  public static final int DEFAULT = DEFAULT();
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::Engine") int CAFFE();
  public static final int CAFFE = CAFFE();
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::Engine") int CUDNN();
  public static final int CUDNN = CUDNN();
  public static native @Cast("bool") boolean Engine_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::Engine") int Engine_MIN();
  public static final int Engine_MIN = Engine_MIN();
  @MemberGetter public static native @Cast("const caffe::PoolingParameter::Engine") int Engine_MAX();
  public static final int Engine_MAX = Engine_MAX();
  @MemberGetter public static native int Engine_ARRAYSIZE();
  public static final int Engine_ARRAYSIZE = Engine_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Engine_descriptor();
  public static native @StdString BytePointer Engine_Name(@Cast("caffe::PoolingParameter::Engine") int value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::PoolingParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::PoolingParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::PoolingParameter::Engine*") int[] value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::PoolingParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::PoolingParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::PoolingParameter::Engine*") int[] value);

  // accessors -------------------------------------------------------

  // optional .caffe.PoolingParameter.PoolMethod pool = 1 [default = MAX];
  public native @Cast("bool") boolean has_pool();
  public native void clear_pool();
  @MemberGetter public static native int kPoolFieldNumber();
  public static final int kPoolFieldNumber = kPoolFieldNumber();
  public native @Cast("caffe::PoolingParameter_PoolMethod") int pool();
  public native void set_pool(@Cast("caffe::PoolingParameter_PoolMethod") int value);

  // optional uint32 pad = 4 [default = 0];
  public native @Cast("bool") boolean has_pad();
  public native void clear_pad();
  @MemberGetter public static native int kPadFieldNumber();
  public static final int kPadFieldNumber = kPadFieldNumber();
  public native @Cast("google::protobuf::uint32") int pad();
  public native void set_pad(@Cast("google::protobuf::uint32") int value);

  // optional uint32 pad_h = 9 [default = 0];
  public native @Cast("bool") boolean has_pad_h();
  public native void clear_pad_h();
  @MemberGetter public static native int kPadHFieldNumber();
  public static final int kPadHFieldNumber = kPadHFieldNumber();
  public native @Cast("google::protobuf::uint32") int pad_h();
  public native void set_pad_h(@Cast("google::protobuf::uint32") int value);

  // optional uint32 pad_w = 10 [default = 0];
  public native @Cast("bool") boolean has_pad_w();
  public native void clear_pad_w();
  @MemberGetter public static native int kPadWFieldNumber();
  public static final int kPadWFieldNumber = kPadWFieldNumber();
  public native @Cast("google::protobuf::uint32") int pad_w();
  public native void set_pad_w(@Cast("google::protobuf::uint32") int value);

  // optional uint32 kernel_size = 2;
  public native @Cast("bool") boolean has_kernel_size();
  public native void clear_kernel_size();
  @MemberGetter public static native int kKernelSizeFieldNumber();
  public static final int kKernelSizeFieldNumber = kKernelSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int kernel_size();
  public native void set_kernel_size(@Cast("google::protobuf::uint32") int value);

  // optional uint32 kernel_h = 5;
  public native @Cast("bool") boolean has_kernel_h();
  public native void clear_kernel_h();
  @MemberGetter public static native int kKernelHFieldNumber();
  public static final int kKernelHFieldNumber = kKernelHFieldNumber();
  public native @Cast("google::protobuf::uint32") int kernel_h();
  public native void set_kernel_h(@Cast("google::protobuf::uint32") int value);

  // optional uint32 kernel_w = 6;
  public native @Cast("bool") boolean has_kernel_w();
  public native void clear_kernel_w();
  @MemberGetter public static native int kKernelWFieldNumber();
  public static final int kKernelWFieldNumber = kKernelWFieldNumber();
  public native @Cast("google::protobuf::uint32") int kernel_w();
  public native void set_kernel_w(@Cast("google::protobuf::uint32") int value);

  // optional uint32 stride = 3 [default = 1];
  public native @Cast("bool") boolean has_stride();
  public native void clear_stride();
  @MemberGetter public static native int kStrideFieldNumber();
  public static final int kStrideFieldNumber = kStrideFieldNumber();
  public native @Cast("google::protobuf::uint32") int stride();
  public native void set_stride(@Cast("google::protobuf::uint32") int value);

  // optional uint32 stride_h = 7;
  public native @Cast("bool") boolean has_stride_h();
  public native void clear_stride_h();
  @MemberGetter public static native int kStrideHFieldNumber();
  public static final int kStrideHFieldNumber = kStrideHFieldNumber();
  public native @Cast("google::protobuf::uint32") int stride_h();
  public native void set_stride_h(@Cast("google::protobuf::uint32") int value);

  // optional uint32 stride_w = 8;
  public native @Cast("bool") boolean has_stride_w();
  public native void clear_stride_w();
  @MemberGetter public static native int kStrideWFieldNumber();
  public static final int kStrideWFieldNumber = kStrideWFieldNumber();
  public native @Cast("google::protobuf::uint32") int stride_w();
  public native void set_stride_w(@Cast("google::protobuf::uint32") int value);

  // optional .caffe.PoolingParameter.Engine engine = 11 [default = DEFAULT];
  public native @Cast("bool") boolean has_engine();
  public native void clear_engine();
  @MemberGetter public static native int kEngineFieldNumber();
  public static final int kEngineFieldNumber = kEngineFieldNumber();
  public native @Cast("caffe::PoolingParameter_Engine") int engine();
  public native void set_engine(@Cast("caffe::PoolingParameter_Engine") int value);

  // optional bool global_pooling = 12 [default = false];
  public native @Cast("bool") boolean has_global_pooling();
  public native void clear_global_pooling();
  @MemberGetter public static native int kGlobalPoolingFieldNumber();
  public static final int kGlobalPoolingFieldNumber = kGlobalPoolingFieldNumber();
  public native @Cast("bool") boolean global_pooling();
  public native void set_global_pooling(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class PowerParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PowerParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PowerParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PowerParameter position(int position) {
        return (PowerParameter)super.position(position);
    }

  public PowerParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public PowerParameter(@Const @ByRef PowerParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef PowerParameter from);

  public native @ByRef @Name("operator =") PowerParameter put(@Const @ByRef PowerParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef PowerParameter default_instance();

  public native void Swap(PowerParameter other);

  // implements Message ----------------------------------------------

  public native PowerParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef PowerParameter from);
  public native void MergeFrom(@Const @ByRef PowerParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional float power = 1 [default = 1];
  public native @Cast("bool") boolean has_power();
  public native void clear_power();
  @MemberGetter public static native int kPowerFieldNumber();
  public static final int kPowerFieldNumber = kPowerFieldNumber();
  public native float power();
  public native void set_power(float value);

  // optional float scale = 2 [default = 1];
  public native @Cast("bool") boolean has_scale();
  public native void clear_scale();
  @MemberGetter public static native int kScaleFieldNumber();
  public static final int kScaleFieldNumber = kScaleFieldNumber();
  public native float scale();
  public native void set_scale(float value);

  // optional float shift = 3 [default = 0];
  public native @Cast("bool") boolean has_shift();
  public native void clear_shift();
  @MemberGetter public static native int kShiftFieldNumber();
  public static final int kShiftFieldNumber = kShiftFieldNumber();
  public native float shift();
  public native void set_shift(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class PythonParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PythonParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PythonParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PythonParameter position(int position) {
        return (PythonParameter)super.position(position);
    }

  public PythonParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public PythonParameter(@Const @ByRef PythonParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef PythonParameter from);

  public native @ByRef @Name("operator =") PythonParameter put(@Const @ByRef PythonParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef PythonParameter default_instance();

  public native void Swap(PythonParameter other);

  // implements Message ----------------------------------------------

  public native PythonParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef PythonParameter from);
  public native void MergeFrom(@Const @ByRef PythonParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string module = 1;
  public native @Cast("bool") boolean has_module();
  public native void clear_module();
  @MemberGetter public static native int kModuleFieldNumber();
  public static final int kModuleFieldNumber = kModuleFieldNumber();
  public native @StdString BytePointer module();
  public native void set_module(@StdString BytePointer value);
  public native void set_module(@StdString String value);
  public native void set_module(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_module(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_module();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_module();
  public native void set_allocated_module(@StdString @Cast({"char*", "std::string*"}) BytePointer module);

  // optional string layer = 2;
  public native @Cast("bool") boolean has_layer();
  public native void clear_layer();
  @MemberGetter public static native int kLayerFieldNumber();
  public static final int kLayerFieldNumber = kLayerFieldNumber();
  public native @StdString BytePointer layer();
  public native void set_layer(@StdString BytePointer value);
  public native void set_layer(@StdString String value);
  public native void set_layer(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_layer(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_layer();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_layer();
  public native void set_allocated_layer(@StdString @Cast({"char*", "std::string*"}) BytePointer layer);

  // optional string param_str = 3 [default = ""];
  public native @Cast("bool") boolean has_param_str();
  public native void clear_param_str();
  @MemberGetter public static native int kParamStrFieldNumber();
  public static final int kParamStrFieldNumber = kParamStrFieldNumber();
  public native @StdString BytePointer param_str();
  public native void set_param_str(@StdString BytePointer value);
  public native void set_param_str(@StdString String value);
  public native void set_param_str(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_param_str(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_param_str();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_param_str();
  public native void set_allocated_param_str(@StdString @Cast({"char*", "std::string*"}) BytePointer param_str);

  // optional bool share_in_parallel = 4 [default = false];
  public native @Cast("bool") boolean has_share_in_parallel();
  public native void clear_share_in_parallel();
  @MemberGetter public static native int kShareInParallelFieldNumber();
  public static final int kShareInParallelFieldNumber = kShareInParallelFieldNumber();
  public native @Cast("bool") boolean share_in_parallel();
  public native void set_share_in_parallel(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ReductionParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ReductionParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ReductionParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ReductionParameter position(int position) {
        return (ReductionParameter)super.position(position);
    }

  public ReductionParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ReductionParameter(@Const @ByRef ReductionParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ReductionParameter from);

  public native @ByRef @Name("operator =") ReductionParameter put(@Const @ByRef ReductionParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ReductionParameter default_instance();

  public native void Swap(ReductionParameter other);

  // implements Message ----------------------------------------------

  public native ReductionParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ReductionParameter from);
  public native void MergeFrom(@Const @ByRef ReductionParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::ReductionParameter::ReductionOp") int SUM();
  public static final int SUM = SUM();
  @MemberGetter public static native @Cast("const caffe::ReductionParameter::ReductionOp") int ASUM();
  public static final int ASUM = ASUM();
  @MemberGetter public static native @Cast("const caffe::ReductionParameter::ReductionOp") int SUMSQ();
  public static final int SUMSQ = SUMSQ();
  @MemberGetter public static native @Cast("const caffe::ReductionParameter::ReductionOp") int MEAN();
  public static final int MEAN = MEAN();
  public static native @Cast("bool") boolean ReductionOp_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::ReductionParameter::ReductionOp") int ReductionOp_MIN();
  public static final int ReductionOp_MIN = ReductionOp_MIN();
  @MemberGetter public static native @Cast("const caffe::ReductionParameter::ReductionOp") int ReductionOp_MAX();
  public static final int ReductionOp_MAX = ReductionOp_MAX();
  @MemberGetter public static native int ReductionOp_ARRAYSIZE();
  public static final int ReductionOp_ARRAYSIZE = ReductionOp_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer ReductionOp_descriptor();
  public static native @StdString BytePointer ReductionOp_Name(@Cast("caffe::ReductionParameter::ReductionOp") int value);
  public static native @Cast("bool") boolean ReductionOp_Parse(@StdString BytePointer name,
        @Cast("caffe::ReductionParameter::ReductionOp*") IntPointer value);
  public static native @Cast("bool") boolean ReductionOp_Parse(@StdString String name,
        @Cast("caffe::ReductionParameter::ReductionOp*") IntBuffer value);
  public static native @Cast("bool") boolean ReductionOp_Parse(@StdString BytePointer name,
        @Cast("caffe::ReductionParameter::ReductionOp*") int[] value);
  public static native @Cast("bool") boolean ReductionOp_Parse(@StdString String name,
        @Cast("caffe::ReductionParameter::ReductionOp*") IntPointer value);
  public static native @Cast("bool") boolean ReductionOp_Parse(@StdString BytePointer name,
        @Cast("caffe::ReductionParameter::ReductionOp*") IntBuffer value);
  public static native @Cast("bool") boolean ReductionOp_Parse(@StdString String name,
        @Cast("caffe::ReductionParameter::ReductionOp*") int[] value);

  // accessors -------------------------------------------------------

  // optional .caffe.ReductionParameter.ReductionOp operation = 1 [default = SUM];
  public native @Cast("bool") boolean has_operation();
  public native void clear_operation();
  @MemberGetter public static native int kOperationFieldNumber();
  public static final int kOperationFieldNumber = kOperationFieldNumber();
  public native @Cast("caffe::ReductionParameter_ReductionOp") int operation();
  public native void set_operation(@Cast("caffe::ReductionParameter_ReductionOp") int value);

  // optional int32 axis = 2 [default = 0];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);

  // optional float coeff = 3 [default = 1];
  public native @Cast("bool") boolean has_coeff();
  public native void clear_coeff();
  @MemberGetter public static native int kCoeffFieldNumber();
  public static final int kCoeffFieldNumber = kCoeffFieldNumber();
  public native float coeff();
  public native void set_coeff(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ReLUParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ReLUParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ReLUParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ReLUParameter position(int position) {
        return (ReLUParameter)super.position(position);
    }

  public ReLUParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ReLUParameter(@Const @ByRef ReLUParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ReLUParameter from);

  public native @ByRef @Name("operator =") ReLUParameter put(@Const @ByRef ReLUParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ReLUParameter default_instance();

  public native void Swap(ReLUParameter other);

  // implements Message ----------------------------------------------

  public native ReLUParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ReLUParameter from);
  public native void MergeFrom(@Const @ByRef ReLUParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::ReLUParameter::Engine") int DEFAULT();
  public static final int DEFAULT = DEFAULT();
  @MemberGetter public static native @Cast("const caffe::ReLUParameter::Engine") int CAFFE();
  public static final int CAFFE = CAFFE();
  @MemberGetter public static native @Cast("const caffe::ReLUParameter::Engine") int CUDNN();
  public static final int CUDNN = CUDNN();
  public static native @Cast("bool") boolean Engine_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::ReLUParameter::Engine") int Engine_MIN();
  public static final int Engine_MIN = Engine_MIN();
  @MemberGetter public static native @Cast("const caffe::ReLUParameter::Engine") int Engine_MAX();
  public static final int Engine_MAX = Engine_MAX();
  @MemberGetter public static native int Engine_ARRAYSIZE();
  public static final int Engine_ARRAYSIZE = Engine_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Engine_descriptor();
  public static native @StdString BytePointer Engine_Name(@Cast("caffe::ReLUParameter::Engine") int value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::ReLUParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::ReLUParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::ReLUParameter::Engine*") int[] value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::ReLUParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::ReLUParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::ReLUParameter::Engine*") int[] value);

  // accessors -------------------------------------------------------

  // optional float negative_slope = 1 [default = 0];
  public native @Cast("bool") boolean has_negative_slope();
  public native void clear_negative_slope();
  @MemberGetter public static native int kNegativeSlopeFieldNumber();
  public static final int kNegativeSlopeFieldNumber = kNegativeSlopeFieldNumber();
  public native float negative_slope();
  public native void set_negative_slope(float value);

  // optional .caffe.ReLUParameter.Engine engine = 2 [default = DEFAULT];
  public native @Cast("bool") boolean has_engine();
  public native void clear_engine();
  @MemberGetter public static native int kEngineFieldNumber();
  public static final int kEngineFieldNumber = kEngineFieldNumber();
  public native @Cast("caffe::ReLUParameter_Engine") int engine();
  public native void set_engine(@Cast("caffe::ReLUParameter_Engine") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ReshapeParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ReshapeParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ReshapeParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ReshapeParameter position(int position) {
        return (ReshapeParameter)super.position(position);
    }

  public ReshapeParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ReshapeParameter(@Const @ByRef ReshapeParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ReshapeParameter from);

  public native @ByRef @Name("operator =") ReshapeParameter put(@Const @ByRef ReshapeParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ReshapeParameter default_instance();

  public native void Swap(ReshapeParameter other);

  // implements Message ----------------------------------------------

  public native ReshapeParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ReshapeParameter from);
  public native void MergeFrom(@Const @ByRef ReshapeParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional .caffe.BlobShape shape = 1;
  public native @Cast("bool") boolean has_shape();
  public native void clear_shape();
  @MemberGetter public static native int kShapeFieldNumber();
  public static final int kShapeFieldNumber = kShapeFieldNumber();
  public native @Const @ByRef BlobShape shape();
  public native BlobShape mutable_shape();
  public native BlobShape release_shape();
  public native void set_allocated_shape(BlobShape shape);

  // optional int32 axis = 2 [default = 0];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);

  // optional int32 num_axes = 3 [default = -1];
  public native @Cast("bool") boolean has_num_axes();
  public native void clear_num_axes();
  @MemberGetter public static native int kNumAxesFieldNumber();
  public static final int kNumAxesFieldNumber = kNumAxesFieldNumber();
  public native @Cast("google::protobuf::int32") int num_axes();
  public native void set_num_axes(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class SigmoidParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SigmoidParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SigmoidParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SigmoidParameter position(int position) {
        return (SigmoidParameter)super.position(position);
    }

  public SigmoidParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public SigmoidParameter(@Const @ByRef SigmoidParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef SigmoidParameter from);

  public native @ByRef @Name("operator =") SigmoidParameter put(@Const @ByRef SigmoidParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef SigmoidParameter default_instance();

  public native void Swap(SigmoidParameter other);

  // implements Message ----------------------------------------------

  public native SigmoidParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef SigmoidParameter from);
  public native void MergeFrom(@Const @ByRef SigmoidParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::SigmoidParameter::Engine") int DEFAULT();
  public static final int DEFAULT = DEFAULT();
  @MemberGetter public static native @Cast("const caffe::SigmoidParameter::Engine") int CAFFE();
  public static final int CAFFE = CAFFE();
  @MemberGetter public static native @Cast("const caffe::SigmoidParameter::Engine") int CUDNN();
  public static final int CUDNN = CUDNN();
  public static native @Cast("bool") boolean Engine_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::SigmoidParameter::Engine") int Engine_MIN();
  public static final int Engine_MIN = Engine_MIN();
  @MemberGetter public static native @Cast("const caffe::SigmoidParameter::Engine") int Engine_MAX();
  public static final int Engine_MAX = Engine_MAX();
  @MemberGetter public static native int Engine_ARRAYSIZE();
  public static final int Engine_ARRAYSIZE = Engine_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Engine_descriptor();
  public static native @StdString BytePointer Engine_Name(@Cast("caffe::SigmoidParameter::Engine") int value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SigmoidParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SigmoidParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SigmoidParameter::Engine*") int[] value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SigmoidParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SigmoidParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SigmoidParameter::Engine*") int[] value);

  // accessors -------------------------------------------------------

  // optional .caffe.SigmoidParameter.Engine engine = 1 [default = DEFAULT];
  public native @Cast("bool") boolean has_engine();
  public native void clear_engine();
  @MemberGetter public static native int kEngineFieldNumber();
  public static final int kEngineFieldNumber = kEngineFieldNumber();
  public native @Cast("caffe::SigmoidParameter_Engine") int engine();
  public native void set_engine(@Cast("caffe::SigmoidParameter_Engine") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class SliceParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SliceParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SliceParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SliceParameter position(int position) {
        return (SliceParameter)super.position(position);
    }

  public SliceParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public SliceParameter(@Const @ByRef SliceParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef SliceParameter from);

  public native @ByRef @Name("operator =") SliceParameter put(@Const @ByRef SliceParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef SliceParameter default_instance();

  public native void Swap(SliceParameter other);

  // implements Message ----------------------------------------------

  public native SliceParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef SliceParameter from);
  public native void MergeFrom(@Const @ByRef SliceParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional int32 axis = 3 [default = 1];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);

  // repeated uint32 slice_point = 2;
  public native int slice_point_size();
  public native void clear_slice_point();
  @MemberGetter public static native int kSlicePointFieldNumber();
  public static final int kSlicePointFieldNumber = kSlicePointFieldNumber();
  public native @Cast("google::protobuf::uint32") int slice_point(int index);
  public native void set_slice_point(int index, @Cast("google::protobuf::uint32") int value);
  public native void add_slice_point(@Cast("google::protobuf::uint32") int value);

  // optional uint32 slice_dim = 1 [default = 1];
  public native @Cast("bool") boolean has_slice_dim();
  public native void clear_slice_dim();
  @MemberGetter public static native int kSliceDimFieldNumber();
  public static final int kSliceDimFieldNumber = kSliceDimFieldNumber();
  public native @Cast("google::protobuf::uint32") int slice_dim();
  public native void set_slice_dim(@Cast("google::protobuf::uint32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class SoftmaxParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SoftmaxParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SoftmaxParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SoftmaxParameter position(int position) {
        return (SoftmaxParameter)super.position(position);
    }

  public SoftmaxParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public SoftmaxParameter(@Const @ByRef SoftmaxParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef SoftmaxParameter from);

  public native @ByRef @Name("operator =") SoftmaxParameter put(@Const @ByRef SoftmaxParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef SoftmaxParameter default_instance();

  public native void Swap(SoftmaxParameter other);

  // implements Message ----------------------------------------------

  public native SoftmaxParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef SoftmaxParameter from);
  public native void MergeFrom(@Const @ByRef SoftmaxParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::SoftmaxParameter::Engine") int DEFAULT();
  public static final int DEFAULT = DEFAULT();
  @MemberGetter public static native @Cast("const caffe::SoftmaxParameter::Engine") int CAFFE();
  public static final int CAFFE = CAFFE();
  @MemberGetter public static native @Cast("const caffe::SoftmaxParameter::Engine") int CUDNN();
  public static final int CUDNN = CUDNN();
  public static native @Cast("bool") boolean Engine_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::SoftmaxParameter::Engine") int Engine_MIN();
  public static final int Engine_MIN = Engine_MIN();
  @MemberGetter public static native @Cast("const caffe::SoftmaxParameter::Engine") int Engine_MAX();
  public static final int Engine_MAX = Engine_MAX();
  @MemberGetter public static native int Engine_ARRAYSIZE();
  public static final int Engine_ARRAYSIZE = Engine_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Engine_descriptor();
  public static native @StdString BytePointer Engine_Name(@Cast("caffe::SoftmaxParameter::Engine") int value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SoftmaxParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SoftmaxParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SoftmaxParameter::Engine*") int[] value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SoftmaxParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SoftmaxParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SoftmaxParameter::Engine*") int[] value);

  // accessors -------------------------------------------------------

  // optional .caffe.SoftmaxParameter.Engine engine = 1 [default = DEFAULT];
  public native @Cast("bool") boolean has_engine();
  public native void clear_engine();
  @MemberGetter public static native int kEngineFieldNumber();
  public static final int kEngineFieldNumber = kEngineFieldNumber();
  public native @Cast("caffe::SoftmaxParameter_Engine") int engine();
  public native void set_engine(@Cast("caffe::SoftmaxParameter_Engine") int value);

  // optional int32 axis = 2 [default = 1];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class TanHParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TanHParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TanHParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TanHParameter position(int position) {
        return (TanHParameter)super.position(position);
    }

  public TanHParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public TanHParameter(@Const @ByRef TanHParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef TanHParameter from);

  public native @ByRef @Name("operator =") TanHParameter put(@Const @ByRef TanHParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef TanHParameter default_instance();

  public native void Swap(TanHParameter other);

  // implements Message ----------------------------------------------

  public native TanHParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef TanHParameter from);
  public native void MergeFrom(@Const @ByRef TanHParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::TanHParameter::Engine") int DEFAULT();
  public static final int DEFAULT = DEFAULT();
  @MemberGetter public static native @Cast("const caffe::TanHParameter::Engine") int CAFFE();
  public static final int CAFFE = CAFFE();
  @MemberGetter public static native @Cast("const caffe::TanHParameter::Engine") int CUDNN();
  public static final int CUDNN = CUDNN();
  public static native @Cast("bool") boolean Engine_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::TanHParameter::Engine") int Engine_MIN();
  public static final int Engine_MIN = Engine_MIN();
  @MemberGetter public static native @Cast("const caffe::TanHParameter::Engine") int Engine_MAX();
  public static final int Engine_MAX = Engine_MAX();
  @MemberGetter public static native int Engine_ARRAYSIZE();
  public static final int Engine_ARRAYSIZE = Engine_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Engine_descriptor();
  public static native @StdString BytePointer Engine_Name(@Cast("caffe::TanHParameter::Engine") int value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::TanHParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::TanHParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::TanHParameter::Engine*") int[] value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::TanHParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::TanHParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::TanHParameter::Engine*") int[] value);

  // accessors -------------------------------------------------------

  // optional .caffe.TanHParameter.Engine engine = 1 [default = DEFAULT];
  public native @Cast("bool") boolean has_engine();
  public native void clear_engine();
  @MemberGetter public static native int kEngineFieldNumber();
  public static final int kEngineFieldNumber = kEngineFieldNumber();
  public native @Cast("caffe::TanHParameter_Engine") int engine();
  public native void set_engine(@Cast("caffe::TanHParameter_Engine") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class TileParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TileParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TileParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TileParameter position(int position) {
        return (TileParameter)super.position(position);
    }

  public TileParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public TileParameter(@Const @ByRef TileParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef TileParameter from);

  public native @ByRef @Name("operator =") TileParameter put(@Const @ByRef TileParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef TileParameter default_instance();

  public native void Swap(TileParameter other);

  // implements Message ----------------------------------------------

  public native TileParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef TileParameter from);
  public native void MergeFrom(@Const @ByRef TileParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional int32 axis = 1 [default = 1];
  public native @Cast("bool") boolean has_axis();
  public native void clear_axis();
  @MemberGetter public static native int kAxisFieldNumber();
  public static final int kAxisFieldNumber = kAxisFieldNumber();
  public native @Cast("google::protobuf::int32") int axis();
  public native void set_axis(@Cast("google::protobuf::int32") int value);

  // optional int32 tiles = 2;
  public native @Cast("bool") boolean has_tiles();
  public native void clear_tiles();
  @MemberGetter public static native int kTilesFieldNumber();
  public static final int kTilesFieldNumber = kTilesFieldNumber();
  public native @Cast("google::protobuf::int32") int tiles();
  public native void set_tiles(@Cast("google::protobuf::int32") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class ThresholdParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ThresholdParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ThresholdParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ThresholdParameter position(int position) {
        return (ThresholdParameter)super.position(position);
    }

  public ThresholdParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ThresholdParameter(@Const @ByRef ThresholdParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ThresholdParameter from);

  public native @ByRef @Name("operator =") ThresholdParameter put(@Const @ByRef ThresholdParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ThresholdParameter default_instance();

  public native void Swap(ThresholdParameter other);

  // implements Message ----------------------------------------------

  public native ThresholdParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ThresholdParameter from);
  public native void MergeFrom(@Const @ByRef ThresholdParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional float threshold = 1 [default = 0];
  public native @Cast("bool") boolean has_threshold();
  public native void clear_threshold();
  @MemberGetter public static native int kThresholdFieldNumber();
  public static final int kThresholdFieldNumber = kThresholdFieldNumber();
  public native float threshold();
  public native void set_threshold(float value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class WindowDataParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public WindowDataParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public WindowDataParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public WindowDataParameter position(int position) {
        return (WindowDataParameter)super.position(position);
    }

  public WindowDataParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public WindowDataParameter(@Const @ByRef WindowDataParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef WindowDataParameter from);

  public native @ByRef @Name("operator =") WindowDataParameter put(@Const @ByRef WindowDataParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef WindowDataParameter default_instance();

  public native void Swap(WindowDataParameter other);

  // implements Message ----------------------------------------------

  public native WindowDataParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef WindowDataParameter from);
  public native void MergeFrom(@Const @ByRef WindowDataParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string source = 1;
  public native @Cast("bool") boolean has_source();
  public native void clear_source();
  @MemberGetter public static native int kSourceFieldNumber();
  public static final int kSourceFieldNumber = kSourceFieldNumber();
  public native @StdString BytePointer source();
  public native void set_source(@StdString BytePointer value);
  public native void set_source(@StdString String value);
  public native void set_source(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_source(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_source();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_source();
  public native void set_allocated_source(@StdString @Cast({"char*", "std::string*"}) BytePointer source);

  // optional float scale = 2 [default = 1];
  public native @Cast("bool") boolean has_scale();
  public native void clear_scale();
  @MemberGetter public static native int kScaleFieldNumber();
  public static final int kScaleFieldNumber = kScaleFieldNumber();
  public native float scale();
  public native void set_scale(float value);

  // optional string mean_file = 3;
  public native @Cast("bool") boolean has_mean_file();
  public native void clear_mean_file();
  @MemberGetter public static native int kMeanFileFieldNumber();
  public static final int kMeanFileFieldNumber = kMeanFileFieldNumber();
  public native @StdString BytePointer mean_file();
  public native void set_mean_file(@StdString BytePointer value);
  public native void set_mean_file(@StdString String value);
  public native void set_mean_file(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_mean_file(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_mean_file();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_mean_file();
  public native void set_allocated_mean_file(@StdString @Cast({"char*", "std::string*"}) BytePointer mean_file);

  // optional uint32 batch_size = 4;
  public native @Cast("bool") boolean has_batch_size();
  public native void clear_batch_size();
  @MemberGetter public static native int kBatchSizeFieldNumber();
  public static final int kBatchSizeFieldNumber = kBatchSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int batch_size();
  public native void set_batch_size(@Cast("google::protobuf::uint32") int value);

  // optional uint32 crop_size = 5 [default = 0];
  public native @Cast("bool") boolean has_crop_size();
  public native void clear_crop_size();
  @MemberGetter public static native int kCropSizeFieldNumber();
  public static final int kCropSizeFieldNumber = kCropSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int crop_size();
  public native void set_crop_size(@Cast("google::protobuf::uint32") int value);

  // optional bool mirror = 6 [default = false];
  public native @Cast("bool") boolean has_mirror();
  public native void clear_mirror();
  @MemberGetter public static native int kMirrorFieldNumber();
  public static final int kMirrorFieldNumber = kMirrorFieldNumber();
  public native @Cast("bool") boolean mirror();
  public native void set_mirror(@Cast("bool") boolean value);

  // optional float fg_threshold = 7 [default = 0.5];
  public native @Cast("bool") boolean has_fg_threshold();
  public native void clear_fg_threshold();
  @MemberGetter public static native int kFgThresholdFieldNumber();
  public static final int kFgThresholdFieldNumber = kFgThresholdFieldNumber();
  public native float fg_threshold();
  public native void set_fg_threshold(float value);

  // optional float bg_threshold = 8 [default = 0.5];
  public native @Cast("bool") boolean has_bg_threshold();
  public native void clear_bg_threshold();
  @MemberGetter public static native int kBgThresholdFieldNumber();
  public static final int kBgThresholdFieldNumber = kBgThresholdFieldNumber();
  public native float bg_threshold();
  public native void set_bg_threshold(float value);

  // optional float fg_fraction = 9 [default = 0.25];
  public native @Cast("bool") boolean has_fg_fraction();
  public native void clear_fg_fraction();
  @MemberGetter public static native int kFgFractionFieldNumber();
  public static final int kFgFractionFieldNumber = kFgFractionFieldNumber();
  public native float fg_fraction();
  public native void set_fg_fraction(float value);

  // optional uint32 context_pad = 10 [default = 0];
  public native @Cast("bool") boolean has_context_pad();
  public native void clear_context_pad();
  @MemberGetter public static native int kContextPadFieldNumber();
  public static final int kContextPadFieldNumber = kContextPadFieldNumber();
  public native @Cast("google::protobuf::uint32") int context_pad();
  public native void set_context_pad(@Cast("google::protobuf::uint32") int value);

  // optional string crop_mode = 11 [default = "warp"];
  public native @Cast("bool") boolean has_crop_mode();
  public native void clear_crop_mode();
  @MemberGetter public static native int kCropModeFieldNumber();
  public static final int kCropModeFieldNumber = kCropModeFieldNumber();
  public native @StdString BytePointer crop_mode();
  public native void set_crop_mode(@StdString BytePointer value);
  public native void set_crop_mode(@StdString String value);
  public native void set_crop_mode(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_crop_mode(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_crop_mode();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_crop_mode();
  public native void set_allocated_crop_mode(@StdString @Cast({"char*", "std::string*"}) BytePointer crop_mode);

  // optional bool cache_images = 12 [default = false];
  public native @Cast("bool") boolean has_cache_images();
  public native void clear_cache_images();
  @MemberGetter public static native int kCacheImagesFieldNumber();
  public static final int kCacheImagesFieldNumber = kCacheImagesFieldNumber();
  public native @Cast("bool") boolean cache_images();
  public native void set_cache_images(@Cast("bool") boolean value);

  // optional string root_folder = 13 [default = ""];
  public native @Cast("bool") boolean has_root_folder();
  public native void clear_root_folder();
  @MemberGetter public static native int kRootFolderFieldNumber();
  public static final int kRootFolderFieldNumber = kRootFolderFieldNumber();
  public native @StdString BytePointer root_folder();
  public native void set_root_folder(@StdString BytePointer value);
  public native void set_root_folder(@StdString String value);
  public native void set_root_folder(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_root_folder(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_root_folder();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_root_folder();
  public native void set_allocated_root_folder(@StdString @Cast({"char*", "std::string*"}) BytePointer root_folder);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class SPPParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SPPParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SPPParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SPPParameter position(int position) {
        return (SPPParameter)super.position(position);
    }

  public SPPParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public SPPParameter(@Const @ByRef SPPParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef SPPParameter from);

  public native @ByRef @Name("operator =") SPPParameter put(@Const @ByRef SPPParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef SPPParameter default_instance();

  public native void Swap(SPPParameter other);

  // implements Message ----------------------------------------------

  public native SPPParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef SPPParameter from);
  public native void MergeFrom(@Const @ByRef SPPParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::SPPParameter::PoolMethod") int MAX();
  public static final int MAX = MAX();
  @MemberGetter public static native @Cast("const caffe::SPPParameter::PoolMethod") int AVE();
  public static final int AVE = AVE();
  @MemberGetter public static native @Cast("const caffe::SPPParameter::PoolMethod") int STOCHASTIC();
  public static final int STOCHASTIC = STOCHASTIC();
  public static native @Cast("bool") boolean PoolMethod_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::SPPParameter::PoolMethod") int PoolMethod_MIN();
  public static final int PoolMethod_MIN = PoolMethod_MIN();
  @MemberGetter public static native @Cast("const caffe::SPPParameter::PoolMethod") int PoolMethod_MAX();
  public static final int PoolMethod_MAX = PoolMethod_MAX();
  @MemberGetter public static native int PoolMethod_ARRAYSIZE();
  public static final int PoolMethod_ARRAYSIZE = PoolMethod_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer PoolMethod_descriptor();
  public static native @StdString BytePointer PoolMethod_Name(@Cast("caffe::SPPParameter::PoolMethod") int value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::SPPParameter::PoolMethod*") IntPointer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::SPPParameter::PoolMethod*") IntBuffer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::SPPParameter::PoolMethod*") int[] value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::SPPParameter::PoolMethod*") IntPointer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::SPPParameter::PoolMethod*") IntBuffer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::SPPParameter::PoolMethod*") int[] value);
  @MemberGetter public static native @Cast("const caffe::SPPParameter::Engine") int DEFAULT();
  public static final int DEFAULT = DEFAULT();
  @MemberGetter public static native @Cast("const caffe::SPPParameter::Engine") int CAFFE();
  public static final int CAFFE = CAFFE();
  @MemberGetter public static native @Cast("const caffe::SPPParameter::Engine") int CUDNN();
  public static final int CUDNN = CUDNN();
  public static native @Cast("bool") boolean Engine_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::SPPParameter::Engine") int Engine_MIN();
  public static final int Engine_MIN = Engine_MIN();
  @MemberGetter public static native @Cast("const caffe::SPPParameter::Engine") int Engine_MAX();
  public static final int Engine_MAX = Engine_MAX();
  @MemberGetter public static native int Engine_ARRAYSIZE();
  public static final int Engine_ARRAYSIZE = Engine_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Engine_descriptor();
  public static native @StdString BytePointer Engine_Name(@Cast("caffe::SPPParameter::Engine") int value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SPPParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SPPParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SPPParameter::Engine*") int[] value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SPPParameter::Engine*") IntPointer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString BytePointer name,
        @Cast("caffe::SPPParameter::Engine*") IntBuffer value);
  public static native @Cast("bool") boolean Engine_Parse(@StdString String name,
        @Cast("caffe::SPPParameter::Engine*") int[] value);

  // accessors -------------------------------------------------------

  // optional uint32 pyramid_height = 1;
  public native @Cast("bool") boolean has_pyramid_height();
  public native void clear_pyramid_height();
  @MemberGetter public static native int kPyramidHeightFieldNumber();
  public static final int kPyramidHeightFieldNumber = kPyramidHeightFieldNumber();
  public native @Cast("google::protobuf::uint32") int pyramid_height();
  public native void set_pyramid_height(@Cast("google::protobuf::uint32") int value);

  // optional .caffe.SPPParameter.PoolMethod pool = 2 [default = MAX];
  public native @Cast("bool") boolean has_pool();
  public native void clear_pool();
  @MemberGetter public static native int kPoolFieldNumber();
  public static final int kPoolFieldNumber = kPoolFieldNumber();
  public native @Cast("caffe::SPPParameter_PoolMethod") int pool();
  public native void set_pool(@Cast("caffe::SPPParameter_PoolMethod") int value);

  // optional .caffe.SPPParameter.Engine engine = 6 [default = DEFAULT];
  public native @Cast("bool") boolean has_engine();
  public native void clear_engine();
  @MemberGetter public static native int kEngineFieldNumber();
  public static final int kEngineFieldNumber = kEngineFieldNumber();
  public native @Cast("caffe::SPPParameter_Engine") int engine();
  public native void set_engine(@Cast("caffe::SPPParameter_Engine") int value);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class V1LayerParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public V1LayerParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public V1LayerParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public V1LayerParameter position(int position) {
        return (V1LayerParameter)super.position(position);
    }

  public V1LayerParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public V1LayerParameter(@Const @ByRef V1LayerParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef V1LayerParameter from);

  public native @ByRef @Name("operator =") V1LayerParameter put(@Const @ByRef V1LayerParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef V1LayerParameter default_instance();

  public native void Swap(V1LayerParameter other);

  // implements Message ----------------------------------------------

  public native V1LayerParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef V1LayerParameter from);
  public native void MergeFrom(@Const @ByRef V1LayerParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int NONE();
  public static final int NONE = NONE();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int ABSVAL();
  public static final int ABSVAL = ABSVAL();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int ACCURACY();
  public static final int ACCURACY = ACCURACY();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int ARGMAX();
  public static final int ARGMAX = ARGMAX();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int BNLL();
  public static final int BNLL = BNLL();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int CONCAT();
  public static final int CONCAT = CONCAT();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int CONTRASTIVE_LOSS();
  public static final int CONTRASTIVE_LOSS = CONTRASTIVE_LOSS();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int CONVOLUTION();
  public static final int CONVOLUTION = CONVOLUTION();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int DATA();
  public static final int DATA = DATA();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int DECONVOLUTION();
  public static final int DECONVOLUTION = DECONVOLUTION();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int DROPOUT();
  public static final int DROPOUT = DROPOUT();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int DUMMY_DATA();
  public static final int DUMMY_DATA = DUMMY_DATA();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int EUCLIDEAN_LOSS();
  public static final int EUCLIDEAN_LOSS = EUCLIDEAN_LOSS();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int ELTWISE();
  public static final int ELTWISE = ELTWISE();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int EXP();
  public static final int EXP = EXP();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int FLATTEN();
  public static final int FLATTEN = FLATTEN();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int HDF5_DATA();
  public static final int HDF5_DATA = HDF5_DATA();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int HDF5_OUTPUT();
  public static final int HDF5_OUTPUT = HDF5_OUTPUT();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int HINGE_LOSS();
  public static final int HINGE_LOSS = HINGE_LOSS();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int IM2COL();
  public static final int IM2COL = IM2COL();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int IMAGE_DATA();
  public static final int IMAGE_DATA = IMAGE_DATA();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int INFOGAIN_LOSS();
  public static final int INFOGAIN_LOSS = INFOGAIN_LOSS();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int INNER_PRODUCT();
  public static final int INNER_PRODUCT = INNER_PRODUCT();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int LRN();
  public static final int LRN = LRN();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int MEMORY_DATA();
  public static final int MEMORY_DATA = MEMORY_DATA();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int MULTINOMIAL_LOGISTIC_LOSS();
  public static final int MULTINOMIAL_LOGISTIC_LOSS = MULTINOMIAL_LOGISTIC_LOSS();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int MVN();
  public static final int MVN = MVN();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int POOLING();
  public static final int POOLING = POOLING();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int POWER();
  public static final int POWER = POWER();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int RELU();
  public static final int RELU = RELU();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int SIGMOID();
  public static final int SIGMOID = SIGMOID();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int SIGMOID_CROSS_ENTROPY_LOSS();
  public static final int SIGMOID_CROSS_ENTROPY_LOSS = SIGMOID_CROSS_ENTROPY_LOSS();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int SILENCE();
  public static final int SILENCE = SILENCE();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int SOFTMAX();
  public static final int SOFTMAX = SOFTMAX();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int SOFTMAX_LOSS();
  public static final int SOFTMAX_LOSS = SOFTMAX_LOSS();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int SPLIT();
  public static final int SPLIT = SPLIT();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int SLICE();
  public static final int SLICE = SLICE();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int TANH();
  public static final int TANH = TANH();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int WINDOW_DATA();
  public static final int WINDOW_DATA = WINDOW_DATA();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int THRESHOLD();
  public static final int THRESHOLD = THRESHOLD();
  public static native @Cast("bool") boolean LayerType_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int LayerType_MIN();
  public static final int LayerType_MIN = LayerType_MIN();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::LayerType") int LayerType_MAX();
  public static final int LayerType_MAX = LayerType_MAX();
  @MemberGetter public static native int LayerType_ARRAYSIZE();
  public static final int LayerType_ARRAYSIZE = LayerType_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer LayerType_descriptor();
  public static native @StdString BytePointer LayerType_Name(@Cast("caffe::V1LayerParameter::LayerType") int value);
  public static native @Cast("bool") boolean LayerType_Parse(@StdString BytePointer name,
        @Cast("caffe::V1LayerParameter::LayerType*") IntPointer value);
  public static native @Cast("bool") boolean LayerType_Parse(@StdString String name,
        @Cast("caffe::V1LayerParameter::LayerType*") IntBuffer value);
  public static native @Cast("bool") boolean LayerType_Parse(@StdString BytePointer name,
        @Cast("caffe::V1LayerParameter::LayerType*") int[] value);
  public static native @Cast("bool") boolean LayerType_Parse(@StdString String name,
        @Cast("caffe::V1LayerParameter::LayerType*") IntPointer value);
  public static native @Cast("bool") boolean LayerType_Parse(@StdString BytePointer name,
        @Cast("caffe::V1LayerParameter::LayerType*") IntBuffer value);
  public static native @Cast("bool") boolean LayerType_Parse(@StdString String name,
        @Cast("caffe::V1LayerParameter::LayerType*") int[] value);
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::DimCheckMode") int STRICT();
  public static final int STRICT = STRICT();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::DimCheckMode") int PERMISSIVE();
  public static final int PERMISSIVE = PERMISSIVE();
  public static native @Cast("bool") boolean DimCheckMode_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::DimCheckMode") int DimCheckMode_MIN();
  public static final int DimCheckMode_MIN = DimCheckMode_MIN();
  @MemberGetter public static native @Cast("const caffe::V1LayerParameter::DimCheckMode") int DimCheckMode_MAX();
  public static final int DimCheckMode_MAX = DimCheckMode_MAX();
  @MemberGetter public static native int DimCheckMode_ARRAYSIZE();
  public static final int DimCheckMode_ARRAYSIZE = DimCheckMode_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer DimCheckMode_descriptor();
  public static native @StdString BytePointer DimCheckMode_Name(@Cast("caffe::V1LayerParameter::DimCheckMode") int value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString BytePointer name,
        @Cast("caffe::V1LayerParameter::DimCheckMode*") IntPointer value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString String name,
        @Cast("caffe::V1LayerParameter::DimCheckMode*") IntBuffer value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString BytePointer name,
        @Cast("caffe::V1LayerParameter::DimCheckMode*") int[] value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString String name,
        @Cast("caffe::V1LayerParameter::DimCheckMode*") IntPointer value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString BytePointer name,
        @Cast("caffe::V1LayerParameter::DimCheckMode*") IntBuffer value);
  public static native @Cast("bool") boolean DimCheckMode_Parse(@StdString String name,
        @Cast("caffe::V1LayerParameter::DimCheckMode*") int[] value);

  // accessors -------------------------------------------------------

  // repeated string bottom = 2;
  public native int bottom_size();
  public native void clear_bottom();
  @MemberGetter public static native int kBottomFieldNumber();
  public static final int kBottomFieldNumber = kBottomFieldNumber();
  public native @StdString BytePointer bottom(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_bottom(int index);
  public native void set_bottom(int index, @StdString BytePointer value);
  public native void set_bottom(int index, @StdString String value);
  public native void set_bottom(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_bottom(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_bottom();
  public native void add_bottom(@StdString BytePointer value);
  public native void add_bottom(@StdString String value);
  public native void add_bottom(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_bottom(String value, @Cast("size_t") long size);

  // repeated string top = 3;
  public native int top_size();
  public native void clear_top();
  @MemberGetter public static native int kTopFieldNumber();
  public static final int kTopFieldNumber = kTopFieldNumber();
  public native @StdString BytePointer top(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_top(int index);
  public native void set_top(int index, @StdString BytePointer value);
  public native void set_top(int index, @StdString String value);
  public native void set_top(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_top(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_top();
  public native void add_top(@StdString BytePointer value);
  public native void add_top(@StdString String value);
  public native void add_top(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_top(String value, @Cast("size_t") long size);

  // optional string name = 4;
  public native @Cast("bool") boolean has_name();
  public native void clear_name();
  @MemberGetter public static native int kNameFieldNumber();
  public static final int kNameFieldNumber = kNameFieldNumber();
  public native @StdString BytePointer name();
  public native void set_name(@StdString BytePointer value);
  public native void set_name(@StdString String value);
  public native void set_name(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_name(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_name();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_name();
  public native void set_allocated_name(@StdString @Cast({"char*", "std::string*"}) BytePointer name);

  // repeated .caffe.NetStateRule include = 32;
  public native int include_size();
  public native void clear_include();
  @MemberGetter public static native int kIncludeFieldNumber();
  public static final int kIncludeFieldNumber = kIncludeFieldNumber();
  public native @Const @ByRef NetStateRule include(int index);
  public native NetStateRule mutable_include(int index);
  public native NetStateRule add_include();

  // repeated .caffe.NetStateRule exclude = 33;
  public native int exclude_size();
  public native void clear_exclude();
  @MemberGetter public static native int kExcludeFieldNumber();
  public static final int kExcludeFieldNumber = kExcludeFieldNumber();
  public native @Const @ByRef NetStateRule exclude(int index);
  public native NetStateRule mutable_exclude(int index);
  public native NetStateRule add_exclude();

  // optional .caffe.V1LayerParameter.LayerType type = 5;
  public native @Cast("bool") boolean has_type();
  public native void clear_type();
  @MemberGetter public static native int kTypeFieldNumber();
  public static final int kTypeFieldNumber = kTypeFieldNumber();
  public native @Cast("caffe::V1LayerParameter_LayerType") int type();
  public native void set_type(@Cast("caffe::V1LayerParameter_LayerType") int value);

  // repeated .caffe.BlobProto blobs = 6;
  public native int blobs_size();
  public native void clear_blobs();
  @MemberGetter public static native int kBlobsFieldNumber();
  public static final int kBlobsFieldNumber = kBlobsFieldNumber();
  public native @Const @ByRef BlobProto blobs(int index);
  public native BlobProto mutable_blobs(int index);
  public native BlobProto add_blobs();

  // repeated string param = 1001;
  public native int param_size();
  public native void clear_param();
  @MemberGetter public static native int kParamFieldNumber();
  public static final int kParamFieldNumber = kParamFieldNumber();
  public native @StdString BytePointer param(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_param(int index);
  public native void set_param(int index, @StdString BytePointer value);
  public native void set_param(int index, @StdString String value);
  public native void set_param(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_param(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_param();
  public native void add_param(@StdString BytePointer value);
  public native void add_param(@StdString String value);
  public native void add_param(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_param(String value, @Cast("size_t") long size);

  // repeated .caffe.V1LayerParameter.DimCheckMode blob_share_mode = 1002;
  public native int blob_share_mode_size();
  public native void clear_blob_share_mode();
  @MemberGetter public static native int kBlobShareModeFieldNumber();
  public static final int kBlobShareModeFieldNumber = kBlobShareModeFieldNumber();
  public native @Cast("caffe::V1LayerParameter_DimCheckMode") int blob_share_mode(int index);
  public native void set_blob_share_mode(int index, @Cast("caffe::V1LayerParameter_DimCheckMode") int value);
  public native void add_blob_share_mode(@Cast("caffe::V1LayerParameter_DimCheckMode") int value);

  // repeated float blobs_lr = 7;
  public native int blobs_lr_size();
  public native void clear_blobs_lr();
  @MemberGetter public static native int kBlobsLrFieldNumber();
  public static final int kBlobsLrFieldNumber = kBlobsLrFieldNumber();
  public native float blobs_lr(int index);
  public native void set_blobs_lr(int index, float value);
  public native void add_blobs_lr(float value);

  // repeated float weight_decay = 8;
  public native int weight_decay_size();
  public native void clear_weight_decay();
  @MemberGetter public static native int kWeightDecayFieldNumber();
  public static final int kWeightDecayFieldNumber = kWeightDecayFieldNumber();
  public native float weight_decay(int index);
  public native void set_weight_decay(int index, float value);
  public native void add_weight_decay(float value);

  // repeated float loss_weight = 35;
  public native int loss_weight_size();
  public native void clear_loss_weight();
  @MemberGetter public static native int kLossWeightFieldNumber();
  public static final int kLossWeightFieldNumber = kLossWeightFieldNumber();
  public native float loss_weight(int index);
  public native void set_loss_weight(int index, float value);
  public native void add_loss_weight(float value);

  // optional .caffe.AccuracyParameter accuracy_param = 27;
  public native @Cast("bool") boolean has_accuracy_param();
  public native void clear_accuracy_param();
  @MemberGetter public static native int kAccuracyParamFieldNumber();
  public static final int kAccuracyParamFieldNumber = kAccuracyParamFieldNumber();
  public native @Const @ByRef AccuracyParameter accuracy_param();
  public native AccuracyParameter mutable_accuracy_param();
  public native AccuracyParameter release_accuracy_param();
  public native void set_allocated_accuracy_param(AccuracyParameter accuracy_param);

  // optional .caffe.ArgMaxParameter argmax_param = 23;
  public native @Cast("bool") boolean has_argmax_param();
  public native void clear_argmax_param();
  @MemberGetter public static native int kArgmaxParamFieldNumber();
  public static final int kArgmaxParamFieldNumber = kArgmaxParamFieldNumber();
  public native @Const @ByRef ArgMaxParameter argmax_param();
  public native ArgMaxParameter mutable_argmax_param();
  public native ArgMaxParameter release_argmax_param();
  public native void set_allocated_argmax_param(ArgMaxParameter argmax_param);

  // optional .caffe.ConcatParameter concat_param = 9;
  public native @Cast("bool") boolean has_concat_param();
  public native void clear_concat_param();
  @MemberGetter public static native int kConcatParamFieldNumber();
  public static final int kConcatParamFieldNumber = kConcatParamFieldNumber();
  public native @Const @ByRef ConcatParameter concat_param();
  public native ConcatParameter mutable_concat_param();
  public native ConcatParameter release_concat_param();
  public native void set_allocated_concat_param(ConcatParameter concat_param);

  // optional .caffe.ContrastiveLossParameter contrastive_loss_param = 40;
  public native @Cast("bool") boolean has_contrastive_loss_param();
  public native void clear_contrastive_loss_param();
  @MemberGetter public static native int kContrastiveLossParamFieldNumber();
  public static final int kContrastiveLossParamFieldNumber = kContrastiveLossParamFieldNumber();
  public native @Const @ByRef ContrastiveLossParameter contrastive_loss_param();
  public native ContrastiveLossParameter mutable_contrastive_loss_param();
  public native ContrastiveLossParameter release_contrastive_loss_param();
  public native void set_allocated_contrastive_loss_param(ContrastiveLossParameter contrastive_loss_param);

  // optional .caffe.ConvolutionParameter convolution_param = 10;
  public native @Cast("bool") boolean has_convolution_param();
  public native void clear_convolution_param();
  @MemberGetter public static native int kConvolutionParamFieldNumber();
  public static final int kConvolutionParamFieldNumber = kConvolutionParamFieldNumber();
  public native @Const @ByRef ConvolutionParameter convolution_param();
  public native ConvolutionParameter mutable_convolution_param();
  public native ConvolutionParameter release_convolution_param();
  public native void set_allocated_convolution_param(ConvolutionParameter convolution_param);

  // optional .caffe.DataParameter data_param = 11;
  public native @Cast("bool") boolean has_data_param();
  public native void clear_data_param();
  @MemberGetter public static native int kDataParamFieldNumber();
  public static final int kDataParamFieldNumber = kDataParamFieldNumber();
  public native @Const @ByRef DataParameter data_param();
  public native DataParameter mutable_data_param();
  public native DataParameter release_data_param();
  public native void set_allocated_data_param(DataParameter data_param);

  // optional .caffe.DropoutParameter dropout_param = 12;
  public native @Cast("bool") boolean has_dropout_param();
  public native void clear_dropout_param();
  @MemberGetter public static native int kDropoutParamFieldNumber();
  public static final int kDropoutParamFieldNumber = kDropoutParamFieldNumber();
  public native @Const @ByRef DropoutParameter dropout_param();
  public native DropoutParameter mutable_dropout_param();
  public native DropoutParameter release_dropout_param();
  public native void set_allocated_dropout_param(DropoutParameter dropout_param);

  // optional .caffe.DummyDataParameter dummy_data_param = 26;
  public native @Cast("bool") boolean has_dummy_data_param();
  public native void clear_dummy_data_param();
  @MemberGetter public static native int kDummyDataParamFieldNumber();
  public static final int kDummyDataParamFieldNumber = kDummyDataParamFieldNumber();
  public native @Const @ByRef DummyDataParameter dummy_data_param();
  public native DummyDataParameter mutable_dummy_data_param();
  public native DummyDataParameter release_dummy_data_param();
  public native void set_allocated_dummy_data_param(DummyDataParameter dummy_data_param);

  // optional .caffe.EltwiseParameter eltwise_param = 24;
  public native @Cast("bool") boolean has_eltwise_param();
  public native void clear_eltwise_param();
  @MemberGetter public static native int kEltwiseParamFieldNumber();
  public static final int kEltwiseParamFieldNumber = kEltwiseParamFieldNumber();
  public native @Const @ByRef EltwiseParameter eltwise_param();
  public native EltwiseParameter mutable_eltwise_param();
  public native EltwiseParameter release_eltwise_param();
  public native void set_allocated_eltwise_param(EltwiseParameter eltwise_param);

  // optional .caffe.ExpParameter exp_param = 41;
  public native @Cast("bool") boolean has_exp_param();
  public native void clear_exp_param();
  @MemberGetter public static native int kExpParamFieldNumber();
  public static final int kExpParamFieldNumber = kExpParamFieldNumber();
  public native @Const @ByRef ExpParameter exp_param();
  public native ExpParameter mutable_exp_param();
  public native ExpParameter release_exp_param();
  public native void set_allocated_exp_param(ExpParameter exp_param);

  // optional .caffe.HDF5DataParameter hdf5_data_param = 13;
  public native @Cast("bool") boolean has_hdf5_data_param();
  public native void clear_hdf5_data_param();
  @MemberGetter public static native int kHdf5DataParamFieldNumber();
  public static final int kHdf5DataParamFieldNumber = kHdf5DataParamFieldNumber();
  public native @Const @ByRef HDF5DataParameter hdf5_data_param();
  public native HDF5DataParameter mutable_hdf5_data_param();
  public native HDF5DataParameter release_hdf5_data_param();
  public native void set_allocated_hdf5_data_param(HDF5DataParameter hdf5_data_param);

  // optional .caffe.HDF5OutputParameter hdf5_output_param = 14;
  public native @Cast("bool") boolean has_hdf5_output_param();
  public native void clear_hdf5_output_param();
  @MemberGetter public static native int kHdf5OutputParamFieldNumber();
  public static final int kHdf5OutputParamFieldNumber = kHdf5OutputParamFieldNumber();
  public native @Const @ByRef HDF5OutputParameter hdf5_output_param();
  public native HDF5OutputParameter mutable_hdf5_output_param();
  public native HDF5OutputParameter release_hdf5_output_param();
  public native void set_allocated_hdf5_output_param(HDF5OutputParameter hdf5_output_param);

  // optional .caffe.HingeLossParameter hinge_loss_param = 29;
  public native @Cast("bool") boolean has_hinge_loss_param();
  public native void clear_hinge_loss_param();
  @MemberGetter public static native int kHingeLossParamFieldNumber();
  public static final int kHingeLossParamFieldNumber = kHingeLossParamFieldNumber();
  public native @Const @ByRef HingeLossParameter hinge_loss_param();
  public native HingeLossParameter mutable_hinge_loss_param();
  public native HingeLossParameter release_hinge_loss_param();
  public native void set_allocated_hinge_loss_param(HingeLossParameter hinge_loss_param);

  // optional .caffe.ImageDataParameter image_data_param = 15;
  public native @Cast("bool") boolean has_image_data_param();
  public native void clear_image_data_param();
  @MemberGetter public static native int kImageDataParamFieldNumber();
  public static final int kImageDataParamFieldNumber = kImageDataParamFieldNumber();
  public native @Const @ByRef ImageDataParameter image_data_param();
  public native ImageDataParameter mutable_image_data_param();
  public native ImageDataParameter release_image_data_param();
  public native void set_allocated_image_data_param(ImageDataParameter image_data_param);

  // optional .caffe.InfogainLossParameter infogain_loss_param = 16;
  public native @Cast("bool") boolean has_infogain_loss_param();
  public native void clear_infogain_loss_param();
  @MemberGetter public static native int kInfogainLossParamFieldNumber();
  public static final int kInfogainLossParamFieldNumber = kInfogainLossParamFieldNumber();
  public native @Const @ByRef InfogainLossParameter infogain_loss_param();
  public native InfogainLossParameter mutable_infogain_loss_param();
  public native InfogainLossParameter release_infogain_loss_param();
  public native void set_allocated_infogain_loss_param(InfogainLossParameter infogain_loss_param);

  // optional .caffe.InnerProductParameter inner_product_param = 17;
  public native @Cast("bool") boolean has_inner_product_param();
  public native void clear_inner_product_param();
  @MemberGetter public static native int kInnerProductParamFieldNumber();
  public static final int kInnerProductParamFieldNumber = kInnerProductParamFieldNumber();
  public native @Const @ByRef InnerProductParameter inner_product_param();
  public native InnerProductParameter mutable_inner_product_param();
  public native InnerProductParameter release_inner_product_param();
  public native void set_allocated_inner_product_param(InnerProductParameter inner_product_param);

  // optional .caffe.LRNParameter lrn_param = 18;
  public native @Cast("bool") boolean has_lrn_param();
  public native void clear_lrn_param();
  @MemberGetter public static native int kLrnParamFieldNumber();
  public static final int kLrnParamFieldNumber = kLrnParamFieldNumber();
  public native @Const @ByRef LRNParameter lrn_param();
  public native LRNParameter mutable_lrn_param();
  public native LRNParameter release_lrn_param();
  public native void set_allocated_lrn_param(LRNParameter lrn_param);

  // optional .caffe.MemoryDataParameter memory_data_param = 22;
  public native @Cast("bool") boolean has_memory_data_param();
  public native void clear_memory_data_param();
  @MemberGetter public static native int kMemoryDataParamFieldNumber();
  public static final int kMemoryDataParamFieldNumber = kMemoryDataParamFieldNumber();
  public native @Const @ByRef MemoryDataParameter memory_data_param();
  public native MemoryDataParameter mutable_memory_data_param();
  public native MemoryDataParameter release_memory_data_param();
  public native void set_allocated_memory_data_param(MemoryDataParameter memory_data_param);

  // optional .caffe.MVNParameter mvn_param = 34;
  public native @Cast("bool") boolean has_mvn_param();
  public native void clear_mvn_param();
  @MemberGetter public static native int kMvnParamFieldNumber();
  public static final int kMvnParamFieldNumber = kMvnParamFieldNumber();
  public native @Const @ByRef MVNParameter mvn_param();
  public native MVNParameter mutable_mvn_param();
  public native MVNParameter release_mvn_param();
  public native void set_allocated_mvn_param(MVNParameter mvn_param);

  // optional .caffe.PoolingParameter pooling_param = 19;
  public native @Cast("bool") boolean has_pooling_param();
  public native void clear_pooling_param();
  @MemberGetter public static native int kPoolingParamFieldNumber();
  public static final int kPoolingParamFieldNumber = kPoolingParamFieldNumber();
  public native @Const @ByRef PoolingParameter pooling_param();
  public native PoolingParameter mutable_pooling_param();
  public native PoolingParameter release_pooling_param();
  public native void set_allocated_pooling_param(PoolingParameter pooling_param);

  // optional .caffe.PowerParameter power_param = 21;
  public native @Cast("bool") boolean has_power_param();
  public native void clear_power_param();
  @MemberGetter public static native int kPowerParamFieldNumber();
  public static final int kPowerParamFieldNumber = kPowerParamFieldNumber();
  public native @Const @ByRef PowerParameter power_param();
  public native PowerParameter mutable_power_param();
  public native PowerParameter release_power_param();
  public native void set_allocated_power_param(PowerParameter power_param);

  // optional .caffe.ReLUParameter relu_param = 30;
  public native @Cast("bool") boolean has_relu_param();
  public native void clear_relu_param();
  @MemberGetter public static native int kReluParamFieldNumber();
  public static final int kReluParamFieldNumber = kReluParamFieldNumber();
  public native @Const @ByRef ReLUParameter relu_param();
  public native ReLUParameter mutable_relu_param();
  public native ReLUParameter release_relu_param();
  public native void set_allocated_relu_param(ReLUParameter relu_param);

  // optional .caffe.SigmoidParameter sigmoid_param = 38;
  public native @Cast("bool") boolean has_sigmoid_param();
  public native void clear_sigmoid_param();
  @MemberGetter public static native int kSigmoidParamFieldNumber();
  public static final int kSigmoidParamFieldNumber = kSigmoidParamFieldNumber();
  public native @Const @ByRef SigmoidParameter sigmoid_param();
  public native SigmoidParameter mutable_sigmoid_param();
  public native SigmoidParameter release_sigmoid_param();
  public native void set_allocated_sigmoid_param(SigmoidParameter sigmoid_param);

  // optional .caffe.SoftmaxParameter softmax_param = 39;
  public native @Cast("bool") boolean has_softmax_param();
  public native void clear_softmax_param();
  @MemberGetter public static native int kSoftmaxParamFieldNumber();
  public static final int kSoftmaxParamFieldNumber = kSoftmaxParamFieldNumber();
  public native @Const @ByRef SoftmaxParameter softmax_param();
  public native SoftmaxParameter mutable_softmax_param();
  public native SoftmaxParameter release_softmax_param();
  public native void set_allocated_softmax_param(SoftmaxParameter softmax_param);

  // optional .caffe.SliceParameter slice_param = 31;
  public native @Cast("bool") boolean has_slice_param();
  public native void clear_slice_param();
  @MemberGetter public static native int kSliceParamFieldNumber();
  public static final int kSliceParamFieldNumber = kSliceParamFieldNumber();
  public native @Const @ByRef SliceParameter slice_param();
  public native SliceParameter mutable_slice_param();
  public native SliceParameter release_slice_param();
  public native void set_allocated_slice_param(SliceParameter slice_param);

  // optional .caffe.TanHParameter tanh_param = 37;
  public native @Cast("bool") boolean has_tanh_param();
  public native void clear_tanh_param();
  @MemberGetter public static native int kTanhParamFieldNumber();
  public static final int kTanhParamFieldNumber = kTanhParamFieldNumber();
  public native @Const @ByRef TanHParameter tanh_param();
  public native TanHParameter mutable_tanh_param();
  public native TanHParameter release_tanh_param();
  public native void set_allocated_tanh_param(TanHParameter tanh_param);

  // optional .caffe.ThresholdParameter threshold_param = 25;
  public native @Cast("bool") boolean has_threshold_param();
  public native void clear_threshold_param();
  @MemberGetter public static native int kThresholdParamFieldNumber();
  public static final int kThresholdParamFieldNumber = kThresholdParamFieldNumber();
  public native @Const @ByRef ThresholdParameter threshold_param();
  public native ThresholdParameter mutable_threshold_param();
  public native ThresholdParameter release_threshold_param();
  public native void set_allocated_threshold_param(ThresholdParameter threshold_param);

  // optional .caffe.WindowDataParameter window_data_param = 20;
  public native @Cast("bool") boolean has_window_data_param();
  public native void clear_window_data_param();
  @MemberGetter public static native int kWindowDataParamFieldNumber();
  public static final int kWindowDataParamFieldNumber = kWindowDataParamFieldNumber();
  public native @Const @ByRef WindowDataParameter window_data_param();
  public native WindowDataParameter mutable_window_data_param();
  public native WindowDataParameter release_window_data_param();
  public native void set_allocated_window_data_param(WindowDataParameter window_data_param);

  // optional .caffe.TransformationParameter transform_param = 36;
  public native @Cast("bool") boolean has_transform_param();
  public native void clear_transform_param();
  @MemberGetter public static native int kTransformParamFieldNumber();
  public static final int kTransformParamFieldNumber = kTransformParamFieldNumber();
  public native @Const @ByRef TransformationParameter transform_param();
  public native TransformationParameter mutable_transform_param();
  public native TransformationParameter release_transform_param();
  public native void set_allocated_transform_param(TransformationParameter transform_param);

  // optional .caffe.LossParameter loss_param = 42;
  public native @Cast("bool") boolean has_loss_param();
  public native void clear_loss_param();
  @MemberGetter public static native int kLossParamFieldNumber();
  public static final int kLossParamFieldNumber = kLossParamFieldNumber();
  public native @Const @ByRef LossParameter loss_param();
  public native LossParameter mutable_loss_param();
  public native LossParameter release_loss_param();
  public native void set_allocated_loss_param(LossParameter loss_param);

  // optional .caffe.V0LayerParameter layer = 1;
  public native @Cast("bool") boolean has_layer();
  public native void clear_layer();
  @MemberGetter public static native int kLayerFieldNumber();
  public static final int kLayerFieldNumber = kLayerFieldNumber();
  public native @Const @ByRef V0LayerParameter layer();
  public native V0LayerParameter mutable_layer();
  public native V0LayerParameter release_layer();
  public native void set_allocated_layer(V0LayerParameter layer);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class V0LayerParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public V0LayerParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public V0LayerParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public V0LayerParameter position(int position) {
        return (V0LayerParameter)super.position(position);
    }

  public V0LayerParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public V0LayerParameter(@Const @ByRef V0LayerParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef V0LayerParameter from);

  public native @ByRef @Name("operator =") V0LayerParameter put(@Const @ByRef V0LayerParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef V0LayerParameter default_instance();

  public native void Swap(V0LayerParameter other);

  // implements Message ----------------------------------------------

  public native V0LayerParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef V0LayerParameter from);
  public native void MergeFrom(@Const @ByRef V0LayerParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------
  @MemberGetter public static native @Cast("const caffe::V0LayerParameter::PoolMethod") int MAX();
  public static final int MAX = MAX();
  @MemberGetter public static native @Cast("const caffe::V0LayerParameter::PoolMethod") int AVE();
  public static final int AVE = AVE();
  @MemberGetter public static native @Cast("const caffe::V0LayerParameter::PoolMethod") int STOCHASTIC();
  public static final int STOCHASTIC = STOCHASTIC();
  public static native @Cast("bool") boolean PoolMethod_IsValid(int value);
  @MemberGetter public static native @Cast("const caffe::V0LayerParameter::PoolMethod") int PoolMethod_MIN();
  public static final int PoolMethod_MIN = PoolMethod_MIN();
  @MemberGetter public static native @Cast("const caffe::V0LayerParameter::PoolMethod") int PoolMethod_MAX();
  public static final int PoolMethod_MAX = PoolMethod_MAX();
  @MemberGetter public static native int PoolMethod_ARRAYSIZE();
  public static final int PoolMethod_ARRAYSIZE = PoolMethod_ARRAYSIZE();
  public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer PoolMethod_descriptor();
  public static native @StdString BytePointer PoolMethod_Name(@Cast("caffe::V0LayerParameter::PoolMethod") int value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::V0LayerParameter::PoolMethod*") IntPointer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::V0LayerParameter::PoolMethod*") IntBuffer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::V0LayerParameter::PoolMethod*") int[] value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::V0LayerParameter::PoolMethod*") IntPointer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString BytePointer name,
        @Cast("caffe::V0LayerParameter::PoolMethod*") IntBuffer value);
  public static native @Cast("bool") boolean PoolMethod_Parse(@StdString String name,
        @Cast("caffe::V0LayerParameter::PoolMethod*") int[] value);

  // accessors -------------------------------------------------------

  // optional string name = 1;
  public native @Cast("bool") boolean has_name();
  public native void clear_name();
  @MemberGetter public static native int kNameFieldNumber();
  public static final int kNameFieldNumber = kNameFieldNumber();
  public native @StdString BytePointer name();
  public native void set_name(@StdString BytePointer value);
  public native void set_name(@StdString String value);
  public native void set_name(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_name(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_name();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_name();
  public native void set_allocated_name(@StdString @Cast({"char*", "std::string*"}) BytePointer name);

  // optional string type = 2;
  public native @Cast("bool") boolean has_type();
  public native void clear_type();
  @MemberGetter public static native int kTypeFieldNumber();
  public static final int kTypeFieldNumber = kTypeFieldNumber();
  public native @StdString BytePointer type();
  public native void set_type(@StdString BytePointer value);
  public native void set_type(@StdString String value);
  public native void set_type(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_type(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_type();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_type();
  public native void set_allocated_type(@StdString @Cast({"char*", "std::string*"}) BytePointer type);

  // optional uint32 num_output = 3;
  public native @Cast("bool") boolean has_num_output();
  public native void clear_num_output();
  @MemberGetter public static native int kNumOutputFieldNumber();
  public static final int kNumOutputFieldNumber = kNumOutputFieldNumber();
  public native @Cast("google::protobuf::uint32") int num_output();
  public native void set_num_output(@Cast("google::protobuf::uint32") int value);

  // optional bool biasterm = 4 [default = true];
  public native @Cast("bool") boolean has_biasterm();
  public native void clear_biasterm();
  @MemberGetter public static native int kBiastermFieldNumber();
  public static final int kBiastermFieldNumber = kBiastermFieldNumber();
  public native @Cast("bool") boolean biasterm();
  public native void set_biasterm(@Cast("bool") boolean value);

  // optional .caffe.FillerParameter weight_filler = 5;
  public native @Cast("bool") boolean has_weight_filler();
  public native void clear_weight_filler();
  @MemberGetter public static native int kWeightFillerFieldNumber();
  public static final int kWeightFillerFieldNumber = kWeightFillerFieldNumber();
  public native @Const @ByRef FillerParameter weight_filler();
  public native FillerParameter mutable_weight_filler();
  public native FillerParameter release_weight_filler();
  public native void set_allocated_weight_filler(FillerParameter weight_filler);

  // optional .caffe.FillerParameter bias_filler = 6;
  public native @Cast("bool") boolean has_bias_filler();
  public native void clear_bias_filler();
  @MemberGetter public static native int kBiasFillerFieldNumber();
  public static final int kBiasFillerFieldNumber = kBiasFillerFieldNumber();
  public native @Const @ByRef FillerParameter bias_filler();
  public native FillerParameter mutable_bias_filler();
  public native FillerParameter release_bias_filler();
  public native void set_allocated_bias_filler(FillerParameter bias_filler);

  // optional uint32 pad = 7 [default = 0];
  public native @Cast("bool") boolean has_pad();
  public native void clear_pad();
  @MemberGetter public static native int kPadFieldNumber();
  public static final int kPadFieldNumber = kPadFieldNumber();
  public native @Cast("google::protobuf::uint32") int pad();
  public native void set_pad(@Cast("google::protobuf::uint32") int value);

  // optional uint32 kernelsize = 8;
  public native @Cast("bool") boolean has_kernelsize();
  public native void clear_kernelsize();
  @MemberGetter public static native int kKernelsizeFieldNumber();
  public static final int kKernelsizeFieldNumber = kKernelsizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int kernelsize();
  public native void set_kernelsize(@Cast("google::protobuf::uint32") int value);

  // optional uint32 group = 9 [default = 1];
  public native @Cast("bool") boolean has_group();
  public native void clear_group();
  @MemberGetter public static native int kGroupFieldNumber();
  public static final int kGroupFieldNumber = kGroupFieldNumber();
  public native @Cast("google::protobuf::uint32") int group();
  public native void set_group(@Cast("google::protobuf::uint32") int value);

  // optional uint32 stride = 10 [default = 1];
  public native @Cast("bool") boolean has_stride();
  public native void clear_stride();
  @MemberGetter public static native int kStrideFieldNumber();
  public static final int kStrideFieldNumber = kStrideFieldNumber();
  public native @Cast("google::protobuf::uint32") int stride();
  public native void set_stride(@Cast("google::protobuf::uint32") int value);

  // optional .caffe.V0LayerParameter.PoolMethod pool = 11 [default = MAX];
  public native @Cast("bool") boolean has_pool();
  public native void clear_pool();
  @MemberGetter public static native int kPoolFieldNumber();
  public static final int kPoolFieldNumber = kPoolFieldNumber();
  public native @Cast("caffe::V0LayerParameter_PoolMethod") int pool();
  public native void set_pool(@Cast("caffe::V0LayerParameter_PoolMethod") int value);

  // optional float dropout_ratio = 12 [default = 0.5];
  public native @Cast("bool") boolean has_dropout_ratio();
  public native void clear_dropout_ratio();
  @MemberGetter public static native int kDropoutRatioFieldNumber();
  public static final int kDropoutRatioFieldNumber = kDropoutRatioFieldNumber();
  public native float dropout_ratio();
  public native void set_dropout_ratio(float value);

  // optional uint32 local_size = 13 [default = 5];
  public native @Cast("bool") boolean has_local_size();
  public native void clear_local_size();
  @MemberGetter public static native int kLocalSizeFieldNumber();
  public static final int kLocalSizeFieldNumber = kLocalSizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int local_size();
  public native void set_local_size(@Cast("google::protobuf::uint32") int value);

  // optional float alpha = 14 [default = 1];
  public native @Cast("bool") boolean has_alpha();
  public native void clear_alpha();
  @MemberGetter public static native int kAlphaFieldNumber();
  public static final int kAlphaFieldNumber = kAlphaFieldNumber();
  public native float alpha();
  public native void set_alpha(float value);

  // optional float beta = 15 [default = 0.75];
  public native @Cast("bool") boolean has_beta();
  public native void clear_beta();
  @MemberGetter public static native int kBetaFieldNumber();
  public static final int kBetaFieldNumber = kBetaFieldNumber();
  public native float beta();
  public native void set_beta(float value);

  // optional float k = 22 [default = 1];
  public native @Cast("bool") boolean has_k();
  public native void clear_k();
  @MemberGetter public static native int kKFieldNumber();
  public static final int kKFieldNumber = kKFieldNumber();
  public native float k();
  public native void set_k(float value);

  // optional string source = 16;
  public native @Cast("bool") boolean has_source();
  public native void clear_source();
  @MemberGetter public static native int kSourceFieldNumber();
  public static final int kSourceFieldNumber = kSourceFieldNumber();
  public native @StdString BytePointer source();
  public native void set_source(@StdString BytePointer value);
  public native void set_source(@StdString String value);
  public native void set_source(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_source(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_source();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_source();
  public native void set_allocated_source(@StdString @Cast({"char*", "std::string*"}) BytePointer source);

  // optional float scale = 17 [default = 1];
  public native @Cast("bool") boolean has_scale();
  public native void clear_scale();
  @MemberGetter public static native int kScaleFieldNumber();
  public static final int kScaleFieldNumber = kScaleFieldNumber();
  public native float scale();
  public native void set_scale(float value);

  // optional string meanfile = 18;
  public native @Cast("bool") boolean has_meanfile();
  public native void clear_meanfile();
  @MemberGetter public static native int kMeanfileFieldNumber();
  public static final int kMeanfileFieldNumber = kMeanfileFieldNumber();
  public native @StdString BytePointer meanfile();
  public native void set_meanfile(@StdString BytePointer value);
  public native void set_meanfile(@StdString String value);
  public native void set_meanfile(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_meanfile(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_meanfile();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_meanfile();
  public native void set_allocated_meanfile(@StdString @Cast({"char*", "std::string*"}) BytePointer meanfile);

  // optional uint32 batchsize = 19;
  public native @Cast("bool") boolean has_batchsize();
  public native void clear_batchsize();
  @MemberGetter public static native int kBatchsizeFieldNumber();
  public static final int kBatchsizeFieldNumber = kBatchsizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int batchsize();
  public native void set_batchsize(@Cast("google::protobuf::uint32") int value);

  // optional uint32 cropsize = 20 [default = 0];
  public native @Cast("bool") boolean has_cropsize();
  public native void clear_cropsize();
  @MemberGetter public static native int kCropsizeFieldNumber();
  public static final int kCropsizeFieldNumber = kCropsizeFieldNumber();
  public native @Cast("google::protobuf::uint32") int cropsize();
  public native void set_cropsize(@Cast("google::protobuf::uint32") int value);

  // optional bool mirror = 21 [default = false];
  public native @Cast("bool") boolean has_mirror();
  public native void clear_mirror();
  @MemberGetter public static native int kMirrorFieldNumber();
  public static final int kMirrorFieldNumber = kMirrorFieldNumber();
  public native @Cast("bool") boolean mirror();
  public native void set_mirror(@Cast("bool") boolean value);

  // repeated .caffe.BlobProto blobs = 50;
  public native int blobs_size();
  public native void clear_blobs();
  @MemberGetter public static native int kBlobsFieldNumber();
  public static final int kBlobsFieldNumber = kBlobsFieldNumber();
  public native @Const @ByRef BlobProto blobs(int index);
  public native BlobProto mutable_blobs(int index);
  public native BlobProto add_blobs();

  // repeated float blobs_lr = 51;
  public native int blobs_lr_size();
  public native void clear_blobs_lr();
  @MemberGetter public static native int kBlobsLrFieldNumber();
  public static final int kBlobsLrFieldNumber = kBlobsLrFieldNumber();
  public native float blobs_lr(int index);
  public native void set_blobs_lr(int index, float value);
  public native void add_blobs_lr(float value);

  // repeated float weight_decay = 52;
  public native int weight_decay_size();
  public native void clear_weight_decay();
  @MemberGetter public static native int kWeightDecayFieldNumber();
  public static final int kWeightDecayFieldNumber = kWeightDecayFieldNumber();
  public native float weight_decay(int index);
  public native void set_weight_decay(int index, float value);
  public native void add_weight_decay(float value);

  // optional uint32 rand_skip = 53 [default = 0];
  public native @Cast("bool") boolean has_rand_skip();
  public native void clear_rand_skip();
  @MemberGetter public static native int kRandSkipFieldNumber();
  public static final int kRandSkipFieldNumber = kRandSkipFieldNumber();
  public native @Cast("google::protobuf::uint32") int rand_skip();
  public native void set_rand_skip(@Cast("google::protobuf::uint32") int value);

  // optional float det_fg_threshold = 54 [default = 0.5];
  public native @Cast("bool") boolean has_det_fg_threshold();
  public native void clear_det_fg_threshold();
  @MemberGetter public static native int kDetFgThresholdFieldNumber();
  public static final int kDetFgThresholdFieldNumber = kDetFgThresholdFieldNumber();
  public native float det_fg_threshold();
  public native void set_det_fg_threshold(float value);

  // optional float det_bg_threshold = 55 [default = 0.5];
  public native @Cast("bool") boolean has_det_bg_threshold();
  public native void clear_det_bg_threshold();
  @MemberGetter public static native int kDetBgThresholdFieldNumber();
  public static final int kDetBgThresholdFieldNumber = kDetBgThresholdFieldNumber();
  public native float det_bg_threshold();
  public native void set_det_bg_threshold(float value);

  // optional float det_fg_fraction = 56 [default = 0.25];
  public native @Cast("bool") boolean has_det_fg_fraction();
  public native void clear_det_fg_fraction();
  @MemberGetter public static native int kDetFgFractionFieldNumber();
  public static final int kDetFgFractionFieldNumber = kDetFgFractionFieldNumber();
  public native float det_fg_fraction();
  public native void set_det_fg_fraction(float value);

  // optional uint32 det_context_pad = 58 [default = 0];
  public native @Cast("bool") boolean has_det_context_pad();
  public native void clear_det_context_pad();
  @MemberGetter public static native int kDetContextPadFieldNumber();
  public static final int kDetContextPadFieldNumber = kDetContextPadFieldNumber();
  public native @Cast("google::protobuf::uint32") int det_context_pad();
  public native void set_det_context_pad(@Cast("google::protobuf::uint32") int value);

  // optional string det_crop_mode = 59 [default = "warp"];
  public native @Cast("bool") boolean has_det_crop_mode();
  public native void clear_det_crop_mode();
  @MemberGetter public static native int kDetCropModeFieldNumber();
  public static final int kDetCropModeFieldNumber = kDetCropModeFieldNumber();
  public native @StdString BytePointer det_crop_mode();
  public native void set_det_crop_mode(@StdString BytePointer value);
  public native void set_det_crop_mode(@StdString String value);
  public native void set_det_crop_mode(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_det_crop_mode(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_det_crop_mode();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_det_crop_mode();
  public native void set_allocated_det_crop_mode(@StdString @Cast({"char*", "std::string*"}) BytePointer det_crop_mode);

  // optional int32 new_num = 60 [default = 0];
  public native @Cast("bool") boolean has_new_num();
  public native void clear_new_num();
  @MemberGetter public static native int kNewNumFieldNumber();
  public static final int kNewNumFieldNumber = kNewNumFieldNumber();
  public native @Cast("google::protobuf::int32") int new_num();
  public native void set_new_num(@Cast("google::protobuf::int32") int value);

  // optional int32 new_channels = 61 [default = 0];
  public native @Cast("bool") boolean has_new_channels();
  public native void clear_new_channels();
  @MemberGetter public static native int kNewChannelsFieldNumber();
  public static final int kNewChannelsFieldNumber = kNewChannelsFieldNumber();
  public native @Cast("google::protobuf::int32") int new_channels();
  public native void set_new_channels(@Cast("google::protobuf::int32") int value);

  // optional int32 new_height = 62 [default = 0];
  public native @Cast("bool") boolean has_new_height();
  public native void clear_new_height();
  @MemberGetter public static native int kNewHeightFieldNumber();
  public static final int kNewHeightFieldNumber = kNewHeightFieldNumber();
  public native @Cast("google::protobuf::int32") int new_height();
  public native void set_new_height(@Cast("google::protobuf::int32") int value);

  // optional int32 new_width = 63 [default = 0];
  public native @Cast("bool") boolean has_new_width();
  public native void clear_new_width();
  @MemberGetter public static native int kNewWidthFieldNumber();
  public static final int kNewWidthFieldNumber = kNewWidthFieldNumber();
  public native @Cast("google::protobuf::int32") int new_width();
  public native void set_new_width(@Cast("google::protobuf::int32") int value);

  // optional bool shuffle_images = 64 [default = false];
  public native @Cast("bool") boolean has_shuffle_images();
  public native void clear_shuffle_images();
  @MemberGetter public static native int kShuffleImagesFieldNumber();
  public static final int kShuffleImagesFieldNumber = kShuffleImagesFieldNumber();
  public native @Cast("bool") boolean shuffle_images();
  public native void set_shuffle_images(@Cast("bool") boolean value);

  // optional uint32 concat_dim = 65 [default = 1];
  public native @Cast("bool") boolean has_concat_dim();
  public native void clear_concat_dim();
  @MemberGetter public static native int kConcatDimFieldNumber();
  public static final int kConcatDimFieldNumber = kConcatDimFieldNumber();
  public native @Cast("google::protobuf::uint32") int concat_dim();
  public native void set_concat_dim(@Cast("google::protobuf::uint32") int value);

  // optional .caffe.HDF5OutputParameter hdf5_output_param = 1001;
  public native @Cast("bool") boolean has_hdf5_output_param();
  public native void clear_hdf5_output_param();
  @MemberGetter public static native int kHdf5OutputParamFieldNumber();
  public static final int kHdf5OutputParamFieldNumber = kHdf5OutputParamFieldNumber();
  public native @Const @ByRef HDF5OutputParameter hdf5_output_param();
  public native HDF5OutputParameter mutable_hdf5_output_param();
  public native HDF5OutputParameter release_hdf5_output_param();
  public native void set_allocated_hdf5_output_param(HDF5OutputParameter hdf5_output_param);
}
// -------------------------------------------------------------------

@Namespace("caffe") @NoOffset public static class PReLUParameter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PReLUParameter(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public PReLUParameter(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public PReLUParameter position(int position) {
        return (PReLUParameter)super.position(position);
    }

  public PReLUParameter() { super((Pointer)null); allocate(); }
  private native void allocate();

  public PReLUParameter(@Const @ByRef PReLUParameter from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef PReLUParameter from);

  public native @ByRef @Name("operator =") PReLUParameter put(@Const @ByRef PReLUParameter from);

  public native @Cast("const google::protobuf::UnknownFieldSet*") @ByRef Pointer unknown_fields();

  public native @Cast("google::protobuf::UnknownFieldSet*") Pointer mutable_unknown_fields();

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef PReLUParameter default_instance();

  public native void Swap(PReLUParameter other);

  // implements Message ----------------------------------------------

  public native PReLUParameter New();
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef PReLUParameter from);
  public native void MergeFrom(@Const @ByRef PReLUParameter from);
  public native void Clear();
  public native @Cast("bool") boolean IsInitialized();

  public native int ByteSize();
  public native @Cast("bool") boolean MergePartialFromCodedStream(
        @Cast("google::protobuf::io::CodedInputStream*") Pointer input);
  public native void SerializeWithCachedSizes(
        @Cast("google::protobuf::io::CodedOutputStream*") Pointer output);
  public native @Cast("google::protobuf::uint8*") BytePointer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") BytePointer output);
  public native @Cast("google::protobuf::uint8*") ByteBuffer SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") ByteBuffer output);
  public native @Cast("google::protobuf::uint8*") byte[] SerializeWithCachedSizesToArray(@Cast("google::protobuf::uint8*") byte[] output);
  public native int GetCachedSize();
  public native @ByVal @Cast("google::protobuf::Metadata*") Pointer GetMetadata();

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional .caffe.FillerParameter filler = 1;
  public native @Cast("bool") boolean has_filler();
  public native void clear_filler();
  @MemberGetter public static native int kFillerFieldNumber();
  public static final int kFillerFieldNumber = kFillerFieldNumber();
  public native @Const @ByRef FillerParameter filler();
  public native FillerParameter mutable_filler();
  public native FillerParameter release_filler();
  public native void set_allocated_filler(FillerParameter filler);

  // optional bool channel_shared = 2 [default = false];
  public native @Cast("bool") boolean has_channel_shared();
  public native void clear_channel_shared();
  @MemberGetter public static native int kChannelSharedFieldNumber();
  public static final int kChannelSharedFieldNumber = kChannelSharedFieldNumber();
  public native @Cast("bool") boolean channel_shared();
  public native void set_channel_shared(@Cast("bool") boolean value);
}
// ===================================================================


// ===================================================================

// BlobShape

// repeated int64 dim = 1 [packed = true];








// -------------------------------------------------------------------

// BlobProto

// optional .caffe.BlobShape shape = 7;









// repeated float data = 5 [packed = true];








// repeated float diff = 6 [packed = true];








// repeated double double_data = 8 [packed = true];








// repeated double double_diff = 9 [packed = true];








// optional int32 num = 1 [default = 0];







// optional int32 channels = 2 [default = 0];







// optional int32 height = 3 [default = 0];







// optional int32 width = 4 [default = 0];







// -------------------------------------------------------------------

// BlobProtoVector

// repeated .caffe.BlobProto blobs = 1;








// -------------------------------------------------------------------

// Datum

// optional int32 channels = 1;







// optional int32 height = 2;







// optional int32 width = 3;







// optional bytes data = 4;












// optional int32 label = 5;







// repeated float float_data = 6;








// optional bool encoded = 7 [default = false];







// -------------------------------------------------------------------

// FillerParameter

// optional string type = 1 [default = "constant"];












// optional float value = 2 [default = 0];







// optional float min = 3 [default = 0];







// optional float max = 4 [default = 1];







// optional float mean = 5 [default = 0];







// optional float std = 6 [default = 1];







// optional int32 sparse = 7 [default = -1];







// optional .caffe.FillerParameter.VarianceNorm variance_norm = 8 [default = FAN_IN];







// -------------------------------------------------------------------

// NetParameter

// optional string name = 1;












// repeated string input = 3;














// repeated .caffe.BlobShape input_shape = 8;








// repeated int32 input_dim = 4;








// optional bool force_backward = 5 [default = false];







// optional .caffe.NetState state = 6;









// optional bool debug_info = 7 [default = false];







// repeated .caffe.LayerParameter layer = 100;








// repeated .caffe.V1LayerParameter layers = 2;








// -------------------------------------------------------------------

// SolverParameter

// optional string net = 24;












// optional .caffe.NetParameter net_param = 25;









// optional string train_net = 1;












// repeated string test_net = 2;














// optional .caffe.NetParameter train_net_param = 21;









// repeated .caffe.NetParameter test_net_param = 22;








// optional .caffe.NetState train_state = 26;









// repeated .caffe.NetState test_state = 27;








// repeated int32 test_iter = 3;








// optional int32 test_interval = 4 [default = 0];







// optional bool test_compute_loss = 19 [default = false];







// optional bool test_initialization = 32 [default = true];







// optional float base_lr = 5;







// optional int32 display = 6;







// optional int32 average_loss = 33 [default = 1];







// optional int32 max_iter = 7;







// optional int32 iter_size = 36 [default = 1];







// optional string lr_policy = 8;












// optional float gamma = 9;







// optional float power = 10;







// optional float momentum = 11;







// optional float weight_decay = 12;







// optional string regularization_type = 29 [default = "L2"];












// optional int32 stepsize = 13;







// repeated int32 stepvalue = 34;








// optional float clip_gradients = 35 [default = -1];







// optional int32 snapshot = 14 [default = 0];







// optional string snapshot_prefix = 15;












// optional bool snapshot_diff = 16 [default = false];







// optional .caffe.SolverParameter.SnapshotFormat snapshot_format = 37 [default = BINARYPROTO];







// optional .caffe.SolverParameter.SolverMode solver_mode = 17 [default = GPU];







// optional int32 device_id = 18 [default = 0];







// optional int64 random_seed = 20 [default = -1];







// optional string type = 40 [default = "SGD"];












// optional float delta = 31 [default = 1e-08];







// optional float momentum2 = 39 [default = 0.999];







// optional float rms_decay = 38;







// optional bool debug_info = 23 [default = false];







// optional bool snapshot_after_train = 28 [default = true];







// optional .caffe.SolverParameter.SolverType solver_type = 30 [default = SGD];







// -------------------------------------------------------------------

// SolverState

// optional int32 iter = 1;







// optional string learned_net = 2;












// repeated .caffe.BlobProto history = 3;








// optional int32 current_step = 4 [default = 0];







// -------------------------------------------------------------------

// NetState

// optional .caffe.Phase phase = 1 [default = TEST];







// optional int32 level = 2 [default = 0];







// repeated string stage = 3;














// -------------------------------------------------------------------

// NetStateRule

// optional .caffe.Phase phase = 1;







// optional int32 min_level = 2;







// optional int32 max_level = 3;







// repeated string stage = 4;














// repeated string not_stage = 5;














// -------------------------------------------------------------------

// ParamSpec

// optional string name = 1;












// optional .caffe.ParamSpec.DimCheckMode share_mode = 2;







// optional float lr_mult = 3 [default = 1];







// optional float decay_mult = 4 [default = 1];







// -------------------------------------------------------------------

// LayerParameter

// optional string name = 1;












// optional string type = 2;












// repeated string bottom = 3;














// repeated string top = 4;














// optional .caffe.Phase phase = 10;







// repeated float loss_weight = 5;








// repeated .caffe.ParamSpec param = 6;








// repeated .caffe.BlobProto blobs = 7;








// repeated bool propagate_down = 11;








// repeated .caffe.NetStateRule include = 8;








// repeated .caffe.NetStateRule exclude = 9;








// optional .caffe.TransformationParameter transform_param = 100;









// optional .caffe.LossParameter loss_param = 101;









// optional .caffe.AccuracyParameter accuracy_param = 102;









// optional .caffe.ArgMaxParameter argmax_param = 103;









// optional .caffe.BatchNormParameter batch_norm_param = 139;









// optional .caffe.ConcatParameter concat_param = 104;









// optional .caffe.ContrastiveLossParameter contrastive_loss_param = 105;









// optional .caffe.ConvolutionParameter convolution_param = 106;









// optional .caffe.DataParameter data_param = 107;









// optional .caffe.DropoutParameter dropout_param = 108;









// optional .caffe.DummyDataParameter dummy_data_param = 109;









// optional .caffe.EltwiseParameter eltwise_param = 110;









// optional .caffe.EmbedParameter embed_param = 137;









// optional .caffe.ExpParameter exp_param = 111;









// optional .caffe.FlattenParameter flatten_param = 135;









// optional .caffe.HDF5DataParameter hdf5_data_param = 112;









// optional .caffe.HDF5OutputParameter hdf5_output_param = 113;









// optional .caffe.HingeLossParameter hinge_loss_param = 114;









// optional .caffe.ImageDataParameter image_data_param = 115;









// optional .caffe.InfogainLossParameter infogain_loss_param = 116;









// optional .caffe.InnerProductParameter inner_product_param = 117;









// optional .caffe.LogParameter log_param = 134;









// optional .caffe.LRNParameter lrn_param = 118;









// optional .caffe.MemoryDataParameter memory_data_param = 119;









// optional .caffe.MVNParameter mvn_param = 120;









// optional .caffe.PoolingParameter pooling_param = 121;









// optional .caffe.PowerParameter power_param = 122;









// optional .caffe.PReLUParameter prelu_param = 131;









// optional .caffe.PythonParameter python_param = 130;









// optional .caffe.ReductionParameter reduction_param = 136;









// optional .caffe.ReLUParameter relu_param = 123;









// optional .caffe.ReshapeParameter reshape_param = 133;









// optional .caffe.SigmoidParameter sigmoid_param = 124;









// optional .caffe.SoftmaxParameter softmax_param = 125;









// optional .caffe.SPPParameter spp_param = 132;









// optional .caffe.SliceParameter slice_param = 126;









// optional .caffe.TanHParameter tanh_param = 127;









// optional .caffe.ThresholdParameter threshold_param = 128;









// optional .caffe.TileParameter tile_param = 138;









// optional .caffe.WindowDataParameter window_data_param = 129;









// -------------------------------------------------------------------

// TransformationParameter

// optional float scale = 1 [default = 1];







// optional bool mirror = 2 [default = false];







// optional uint32 crop_size = 3 [default = 0];







// optional string mean_file = 4;












// repeated float mean_value = 5;








// optional bool force_color = 6 [default = false];







// optional bool force_gray = 7 [default = false];







// -------------------------------------------------------------------

// LossParameter

// optional int32 ignore_label = 1;







// optional .caffe.LossParameter.NormalizationMode normalization = 3 [default = VALID];







// optional bool normalize = 2;







// -------------------------------------------------------------------

// AccuracyParameter

// optional uint32 top_k = 1 [default = 1];







// optional int32 axis = 2 [default = 1];







// optional int32 ignore_label = 3;







// -------------------------------------------------------------------

// ArgMaxParameter

// optional bool out_max_val = 1 [default = false];







// optional uint32 top_k = 2 [default = 1];







// optional int32 axis = 3;







// -------------------------------------------------------------------

// ConcatParameter

// optional int32 axis = 2 [default = 1];







// optional uint32 concat_dim = 1 [default = 1];







// -------------------------------------------------------------------

// BatchNormParameter

// optional bool use_global_stats = 1;







// optional float moving_average_fraction = 2 [default = 0.999];







// optional float eps = 3 [default = 1e-05];







// -------------------------------------------------------------------

// ContrastiveLossParameter

// optional float margin = 1 [default = 1];







// optional bool legacy_version = 2 [default = false];







// -------------------------------------------------------------------

// ConvolutionParameter

// optional uint32 num_output = 1;







// optional bool bias_term = 2 [default = true];







// repeated uint32 pad = 3;








// repeated uint32 kernel_size = 4;








// repeated uint32 stride = 6;








// optional uint32 pad_h = 9 [default = 0];







// optional uint32 pad_w = 10 [default = 0];







// optional uint32 kernel_h = 11;







// optional uint32 kernel_w = 12;







// optional uint32 stride_h = 13;







// optional uint32 stride_w = 14;







// optional uint32 group = 5 [default = 1];







// optional .caffe.FillerParameter weight_filler = 7;









// optional .caffe.FillerParameter bias_filler = 8;









// optional .caffe.ConvolutionParameter.Engine engine = 15 [default = DEFAULT];







// optional int32 axis = 16 [default = 1];







// optional bool force_nd_im2col = 17 [default = false];







// -------------------------------------------------------------------

// DataParameter

// optional string source = 1;












// optional uint32 batch_size = 4;







// optional uint32 rand_skip = 7 [default = 0];







// optional .caffe.DataParameter.DB backend = 8 [default = LEVELDB];







// optional float scale = 2 [default = 1];







// optional string mean_file = 3;












// optional uint32 crop_size = 5 [default = 0];







// optional bool mirror = 6 [default = false];







// optional bool force_encoded_color = 9 [default = false];







// optional uint32 prefetch = 10 [default = 4];







// -------------------------------------------------------------------

// DropoutParameter

// optional float dropout_ratio = 1 [default = 0.5];







// -------------------------------------------------------------------

// DummyDataParameter

// repeated .caffe.FillerParameter data_filler = 1;








// repeated .caffe.BlobShape shape = 6;








// repeated uint32 num = 2;








// repeated uint32 channels = 3;








// repeated uint32 height = 4;








// repeated uint32 width = 5;








// -------------------------------------------------------------------

// EltwiseParameter

// optional .caffe.EltwiseParameter.EltwiseOp operation = 1 [default = SUM];







// repeated float coeff = 2;








// optional bool stable_prod_grad = 3 [default = true];







// -------------------------------------------------------------------

// EmbedParameter

// optional uint32 num_output = 1;







// optional uint32 input_dim = 2;







// optional bool bias_term = 3 [default = true];







// optional .caffe.FillerParameter weight_filler = 4;









// optional .caffe.FillerParameter bias_filler = 5;









// -------------------------------------------------------------------

// ExpParameter

// optional float base = 1 [default = -1];







// optional float scale = 2 [default = 1];







// optional float shift = 3 [default = 0];







// -------------------------------------------------------------------

// FlattenParameter

// optional int32 axis = 1 [default = 1];







// optional int32 end_axis = 2 [default = -1];







// -------------------------------------------------------------------

// HDF5DataParameter

// optional string source = 1;












// optional uint32 batch_size = 2;







// optional bool shuffle = 3 [default = false];







// -------------------------------------------------------------------

// HDF5OutputParameter

// optional string file_name = 1;












// -------------------------------------------------------------------

// HingeLossParameter

// optional .caffe.HingeLossParameter.Norm norm = 1 [default = L1];







// -------------------------------------------------------------------

// ImageDataParameter

// optional string source = 1;












// optional uint32 batch_size = 4 [default = 1];







// optional uint32 rand_skip = 7 [default = 0];







// optional bool shuffle = 8 [default = false];







// optional uint32 new_height = 9 [default = 0];







// optional uint32 new_width = 10 [default = 0];







// optional bool is_color = 11 [default = true];







// optional float scale = 2 [default = 1];







// optional string mean_file = 3;












// optional uint32 crop_size = 5 [default = 0];







// optional bool mirror = 6 [default = false];







// optional string root_folder = 12 [default = ""];












// -------------------------------------------------------------------

// InfogainLossParameter

// optional string source = 1;












// -------------------------------------------------------------------

// InnerProductParameter

// optional uint32 num_output = 1;







// optional bool bias_term = 2 [default = true];







// optional .caffe.FillerParameter weight_filler = 3;









// optional .caffe.FillerParameter bias_filler = 4;









// optional int32 axis = 5 [default = 1];







// -------------------------------------------------------------------

// LogParameter

// optional float base = 1 [default = -1];







// optional float scale = 2 [default = 1];







// optional float shift = 3 [default = 0];







// -------------------------------------------------------------------

// LRNParameter

// optional uint32 local_size = 1 [default = 5];







// optional float alpha = 2 [default = 1];







// optional float beta = 3 [default = 0.75];







// optional .caffe.LRNParameter.NormRegion norm_region = 4 [default = ACROSS_CHANNELS];







// optional float k = 5 [default = 1];







// optional .caffe.LRNParameter.Engine engine = 6 [default = DEFAULT];







// -------------------------------------------------------------------

// MemoryDataParameter

// optional uint32 batch_size = 1;







// optional uint32 channels = 2;







// optional uint32 height = 3;







// optional uint32 width = 4;







// -------------------------------------------------------------------

// MVNParameter

// optional bool normalize_variance = 1 [default = true];







// optional bool across_channels = 2 [default = false];







// optional float eps = 3 [default = 1e-09];







// -------------------------------------------------------------------

// PoolingParameter

// optional .caffe.PoolingParameter.PoolMethod pool = 1 [default = MAX];







// optional uint32 pad = 4 [default = 0];







// optional uint32 pad_h = 9 [default = 0];







// optional uint32 pad_w = 10 [default = 0];







// optional uint32 kernel_size = 2;







// optional uint32 kernel_h = 5;







// optional uint32 kernel_w = 6;







// optional uint32 stride = 3 [default = 1];







// optional uint32 stride_h = 7;







// optional uint32 stride_w = 8;







// optional .caffe.PoolingParameter.Engine engine = 11 [default = DEFAULT];







// optional bool global_pooling = 12 [default = false];







// -------------------------------------------------------------------

// PowerParameter

// optional float power = 1 [default = 1];







// optional float scale = 2 [default = 1];







// optional float shift = 3 [default = 0];







// -------------------------------------------------------------------

// PythonParameter

// optional string module = 1;












// optional string layer = 2;












// optional string param_str = 3 [default = ""];












// optional bool share_in_parallel = 4 [default = false];







// -------------------------------------------------------------------

// ReductionParameter

// optional .caffe.ReductionParameter.ReductionOp operation = 1 [default = SUM];







// optional int32 axis = 2 [default = 0];







// optional float coeff = 3 [default = 1];







// -------------------------------------------------------------------

// ReLUParameter

// optional float negative_slope = 1 [default = 0];







// optional .caffe.ReLUParameter.Engine engine = 2 [default = DEFAULT];







// -------------------------------------------------------------------

// ReshapeParameter

// optional .caffe.BlobShape shape = 1;









// optional int32 axis = 2 [default = 0];







// optional int32 num_axes = 3 [default = -1];







// -------------------------------------------------------------------

// SigmoidParameter

// optional .caffe.SigmoidParameter.Engine engine = 1 [default = DEFAULT];







// -------------------------------------------------------------------

// SliceParameter

// optional int32 axis = 3 [default = 1];







// repeated uint32 slice_point = 2;








// optional uint32 slice_dim = 1 [default = 1];







// -------------------------------------------------------------------

// SoftmaxParameter

// optional .caffe.SoftmaxParameter.Engine engine = 1 [default = DEFAULT];







// optional int32 axis = 2 [default = 1];







// -------------------------------------------------------------------

// TanHParameter

// optional .caffe.TanHParameter.Engine engine = 1 [default = DEFAULT];







// -------------------------------------------------------------------

// TileParameter

// optional int32 axis = 1 [default = 1];







// optional int32 tiles = 2;







// -------------------------------------------------------------------

// ThresholdParameter

// optional float threshold = 1 [default = 0];







// -------------------------------------------------------------------

// WindowDataParameter

// optional string source = 1;












// optional float scale = 2 [default = 1];







// optional string mean_file = 3;












// optional uint32 batch_size = 4;







// optional uint32 crop_size = 5 [default = 0];







// optional bool mirror = 6 [default = false];







// optional float fg_threshold = 7 [default = 0.5];







// optional float bg_threshold = 8 [default = 0.5];







// optional float fg_fraction = 9 [default = 0.25];







// optional uint32 context_pad = 10 [default = 0];







// optional string crop_mode = 11 [default = "warp"];












// optional bool cache_images = 12 [default = false];







// optional string root_folder = 13 [default = ""];












// -------------------------------------------------------------------

// SPPParameter

// optional uint32 pyramid_height = 1;







// optional .caffe.SPPParameter.PoolMethod pool = 2 [default = MAX];







// optional .caffe.SPPParameter.Engine engine = 6 [default = DEFAULT];







// -------------------------------------------------------------------

// V1LayerParameter

// repeated string bottom = 2;














// repeated string top = 3;














// optional string name = 4;












// repeated .caffe.NetStateRule include = 32;








// repeated .caffe.NetStateRule exclude = 33;








// optional .caffe.V1LayerParameter.LayerType type = 5;







// repeated .caffe.BlobProto blobs = 6;








// repeated string param = 1001;














// repeated .caffe.V1LayerParameter.DimCheckMode blob_share_mode = 1002;








// repeated float blobs_lr = 7;








// repeated float weight_decay = 8;








// repeated float loss_weight = 35;








// optional .caffe.AccuracyParameter accuracy_param = 27;









// optional .caffe.ArgMaxParameter argmax_param = 23;









// optional .caffe.ConcatParameter concat_param = 9;









// optional .caffe.ContrastiveLossParameter contrastive_loss_param = 40;









// optional .caffe.ConvolutionParameter convolution_param = 10;









// optional .caffe.DataParameter data_param = 11;









// optional .caffe.DropoutParameter dropout_param = 12;









// optional .caffe.DummyDataParameter dummy_data_param = 26;









// optional .caffe.EltwiseParameter eltwise_param = 24;









// optional .caffe.ExpParameter exp_param = 41;









// optional .caffe.HDF5DataParameter hdf5_data_param = 13;









// optional .caffe.HDF5OutputParameter hdf5_output_param = 14;









// optional .caffe.HingeLossParameter hinge_loss_param = 29;









// optional .caffe.ImageDataParameter image_data_param = 15;









// optional .caffe.InfogainLossParameter infogain_loss_param = 16;









// optional .caffe.InnerProductParameter inner_product_param = 17;









// optional .caffe.LRNParameter lrn_param = 18;









// optional .caffe.MemoryDataParameter memory_data_param = 22;









// optional .caffe.MVNParameter mvn_param = 34;









// optional .caffe.PoolingParameter pooling_param = 19;









// optional .caffe.PowerParameter power_param = 21;









// optional .caffe.ReLUParameter relu_param = 30;









// optional .caffe.SigmoidParameter sigmoid_param = 38;









// optional .caffe.SoftmaxParameter softmax_param = 39;









// optional .caffe.SliceParameter slice_param = 31;









// optional .caffe.TanHParameter tanh_param = 37;









// optional .caffe.ThresholdParameter threshold_param = 25;









// optional .caffe.WindowDataParameter window_data_param = 20;









// optional .caffe.TransformationParameter transform_param = 36;









// optional .caffe.LossParameter loss_param = 42;









// optional .caffe.V0LayerParameter layer = 1;









// -------------------------------------------------------------------

// V0LayerParameter

// optional string name = 1;












// optional string type = 2;












// optional uint32 num_output = 3;







// optional bool biasterm = 4 [default = true];







// optional .caffe.FillerParameter weight_filler = 5;









// optional .caffe.FillerParameter bias_filler = 6;









// optional uint32 pad = 7 [default = 0];







// optional uint32 kernelsize = 8;







// optional uint32 group = 9 [default = 1];







// optional uint32 stride = 10 [default = 1];







// optional .caffe.V0LayerParameter.PoolMethod pool = 11 [default = MAX];







// optional float dropout_ratio = 12 [default = 0.5];







// optional uint32 local_size = 13 [default = 5];







// optional float alpha = 14 [default = 1];







// optional float beta = 15 [default = 0.75];







// optional float k = 22 [default = 1];







// optional string source = 16;












// optional float scale = 17 [default = 1];







// optional string meanfile = 18;












// optional uint32 batchsize = 19;







// optional uint32 cropsize = 20 [default = 0];







// optional bool mirror = 21 [default = false];







// repeated .caffe.BlobProto blobs = 50;








// repeated float blobs_lr = 51;








// repeated float weight_decay = 52;








// optional uint32 rand_skip = 53 [default = 0];







// optional float det_fg_threshold = 54 [default = 0.5];







// optional float det_bg_threshold = 55 [default = 0.5];







// optional float det_fg_fraction = 56 [default = 0.25];







// optional uint32 det_context_pad = 58 [default = 0];







// optional string det_crop_mode = 59 [default = "warp"];












// optional int32 new_num = 60 [default = 0];







// optional int32 new_channels = 61 [default = 0];







// optional int32 new_height = 62 [default = 0];







// optional int32 new_width = 63 [default = 0];







// optional bool shuffle_images = 64 [default = false];







// optional uint32 concat_dim = 65 [default = 1];







// optional .caffe.HDF5OutputParameter hdf5_output_param = 1001;









// -------------------------------------------------------------------

// PReLUParameter

// optional .caffe.FillerParameter filler = 1;









// optional bool channel_shared = 2 [default = false];








// @@protoc_insertion_point(namespace_scope)

  // namespace caffe

// #ifndef SWIG
// #endif  // SWIG

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_caffe_2eproto__INCLUDED


// Parsed from caffe/util/blocking_queue.hpp

// #ifndef CAFFE_UTIL_BLOCKING_QUEUE_HPP_
// #define CAFFE_UTIL_BLOCKING_QUEUE_HPP_

// #include <queue>
// #include <string>

@Name("caffe::BlockingQueue<caffe::Datum*>") @NoOffset public static class DatumBlockingQueue extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DatumBlockingQueue(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DatumBlockingQueue(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DatumBlockingQueue position(int position) {
        return (DatumBlockingQueue)super.position(position);
    }

  public DatumBlockingQueue() { super((Pointer)null); allocate(); }
  private native void allocate();

  public native void push(@ByPtrRef Datum t);

  public native @Cast("bool") boolean try_pop(@Cast("caffe::Datum**") PointerPointer t);
  public native @Cast("bool") boolean try_pop(@ByPtrPtr Datum t);

  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  public native Datum pop(@StdString BytePointer log_on_wait/*=""*/);
  public native Datum pop();
  public native Datum pop(@StdString String log_on_wait/*=""*/);

  public native @Cast("bool") boolean try_peek(@Cast("caffe::Datum**") PointerPointer t);
  public native @Cast("bool") boolean try_peek(@ByPtrPtr Datum t);

  // Return element without removing it
  public native Datum peek();

  public native @Cast("size_t") long size();
}

  // namespace caffe

// #endif


// Parsed from caffe/data_reader.hpp

// #ifndef CAFFE_DATA_READER_HPP_
// #define CAFFE_DATA_READER_HPP_

// #include <map>
// #include <string>
// #include <vector>

// #include "caffe/common.hpp"
// #include "caffe/internal_thread.hpp"
// #include "caffe/util/blocking_queue.hpp"
// #include "caffe/util/db.hpp"

/**
 * \brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
@Namespace("caffe") @NoOffset public static class DataReader extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DataReader(Pointer p) { super(p); }

  public DataReader(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  public native @ByRef DatumBlockingQueue free();
  public native @ByRef DatumBlockingQueue full();
}

  // namespace caffe

// #endif  // CAFFE_DATA_READER_HPP_


// Parsed from caffe/util/math_functions.hpp

// #ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
// #define CAFFE_UTIL_MATH_FUNCTIONS_H_

// #include <stdint.h>
// #include <cmath>  // for std::fabs and std::signbit

// #include "glog/logging.h"

// #include "caffe/common.hpp"
// #include "caffe/util/device_alternate.hpp"
// #include "caffe/util/mkl_alternate.hpp"

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
@Namespace("caffe") public static native @Name("caffe_cpu_gemm<float>") void caffe_cpu_gemm_float(@Cast("const CBLAS_TRANSPOSE") int TransA,
    @Cast("const CBLAS_TRANSPOSE") int TransB, int M, int N, int K,
    float alpha, @Const FloatPointer A, @Const FloatPointer B, float beta,
    FloatPointer C);
@Namespace("caffe") public static native @Name("caffe_cpu_gemm<float>") void caffe_cpu_gemm_float(@Cast("const CBLAS_TRANSPOSE") int TransA,
    @Cast("const CBLAS_TRANSPOSE") int TransB, int M, int N, int K,
    float alpha, @Const FloatBuffer A, @Const FloatBuffer B, float beta,
    FloatBuffer C);
@Namespace("caffe") public static native @Name("caffe_cpu_gemm<float>") void caffe_cpu_gemm_float(@Cast("const CBLAS_TRANSPOSE") int TransA,
    @Cast("const CBLAS_TRANSPOSE") int TransB, int M, int N, int K,
    float alpha, @Const float[] A, @Const float[] B, float beta,
    float[] C);
@Namespace("caffe") public static native @Name("caffe_cpu_gemm<double>") void caffe_cpu_gemm_double(@Cast("const CBLAS_TRANSPOSE") int TransA,
    @Cast("const CBLAS_TRANSPOSE") int TransB, int M, int N, int K,
    double alpha, @Const DoublePointer A, @Const DoublePointer B, double beta,
    DoublePointer C);
@Namespace("caffe") public static native @Name("caffe_cpu_gemm<double>") void caffe_cpu_gemm_double(@Cast("const CBLAS_TRANSPOSE") int TransA,
    @Cast("const CBLAS_TRANSPOSE") int TransB, int M, int N, int K,
    double alpha, @Const DoubleBuffer A, @Const DoubleBuffer B, double beta,
    DoubleBuffer C);
@Namespace("caffe") public static native @Name("caffe_cpu_gemm<double>") void caffe_cpu_gemm_double(@Cast("const CBLAS_TRANSPOSE") int TransA,
    @Cast("const CBLAS_TRANSPOSE") int TransB, int M, int N, int K,
    double alpha, @Const double[] A, @Const double[] B, double beta,
    double[] C);

@Namespace("caffe") public static native @Name("caffe_cpu_gemv<float>") void caffe_cpu_gemv_float(@Cast("const CBLAS_TRANSPOSE") int TransA, int M, int N,
    float alpha, @Const FloatPointer A, @Const FloatPointer x, float beta,
    FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_cpu_gemv<float>") void caffe_cpu_gemv_float(@Cast("const CBLAS_TRANSPOSE") int TransA, int M, int N,
    float alpha, @Const FloatBuffer A, @Const FloatBuffer x, float beta,
    FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_cpu_gemv<float>") void caffe_cpu_gemv_float(@Cast("const CBLAS_TRANSPOSE") int TransA, int M, int N,
    float alpha, @Const float[] A, @Const float[] x, float beta,
    float[] y);

@Namespace("caffe") public static native @Name("caffe_cpu_gemv<double>") void caffe_cpu_gemv_double(@Cast("const CBLAS_TRANSPOSE") int TransA, int M, int N,
    double alpha, @Const DoublePointer A, @Const DoublePointer x, double beta,
    DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_cpu_gemv<double>") void caffe_cpu_gemv_double(@Cast("const CBLAS_TRANSPOSE") int TransA, int M, int N,
    double alpha, @Const DoubleBuffer A, @Const DoubleBuffer x, double beta,
    DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_cpu_gemv<double>") void caffe_cpu_gemv_double(@Cast("const CBLAS_TRANSPOSE") int TransA, int M, int N,
    double alpha, @Const double[] A, @Const double[] x, double beta,
    double[] y);

@Namespace("caffe") public static native @Name("caffe_axpy<float>") void caffe_axpy_float(int N, float alpha, @Const FloatPointer X,
    FloatPointer Y);
@Namespace("caffe") public static native @Name("caffe_axpy<float>") void caffe_axpy_float(int N, float alpha, @Const FloatBuffer X,
    FloatBuffer Y);
@Namespace("caffe") public static native @Name("caffe_axpy<float>") void caffe_axpy_float(int N, float alpha, @Const float[] X,
    float[] Y);

@Namespace("caffe") public static native @Name("caffe_axpy<double>") void caffe_axpy_double(int N, double alpha, @Const DoublePointer X,
    DoublePointer Y);
@Namespace("caffe") public static native @Name("caffe_axpy<double>") void caffe_axpy_double(int N, double alpha, @Const DoubleBuffer X,
    DoubleBuffer Y);
@Namespace("caffe") public static native @Name("caffe_axpy<double>") void caffe_axpy_double(int N, double alpha, @Const double[] X,
    double[] Y);

@Namespace("caffe") public static native @Name("caffe_cpu_axpby<float>") void caffe_cpu_axpby_float(int N, float alpha, @Const FloatPointer X,
    float beta, FloatPointer Y);
@Namespace("caffe") public static native @Name("caffe_cpu_axpby<float>") void caffe_cpu_axpby_float(int N, float alpha, @Const FloatBuffer X,
    float beta, FloatBuffer Y);
@Namespace("caffe") public static native @Name("caffe_cpu_axpby<float>") void caffe_cpu_axpby_float(int N, float alpha, @Const float[] X,
    float beta, float[] Y);

@Namespace("caffe") public static native @Name("caffe_cpu_axpby<double>") void caffe_cpu_axpby_double(int N, double alpha, @Const DoublePointer X,
    double beta, DoublePointer Y);
@Namespace("caffe") public static native @Name("caffe_cpu_axpby<double>") void caffe_cpu_axpby_double(int N, double alpha, @Const DoubleBuffer X,
    double beta, DoubleBuffer Y);
@Namespace("caffe") public static native @Name("caffe_cpu_axpby<double>") void caffe_cpu_axpby_double(int N, double alpha, @Const double[] X,
    double beta, double[] Y);

@Namespace("caffe") public static native @Name("caffe_copy<float>") void caffe_copy_float(int N, @Const FloatPointer X, FloatPointer Y);
@Namespace("caffe") public static native @Name("caffe_copy<float>") void caffe_copy_float(int N, @Const FloatBuffer X, FloatBuffer Y);
@Namespace("caffe") public static native @Name("caffe_copy<float>") void caffe_copy_float(int N, @Const float[] X, float[] Y);

@Namespace("caffe") public static native @Name("caffe_copy<double>") void caffe_copy_double(int N, @Const DoublePointer X, DoublePointer Y);
@Namespace("caffe") public static native @Name("caffe_copy<double>") void caffe_copy_double(int N, @Const DoubleBuffer X, DoubleBuffer Y);
@Namespace("caffe") public static native @Name("caffe_copy<double>") void caffe_copy_double(int N, @Const double[] X, double[] Y);

@Namespace("caffe") public static native @Name("caffe_set<float>") void caffe_set_float(int N, float alpha, FloatPointer X);
@Namespace("caffe") public static native @Name("caffe_set<float>") void caffe_set_float(int N, float alpha, FloatBuffer X);
@Namespace("caffe") public static native @Name("caffe_set<float>") void caffe_set_float(int N, float alpha, float[] X);

@Namespace("caffe") public static native @Name("caffe_set<double>") void caffe_set_double(int N, double alpha, DoublePointer X);
@Namespace("caffe") public static native @Name("caffe_set<double>") void caffe_set_double(int N, double alpha, DoubleBuffer X);
@Namespace("caffe") public static native @Name("caffe_set<double>") void caffe_set_double(int N, double alpha, double[] X);

@Namespace("caffe") public static native void caffe_memset(@Cast("const size_t") long N, int alpha, Pointer X);

@Namespace("caffe") public static native @Name("caffe_add_scalar<float>") void caffe_add_scalar_float(int N, float alpha, FloatPointer X);
@Namespace("caffe") public static native @Name("caffe_add_scalar<float>") void caffe_add_scalar_float(int N, float alpha, FloatBuffer X);
@Namespace("caffe") public static native @Name("caffe_add_scalar<float>") void caffe_add_scalar_float(int N, float alpha, float[] X);

@Namespace("caffe") public static native @Name("caffe_add_scalar<double>") void caffe_add_scalar_double(int N, double alpha, DoublePointer X);
@Namespace("caffe") public static native @Name("caffe_add_scalar<double>") void caffe_add_scalar_double(int N, double alpha, DoubleBuffer X);
@Namespace("caffe") public static native @Name("caffe_add_scalar<double>") void caffe_add_scalar_double(int N, double alpha, double[] X);

@Namespace("caffe") public static native @Name("caffe_scal<float>") void caffe_scal_float(int N, float alpha, FloatPointer X);
@Namespace("caffe") public static native @Name("caffe_scal<float>") void caffe_scal_float(int N, float alpha, FloatBuffer X);
@Namespace("caffe") public static native @Name("caffe_scal<float>") void caffe_scal_float(int N, float alpha, float[] X);

@Namespace("caffe") public static native @Name("caffe_scal<double>") void caffe_scal_double(int N, double alpha, DoublePointer X);
@Namespace("caffe") public static native @Name("caffe_scal<double>") void caffe_scal_double(int N, double alpha, DoubleBuffer X);
@Namespace("caffe") public static native @Name("caffe_scal<double>") void caffe_scal_double(int N, double alpha, double[] X);

@Namespace("caffe") public static native @Name("caffe_sqr<float>") void caffe_sqr_float(int N, @Const FloatPointer a, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_sqr<float>") void caffe_sqr_float(int N, @Const FloatBuffer a, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_sqr<float>") void caffe_sqr_float(int N, @Const float[] a, float[] y);

@Namespace("caffe") public static native @Name("caffe_sqr<double>") void caffe_sqr_double(int N, @Const DoublePointer a, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_sqr<double>") void caffe_sqr_double(int N, @Const DoubleBuffer a, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_sqr<double>") void caffe_sqr_double(int N, @Const double[] a, double[] y);

@Namespace("caffe") public static native @Name("caffe_add<float>") void caffe_add_float(int N, @Const FloatPointer a, @Const FloatPointer b, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_add<float>") void caffe_add_float(int N, @Const FloatBuffer a, @Const FloatBuffer b, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_add<float>") void caffe_add_float(int N, @Const float[] a, @Const float[] b, float[] y);

@Namespace("caffe") public static native @Name("caffe_add<double>") void caffe_add_double(int N, @Const DoublePointer a, @Const DoublePointer b, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_add<double>") void caffe_add_double(int N, @Const DoubleBuffer a, @Const DoubleBuffer b, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_add<double>") void caffe_add_double(int N, @Const double[] a, @Const double[] b, double[] y);

@Namespace("caffe") public static native @Name("caffe_sub<float>") void caffe_sub_float(int N, @Const FloatPointer a, @Const FloatPointer b, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_sub<float>") void caffe_sub_float(int N, @Const FloatBuffer a, @Const FloatBuffer b, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_sub<float>") void caffe_sub_float(int N, @Const float[] a, @Const float[] b, float[] y);

@Namespace("caffe") public static native @Name("caffe_sub<double>") void caffe_sub_double(int N, @Const DoublePointer a, @Const DoublePointer b, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_sub<double>") void caffe_sub_double(int N, @Const DoubleBuffer a, @Const DoubleBuffer b, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_sub<double>") void caffe_sub_double(int N, @Const double[] a, @Const double[] b, double[] y);

@Namespace("caffe") public static native @Name("caffe_mul<float>") void caffe_mul_float(int N, @Const FloatPointer a, @Const FloatPointer b, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_mul<float>") void caffe_mul_float(int N, @Const FloatBuffer a, @Const FloatBuffer b, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_mul<float>") void caffe_mul_float(int N, @Const float[] a, @Const float[] b, float[] y);

@Namespace("caffe") public static native @Name("caffe_mul<double>") void caffe_mul_double(int N, @Const DoublePointer a, @Const DoublePointer b, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_mul<double>") void caffe_mul_double(int N, @Const DoubleBuffer a, @Const DoubleBuffer b, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_mul<double>") void caffe_mul_double(int N, @Const double[] a, @Const double[] b, double[] y);

@Namespace("caffe") public static native @Name("caffe_div<float>") void caffe_div_float(int N, @Const FloatPointer a, @Const FloatPointer b, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_div<float>") void caffe_div_float(int N, @Const FloatBuffer a, @Const FloatBuffer b, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_div<float>") void caffe_div_float(int N, @Const float[] a, @Const float[] b, float[] y);

@Namespace("caffe") public static native @Name("caffe_div<double>") void caffe_div_double(int N, @Const DoublePointer a, @Const DoublePointer b, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_div<double>") void caffe_div_double(int N, @Const DoubleBuffer a, @Const DoubleBuffer b, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_div<double>") void caffe_div_double(int N, @Const double[] a, @Const double[] b, double[] y);

@Namespace("caffe") public static native @Name("caffe_powx<float>") void caffe_powx_float(int n, @Const FloatPointer a, float b, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_powx<float>") void caffe_powx_float(int n, @Const FloatBuffer a, float b, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_powx<float>") void caffe_powx_float(int n, @Const float[] a, float b, float[] y);

@Namespace("caffe") public static native @Name("caffe_powx<double>") void caffe_powx_double(int n, @Const DoublePointer a, double b, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_powx<double>") void caffe_powx_double(int n, @Const DoubleBuffer a, double b, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_powx<double>") void caffe_powx_double(int n, @Const double[] a, double b, double[] y);

@Namespace("caffe") public static native @Cast("unsigned int") int caffe_rng_rand();

@Namespace("caffe") public static native @Name("caffe_nextafter<float>") float caffe_nextafter_float(float b);

@Namespace("caffe") public static native @Name("caffe_nextafter<double>") double caffe_nextafter_double(double b);

@Namespace("caffe") public static native @Name("caffe_rng_uniform<float>") void caffe_rng_uniform_float(int n, float a, float b, FloatPointer r);
@Namespace("caffe") public static native @Name("caffe_rng_uniform<float>") void caffe_rng_uniform_float(int n, float a, float b, FloatBuffer r);
@Namespace("caffe") public static native @Name("caffe_rng_uniform<float>") void caffe_rng_uniform_float(int n, float a, float b, float[] r);

@Namespace("caffe") public static native @Name("caffe_rng_uniform<double>") void caffe_rng_uniform_double(int n, double a, double b, DoublePointer r);
@Namespace("caffe") public static native @Name("caffe_rng_uniform<double>") void caffe_rng_uniform_double(int n, double a, double b, DoubleBuffer r);
@Namespace("caffe") public static native @Name("caffe_rng_uniform<double>") void caffe_rng_uniform_double(int n, double a, double b, double[] r);

@Namespace("caffe") public static native @Name("caffe_rng_gaussian<float>") void caffe_rng_gaussian_float(int n, float mu, float sigma,
                        FloatPointer r);
@Namespace("caffe") public static native @Name("caffe_rng_gaussian<float>") void caffe_rng_gaussian_float(int n, float mu, float sigma,
                        FloatBuffer r);
@Namespace("caffe") public static native @Name("caffe_rng_gaussian<float>") void caffe_rng_gaussian_float(int n, float mu, float sigma,
                        float[] r);

@Namespace("caffe") public static native @Name("caffe_rng_gaussian<double>") void caffe_rng_gaussian_double(int n, double mu, double sigma,
                        DoublePointer r);
@Namespace("caffe") public static native @Name("caffe_rng_gaussian<double>") void caffe_rng_gaussian_double(int n, double mu, double sigma,
                        DoubleBuffer r);
@Namespace("caffe") public static native @Name("caffe_rng_gaussian<double>") void caffe_rng_gaussian_double(int n, double mu, double sigma,
                        double[] r);

@Namespace("caffe") public static native @Name("caffe_rng_bernoulli<float>") void caffe_rng_bernoulli_float(int n, float p, IntPointer r);
@Namespace("caffe") public static native @Name("caffe_rng_bernoulli<float>") void caffe_rng_bernoulli_float(int n, float p, IntBuffer r);
@Namespace("caffe") public static native @Name("caffe_rng_bernoulli<float>") void caffe_rng_bernoulli_float(int n, float p, int[] r);

@Namespace("caffe") public static native @Name("caffe_rng_bernoulli<double>") void caffe_rng_bernoulli_double(int n, double p, IntPointer r);
@Namespace("caffe") public static native @Name("caffe_rng_bernoulli<double>") void caffe_rng_bernoulli_double(int n, double p, IntBuffer r);
@Namespace("caffe") public static native @Name("caffe_rng_bernoulli<double>") void caffe_rng_bernoulli_double(int n, double p, int[] r);

@Namespace("caffe") public static native @Name("caffe_exp<float>") void caffe_exp_float(int n, @Const FloatPointer a, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_exp<float>") void caffe_exp_float(int n, @Const FloatBuffer a, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_exp<float>") void caffe_exp_float(int n, @Const float[] a, float[] y);

@Namespace("caffe") public static native @Name("caffe_exp<double>") void caffe_exp_double(int n, @Const DoublePointer a, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_exp<double>") void caffe_exp_double(int n, @Const DoubleBuffer a, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_exp<double>") void caffe_exp_double(int n, @Const double[] a, double[] y);

@Namespace("caffe") public static native @Name("caffe_log<float>") void caffe_log_float(int n, @Const FloatPointer a, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_log<float>") void caffe_log_float(int n, @Const FloatBuffer a, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_log<float>") void caffe_log_float(int n, @Const float[] a, float[] y);

@Namespace("caffe") public static native @Name("caffe_log<double>") void caffe_log_double(int n, @Const DoublePointer a, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_log<double>") void caffe_log_double(int n, @Const DoubleBuffer a, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_log<double>") void caffe_log_double(int n, @Const double[] a, double[] y);

@Namespace("caffe") public static native @Name("caffe_abs<float>") void caffe_abs_float(int n, @Const FloatPointer a, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_abs<float>") void caffe_abs_float(int n, @Const FloatBuffer a, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_abs<float>") void caffe_abs_float(int n, @Const float[] a, float[] y);

@Namespace("caffe") public static native @Name("caffe_abs<double>") void caffe_abs_double(int n, @Const DoublePointer a, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_abs<double>") void caffe_abs_double(int n, @Const DoubleBuffer a, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_abs<double>") void caffe_abs_double(int n, @Const double[] a, double[] y);

@Namespace("caffe") public static native @Name("caffe_cpu_dot<float>") float caffe_cpu_dot_float(int n, @Const FloatPointer x, @Const FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_cpu_dot<float>") float caffe_cpu_dot_float(int n, @Const FloatBuffer x, @Const FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_cpu_dot<float>") float caffe_cpu_dot_float(int n, @Const float[] x, @Const float[] y);

@Namespace("caffe") public static native @Name("caffe_cpu_dot<double>") double caffe_cpu_dot_double(int n, @Const DoublePointer x, @Const DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_cpu_dot<double>") double caffe_cpu_dot_double(int n, @Const DoubleBuffer x, @Const DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_cpu_dot<double>") double caffe_cpu_dot_double(int n, @Const double[] x, @Const double[] y);

@Namespace("caffe") public static native @Name("caffe_cpu_strided_dot<float>") float caffe_cpu_strided_dot_float(int n, @Const FloatPointer x, int incx,
    @Const FloatPointer y, int incy);
@Namespace("caffe") public static native @Name("caffe_cpu_strided_dot<float>") float caffe_cpu_strided_dot_float(int n, @Const FloatBuffer x, int incx,
    @Const FloatBuffer y, int incy);
@Namespace("caffe") public static native @Name("caffe_cpu_strided_dot<float>") float caffe_cpu_strided_dot_float(int n, @Const float[] x, int incx,
    @Const float[] y, int incy);

@Namespace("caffe") public static native @Name("caffe_cpu_strided_dot<double>") double caffe_cpu_strided_dot_double(int n, @Const DoublePointer x, int incx,
    @Const DoublePointer y, int incy);
@Namespace("caffe") public static native @Name("caffe_cpu_strided_dot<double>") double caffe_cpu_strided_dot_double(int n, @Const DoubleBuffer x, int incx,
    @Const DoubleBuffer y, int incy);
@Namespace("caffe") public static native @Name("caffe_cpu_strided_dot<double>") double caffe_cpu_strided_dot_double(int n, @Const double[] x, int incx,
    @Const double[] y, int incy);

// Returns the sum of the absolute values of the elements of vector x
@Namespace("caffe") public static native @Name("caffe_cpu_asum<float>") float caffe_cpu_asum_float(int n, @Const FloatPointer x);
@Namespace("caffe") public static native @Name("caffe_cpu_asum<float>") float caffe_cpu_asum_float(int n, @Const FloatBuffer x);
@Namespace("caffe") public static native @Name("caffe_cpu_asum<float>") float caffe_cpu_asum_float(int n, @Const float[] x);
@Namespace("caffe") public static native @Name("caffe_cpu_asum<double>") double caffe_cpu_asum_double(int n, @Const DoublePointer x);
@Namespace("caffe") public static native @Name("caffe_cpu_asum<double>") double caffe_cpu_asum_double(int n, @Const DoubleBuffer x);
@Namespace("caffe") public static native @Name("caffe_cpu_asum<double>") double caffe_cpu_asum_double(int n, @Const double[] x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
@Namespace("caffe") public static native @Name("caffe_sign<float>") byte caffe_sign_float(float val);
@Namespace("caffe") public static native @Name("caffe_sign<double>") byte caffe_sign_double(double val);

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
// #define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation)
//   template<typename Dtype>
//   void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) {
//     CHECK_GT(n, 0); CHECK(x); CHECK(y);
//     for (int i = 0; i < n; ++i) {
//       operation;
//     }
//   }

// output is 1 for the positives, 0 for zero, and -1 for the negatives

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.

@Namespace("caffe") public static native @Name("caffe_cpu_scale<float>") void caffe_cpu_scale_float(int n, float alpha, @Const FloatPointer x, FloatPointer y);
@Namespace("caffe") public static native @Name("caffe_cpu_scale<float>") void caffe_cpu_scale_float(int n, float alpha, @Const FloatBuffer x, FloatBuffer y);
@Namespace("caffe") public static native @Name("caffe_cpu_scale<float>") void caffe_cpu_scale_float(int n, float alpha, @Const float[] x, float[] y);

@Namespace("caffe") public static native @Name("caffe_cpu_scale<double>") void caffe_cpu_scale_double(int n, double alpha, @Const DoublePointer x, DoublePointer y);
@Namespace("caffe") public static native @Name("caffe_cpu_scale<double>") void caffe_cpu_scale_double(int n, double alpha, @Const DoubleBuffer x, DoubleBuffer y);
@Namespace("caffe") public static native @Name("caffe_cpu_scale<double>") void caffe_cpu_scale_double(int n, double alpha, @Const double[] x, double[] y);

// #ifndef CPU_ONLY  // GPU

// #endif  // !CPU_ONLY

  // namespace caffe

// #endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_


// Parsed from caffe/syncedmem.hpp

// #ifndef CAFFE_SYNCEDMEM_HPP_
// #define CAFFE_SYNCEDMEM_HPP_

// #include <cstdlib>

// #include "caffe/common.hpp"

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
@Namespace("caffe") public static native void CaffeMallocHost(@Cast("void**") PointerPointer ptr, @Cast("size_t") long size, @Cast("bool*") BoolPointer use_cuda);
@Namespace("caffe") public static native void CaffeMallocHost(@Cast("void**") @ByPtrPtr Pointer ptr, @Cast("size_t") long size, @Cast("bool*") BoolPointer use_cuda);
@Namespace("caffe") public static native void CaffeMallocHost(@Cast("void**") @ByPtrPtr Pointer ptr, @Cast("size_t") long size, @Cast("bool*") boolean[] use_cuda);

@Namespace("caffe") public static native void CaffeFreeHost(Pointer ptr, @Cast("bool") boolean use_cuda);


/**
 * \brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
@Namespace("caffe") @NoOffset public static class SyncedMemory extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SyncedMemory(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SyncedMemory(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SyncedMemory position(int position) {
        return (SyncedMemory)super.position(position);
    }

  public SyncedMemory() { super((Pointer)null); allocate(); }
  private native void allocate();
  public SyncedMemory(@Cast("size_t") long size) { super((Pointer)null); allocate(size); }
  private native void allocate(@Cast("size_t") long size);
  public native @Const Pointer cpu_data();
  public native void set_cpu_data(Pointer data);
  public native @Const Pointer gpu_data();
  public native void set_gpu_data(Pointer data);
  public native Pointer mutable_cpu_data();
  public native Pointer mutable_gpu_data();
  /** enum caffe::SyncedMemory::SyncedHead */
  public static final int UNINITIALIZED = 0, HEAD_AT_CPU = 1, HEAD_AT_GPU = 2, SYNCED = 3;
  public native @Cast("caffe::SyncedMemory::SyncedHead") int head();
  public native @Cast("size_t") long size();
}  // class SyncedMemory

  // namespace caffe

// #endif  // CAFFE_SYNCEDMEM_HPP_


// Parsed from caffe/blob.hpp

// #ifndef CAFFE_BLOB_HPP_
// #define CAFFE_BLOB_HPP_

// #include <algorithm>
// #include <string>
// #include <vector>

// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/syncedmem.hpp"

@MemberGetter public static native int kMaxBlobAxes();

/**
 * \brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
@Name("caffe::Blob<float>") @NoOffset public static class FloatBlob extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBlob(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FloatBlob(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FloatBlob position(int position) {
        return (FloatBlob)super.position(position);
    }

  public FloatBlob() { super((Pointer)null); allocate(); }
  private native void allocate();

  /** \brief Deprecated; use <code>Blob(const vector<int>& shape)</code>. */
  public FloatBlob(int num, int channels, int height,
        int width) { super((Pointer)null); allocate(num, channels, height, width); }
  private native void allocate(int num, int channels, int height,
        int width);
  public FloatBlob(@StdVector IntPointer shape) { super((Pointer)null); allocate(shape); }
  private native void allocate(@StdVector IntPointer shape);
  public FloatBlob(@StdVector IntBuffer shape) { super((Pointer)null); allocate(shape); }
  private native void allocate(@StdVector IntBuffer shape);
  public FloatBlob(@StdVector int[] shape) { super((Pointer)null); allocate(shape); }
  private native void allocate(@StdVector int[] shape);

  /** \brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>. */
  public native void Reshape(int num, int channels, int height,
        int width);
  /**
   * \brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  public native void Reshape(@StdVector IntPointer shape);
  public native void Reshape(@StdVector IntBuffer shape);
  public native void Reshape(@StdVector int[] shape);
  public native void Reshape(@Const @ByRef BlobShape shape);
  public native void ReshapeLike(@Const @ByRef FloatBlob other);
  public native @StdString BytePointer shape_string();
  public native @StdVector IntPointer shape();
  /**
   * \brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  public native int shape(int index);
  public native int num_axes();
  public native int count();

  /**
   * \brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  public native int count(int start_axis, int end_axis);
  /**
   * \brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  public native int count(int start_axis);

  /**
   * \brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  public native int CanonicalAxisIndex(int axis_index);

  /** \brief Deprecated legacy shape accessor num: use shape(0) instead. */
  public native int num();
  /** \brief Deprecated legacy shape accessor channels: use shape(1) instead. */
  public native int channels();
  /** \brief Deprecated legacy shape accessor height: use shape(2) instead. */
  public native int height();
  /** \brief Deprecated legacy shape accessor width: use shape(3) instead. */
  public native int width();
  public native int LegacyShape(int index);

  public native int offset(int n, int c/*=0*/, int h/*=0*/,
        int w/*=0*/);
  public native int offset(int n);

  public native int offset(@StdVector IntPointer indices);
  public native int offset(@StdVector IntBuffer indices);
  public native int offset(@StdVector int[] indices);
  /**
   * \brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  public native void CopyFrom(@Const @ByRef FloatBlob source, @Cast("bool") boolean copy_diff/*=false*/,
        @Cast("bool") boolean reshape/*=false*/);
  public native void CopyFrom(@Const @ByRef FloatBlob source);

  public native float data_at(int n, int c, int h,
        int w);

  public native float diff_at(int n, int c, int h,
        int w);

  public native float data_at(@StdVector IntPointer index);
  public native float data_at(@StdVector IntBuffer index);
  public native float data_at(@StdVector int[] index);

  public native float diff_at(@StdVector IntPointer index);
  public native float diff_at(@StdVector IntBuffer index);
  public native float diff_at(@StdVector int[] index);

  public native @SharedPtr SyncedMemory data();

  public native @SharedPtr SyncedMemory diff();

  public native @Const FloatPointer cpu_data();
  public native void set_cpu_data(FloatPointer data);
  public native void set_cpu_data(FloatBuffer data);
  public native void set_cpu_data(float[] data);
  public native @Const IntPointer gpu_shape();
  public native @Const FloatPointer gpu_data();
  public native @Const FloatPointer cpu_diff();
  public native @Const FloatPointer gpu_diff();
  public native FloatPointer mutable_cpu_data();
  public native FloatPointer mutable_gpu_data();
  public native FloatPointer mutable_cpu_diff();
  public native FloatPointer mutable_gpu_diff();
  public native void Update();
  public native void FromProto(@Const @ByRef BlobProto proto, @Cast("bool") boolean reshape/*=true*/);
  public native void FromProto(@Const @ByRef BlobProto proto);
  public native void ToProto(BlobProto proto, @Cast("bool") boolean write_diff/*=false*/);
  public native void ToProto(BlobProto proto);

  /** \brief Compute the sum of absolute values (L1 norm) of the data. */
  public native float asum_data();
  /** \brief Compute the sum of absolute values (L1 norm) of the diff. */
  public native float asum_diff();
  /** \brief Compute the sum of squares (L2 norm squared) of the data. */
  public native float sumsq_data();
  /** \brief Compute the sum of squares (L2 norm squared) of the diff. */
  public native float sumsq_diff();

  /** \brief Scale the blob data by a constant factor. */
  public native void scale_data(float scale_factor);
  /** \brief Scale the blob diff by a constant factor. */
  public native void scale_diff(float scale_factor);

  /**
   * \brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  public native void ShareData(@Const @ByRef FloatBlob other);
  /**
   * \brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  public native void ShareDiff(@Const @ByRef FloatBlob other);

  public native @Cast("bool") boolean ShapeEquals(@Const @ByRef BlobProto other);
}
@Name("caffe::Blob<double>") @NoOffset public static class DoubleBlob extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBlob(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DoubleBlob(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public DoubleBlob position(int position) {
        return (DoubleBlob)super.position(position);
    }

  public DoubleBlob() { super((Pointer)null); allocate(); }
  private native void allocate();

  /** \brief Deprecated; use <code>Blob(const vector<int>& shape)</code>. */
  public DoubleBlob(int num, int channels, int height,
        int width) { super((Pointer)null); allocate(num, channels, height, width); }
  private native void allocate(int num, int channels, int height,
        int width);
  public DoubleBlob(@StdVector IntPointer shape) { super((Pointer)null); allocate(shape); }
  private native void allocate(@StdVector IntPointer shape);
  public DoubleBlob(@StdVector IntBuffer shape) { super((Pointer)null); allocate(shape); }
  private native void allocate(@StdVector IntBuffer shape);
  public DoubleBlob(@StdVector int[] shape) { super((Pointer)null); allocate(shape); }
  private native void allocate(@StdVector int[] shape);

  /** \brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>. */
  public native void Reshape(int num, int channels, int height,
        int width);
  /**
   * \brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  public native void Reshape(@StdVector IntPointer shape);
  public native void Reshape(@StdVector IntBuffer shape);
  public native void Reshape(@StdVector int[] shape);
  public native void Reshape(@Const @ByRef BlobShape shape);
  public native void ReshapeLike(@Const @ByRef DoubleBlob other);
  public native @StdString BytePointer shape_string();
  public native @StdVector IntPointer shape();
  /**
   * \brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  public native int shape(int index);
  public native int num_axes();
  public native int count();

  /**
   * \brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  public native int count(int start_axis, int end_axis);
  /**
   * \brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  public native int count(int start_axis);

  /**
   * \brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  public native int CanonicalAxisIndex(int axis_index);

  /** \brief Deprecated legacy shape accessor num: use shape(0) instead. */
  public native int num();
  /** \brief Deprecated legacy shape accessor channels: use shape(1) instead. */
  public native int channels();
  /** \brief Deprecated legacy shape accessor height: use shape(2) instead. */
  public native int height();
  /** \brief Deprecated legacy shape accessor width: use shape(3) instead. */
  public native int width();
  public native int LegacyShape(int index);

  public native int offset(int n, int c/*=0*/, int h/*=0*/,
        int w/*=0*/);
  public native int offset(int n);

  public native int offset(@StdVector IntPointer indices);
  public native int offset(@StdVector IntBuffer indices);
  public native int offset(@StdVector int[] indices);
  /**
   * \brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  public native void CopyFrom(@Const @ByRef DoubleBlob source, @Cast("bool") boolean copy_diff/*=false*/,
        @Cast("bool") boolean reshape/*=false*/);
  public native void CopyFrom(@Const @ByRef DoubleBlob source);

  public native double data_at(int n, int c, int h,
        int w);

  public native double diff_at(int n, int c, int h,
        int w);

  public native double data_at(@StdVector IntPointer index);
  public native double data_at(@StdVector IntBuffer index);
  public native double data_at(@StdVector int[] index);

  public native double diff_at(@StdVector IntPointer index);
  public native double diff_at(@StdVector IntBuffer index);
  public native double diff_at(@StdVector int[] index);

  public native @SharedPtr SyncedMemory data();

  public native @SharedPtr SyncedMemory diff();

  public native @Const DoublePointer cpu_data();
  public native void set_cpu_data(DoublePointer data);
  public native void set_cpu_data(DoubleBuffer data);
  public native void set_cpu_data(double[] data);
  public native @Const IntPointer gpu_shape();
  public native @Const DoublePointer gpu_data();
  public native @Const DoublePointer cpu_diff();
  public native @Const DoublePointer gpu_diff();
  public native DoublePointer mutable_cpu_data();
  public native DoublePointer mutable_gpu_data();
  public native DoublePointer mutable_cpu_diff();
  public native DoublePointer mutable_gpu_diff();
  public native void Update();
  public native void FromProto(@Const @ByRef BlobProto proto, @Cast("bool") boolean reshape/*=true*/);
  public native void FromProto(@Const @ByRef BlobProto proto);
  public native void ToProto(BlobProto proto, @Cast("bool") boolean write_diff/*=false*/);
  public native void ToProto(BlobProto proto);

  /** \brief Compute the sum of absolute values (L1 norm) of the data. */
  public native double asum_data();
  /** \brief Compute the sum of absolute values (L1 norm) of the diff. */
  public native double asum_diff();
  /** \brief Compute the sum of squares (L2 norm squared) of the data. */
  public native double sumsq_data();
  /** \brief Compute the sum of squares (L2 norm squared) of the diff. */
  public native double sumsq_diff();

  /** \brief Scale the blob data by a constant factor. */
  public native void scale_data(double scale_factor);
  /** \brief Scale the blob diff by a constant factor. */
  public native void scale_diff(double scale_factor);

  /**
   * \brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  public native void ShareData(@Const @ByRef DoubleBlob other);
  /**
   * \brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  public native void ShareDiff(@Const @ByRef DoubleBlob other);

  public native @Cast("bool") boolean ShapeEquals(@Const @ByRef BlobProto other);
}  // class Blob

  // namespace caffe

// #endif  // CAFFE_BLOB_HPP_


// Parsed from caffe/data_transformer.hpp

// #ifndef CAFFE_DATA_TRANSFORMER_HPP
// #define CAFFE_DATA_TRANSFORMER_HPP

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
@Name("caffe::DataTransformer<float>") @NoOffset public static class FloatDataTransformer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatDataTransformer(Pointer p) { super(p); }

  public FloatDataTransformer(@Const @ByRef TransformationParameter param, @Cast("caffe::Phase") int phase) { super((Pointer)null); allocate(param, phase); }
  private native void allocate(@Const @ByRef TransformationParameter param, @Cast("caffe::Phase") int phase);

  /**
   * \brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  public native void InitRand();

  /**
   * \brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  public native void Transform(@Const @ByRef Datum datum, FloatBlob transformed_blob);

  /**
   * \brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  public native void Transform(@Const @ByRef DatumVector datum_vector,
                  FloatBlob transformed_blob);

// #ifdef USE_OPENCV
  /**
   * \brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  public native void Transform(@Const @ByRef MatVector mat_vector,
                  FloatBlob transformed_blob);

  /**
   * \brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  public native void Transform(@Const @ByRef Mat cv_img, FloatBlob transformed_blob);
// #endif  // USE_OPENCV

  /**
   * \brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  public native void Transform(FloatBlob input_blob, FloatBlob transformed_blob);

  /**
   * \brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  public native @StdVector IntPointer InferBlobShape(@Const @ByRef Datum datum);
  /**
   * \brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  public native @StdVector IntPointer InferBlobShape(@Const @ByRef DatumVector datum_vector);
  /**
   * \brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
// #ifdef USE_OPENCV
  public native @StdVector IntPointer InferBlobShape(@Const @ByRef MatVector mat_vector);
  /**
   * \brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  public native @StdVector IntPointer InferBlobShape(@Const @ByRef Mat cv_img);
}
@Name("caffe::DataTransformer<double>") @NoOffset public static class DoubleDataTransformer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleDataTransformer(Pointer p) { super(p); }

  public DoubleDataTransformer(@Const @ByRef TransformationParameter param, @Cast("caffe::Phase") int phase) { super((Pointer)null); allocate(param, phase); }
  private native void allocate(@Const @ByRef TransformationParameter param, @Cast("caffe::Phase") int phase);

  /**
   * \brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  public native void InitRand();

  /**
   * \brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  public native void Transform(@Const @ByRef Datum datum, DoubleBlob transformed_blob);

  /**
   * \brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  public native void Transform(@Const @ByRef DatumVector datum_vector,
                  DoubleBlob transformed_blob);

// #ifdef USE_OPENCV
  /**
   * \brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  public native void Transform(@Const @ByRef MatVector mat_vector,
                  DoubleBlob transformed_blob);

  /**
   * \brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  public native void Transform(@Const @ByRef Mat cv_img, DoubleBlob transformed_blob);
// #endif  // USE_OPENCV

  /**
   * \brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  public native void Transform(DoubleBlob input_blob, DoubleBlob transformed_blob);

  /**
   * \brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  public native @StdVector IntPointer InferBlobShape(@Const @ByRef Datum datum);
  /**
   * \brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  public native @StdVector IntPointer InferBlobShape(@Const @ByRef DatumVector datum_vector);
  /**
   * \brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
// #ifdef USE_OPENCV
  public native @StdVector IntPointer InferBlobShape(@Const @ByRef MatVector mat_vector);
  /**
   * \brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  public native @StdVector IntPointer InferBlobShape(@Const @ByRef Mat cv_img);
}

  // namespace caffe

// #endif  // CAFFE_DATA_TRANSFORMER_HPP_


// Parsed from caffe/filler.hpp

// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

// #ifndef CAFFE_FILLER_HPP
// #define CAFFE_FILLER_HPP

// #include <string>

// #include "caffe/blob.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/syncedmem.hpp"
// #include "caffe/util/math_functions.hpp"

/** \brief Fills a Blob with constant or randomly-generated data. */
@Name("caffe::Filler<float>") @NoOffset public static class FloatFiller extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatFiller(Pointer p) { super(p); }

  public native void Fill(FloatBlob blob);
}
@Name("caffe::Filler<double>") @NoOffset public static class DoubleFiller extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleFiller(Pointer p) { super(p); }

  public native void Fill(DoubleBlob blob);
}  // class Filler


/** \brief Fills a Blob with constant values \f$ x = 0 \f$. */
@Name("caffe::ConstantFiller<float>") public static class FloatConstantFiller extends FloatFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatConstantFiller(Pointer p) { super(p); }

  public FloatConstantFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(FloatBlob blob);
}
@Name("caffe::ConstantFiller<double>") public static class DoubleConstantFiller extends DoubleFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleConstantFiller(Pointer p) { super(p); }

  public DoubleConstantFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(DoubleBlob blob);
}

/** \brief Fills a Blob with uniformly distributed values \f$ x\sim U(a, b) \f$. */
@Name("caffe::UniformFiller<float>") public static class FloatUniformFiller extends FloatFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatUniformFiller(Pointer p) { super(p); }

  public FloatUniformFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(FloatBlob blob);
}
@Name("caffe::UniformFiller<double>") public static class DoubleUniformFiller extends DoubleFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleUniformFiller(Pointer p) { super(p); }

  public DoubleUniformFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(DoubleBlob blob);
}

/** \brief Fills a Blob with Gaussian-distributed values \f$ x = a \f$. */
@Name("caffe::GaussianFiller<float>") @NoOffset public static class FloatGaussianFiller extends FloatFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatGaussianFiller(Pointer p) { super(p); }

  public FloatGaussianFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(FloatBlob blob);
}
@Name("caffe::GaussianFiller<double>") @NoOffset public static class DoubleGaussianFiller extends DoubleFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleGaussianFiller(Pointer p) { super(p); }

  public DoubleGaussianFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(DoubleBlob blob);
}

/** \brief Fills a Blob with values \f$ x \in [0, 1] \f$
 *         such that \f$ \forall i \sum_j x_{ij} = 1 \f$.
 */
@Name("caffe::PositiveUnitballFiller<float>") public static class FloatPositiveUnitballFiller extends FloatFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatPositiveUnitballFiller(Pointer p) { super(p); }

  public FloatPositiveUnitballFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(FloatBlob blob);
}
@Name("caffe::PositiveUnitballFiller<double>") public static class DoublePositiveUnitballFiller extends DoubleFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoublePositiveUnitballFiller(Pointer p) { super(p); }

  public DoublePositiveUnitballFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(DoubleBlob blob);
}

/**
 * \brief Fills a Blob with values \f$ x \sim U(-a, +a) \f$ where \f$ a \f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
@Name("caffe::XavierFiller<float>") public static class FloatXavierFiller extends FloatFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatXavierFiller(Pointer p) { super(p); }

  public FloatXavierFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(FloatBlob blob);
}
@Name("caffe::XavierFiller<double>") public static class DoubleXavierFiller extends DoubleFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleXavierFiller(Pointer p) { super(p); }

  public DoubleXavierFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(DoubleBlob blob);
}

/**
 * \brief Fills a Blob with values \f$ x \sim N(0, \sigma^2) \f$ where
 *        \f$ \sigma^2 \f$ is set inversely proportional to number of incoming
 *        nodes, outgoing nodes, or their average.
 *
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
 * the variance_norm option. You should make sure the input blob has shape (num,
 * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
 * is currently not the case for inner product layers.
 */
@Name("caffe::MSRAFiller<float>") public static class FloatMSRAFiller extends FloatFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatMSRAFiller(Pointer p) { super(p); }

  public FloatMSRAFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(FloatBlob blob);
}
@Name("caffe::MSRAFiller<double>") public static class DoubleMSRAFiller extends DoubleFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleMSRAFiller(Pointer p) { super(p); }

  public DoubleMSRAFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(DoubleBlob blob);
}

/**
\brief Fills a Blob with coefficients for bilinear interpolation.
<p>
A common use case is with the DeconvolutionLayer acting as upsampling.
You can upsample a feature map with shape of (B, C, H, W) by any integer factor
using the following proto.
<pre>{@code
layer {
  name: "upsample", type: "Deconvolution"
  bottom: "{{bottom_name}}" top: "{{top_name}}"
  convolution_param {
    kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    num_output: {{C}} group: {{C}}
    pad: {{ceil((factor - 1) / 2.)}}
    weight_filler: { type: "bilinear" } bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
}</pre>
Please use this by replacing {@code {{}}} with your values. By specifying
{@code num_output: {{C}} group: {{C}}}, it behaves as
channel-wise convolution. The filter shape of this deconvolution layer will be
(C, 1, K, K) where K is {@code kernel_size}, and this filler will set a (K, K)
interpolation kernel for every channel of the filter identically. The resulting
shape of the top feature map will be (B, C, factor * H, factor * W).
Note that the learning rate and the
weight decay are set to 0 in order to keep coefficient values of bilinear
interpolation unchanged during training. If you apply this to an image, this
operation is equivalent to the following call in Python with Scikit.Image.
<pre>{@code {.py}
out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
}</pre>
 */
@Name("caffe::BilinearFiller<float>") public static class FloatBilinearFiller extends FloatFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBilinearFiller(Pointer p) { super(p); }

  public FloatBilinearFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(FloatBlob blob);
}
@Name("caffe::BilinearFiller<double>") public static class DoubleBilinearFiller extends DoubleFiller {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBilinearFiller(Pointer p) { super(p); }

  public DoubleBilinearFiller(@Const @ByRef FillerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef FillerParameter param);
  public native void Fill(DoubleBlob blob);
}

/**
 * \brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
@Namespace("caffe") public static native @Name("GetFiller<float>") FloatFiller GetFloatFiller(@Const @ByRef FillerParameter param);
@Namespace("caffe") public static native @Name("GetFiller<double>") DoubleFiller GetDoubleFiller(@Const @ByRef FillerParameter param);

  // namespace caffe

// #endif  // CAFFE_FILLER_HPP_


// Parsed from caffe/internal_thread.hpp

// #ifndef CAFFE_INTERNAL_THREAD_HPP_
// #define CAFFE_INTERNAL_THREAD_HPP_

// #include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
@Namespace("boost") @Opaque public static class thread extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public thread() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public thread(Pointer p) { super(p); }
} 

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
@Namespace("caffe") @NoOffset public static class InternalThread extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public InternalThread(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public InternalThread(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public InternalThread position(int position) {
        return (InternalThread)super.position(position);
    }

  public InternalThread() { super((Pointer)null); allocate(); }
  private native void allocate();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  public native void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  public native void StopInternalThread();

  public native @Cast("bool") boolean is_started();
}

  // namespace caffe

// #endif  // CAFFE_INTERNAL_THREAD_HPP_


// Parsed from caffe/util/hdf5.hpp

// #ifndef CAFFE_UTIL_HDF5_H_
// #define CAFFE_UTIL_HDF5_H_

// #include <string>

// #include "hdf5.h"
// #include "hdf5_hl.h"

// #include "caffe/blob.hpp"

@Namespace("caffe") public static native @Name("hdf5_load_nd_dataset_helper<float>") void hdf5_load_nd_dataset_helper_float(
    @Cast("hid_t") int file_id, @Cast("const char*") BytePointer dataset_name_, int min_dim, int max_dim,
    FloatBlob blob);
@Namespace("caffe") public static native @Name("hdf5_load_nd_dataset_helper<float>") void hdf5_load_nd_dataset_helper_float(
    @Cast("hid_t") int file_id, String dataset_name_, int min_dim, int max_dim,
    FloatBlob blob);

@Namespace("caffe") public static native @Name("hdf5_load_nd_dataset_helper<double>") void hdf5_load_nd_dataset_helper_double(
    @Cast("hid_t") int file_id, @Cast("const char*") BytePointer dataset_name_, int min_dim, int max_dim,
    DoubleBlob blob);
@Namespace("caffe") public static native @Name("hdf5_load_nd_dataset_helper<double>") void hdf5_load_nd_dataset_helper_double(
    @Cast("hid_t") int file_id, String dataset_name_, int min_dim, int max_dim,
    DoubleBlob blob);

@Namespace("caffe") public static native @Name("hdf5_load_nd_dataset<float>") void hdf5_load_nd_dataset_float(
    @Cast("hid_t") int file_id, @Cast("const char*") BytePointer dataset_name_, int min_dim, int max_dim,
    FloatBlob blob);
@Namespace("caffe") public static native @Name("hdf5_load_nd_dataset<float>") void hdf5_load_nd_dataset_float(
    @Cast("hid_t") int file_id, String dataset_name_, int min_dim, int max_dim,
    FloatBlob blob);

@Namespace("caffe") public static native @Name("hdf5_load_nd_dataset<double>") void hdf5_load_nd_dataset_double(
    @Cast("hid_t") int file_id, @Cast("const char*") BytePointer dataset_name_, int min_dim, int max_dim,
    DoubleBlob blob);
@Namespace("caffe") public static native @Name("hdf5_load_nd_dataset<double>") void hdf5_load_nd_dataset_double(
    @Cast("hid_t") int file_id, String dataset_name_, int min_dim, int max_dim,
    DoubleBlob blob);

@Namespace("caffe") public static native @Name("hdf5_save_nd_dataset<float>") void hdf5_save_nd_dataset_float(
    @Cast("const hid_t") int file_id, @StdString BytePointer dataset_name, @Const @ByRef FloatBlob blob,
    @Cast("bool") boolean write_diff/*=false*/);
@Namespace("caffe") public static native @Name("hdf5_save_nd_dataset<float>") void hdf5_save_nd_dataset_float(
    @Cast("const hid_t") int file_id, @StdString BytePointer dataset_name, @Const @ByRef FloatBlob blob);
@Namespace("caffe") public static native @Name("hdf5_save_nd_dataset<float>") void hdf5_save_nd_dataset_float(
    @Cast("const hid_t") int file_id, @StdString String dataset_name, @Const @ByRef FloatBlob blob,
    @Cast("bool") boolean write_diff/*=false*/);
@Namespace("caffe") public static native @Name("hdf5_save_nd_dataset<float>") void hdf5_save_nd_dataset_float(
    @Cast("const hid_t") int file_id, @StdString String dataset_name, @Const @ByRef FloatBlob blob);

@Namespace("caffe") public static native @Name("hdf5_save_nd_dataset<double>") void hdf5_save_nd_dataset_double(
    @Cast("const hid_t") int file_id, @StdString BytePointer dataset_name, @Const @ByRef DoubleBlob blob,
    @Cast("bool") boolean write_diff/*=false*/);
@Namespace("caffe") public static native @Name("hdf5_save_nd_dataset<double>") void hdf5_save_nd_dataset_double(
    @Cast("const hid_t") int file_id, @StdString BytePointer dataset_name, @Const @ByRef DoubleBlob blob);
@Namespace("caffe") public static native @Name("hdf5_save_nd_dataset<double>") void hdf5_save_nd_dataset_double(
    @Cast("const hid_t") int file_id, @StdString String dataset_name, @Const @ByRef DoubleBlob blob,
    @Cast("bool") boolean write_diff/*=false*/);
@Namespace("caffe") public static native @Name("hdf5_save_nd_dataset<double>") void hdf5_save_nd_dataset_double(
    @Cast("const hid_t") int file_id, @StdString String dataset_name, @Const @ByRef DoubleBlob blob);

@Namespace("caffe") public static native int hdf5_load_int(@Cast("hid_t") int loc_id, @StdString BytePointer dataset_name);
@Namespace("caffe") public static native int hdf5_load_int(@Cast("hid_t") int loc_id, @StdString String dataset_name);
@Namespace("caffe") public static native void hdf5_save_int(@Cast("hid_t") int loc_id, @StdString BytePointer dataset_name, int i);
@Namespace("caffe") public static native void hdf5_save_int(@Cast("hid_t") int loc_id, @StdString String dataset_name, int i);
@Namespace("caffe") public static native @StdString BytePointer hdf5_load_string(@Cast("hid_t") int loc_id, @StdString BytePointer dataset_name);
@Namespace("caffe") public static native @StdString String hdf5_load_string(@Cast("hid_t") int loc_id, @StdString String dataset_name);
@Namespace("caffe") public static native void hdf5_save_string(@Cast("hid_t") int loc_id, @StdString BytePointer dataset_name,
                      @StdString BytePointer s);
@Namespace("caffe") public static native void hdf5_save_string(@Cast("hid_t") int loc_id, @StdString String dataset_name,
                      @StdString String s);

@Namespace("caffe") public static native int hdf5_get_num_links(@Cast("hid_t") int loc_id);
@Namespace("caffe") public static native @StdString BytePointer hdf5_get_name_by_idx(@Cast("hid_t") int loc_id, int idx);

  // namespace caffe

// #endif   // CAFFE_UTIL_HDF5_H_


// Parsed from caffe/layers/base_data_layer.hpp

// #ifndef CAFFE_DATA_LAYERS_HPP_
// #define CAFFE_DATA_LAYERS_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/data_transformer.hpp"
// #include "caffe/internal_thread.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/blocking_queue.hpp"

/**
 * \brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
@Name("caffe::BaseDataLayer<float>") @NoOffset public static class FloatBaseDataLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBaseDataLayer(Pointer p) { super(p); }

  public FloatBaseDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  // Data layers should be shared by multiple solvers in parallel
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  @Virtual public native void DataLayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  // Data layers have no bottoms, so reshaping is trivial.
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual public native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::BaseDataLayer<double>") @NoOffset public static class DoubleBaseDataLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBaseDataLayer(Pointer p) { super(p); }

  public DoubleBaseDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  // Data layers should be shared by multiple solvers in parallel
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  @Virtual public native void DataLayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  // Data layers have no bottoms, so reshaping is trivial.
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual public native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

@Name("caffe::Batch<float>") public static class FloatBatch extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public FloatBatch() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FloatBatch(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBatch(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public FloatBatch position(int position) {
        return (FloatBatch)super.position(position);
    }

  @MemberGetter public native @ByRef FloatBlob data_();
  @MemberGetter public native @ByRef FloatBlob label_();
}

@Name("caffe::Batch<double>") public static class DoubleBatch extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public DoubleBatch() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DoubleBatch(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBatch(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public DoubleBatch position(int position) {
        return (DoubleBatch)super.position(position);
    }

  @MemberGetter public native @ByRef DoubleBlob data_();
  @MemberGetter public native @ByRef DoubleBlob label_();
}

@Name("caffe::BasePrefetchingDataLayer<float>") @NoOffset public static class FloatBasePrefetchingDataLayer extends FloatBaseDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBasePrefetchingDataLayer(Pointer p) { super(p); }
    public InternalThread asInternalThread() { return asInternalThread(this); }
    @Namespace public static native @Name("static_cast<caffe::InternalThread*>") InternalThread asInternalThread(FloatBasePrefetchingDataLayer pointer);

  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  // Prefetches batches (asynchronously if to GPU memory)
  @MemberGetter public static native int PREFETCH_COUNT();
  public static final int PREFETCH_COUNT = PREFETCH_COUNT();
  @Virtual(true) protected native void load_batch(FloatBatch batch);
}

@Name("caffe::BasePrefetchingDataLayer<double>") @NoOffset public static class DoubleBasePrefetchingDataLayer extends DoubleBaseDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBasePrefetchingDataLayer(Pointer p) { super(p); }
    public InternalThread asInternalThread() { return asInternalThread(this); }
    @Namespace public static native @Name("static_cast<caffe::InternalThread*>") InternalThread asInternalThread(DoubleBasePrefetchingDataLayer pointer);

  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  // Prefetches batches (asynchronously if to GPU memory)
  @MemberGetter public static native int PREFETCH_COUNT();
  public static final int PREFETCH_COUNT = PREFETCH_COUNT();
  @Virtual(true) protected native void load_batch(DoubleBatch batch);
}

  // namespace caffe

// #endif  // CAFFE_DATA_LAYERS_HPP_


// Parsed from caffe/layers/data_layer.hpp

// #ifndef CAFFE_DATA_LAYER_HPP_
// #define CAFFE_DATA_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/data_reader.hpp"
// #include "caffe/data_transformer.hpp"
// #include "caffe/internal_thread.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/layers/base_data_layer.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/db.hpp"

@Name("caffe::DataLayer<float>") @NoOffset public static class FloatDataLayer extends FloatBasePrefetchingDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatDataLayer(Pointer p) { super(p); }

  public FloatDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void DataLayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  // DataLayer uses DataReader instead for sharing for parallelism
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual public native int MaxTopBlobs();
  @Virtual protected native void load_batch(FloatBatch batch);
}

@Name("caffe::DataLayer<double>") @NoOffset public static class DoubleDataLayer extends DoubleBasePrefetchingDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleDataLayer(Pointer p) { super(p); }

  public DoubleDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void DataLayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  // DataLayer uses DataReader instead for sharing for parallelism
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual public native int MaxTopBlobs();
  @Virtual protected native void load_batch(DoubleBatch batch);
}

  // namespace caffe

// #endif  // CAFFE_DATA_LAYER_HPP_


// Parsed from caffe/layers/dummy_data_layer.hpp

// #ifndef CAFFE_DUMMY_DATA_LAYER_HPP_
// #define CAFFE_DUMMY_DATA_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/filler.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Provides data to the Net generated by a Filler.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
@Name("caffe::DummyDataLayer<float>") @NoOffset public static class FloatDummyDataLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatDummyDataLayer(Pointer p) { super(p); }

  public FloatDummyDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  // Data layers should be shared by multiple solvers in parallel
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  // Data layers have no bottoms, so reshaping is trivial.
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::DummyDataLayer<double>") @NoOffset public static class DoubleDummyDataLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleDummyDataLayer(Pointer p) { super(p); }

  public DoubleDummyDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  // Data layers should be shared by multiple solvers in parallel
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  // Data layers have no bottoms, so reshaping is trivial.
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_DUMMY_DATA_LAYER_HPP_


// Parsed from caffe/layers/hdf5_data_layer.hpp

// #ifndef CAFFE_HDF5_DATA_LAYER_HPP_
// #define CAFFE_HDF5_DATA_LAYER_HPP_

// #include "hdf5.h"

// #include <string>
// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/base_data_layer.hpp"

/**
 * \brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
@Name("caffe::HDF5DataLayer<float>") @NoOffset public static class FloatHDF5DataLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatHDF5DataLayer(Pointer p) { super(p); }

  public FloatHDF5DataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  // Data layers should be shared by multiple solvers in parallel
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  // Data layers have no bottoms, so reshaping is trivial.
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void LoadHDF5FileData(@Cast("const char*") BytePointer filename);
}
@Name("caffe::HDF5DataLayer<double>") @NoOffset public static class DoubleHDF5DataLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleHDF5DataLayer(Pointer p) { super(p); }

  public DoubleHDF5DataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  // Data layers should be shared by multiple solvers in parallel
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  // Data layers have no bottoms, so reshaping is trivial.
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void LoadHDF5FileData(@Cast("const char*") BytePointer filename);
}

  // namespace caffe

// #endif  // CAFFE_HDF5_DATA_LAYER_HPP_


// Parsed from caffe/layers/hdf5_output_layer.hpp

// #ifndef CAFFE_HDF5_OUTPUT_LAYER_HPP_
// #define CAFFE_HDF5_OUTPUT_LAYER_HPP_

// #include "hdf5.h"

// #include <string>
// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

public static final String HDF5_DATA_DATASET_NAME = "data";
public static final String HDF5_DATA_LABEL_NAME = "label";

/**
 * \brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
@Name("caffe::HDF5OutputLayer<float>") @NoOffset public static class FloatHDF5OutputLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatHDF5OutputLayer(Pointer p) { super(p); }

  public FloatHDF5OutputLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  // Data layers should be shared by multiple solvers in parallel
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  // Data layers have no bottoms, so reshaping is trivial.
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  // TODO: no limit on the number of blobs
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();

  public native @StdString BytePointer file_name();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void SaveBlobs();
}
@Name("caffe::HDF5OutputLayer<double>") @NoOffset public static class DoubleHDF5OutputLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleHDF5OutputLayer(Pointer p) { super(p); }

  public DoubleHDF5OutputLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  // Data layers should be shared by multiple solvers in parallel
  @Virtual public native @Cast("bool") boolean ShareInParallel();
  // Data layers have no bottoms, so reshaping is trivial.
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  // TODO: no limit on the number of blobs
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();

  public native @StdString BytePointer file_name();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void SaveBlobs();
}

  // namespace caffe

// #endif  // CAFFE_HDF5_OUTPUT_LAYER_HPP_


// Parsed from caffe/layers/image_data_layer.hpp

// #ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
// #define CAFFE_IMAGE_DATA_LAYER_HPP_

// #include <string>
// #include <utility>
// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/data_transformer.hpp"
// #include "caffe/internal_thread.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/layers/base_data_layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
@Name("caffe::ImageDataLayer<float>") @NoOffset public static class FloatImageDataLayer extends FloatBasePrefetchingDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatImageDataLayer(Pointer p) { super(p); }

  public FloatImageDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void DataLayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void ShuffleImages();
  @Virtual protected native void load_batch(FloatBatch batch);
}
@Name("caffe::ImageDataLayer<double>") @NoOffset public static class DoubleImageDataLayer extends DoubleBasePrefetchingDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleImageDataLayer(Pointer p) { super(p); }

  public DoubleImageDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void DataLayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void ShuffleImages();
  @Virtual protected native void load_batch(DoubleBatch batch);
}


  // namespace caffe

// #endif  // CAFFE_IMAGE_DATA_LAYER_HPP_


// Parsed from caffe/layers/memory_data_layer.hpp

// #ifndef CAFFE_MEMORY_DATA_LAYER_HPP_
// #define CAFFE_MEMORY_DATA_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/base_data_layer.hpp"

/**
 * \brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
@Name("caffe::MemoryDataLayer<float>") @NoOffset public static class FloatMemoryDataLayer extends FloatBaseDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatMemoryDataLayer(Pointer p) { super(p); }

  public FloatMemoryDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void DataLayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();

  @Virtual public native void AddDatumVector(@Const @ByRef DatumVector datum_vector);
// #ifdef USE_OPENCV
  @Virtual public native void AddMatVector(@Const @ByRef MatVector mat_vector,
        @Cast({"int*", "std::vector<int>&"}) @StdVector IntPointer labels);
// #endif  // USE_OPENCV

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  public native void Reset(FloatPointer data, FloatPointer label, int n);
  public native void Reset(FloatBuffer data, FloatBuffer label, int n);
  public native void Reset(float[] data, float[] label, int n);
  public native void set_batch_size(int new_size);

  public native int batch_size();
  public native int channels();
  public native int height();
  public native int width();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
}
@Name("caffe::MemoryDataLayer<double>") @NoOffset public static class DoubleMemoryDataLayer extends DoubleBaseDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleMemoryDataLayer(Pointer p) { super(p); }

  public DoubleMemoryDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void DataLayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();

  @Virtual public native void AddDatumVector(@Const @ByRef DatumVector datum_vector);
// #ifdef USE_OPENCV
  @Virtual public native void AddMatVector(@Const @ByRef MatVector mat_vector,
        @Cast({"int*", "std::vector<int>&"}) @StdVector IntPointer labels);
// #endif  // USE_OPENCV

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  public native void Reset(DoublePointer data, DoublePointer label, int n);
  public native void Reset(DoubleBuffer data, DoubleBuffer label, int n);
  public native void Reset(double[] data, double[] label, int n);
  public native void set_batch_size(int new_size);

  public native int batch_size();
  public native int channels();
  public native int height();
  public native int width();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
}

  // namespace caffe

// #endif  // CAFFE_MEMORY_DATA_LAYER_HPP_


// Parsed from caffe/layers/window_data_layer.hpp

// #ifndef CAFFE_WINDOW_DATA_LAYER_HPP_
// #define CAFFE_WINDOW_DATA_LAYER_HPP_

// #include <string>
// #include <utility>
// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/data_transformer.hpp"
// #include "caffe/internal_thread.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/layers/base_data_layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
@Name("caffe::WindowDataLayer<float>") @NoOffset public static class FloatWindowDataLayer extends FloatBasePrefetchingDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatWindowDataLayer(Pointer p) { super(p); }

  public FloatWindowDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void DataLayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native @Cast("unsigned int") int PrefetchRand();
  @Virtual protected native void load_batch(FloatBatch batch);
}
@Name("caffe::WindowDataLayer<double>") @NoOffset public static class DoubleWindowDataLayer extends DoubleBasePrefetchingDataLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleWindowDataLayer(Pointer p) { super(p); }

  public DoubleWindowDataLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void DataLayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native @Cast("unsigned int") int PrefetchRand();
  @Virtual protected native void load_batch(DoubleBatch batch);
}

  // namespace caffe

// #endif  // CAFFE_WINDOW_DATA_LAYER_HPP_


// Parsed from caffe/layer_factory.hpp

/**
 * \brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

// #ifndef CAFFE_LAYER_FACTORY_H_
// #define CAFFE_LAYER_FACTORY_H_

// #include <map>
// #include <string>
// #include <vector>

// #include "caffe/common.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

@Name("caffe::LayerRegistry<float>") public static class FloatLayerRegistry extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatLayerRegistry(Pointer p) { super(p); }

  public static class Creator extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Creator(Pointer p) { super(p); }
      protected Creator() { allocate(); }
      private native void allocate();
      public native @Cast({"", "boost::shared_ptr<caffe::Layer<float> >"}) @SharedPtr @ByVal FloatLayer call(@Const @ByRef LayerParameter arg0);
  }

  public static native @Cast("caffe::LayerRegistry<float>::CreatorRegistry*") @ByRef FloatRegistry Registry();

  // Adds a creator.
  public static native void AddCreator(@StdString BytePointer type, Creator creator);
  public static native void AddCreator(@StdString String type, Creator creator);

  // Get a layer using a LayerParameter.
  public static native @Cast({"", "boost::shared_ptr<caffe::Layer<float> >"}) @SharedPtr @ByVal FloatLayer CreateLayer(@Const @ByRef LayerParameter param);

  public static native @ByVal StringVector LayerTypeList();
}

@Name("caffe::LayerRegistry<double>") public static class DoubleLayerRegistry extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleLayerRegistry(Pointer p) { super(p); }

  public static class Creator extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Creator(Pointer p) { super(p); }
      protected Creator() { allocate(); }
      private native void allocate();
      public native @Cast({"", "boost::shared_ptr<caffe::Layer<double> >"}) @SharedPtr @ByVal DoubleLayer call(@Const @ByRef LayerParameter arg0);
  }

  public static native @Cast("caffe::LayerRegistry<double>::CreatorRegistry*") @ByRef DoubleRegistry Registry();

  // Adds a creator.
  public static native void AddCreator(@StdString BytePointer type, Creator creator);
  public static native void AddCreator(@StdString String type, Creator creator);

  // Get a layer using a LayerParameter.
  public static native @Cast({"", "boost::shared_ptr<caffe::Layer<double> >"}) @SharedPtr @ByVal DoubleLayer CreateLayer(@Const @ByRef LayerParameter param);

  public static native @ByVal StringVector LayerTypeList();
}


@Name("caffe::LayerRegisterer<float>") public static class FloatLayerRegisterer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatLayerRegisterer(Pointer p) { super(p); }

  public static class Creator_LayerParameter extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Creator_LayerParameter(Pointer p) { super(p); }
      protected Creator_LayerParameter() { allocate(); }
      private native void allocate();
      public native @Cast({"", "boost::shared_ptr<caffe::Layer<float> >"}) @SharedPtr @ByVal FloatLayer call(@Const @ByRef LayerParameter arg0);
  }
  public FloatLayerRegisterer(@StdString BytePointer type,
                    Creator_LayerParameter creator) { super((Pointer)null); allocate(type, creator); }
  private native void allocate(@StdString BytePointer type,
                    Creator_LayerParameter creator);
  public FloatLayerRegisterer(@StdString String type,
                    Creator_LayerParameter creator) { super((Pointer)null); allocate(type, creator); }
  private native void allocate(@StdString String type,
                    Creator_LayerParameter creator);
}


@Name("caffe::LayerRegisterer<double>") public static class DoubleLayerRegisterer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleLayerRegisterer(Pointer p) { super(p); }

  public static class Creator_LayerParameter extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Creator_LayerParameter(Pointer p) { super(p); }
      protected Creator_LayerParameter() { allocate(); }
      private native void allocate();
      public native @Cast({"", "boost::shared_ptr<caffe::Layer<double> >"}) @SharedPtr @ByVal DoubleLayer call(@Const @ByRef LayerParameter arg0);
  }
  public DoubleLayerRegisterer(@StdString BytePointer type,
                    Creator_LayerParameter creator) { super((Pointer)null); allocate(type, creator); }
  private native void allocate(@StdString BytePointer type,
                    Creator_LayerParameter creator);
  public DoubleLayerRegisterer(@StdString String type,
                    Creator_LayerParameter creator) { super((Pointer)null); allocate(type, creator); }
  private native void allocate(@StdString String type,
                    Creator_LayerParameter creator);
}


// #define REGISTER_LAYER_CREATOR(type, creator)
//   static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);
//   static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    

// #define REGISTER_LAYER_CLASS(type)
//   template <typename Dtype>
//   shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param)
//   {
//     return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));
//   }
//   REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

  // namespace caffe

// #endif  // CAFFE_LAYER_FACTORY_H_


// Parsed from caffe/layer.hpp

// #ifndef CAFFE_LAYER_H_
// #define CAFFE_LAYER_H_

// #include <algorithm>
// #include <string>
// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/layer_factory.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
@Namespace("boost") @Opaque public static class mutex extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public mutex() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public mutex(Pointer p) { super(p); }
} 

/**
 * \brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
@Name("caffe::Layer<float>") @NoOffset public static class FloatLayer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatLayer(Pointer p) { super(p); }

  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */

  /**
   * \brief Implements common layer setup functionality.
   *
   * @param bottom the preshaped input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   */
  public native void SetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  /**
   * \brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  /**
   * \brief Whether a layer should be shared by multiple nets during data
   *        parallelism. By default, all layers except for data layers should
   *        not be shared. data layers should be shared to ensure each worker
   *        solver access data sequentially during data parallelism.
   */
  @Virtual public native @Cast("bool") boolean ShareInParallel();

  /** \brief Return whether this layer is actually shared by other nets.
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then this function is expected return true.
   */
  public native @Cast("bool") boolean IsShared();

  /** \brief Set whether this layer is actually shared by other nets
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then is_shared should be set true.
   */
  public native void SetShared(@Cast("bool") boolean is_shared);

  /**
   * \brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  @Virtual(true) public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  /**
   * \brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * @return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  public native float Forward(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  /**
   * \brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  public native void Backward(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down,
        @Const @ByRef FloatBlobVector bottom);

  /**
   * \brief Returns the vector of learnable parameter blobs.
   */
  public native @ByRef FloatBlobSharedVector blobs();

  /**
   * \brief Returns the layer parameter.
   */
  public native @Const @ByRef LayerParameter layer_param();

  /**
   * \brief Writes the layer parameter to a protocol buffer
   */
  @Virtual public native void ToProto(LayerParameter param, @Cast("bool") boolean write_diff/*=false*/);

  /**
   * \brief Returns the scalar loss associated with a top blob at a given index.
   */
  public native float loss(int top_index);

  /**
   * \brief Sets the loss associated with a top blob at a given index.
   */
  public native void set_loss(int top_index, float value);

  /**
   * \brief Returns the layer type.
   */
  @Virtual public native @Cast("const char*") BytePointer type();

  /**
   * \brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  @Virtual public native int ExactNumBottomBlobs();
  /**
   * \brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  @Virtual public native int MinBottomBlobs();
  /**
   * \brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  @Virtual public native int MaxBottomBlobs();
  /**
   * \brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  @Virtual public native int ExactNumTopBlobs();
  /**
   * \brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  @Virtual public native int MinTopBlobs();
  /**
   * \brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  @Virtual public native int MaxTopBlobs();
  /**
   * \brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  @Virtual public native @Cast("bool") boolean EqualNumBottomTopBlobs();

  /**
   * \brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  @Virtual public native @Cast("bool") boolean AutoTopBlobs();

  /**
   * \brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  @Virtual public native @Cast("bool") boolean AllowForceBackward(int bottom_index);

  /**
   * \brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  public native @Cast("bool") boolean param_propagate_down(int param_id);
  /**
   * \brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  public native void set_param_propagate_down(int param_id, @Cast("const bool") boolean value);
  @Virtual(true) protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual(true) protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down,
        @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down,
        @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void CheckBlobCounts(@Const @ByRef FloatBlobVector bottom,
                                 @Const @ByRef FloatBlobVector top);
}
@Name("caffe::Layer<double>") @NoOffset public static class DoubleLayer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleLayer(Pointer p) { super(p); }

  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */

  /**
   * \brief Implements common layer setup functionality.
   *
   * @param bottom the preshaped input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   */
  public native void SetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  /**
   * \brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  /**
   * \brief Whether a layer should be shared by multiple nets during data
   *        parallelism. By default, all layers except for data layers should
   *        not be shared. data layers should be shared to ensure each worker
   *        solver access data sequentially during data parallelism.
   */
  @Virtual public native @Cast("bool") boolean ShareInParallel();

  /** \brief Return whether this layer is actually shared by other nets.
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then this function is expected return true.
   */
  public native @Cast("bool") boolean IsShared();

  /** \brief Set whether this layer is actually shared by other nets
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then is_shared should be set true.
   */
  public native void SetShared(@Cast("bool") boolean is_shared);

  /**
   * \brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  @Virtual(true) public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  /**
   * \brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * @return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  public native double Forward(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  /**
   * \brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  public native void Backward(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down,
        @Const @ByRef DoubleBlobVector bottom);

  /**
   * \brief Returns the vector of learnable parameter blobs.
   */
  public native @ByRef DoubleBlobSharedVector blobs();

  /**
   * \brief Returns the layer parameter.
   */
  public native @Const @ByRef LayerParameter layer_param();

  /**
   * \brief Writes the layer parameter to a protocol buffer
   */
  @Virtual public native void ToProto(LayerParameter param, @Cast("bool") boolean write_diff/*=false*/);

  /**
   * \brief Returns the scalar loss associated with a top blob at a given index.
   */
  public native double loss(int top_index);

  /**
   * \brief Sets the loss associated with a top blob at a given index.
   */
  public native void set_loss(int top_index, double value);

  /**
   * \brief Returns the layer type.
   */
  @Virtual public native @Cast("const char*") BytePointer type();

  /**
   * \brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  @Virtual public native int ExactNumBottomBlobs();
  /**
   * \brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  @Virtual public native int MinBottomBlobs();
  /**
   * \brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  @Virtual public native int MaxBottomBlobs();
  /**
   * \brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  @Virtual public native int ExactNumTopBlobs();
  /**
   * \brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  @Virtual public native int MinTopBlobs();
  /**
   * \brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  @Virtual public native int MaxTopBlobs();
  /**
   * \brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  @Virtual public native @Cast("bool") boolean EqualNumBottomTopBlobs();

  /**
   * \brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  @Virtual public native @Cast("bool") boolean AutoTopBlobs();

  /**
   * \brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  @Virtual public native @Cast("bool") boolean AllowForceBackward(int bottom_index);

  /**
   * \brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  public native @Cast("bool") boolean param_propagate_down(int param_id);
  /**
   * \brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  public native void set_param_propagate_down(int param_id, @Cast("const bool") boolean value);
  @Virtual(true) protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual(true) protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down,
        @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down,
        @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void CheckBlobCounts(@Const @ByRef DoubleBlobVector bottom,
                                 @Const @ByRef DoubleBlobVector top);
}  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.




// Serialize LayerParameter to protocol buffer


  // namespace caffe

// #endif  // CAFFE_LAYER_H_


// Parsed from caffe/layers/accuracy_layer.hpp

// #ifndef CAFFE_ACCURACY_LAYER_HPP_
// #define CAFFE_ACCURACY_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"

/**
 * \brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
@Name("caffe::AccuracyLayer<float>") @NoOffset public static class FloatAccuracyLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatAccuracyLayer(Pointer p) { super(p); }

  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with AccuracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank \f$ k \f$ at which a prediction is considered
   *     correct.  For example, if \f$ k = 5 \f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  public FloatAccuracyLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();

  // If there are two top blobs, then the second blob will contain
  // accuracies per class.
  @Virtual public native int MinTopBlobs();
  @Virtual public native int MaxTopBlos();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::AccuracyLayer<double>") @NoOffset public static class DoubleAccuracyLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleAccuracyLayer(Pointer p) { super(p); }

  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with AccuracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank \f$ k \f$ at which a prediction is considered
   *     correct.  For example, if \f$ k = 5 \f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  public DoubleAccuracyLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();

  // If there are two top blobs, then the second blob will contain
  // accuracies per class.
  @Virtual public native int MinTopBlobs();
  @Virtual public native int MaxTopBlos();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_ACCURACY_LAYER_HPP_


// Parsed from caffe/layers/loss_layer.hpp

// #ifndef CAFFE_LOSS_LAYER_HPP_
// #define CAFFE_LOSS_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

@Namespace("caffe") @MemberGetter public static native float kLOG_THRESHOLD();

/**
 * \brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */
@Name("caffe::LossLayer<float>") public static class FloatLossLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatLossLayer(Pointer p) { super(p); }

  public FloatLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(
        @Const @ByRef FloatBlobVector bottom, @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(
        @Const @ByRef FloatBlobVector bottom, @Const @ByRef FloatBlobVector top);

  @Virtual public native int ExactNumBottomBlobs();

  /**
   * \brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
  @Virtual public native @Cast("bool") boolean AutoTopBlobs();
  @Virtual public native int ExactNumTopBlobs();
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  @Virtual public native @Cast("bool") boolean AllowForceBackward(int bottom_index);
}
@Name("caffe::LossLayer<double>") public static class DoubleLossLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleLossLayer(Pointer p) { super(p); }

  public DoubleLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(
        @Const @ByRef DoubleBlobVector bottom, @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(
        @Const @ByRef DoubleBlobVector bottom, @Const @ByRef DoubleBlobVector top);

  @Virtual public native int ExactNumBottomBlobs();

  /**
   * \brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
  @Virtual public native @Cast("bool") boolean AutoTopBlobs();
  @Virtual public native int ExactNumTopBlobs();
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  @Virtual public native @Cast("bool") boolean AllowForceBackward(int bottom_index);
}

  // namespace caffe

// #endif  // CAFFE_LOSS_LAYER_HPP_


// Parsed from caffe/layers/contrastive_loss_layer.hpp

// #ifndef CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_
// #define CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"

/**
 * \brief Computes the contrastive loss \f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left(y\right) d^2 +
 *              \left(1-y\right) \max \left(margin-d, 0\right)^2
 *          \f$ where \f$
 *          d = \left| \left| a_n - b_n \right| \right|_2 \f$. This can be
 *          used to train siamese networks.
 *
 * @param bottom input Blob vector (length 3)
 *   -# \f$ (N \times C \times 1 \times 1) \f$
 *      the features \f$ a \in [-\infty, +\infty]\f$
 *   -# \f$ (N \times C \times 1 \times 1) \f$
 *      the features \f$ b \in [-\infty, +\infty]\f$
 *   -# \f$ (N \times 1 \times 1 \times 1) \f$
 *      the binary similarity \f$ s \in [0, 1]\f$
 * @param top output Blob vector (length 1)
 *   -# \f$ (1 \times 1 \times 1 \times 1) \f$
 *      the computed contrastive loss: \f$ E =
 *          \frac{1}{2N} \sum\limits_{n=1}^N \left(y\right) d^2 +
 *          \left(1-y\right) \max \left(margin-d, 0\right)^2
 *          \f$ where \f$
 *          d = \left| \left| a_n - b_n \right| \right|_2 \f$.
 * This can be used to train siamese networks.
 */
@Name("caffe::ContrastiveLossLayer<float>") @NoOffset public static class FloatContrastiveLossLayer extends FloatLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatContrastiveLossLayer(Pointer p) { super(p); }

  public FloatContrastiveLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native @Cast("const char*") BytePointer type();
  /**
   * Unlike most loss layers, in the ContrastiveLossLayer we can backpropagate
   * to the first two inputs.
   */
  @Virtual public native @Cast("bool") boolean AllowForceBackward(int bottom_index);
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::ContrastiveLossLayer<double>") @NoOffset public static class DoubleContrastiveLossLayer extends DoubleLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleContrastiveLossLayer(Pointer p) { super(p); }

  public DoubleContrastiveLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native @Cast("const char*") BytePointer type();
  /**
   * Unlike most loss layers, in the ContrastiveLossLayer we can backpropagate
   * to the first two inputs.
   */
  @Virtual public native @Cast("bool") boolean AllowForceBackward(int bottom_index);
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_


// Parsed from caffe/layers/euclidean_loss_layer.hpp

// #ifndef CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
// #define CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"

/**
 * \brief Computes the Euclidean (L2) loss \f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 \f$ for real-valued regression tasks.
 *
 * @param bottom input Blob vector (length 2)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the predictions \f$ \hat{y} \in [-\infty, +\infty]\f$
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the targets \f$ y \in [-\infty, +\infty]\f$
 * @param top output Blob vector (length 1)
 *   -# \f$ (1 \times 1 \times 1 \times 1) \f$
 *      the computed Euclidean loss: \f$ E =
 *          \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 \f$
 *
 * This can be used for least-squares regression tasks.  An InnerProductLayer
 * input to a EuclideanLossLayer exactly formulates a linear least squares
 * regression problem. With non-zero weight decay the problem becomes one of
 * ridge regression -- see src/caffe/test/test_sgd_solver.cpp for a concrete
 * example wherein we check that the gradients computed for a Net with exactly
 * this structure match hand-computed gradient formulas for ridge regression.
 *
 * (Note: Caffe, and SGD in general, is certainly \b not the best way to solve
 * linear least squares problems! We use it only as an instructive example.)
 */
@Name("caffe::EuclideanLossLayer<float>") @NoOffset public static class FloatEuclideanLossLayer extends FloatLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatEuclideanLossLayer(Pointer p) { super(p); }

  public FloatEuclideanLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  /**
   * Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  @Virtual public native @Cast("bool") boolean AllowForceBackward(int bottom_index);
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::EuclideanLossLayer<double>") @NoOffset public static class DoubleEuclideanLossLayer extends DoubleLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleEuclideanLossLayer(Pointer p) { super(p); }

  public DoubleEuclideanLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  /**
   * Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  @Virtual public native @Cast("bool") boolean AllowForceBackward(int bottom_index);
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_


// Parsed from caffe/layers/hinge_loss_layer.hpp

// #ifndef CAFFE_HINGE_LOSS_LAYER_HPP_
// #define CAFFE_HINGE_LOSS_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"

/**
 * \brief Computes the hinge loss for a one-of-many classification task.
 *
 * @param bottom input Blob vector (length 2)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the predictions \f$ t \f$, a Blob with values in
 *      \f$ [-\infty, +\infty] \f$ indicating the predicted score for each of
 *      the \f$ K = CHW \f$ classes. In an SVM, \f$ t \f$ is the result of
 *      taking the inner product \f$ X^T W \f$ of the D-dimensional features
 *      \f$ X \in \mathcal{R}^{D \times N} \f$ and the learned hyperplane
 *      parameters \f$ W \in \mathcal{R}^{D \times K} \f$, so a Net with just
 *      an InnerProductLayer (with num_output = D) providing predictions to a
 *      HingeLossLayer and no other learnable parameters or losses is
 *      equivalent to an SVM.
 *   -# \f$ (N \times 1 \times 1 \times 1) \f$
 *      the labels \f$ l \f$, an integer-valued Blob with values
 *      \f$ l_n \in [0, 1, 2, ..., K - 1] \f$
 *      indicating the correct class label among the \f$ K \f$ classes
 * @param top output Blob vector (length 1)
 *   -# \f$ (1 \times 1 \times 1 \times 1) \f$
 *      the computed hinge loss: \f$ E =
 *        \frac{1}{N} \sum\limits_{n=1}^N \sum\limits_{k=1}^K
 *        [\max(0, 1 - \delta\{l_n = k\} t_{nk})] ^ p
 *      \f$, for the \f$ L^p \f$ norm
 *      (defaults to \f$ p = 1 \f$, the L1 norm; L2 norm, as in L2-SVM,
 *      is also available), and \f$
 *      \delta\{\mathrm{condition}\} = \left\{
 *         \begin{array}{lr}
 *            1 & \mbox{if condition} \\
 *           -1 & \mbox{otherwise}
 *         \end{array} \right.
 *      \f$
 *
 * In an SVM, \f$ t \in \mathcal{R}^{N \times K} \f$ is the result of taking
 * the inner product \f$ X^T W \f$ of the features
 * \f$ X \in \mathcal{R}^{D \times N} \f$
 * and the learned hyperplane parameters
 * \f$ W \in \mathcal{R}^{D \times K} \f$. So, a Net with just an
 * InnerProductLayer (with num_output = \f$k\f$) providing predictions to a
 * HingeLossLayer is equivalent to an SVM (assuming it has no other learned
 * outside the InnerProductLayer and no other losses outside the
 * HingeLossLayer).
 */
@Name("caffe::HingeLossLayer<float>") public static class FloatHingeLossLayer extends FloatLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatHingeLossLayer(Pointer p) { super(p); }

  public FloatHingeLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::HingeLossLayer<double>") public static class DoubleHingeLossLayer extends DoubleLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleHingeLossLayer(Pointer p) { super(p); }

  public DoubleHingeLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}


  // namespace caffe

// #endif  // CAFFE_HINGE_LOSS_LAYER_HPP_


// Parsed from caffe/layers/infogain_loss_layer.hpp

// #ifndef CAFFE_INFOGAIN_LOSS_LAYER_HPP_
// #define CAFFE_INFOGAIN_LOSS_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"

/**
 * \brief A generalization of MultinomialLogisticLossLayer that takes an
 *        "information gain" (infogain) matrix specifying the "value" of all label
 *        pairs.
 *
 * Equivalent to the MultinomialLogisticLossLayer if the infogain matrix is the
 * identity.
 *
 * @param bottom input Blob vector (length 2-3)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the predictions \f$ \hat{p} \f$, a Blob with values in
 *      \f$ [0, 1] \f$ indicating the predicted probability of each of the
 *      \f$ K = CHW \f$ classes.  Each prediction vector \f$ \hat{p}_n \f$
 *      should sum to 1 as in a probability distribution: \f$
 *      \forall n \sum\limits_{k=1}^K \hat{p}_{nk} = 1 \f$.
 *   -# \f$ (N \times 1 \times 1 \times 1) \f$
 *      the labels \f$ l \f$, an integer-valued Blob with values
 *      \f$ l_n \in [0, 1, 2, ..., K - 1] \f$
 *      indicating the correct class label among the \f$ K \f$ classes
 *   -# \f$ (1 \times 1 \times K \times K) \f$
 *      (\b optional) the infogain matrix \f$ H \f$.  This must be provided as
 *      the third bottom blob input if not provided as the infogain_mat in the
 *      InfogainLossParameter. If \f$ H = I \f$, this layer is equivalent to the
 *      MultinomialLogisticLossLayer.
 * @param top output Blob vector (length 1)
 *   -# \f$ (1 \times 1 \times 1 \times 1) \f$
 *      the computed infogain multinomial logistic loss: \f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N H_{l_n} \log(\hat{p}_n) =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \sum\limits_{k=1}^{K} H_{l_n,k}
 *        \log(\hat{p}_{n,k})
 *      \f$, where \f$ H_{l_n} \f$ denotes row \f$l_n\f$ of \f$H\f$.
 */
@Name("caffe::InfogainLossLayer<float>") @NoOffset public static class FloatInfogainLossLayer extends FloatLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatInfogainLossLayer(Pointer p) { super(p); }

  public FloatInfogainLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  // InfogainLossLayer takes 2-3 bottom Blobs; if there are 3 the third should
  // be the infogain matrix.  (Otherwise the infogain matrix is loaded from a
  // file specified by LayerParameter.)
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int MaxBottomBlobs();

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::InfogainLossLayer<double>") @NoOffset public static class DoubleInfogainLossLayer extends DoubleLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleInfogainLossLayer(Pointer p) { super(p); }

  public DoubleInfogainLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  // InfogainLossLayer takes 2-3 bottom Blobs; if there are 3 the third should
  // be the infogain matrix.  (Otherwise the infogain matrix is loaded from a
  // file specified by LayerParameter.)
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int MaxBottomBlobs();

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_INFOGAIN_LOSS_LAYER_HPP_


// Parsed from caffe/layers/multinomial_logistic_loss_layer.hpp

// #ifndef CAFFE_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_
// #define CAFFE_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"

/**
 * \brief Computes the multinomial logistic loss for a one-of-many
 *        classification task, directly taking a predicted probability
 *        distribution as input.
 *
 * When predictions are not already a probability distribution, you should
 * instead use the SoftmaxWithLossLayer, which maps predictions to a
 * distribution using the SoftmaxLayer, before computing the multinomial
 * logistic loss. The SoftmaxWithLossLayer should be preferred over separate
 * SoftmaxLayer + MultinomialLogisticLossLayer
 * as its gradient computation is more numerically stable.
 *
 * @param bottom input Blob vector (length 2)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the predictions \f$ \hat{p} \f$, a Blob with values in
 *      \f$ [0, 1] \f$ indicating the predicted probability of each of the
 *      \f$ K = CHW \f$ classes.  Each prediction vector \f$ \hat{p}_n \f$
 *      should sum to 1 as in a probability distribution: \f$
 *      \forall n \sum\limits_{k=1}^K \hat{p}_{nk} = 1 \f$.
 *   -# \f$ (N \times 1 \times 1 \times 1) \f$
 *      the labels \f$ l \f$, an integer-valued Blob with values
 *      \f$ l_n \in [0, 1, 2, ..., K - 1] \f$
 *      indicating the correct class label among the \f$ K \f$ classes
 * @param top output Blob vector (length 1)
 *   -# \f$ (1 \times 1 \times 1 \times 1) \f$
 *      the computed multinomial logistic loss: \f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \log(\hat{p}_{n,l_n})
 *      \f$
 */
@Name("caffe::MultinomialLogisticLossLayer<float>") public static class FloatMultinomialLogisticLossLayer extends FloatLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatMultinomialLogisticLossLayer(Pointer p) { super(p); }

  public FloatMultinomialLogisticLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::MultinomialLogisticLossLayer<double>") public static class DoubleMultinomialLogisticLossLayer extends DoubleLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleMultinomialLogisticLossLayer(Pointer p) { super(p); }

  public DoubleMultinomialLogisticLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_


// Parsed from caffe/layers/sigmoid_cross_entropy_loss_layer.hpp

// #ifndef CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
// #define CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"
// #include "caffe/layers/sigmoid_layer.hpp"

/**
 * \brief Computes the cross-entropy (logistic) loss \f$
 *          E = \frac{-1}{n} \sum\limits_{n=1}^N \left[
 *                  p_n \log \hat{p}_n +
 *                  (1 - p_n) \log(1 - \hat{p}_n)
 *              \right]
 *        \f$, often used for predicting targets interpreted as probabilities.
 *
 * This layer is implemented rather than separate
 * SigmoidLayer + CrossEntropyLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SigmoidLayer.
 *
 * @param bottom input Blob vector (length 2)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the scores \f$ x \in [-\infty, +\infty]\f$,
 *      which this layer maps to probability predictions
 *      \f$ \hat{p}_n = \sigma(x_n) \in [0, 1] \f$
 *      using the sigmoid function \f$ \sigma(.) \f$ (see SigmoidLayer).
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the targets \f$ y \in [0, 1] \f$
 * @param top output Blob vector (length 1)
 *   -# \f$ (1 \times 1 \times 1 \times 1) \f$
 *      the computed cross-entropy loss: \f$
 *          E = \frac{-1}{n} \sum\limits_{n=1}^N \left[
 *                  p_n \log \hat{p}_n + (1 - p_n) \log(1 - \hat{p}_n)
 *              \right]
 *      \f$
 */
@Name("caffe::SigmoidCrossEntropyLossLayer<float>") @NoOffset public static class FloatSigmoidCrossEntropyLossLayer extends FloatLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSigmoidCrossEntropyLossLayer(Pointer p) { super(p); }

  public FloatSigmoidCrossEntropyLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::SigmoidCrossEntropyLossLayer<double>") @NoOffset public static class DoubleSigmoidCrossEntropyLossLayer extends DoubleLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSigmoidCrossEntropyLossLayer(Pointer p) { super(p); }

  public DoubleSigmoidCrossEntropyLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_


// Parsed from caffe/layers/softmax_loss_layer.hpp

// #ifndef CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
// #define CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"
// #include "caffe/layers/softmax_layer.hpp"

/**
 * \brief Computes the multinomial logistic loss for a one-of-many
 *        classification task, passing real-valued predictions through a
 *        softmax to get a probability distribution over classes.
 *
 * This layer should be preferred over separate
 * SoftmaxLayer + MultinomialLogisticLossLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SoftmaxLayer.
 *
 * @param bottom input Blob vector (length 2)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the predictions \f$ x \f$, a Blob with values in
 *      \f$ [-\infty, +\infty] \f$ indicating the predicted score for each of
 *      the \f$ K = CHW \f$ classes. This layer maps these scores to a
 *      probability distribution over classes using the softmax function
 *      \f$ \hat{p}_{nk} = \exp(x_{nk}) /
 *      \left[\sum_{k'} \exp(x_{nk'})\right] \f$ (see SoftmaxLayer).
 *   -# \f$ (N \times 1 \times 1 \times 1) \f$
 *      the labels \f$ l \f$, an integer-valued Blob with values
 *      \f$ l_n \in [0, 1, 2, ..., K - 1] \f$
 *      indicating the correct class label among the \f$ K \f$ classes
 * @param top output Blob vector (length 1)
 *   -# \f$ (1 \times 1 \times 1 \times 1) \f$
 *      the computed cross-entropy classification loss: \f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \log(\hat{p}_{n,l_n})
 *      \f$, for softmax output class probabilites \f$ \hat{p} \f$
 */
@Name("caffe::SoftmaxWithLossLayer<float>") @NoOffset public static class FloatSoftmaxWithLossLayer extends FloatLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSoftmaxWithLossLayer(Pointer p) { super(p); }

   /**
    * @param param provides LossParameter loss_param, with options:
    *  - ignore_label (optional)
    *    Specify a label value that should be ignored when computing the loss.
    *  - normalize (optional, default true)
    *    If true, the loss is normalized by the number of (nonignored) labels
    *    present; otherwise the loss is simply summed over spatial locations.
    */
  public FloatSoftmaxWithLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual public native int MaxTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native float get_normalizer(
        @Cast("caffe::LossParameter_NormalizationMode") int normalization_mode, int valid_count);
}
@Name("caffe::SoftmaxWithLossLayer<double>") @NoOffset public static class DoubleSoftmaxWithLossLayer extends DoubleLossLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSoftmaxWithLossLayer(Pointer p) { super(p); }

   /**
    * @param param provides LossParameter loss_param, with options:
    *  - ignore_label (optional)
    *    Specify a label value that should be ignored when computing the loss.
    *  - normalize (optional, default true)
    *    If true, the loss is normalized by the number of (nonignored) labels
    *    present; otherwise the loss is simply summed over spatial locations.
    */
  public DoubleSoftmaxWithLossLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual public native int MaxTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native double get_normalizer(
        @Cast("caffe::LossParameter_NormalizationMode") int normalization_mode, int valid_count);
}

  // namespace caffe

// #endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_


// Parsed from caffe/layers/neuron_layer.hpp

// #ifndef CAFFE_NEURON_LAYER_HPP_
// #define CAFFE_NEURON_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief An interface for layers that take one blob as input (\f$ x \f$)
 *        and produce one equally-sized blob as output (\f$ y \f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 */
@Name("caffe::NeuronLayer<float>") public static class FloatNeuronLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatNeuronLayer(Pointer p) { super(p); }

  public FloatNeuronLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
}
@Name("caffe::NeuronLayer<double>") public static class DoubleNeuronLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleNeuronLayer(Pointer p) { super(p); }

  public DoubleNeuronLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
}

  // namespace caffe

// #endif  // CAFFE_NEURON_LAYER_HPP_


// Parsed from caffe/layers/absval_layer.hpp

// #ifndef CAFFE_ABSVAL_LAYER_HPP_
// #define CAFFE_ABSVAL_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Computes \f$ y = |x| \f$
 *
 * @param bottom input Blob vector (length 1)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the inputs \f$ x \f$
 * @param top output Blob vector (length 1)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the computed outputs \f$ y = |x| \f$
 */
@Name("caffe::AbsValLayer<float>") public static class FloatAbsValLayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatAbsValLayer(Pointer p) { super(p); }

  public FloatAbsValLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::AbsValLayer<double>") public static class DoubleAbsValLayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleAbsValLayer(Pointer p) { super(p); }

  public DoubleAbsValLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_ABSVAL_LAYER_HPP_


// Parsed from caffe/layers/bnll_layer.hpp

// #ifndef CAFFE_BNLL_LAYER_HPP_
// #define CAFFE_BNLL_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Computes \f$ y = x + \log(1 + \exp(-x)) \f$ if \f$ x > 0 \f$;
 *        \f$ y = \log(1 + \exp(x)) \f$ otherwise.
 *
 * @param bottom input Blob vector (length 1)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the inputs \f$ x \f$
 * @param top output Blob vector (length 1)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the computed outputs \f$
 *      y = \left\{
 *         \begin{array}{ll}
 *            x + \log(1 + \exp(-x)) & \mbox{if } x > 0 \\
 *            \log(1 + \exp(x)) & \mbox{otherwise}
 *         \end{array} \right.
 *      \f$
 */
@Name("caffe::BNLLLayer<float>") public static class FloatBNLLLayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBNLLLayer(Pointer p) { super(p); }

  public FloatBNLLLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::BNLLLayer<double>") public static class DoubleBNLLLayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBNLLLayer(Pointer p) { super(p); }

  public DoubleBNLLLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_BNLL_LAYER_HPP_


// Parsed from caffe/layers/dropout_layer.hpp

// #ifndef CAFFE_DROPOUT_LAYER_HPP_
// #define CAFFE_DROPOUT_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief During training only, sets a random portion of \f$x\f$ to 0, adjusting
 *        the rest of the vector magnitude accordingly.
 *
 * @param bottom input Blob vector (length 1)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the inputs \f$ x \f$
 * @param top output Blob vector (length 1)
 *   -# \f$ (N \times C \times H \times W) \f$
 *      the computed outputs \f$ y = |x| \f$
 */
@Name("caffe::DropoutLayer<float>") @NoOffset public static class FloatDropoutLayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatDropoutLayer(Pointer p) { super(p); }

  /**
   * @param param provides DropoutParameter dropout_param,
   *     with DropoutLayer options:
   *   - dropout_ratio (\b optional, default 0.5).
   *     Sets the probability \f$ p \f$ that any given unit is dropped.
   */
  public FloatDropoutLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::DropoutLayer<double>") @NoOffset public static class DoubleDropoutLayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleDropoutLayer(Pointer p) { super(p); }

  /**
   * @param param provides DropoutParameter dropout_param,
   *     with DropoutLayer options:
   *   - dropout_ratio (\b optional, default 0.5).
   *     Sets the probability \f$ p \f$ that any given unit is dropped.
   */
  public DoubleDropoutLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_DROPOUT_LAYER_HPP_


// Parsed from caffe/layers/exp_layer.hpp

// #ifndef CAFFE_EXP_LAYER_HPP_
// #define CAFFE_EXP_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Computes \f$ y = \gamma ^ {\alpha x + \beta} \f$,
 *        as specified by the scale \f$ \alpha \f$, shift \f$ \beta \f$,
 *        and base \f$ \gamma \f$.
 */
@Name("caffe::ExpLayer<float>") @NoOffset public static class FloatExpLayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatExpLayer(Pointer p) { super(p); }

  /**
   * @param param provides ExpParameter exp_param,
   *     with ExpLayer options:
   *   - scale (\b optional, default 1) the scale \f$ \alpha \f$
   *   - shift (\b optional, default 0) the shift \f$ \beta \f$
   *   - base (\b optional, default -1 for a value of \f$ e \approx 2.718 \f$)
   *         the base \f$ \gamma \f$
   */
  public FloatExpLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::ExpLayer<double>") @NoOffset public static class DoubleExpLayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleExpLayer(Pointer p) { super(p); }

  /**
   * @param param provides ExpParameter exp_param,
   *     with ExpLayer options:
   *   - scale (\b optional, default 1) the scale \f$ \alpha \f$
   *   - shift (\b optional, default 0) the shift \f$ \beta \f$
   *   - base (\b optional, default -1 for a value of \f$ e \approx 2.718 \f$)
   *         the base \f$ \gamma \f$
   */
  public DoubleExpLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_EXP_LAYER_HPP_


// Parsed from caffe/layers/log_layer.hpp

// #ifndef CAFFE_LOG_LAYER_HPP_
// #define CAFFE_LOG_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Computes \f$ y = log_{\gamma}(\alpha x + \beta) \f$,
 *        as specified by the scale \f$ \alpha \f$, shift \f$ \beta \f$,
 *        and base \f$ \gamma \f$.
 */

  // namespace caffe

// #endif  // CAFFE_LOG_LAYER_HPP_


// Parsed from caffe/layers/power_layer.hpp

// #ifndef CAFFE_POWER_LAYER_HPP_
// #define CAFFE_POWER_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Computes \f$ y = (\alpha x + \beta) ^ \gamma \f$,
 *        as specified by the scale \f$ \alpha \f$, shift \f$ \beta \f$,
 *        and power \f$ \gamma \f$.
 */
@Name("caffe::PowerLayer<float>") @NoOffset public static class FloatPowerLayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatPowerLayer(Pointer p) { super(p); }

  /**
   * @param param provides PowerParameter power_param,
   *     with PowerLayer options:
   *   - scale (\b optional, default 1) the scale \f$ \alpha \f$
   *   - shift (\b optional, default 0) the shift \f$ \beta \f$
   *   - power (\b optional, default 1) the power \f$ \gamma \f$
   */
  public FloatPowerLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::PowerLayer<double>") @NoOffset public static class DoublePowerLayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoublePowerLayer(Pointer p) { super(p); }

  /**
   * @param param provides PowerParameter power_param,
   *     with PowerLayer options:
   *   - scale (\b optional, default 1) the scale \f$ \alpha \f$
   *   - shift (\b optional, default 0) the shift \f$ \beta \f$
   *   - power (\b optional, default 1) the power \f$ \gamma \f$
   */
  public DoublePowerLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_POWER_LAYER_HPP_


// Parsed from caffe/layers/relu_layer.hpp

// #ifndef CAFFE_RELU_LAYER_HPP_
// #define CAFFE_RELU_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Rectified Linear Unit non-linearity \f$ y = \max(0, x) \f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
@Name("caffe::ReLULayer<float>") public static class FloatReLULayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatReLULayer(Pointer p) { super(p); }

  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value \f$ \nu \f$ by which negative values are multiplied.
   */
  public FloatReLULayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::ReLULayer<double>") public static class DoubleReLULayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleReLULayer(Pointer p) { super(p); }

  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value \f$ \nu \f$ by which negative values are multiplied.
   */
  public DoubleReLULayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_RELU_LAYER_HPP_


// Parsed from caffe/layers/cudnn_relu_layer.hpp

// #ifndef CAFFE_CUDNN_RELU_LAYER_HPP_
// #define CAFFE_CUDNN_RELU_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"
// #include "caffe/layers/relu_layer.hpp"

// #ifdef USE_CUDNN
// #endif

  // namespace caffe

// #endif  // CAFFE_CUDNN_RELU_LAYER_HPP_


// Parsed from caffe/layers/sigmoid_layer.hpp

// #ifndef CAFFE_SIGMOID_LAYER_HPP_
// #define CAFFE_SIGMOID_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Sigmoid function non-linearity \f$
 *         y = (1 + \exp(-x))^{-1}
 *     \f$, a classic choice in neural networks.
 *
 * Note that the gradient vanishes as the values move away from 0.
 * The ReLULayer is often a better choice for this reason.
 */
@Name("caffe::SigmoidLayer<float>") public static class FloatSigmoidLayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSigmoidLayer(Pointer p) { super(p); }

  public FloatSigmoidLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::SigmoidLayer<double>") public static class DoubleSigmoidLayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSigmoidLayer(Pointer p) { super(p); }

  public DoubleSigmoidLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_SIGMOID_LAYER_HPP_


// Parsed from caffe/layers/cudnn_sigmoid_layer.hpp

// #ifndef CAFFE_CUDNN_SIGMOID_LAYER_HPP_
// #define CAFFE_CUDNN_SIGMOID_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"
// #include "caffe/layers/sigmoid_layer.hpp"

// #ifdef USE_CUDNN
// #endif

  // namespace caffe

// #endif  // CAFFE_CUDNN_SIGMOID_LAYER_HPP_


// Parsed from caffe/layers/tanh_layer.hpp

// #ifndef CAFFE_TANH_LAYER_HPP_
// #define CAFFE_TANH_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief TanH hyperbolic tangent non-linearity \f$
 *         y = \frac{\exp(2x) - 1}{\exp(2x) + 1}
 *     \f$, popular in auto-encoders.
 *
 * Note that the gradient vanishes as the values move away from 0.
 * The ReLULayer is often a better choice for this reason.
 */
@Name("caffe::TanHLayer<float>") public static class FloatTanHLayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatTanHLayer(Pointer p) { super(p); }

  public FloatTanHLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::TanHLayer<double>") public static class DoubleTanHLayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleTanHLayer(Pointer p) { super(p); }

  public DoubleTanHLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_TANH_LAYER_HPP_


// Parsed from caffe/layers/cudnn_tanh_layer.hpp

// #ifndef CAFFE_CUDNN_TANH_LAYER_HPP_
// #define CAFFE_CUDNN_TANH_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"
// #include "caffe/layers/tanh_layer.hpp"

// #ifdef USE_CUDNN
// #endif

  // namespace caffe

// #endif  // CAFFE_CUDNN_TANH_LAYER_HPP_


// Parsed from caffe/layers/threshold_layer.hpp

// #ifndef CAFFE_THRESHOLD_LAYER_HPP_
// #define CAFFE_THRESHOLD_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Tests whether the input exceeds a threshold: outputs 1 for inputs
 *        above threshold; 0 otherwise.
 */
@Name("caffe::ThresholdLayer<float>") @NoOffset public static class FloatThresholdLayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatThresholdLayer(Pointer p) { super(p); }

  /**
   * @param param provides ThresholdParameter threshold_param,
   *     with ThresholdLayer options:
   *   - threshold (\b optional, default 0).
   *     the threshold value \f$ t \f$ to which the input values are compared.
   */
  public FloatThresholdLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::ThresholdLayer<double>") @NoOffset public static class DoubleThresholdLayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleThresholdLayer(Pointer p) { super(p); }

  /**
   * @param param provides ThresholdParameter threshold_param,
   *     with ThresholdLayer options:
   *   - threshold (\b optional, default 0).
   *     the threshold value \f$ t \f$ to which the input values are compared.
   */
  public DoubleThresholdLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_THRESHOLD_LAYER_HPP_


// Parsed from caffe/layers/prelu_layer.hpp

// #ifndef CAFFE_PRELU_LAYER_HPP_
// #define CAFFE_PRELU_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/neuron_layer.hpp"

/**
 * \brief Parameterized Rectified Linear Unit non-linearity \f$
 *        y_i = \max(0, x_i) + a_i \min(0, x_i)
 *        \f$. The differences from ReLULayer are 1) negative slopes are
 *        learnable though backprop and 2) negative slopes can vary across
 *        channels. The number of axes of input blob should be greater than or
 *        equal to 2. The 1st axis (0-based) is seen as channels.
 */
@Name("caffe::PReLULayer<float>") @NoOffset public static class FloatPReLULayer extends FloatNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatPReLULayer(Pointer p) { super(p); }

  /**
   * @param param provides PReLUParameter prelu_param,
   *     with PReLULayer options:
   *   - filler (\b optional, FillerParameter,
   *     default {'type': constant 'value':0.25}).
   *   - channel_shared (\b optional, default false).
   *     negative slopes are shared across channels.
   */
  public FloatPReLULayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::PReLULayer<double>") @NoOffset public static class DoublePReLULayer extends DoubleNeuronLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoublePReLULayer(Pointer p) { super(p); }

  /**
   * @param param provides PReLUParameter prelu_param,
   *     with PReLULayer options:
   *   - filler (\b optional, FillerParameter,
   *     default {'type': constant 'value':0.25}).
   *   - channel_shared (\b optional, default false).
   *     negative slopes are shared across channels.
   */
  public DoublePReLULayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_PRELU_LAYER_HPP_


// Parsed from caffe/layers/argmax_layer.hpp

// #ifndef CAFFE_ARGMAX_LAYER_HPP_
// #define CAFFE_ARGMAX_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Compute the index of the \f$ K \f$ max values for each datum across
 *        all dimensions \f$ (C \times H \times W) \f$.
 *
 * Intended for use after a classification layer to produce a prediction.
 * If parameter out_max_val is set to true, output is a vector of pairs
 * (max_ind, max_val) for each image. The axis parameter specifies an axis
 * along which to maximise.
 *
 * NOTE: does not implement Backwards operation.
 */
@Name("caffe::ArgMaxLayer<float>") @NoOffset public static class FloatArgMaxLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatArgMaxLayer(Pointer p) { super(p); }

  /**
   * @param param provides ArgMaxParameter argmax_param,
   *     with ArgMaxLayer options:
   *   - top_k (\b optional uint, default 1).
   *     the number \f$ K \f$ of maximal items to output.
   *   - out_max_val (\b optional bool, default false).
   *     if set, output a vector of pairs (max_ind, max_val) unless axis is set then
   *     output max_val along the specified axis.
   *   - axis (\b optional int).
   *     if set, maximise along the specified axis else maximise the flattened
   *     trailing dimensions for each index of the first / num dimension.
   */
  public FloatArgMaxLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::ArgMaxLayer<double>") @NoOffset public static class DoubleArgMaxLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleArgMaxLayer(Pointer p) { super(p); }

  /**
   * @param param provides ArgMaxParameter argmax_param,
   *     with ArgMaxLayer options:
   *   - top_k (\b optional uint, default 1).
   *     the number \f$ K \f$ of maximal items to output.
   *   - out_max_val (\b optional bool, default false).
   *     if set, output a vector of pairs (max_ind, max_val) unless axis is set then
   *     output max_val along the specified axis.
   *   - axis (\b optional int).
   *     if set, maximise along the specified axis else maximise the flattened
   *     trailing dimensions for each index of the first / num dimension.
   */
  public DoubleArgMaxLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_ARGMAX_LAYER_HPP_


// Parsed from caffe/layers/batch_norm_layer.hpp

// #ifndef CAFFE_BATCHNORM_LAYER_HPP_
// #define CAFFE_BATCHNORM_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization described in [1].  For
 * each channel in the data (i.e. axis 1), it subtracts the mean and divides
 * by the variance, where both statistics are computed across both spatial
 * dimensions and across the different examples in the batch.
 *
 * By default, during training time, the network is computing global mean/
 * variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input.  You can manually
 * toggle whether the network is accumulating or using the statistics via the
 * use_global_stats option.  IMPORTANT: for this feature to work, you MUST
 * set the learning rate to zero for all three parameter blobs, i.e.,
 * param {lr_mult: 0} three times in the layer definition.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor.  It is possible (though a bit cumbersome) to implement
 * this in caffe using a single-channel DummyDataLayer filled with zeros,
 * followed by a Convolution layer with output the same size as the current.
 * This produces a channel-specific value that can be added or multiplied by
 * the BatchNorm layer's output.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::BatchNormLayer<float>") @NoOffset public static class FloatBatchNormLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBatchNormLayer(Pointer p) { super(p); }

  public FloatBatchNormLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
       @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::BatchNormLayer<double>") @NoOffset public static class DoubleBatchNormLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBatchNormLayer(Pointer p) { super(p); }

  public DoubleBatchNormLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
       @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_BATCHNORM_LAYER_HPP_


// Parsed from caffe/layers/batch_reindex_layer.hpp

// #ifndef CAFFE_BATCHREINDEX_LAYER_HPP_
// #define CAFFE_BATCHREINDEX_LAYER_HPP_

// #include <utility>
// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Index into the input blob along its first axis.
 *
 * This layer can be used to select, reorder, and even replicate examples in a
 * batch.  The second blob is cast to int and treated as an index into the
 * first axis of the first blob.
 */
@Name("caffe::BatchReindexLayer<float>") public static class FloatBatchReindexLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBatchReindexLayer(Pointer p) { super(p); }

  public FloatBatchReindexLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::BatchReindexLayer<double>") public static class DoubleBatchReindexLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBatchReindexLayer(Pointer p) { super(p); }

  public DoubleBatchReindexLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_BATCHREINDEX_LAYER_HPP_


// Parsed from caffe/layers/concat_layer.hpp

// #ifndef CAFFE_CONCAT_LAYER_HPP_
// #define CAFFE_CONCAT_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Takes at least two Blob%s and concatenates them along either the num
 *        or channel dimension, outputting the result.
 */
@Name("caffe::ConcatLayer<float>") @NoOffset public static class FloatConcatLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatConcatLayer(Pointer p) { super(p); }

  public FloatConcatLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::ConcatLayer<double>") @NoOffset public static class DoubleConcatLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleConcatLayer(Pointer p) { super(p); }

  public DoubleConcatLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_CONCAT_LAYER_HPP_


// Parsed from caffe/layers/eltwise_layer.hpp

// #ifndef CAFFE_ELTWISE_LAYER_HPP_
// #define CAFFE_ELTWISE_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::EltwiseLayer<float>") @NoOffset public static class FloatEltwiseLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatEltwiseLayer(Pointer p) { super(p); }

  public FloatEltwiseLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::EltwiseLayer<double>") @NoOffset public static class DoubleEltwiseLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleEltwiseLayer(Pointer p) { super(p); }

  public DoubleEltwiseLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_ELTWISE_LAYER_HPP_


// Parsed from caffe/layers/embed_layer.hpp

// #ifndef CAFFE_EMBED_LAYER_HPP_
// #define CAFFE_EMBED_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief A layer for learning "embeddings" of one-hot vector input.
 *        Equivalent to an InnerProductLayer with one-hot vectors as input, but
 *        for efficiency the input is the "hot" index of each column itself.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::EmbedLayer<float>") @NoOffset public static class FloatEmbedLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatEmbedLayer(Pointer p) { super(p); }

  public FloatEmbedLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::EmbedLayer<double>") @NoOffset public static class DoubleEmbedLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleEmbedLayer(Pointer p) { super(p); }

  public DoubleEmbedLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_EMBED_LAYER_HPP_


// Parsed from caffe/layers/filter_layer.hpp

// #ifndef CAFFE_FILTER_LAYER_HPP_
// #define CAFFE_FILTER_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Takes two+ Blobs, interprets last Blob as a selector and
 *  filter remaining Blobs accordingly with selector data (0 means that
 * the corresponding item has to be filtered, non-zero means that corresponding
 * item needs to stay).
 */
@Name("caffe::FilterLayer<float>") @NoOffset public static class FloatFilterLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatFilterLayer(Pointer p) { super(p); }

  public FloatFilterLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
      @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
      @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::FilterLayer<double>") @NoOffset public static class DoubleFilterLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleFilterLayer(Pointer p) { super(p); }

  public DoubleFilterLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
      @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
      @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_FILTER_LAYER_HPP_


// Parsed from caffe/layers/flatten_layer.hpp

// #ifndef CAFFE_FLATTEN_LAYER_HPP_
// #define CAFFE_FLATTEN_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Reshapes the input Blob into flat vectors.
 *
 * Note: because this layer does not change the input values -- merely the
 * dimensions -- it can simply copy the input. The copy happens "virtually"
 * (thus taking effectively 0 real time) by setting, in Forward, the data
 * pointer of the top Blob to that of the bottom Blob (see Blob::ShareData),
 * and in Backward, the diff pointer of the bottom Blob to that of the top Blob
 * (see Blob::ShareDiff).
 */
@Name("caffe::FlattenLayer<float>") public static class FloatFlattenLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatFlattenLayer(Pointer p) { super(p); }

  public FloatFlattenLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::FlattenLayer<double>") public static class DoubleFlattenLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleFlattenLayer(Pointer p) { super(p); }

  public DoubleFlattenLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_FLATTEN_LAYER_HPP_


// Parsed from caffe/layers/inner_product_layer.hpp

// #ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
// #define CAFFE_INNER_PRODUCT_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::InnerProductLayer<float>") @NoOffset public static class FloatInnerProductLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatInnerProductLayer(Pointer p) { super(p); }

  public FloatInnerProductLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::InnerProductLayer<double>") @NoOffset public static class DoubleInnerProductLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleInnerProductLayer(Pointer p) { super(p); }

  public DoubleInnerProductLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_


// Parsed from caffe/layers/mvn_layer.hpp

// #ifndef CAFFE_MVN_LAYER_HPP_
// #define CAFFE_MVN_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Normalizes the input to have 0-mean and/or unit (1) variance.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::MVNLayer<float>") @NoOffset public static class FloatMVNLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatMVNLayer(Pointer p) { super(p); }

  public FloatMVNLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
       @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::MVNLayer<double>") @NoOffset public static class DoubleMVNLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleMVNLayer(Pointer p) { super(p); }

  public DoubleMVNLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
       @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_MVN_LAYER_HPP_


// Parsed from caffe/layers/reshape_layer.hpp

// #ifndef CAFFE_XXX_LAYER_HPP_
// #define CAFFE_XXX_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
@Name("caffe::ReshapeLayer<float>") @NoOffset public static class FloatReshapeLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatReshapeLayer(Pointer p) { super(p); }

  public FloatReshapeLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::ReshapeLayer<double>") @NoOffset public static class DoubleReshapeLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleReshapeLayer(Pointer p) { super(p); }

  public DoubleReshapeLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_XXX_LAYER_HPP_


// Parsed from caffe/layers/reduction_layer.hpp

// #ifndef CAFFE_REDUCTION_LAYER_HPP_
// #define CAFFE_REDUCTION_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Compute "reductions" -- operations that return a scalar output Blob
 *        for an input Blob of arbitrary size, such as the sum, absolute sum,
 *        and sum of squares.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::ReductionLayer<float>") @NoOffset public static class FloatReductionLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatReductionLayer(Pointer p) { super(p); }

  public FloatReductionLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::ReductionLayer<double>") @NoOffset public static class DoubleReductionLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleReductionLayer(Pointer p) { super(p); }

  public DoubleReductionLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_REDUCTION_LAYER_HPP_


// Parsed from caffe/layers/silence_layer.hpp

// #ifndef CAFFE_SILENCE_LAYER_HPP_
// #define CAFFE_SILENCE_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Ignores bottom blobs while producing no top blobs. (This is useful
 *        to suppress outputs during testing.)
 */
@Name("caffe::SilenceLayer<float>") public static class FloatSilenceLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSilenceLayer(Pointer p) { super(p); }

  public FloatSilenceLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::SilenceLayer<double>") public static class DoubleSilenceLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSilenceLayer(Pointer p) { super(p); }

  public DoubleSilenceLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int MinBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_SILENCE_LAYER_HPP_


// Parsed from caffe/layers/softmax_layer.hpp

// #ifndef CAFFE_SOFTMAX_LAYER_HPP_
// #define CAFFE_SOFTMAX_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::SoftmaxLayer<float>") @NoOffset public static class FloatSoftmaxLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSoftmaxLayer(Pointer p) { super(p); }

  public FloatSoftmaxLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
       @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::SoftmaxLayer<double>") @NoOffset public static class DoubleSoftmaxLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSoftmaxLayer(Pointer p) { super(p); }

  public DoubleSoftmaxLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
       @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_SOFTMAX_LAYER_HPP_


// Parsed from caffe/layers/cudnn_softmax_layer.hpp

// #ifndef CAFFE_CUDNN_SOFTMAX_LAYER_HPP_
// #define CAFFE_CUDNN_SOFTMAX_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/softmax_layer.hpp"

// #ifdef USE_CUDNN
// #endif

  // namespace caffe

// #endif  // CAFFE_CUDNN_SOFTMAX_LAYER_HPP_


// Parsed from caffe/layers/split_layer.hpp

// #ifndef CAFFE_SPLIT_LAYER_HPP_
// #define CAFFE_SPLIT_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::SplitLayer<float>") @NoOffset public static class FloatSplitLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSplitLayer(Pointer p) { super(p); }

  public FloatSplitLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::SplitLayer<double>") @NoOffset public static class DoubleSplitLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSplitLayer(Pointer p) { super(p); }

  public DoubleSplitLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_SPLIT_LAYER_HPP_


// Parsed from caffe/layers/slice_layer.hpp

// #ifndef CAFFE_SLICE_LAYER_HPP_
// #define CAFFE_SLICE_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Takes a Blob and slices it along either the num or channel dimension,
 *        outputting multiple sliced Blob results.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::SliceLayer<float>") @NoOffset public static class FloatSliceLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSliceLayer(Pointer p) { super(p); }

  public FloatSliceLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::SliceLayer<double>") @NoOffset public static class DoubleSliceLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSliceLayer(Pointer p) { super(p); }

  public DoubleSliceLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_SLICE_LAYER_HPP_


// Parsed from caffe/layers/tile_layer.hpp

// #ifndef CAFFE_TILE_LAYER_HPP_
// #define CAFFE_TILE_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Copy a Blob along specified dimensions.
 */
@Name("caffe::TileLayer<float>") @NoOffset public static class FloatTileLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatTileLayer(Pointer p) { super(p); }

  public FloatTileLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::TileLayer<double>") @NoOffset public static class DoubleTileLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleTileLayer(Pointer p) { super(p); }

  public DoubleTileLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_TILE_LAYER_HPP_


// Parsed from caffe/net.hpp

// #ifndef CAFFE_NET_HPP_
// #define CAFFE_NET_HPP_

// #include <map>
// #include <set>
// #include <string>
// #include <utility>
// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
@Name("caffe::Net<float>") @NoOffset public static class FloatNet extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatNet(Pointer p) { super(p); }

  public FloatNet(@Const @ByRef NetParameter param, @Const FloatNet root_net/*=NULL*/) { super((Pointer)null); allocate(param, root_net); }
  private native void allocate(@Const @ByRef NetParameter param, @Const FloatNet root_net/*=NULL*/);
  public FloatNet(@Const @ByRef NetParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef NetParameter param);
  public FloatNet(@StdString BytePointer param_file, @Cast("caffe::Phase") int phase,
        @Const FloatNet root_net/*=NULL*/) { super((Pointer)null); allocate(param_file, phase, root_net); }
  private native void allocate(@StdString BytePointer param_file, @Cast("caffe::Phase") int phase,
        @Const FloatNet root_net/*=NULL*/);
  public FloatNet(@StdString BytePointer param_file, @Cast("caffe::Phase") int phase) { super((Pointer)null); allocate(param_file, phase); }
  private native void allocate(@StdString BytePointer param_file, @Cast("caffe::Phase") int phase);
  public FloatNet(@StdString String param_file, @Cast("caffe::Phase") int phase,
        @Const FloatNet root_net/*=NULL*/) { super((Pointer)null); allocate(param_file, phase, root_net); }
  private native void allocate(@StdString String param_file, @Cast("caffe::Phase") int phase,
        @Const FloatNet root_net/*=NULL*/);
  public FloatNet(@StdString String param_file, @Cast("caffe::Phase") int phase) { super((Pointer)null); allocate(param_file, phase); }
  private native void allocate(@StdString String param_file, @Cast("caffe::Phase") int phase);

  /** \brief Initialize a network with a NetParameter. */
  public native void Init(@Const @ByRef NetParameter param);

  /**
   * \brief Run Forward with the input Blob%s already fed separately.
   *
   * You can get the input blobs using input_blobs().
   */
  public native @Const @ByRef FloatBlobVector ForwardPrefilled(FloatPointer loss/*=NULL*/);
  public native @Const @ByRef FloatBlobVector ForwardPrefilled();
  public native @Const @ByRef FloatBlobVector ForwardPrefilled(FloatBuffer loss/*=NULL*/);
  public native @Const @ByRef FloatBlobVector ForwardPrefilled(float[] loss/*=NULL*/);

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  public native float ForwardFromTo(int start, int end);
  public native float ForwardFrom(int start);
  public native float ForwardTo(int end);
  /** \brief Run forward using a set of bottom blobs, and return the result. */
  public native @Const @ByRef FloatBlobVector Forward(@Const @ByRef FloatBlobVector bottom,
        FloatPointer loss/*=NULL*/);
  public native @Const @ByRef FloatBlobVector Forward(@Const @ByRef FloatBlobVector bottom);
  public native @Const @ByRef FloatBlobVector Forward(@Const @ByRef FloatBlobVector bottom,
        FloatBuffer loss/*=NULL*/);
  public native @Const @ByRef FloatBlobVector Forward(@Const @ByRef FloatBlobVector bottom,
        float[] loss/*=NULL*/);
  /**
   * \brief Run forward using a serialized BlobProtoVector and return the
   *        result as a serialized BlobProtoVector
   */
  public native @StdString BytePointer Forward(@StdString BytePointer input_blob_protos, FloatPointer loss/*=NULL*/);
  public native @StdString BytePointer Forward(@StdString BytePointer input_blob_protos);
  public native @StdString String Forward(@StdString String input_blob_protos, FloatBuffer loss/*=NULL*/);
  public native @StdString String Forward(@StdString String input_blob_protos);
  public native @StdString BytePointer Forward(@StdString BytePointer input_blob_protos, float[] loss/*=NULL*/);
  public native @StdString String Forward(@StdString String input_blob_protos, FloatPointer loss/*=NULL*/);
  public native @StdString BytePointer Forward(@StdString BytePointer input_blob_protos, FloatBuffer loss/*=NULL*/);
  public native @StdString String Forward(@StdString String input_blob_protos, float[] loss/*=NULL*/);

  /**
   * \brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  public native void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  public native void Backward();
  public native void BackwardFromTo(int start, int end);
  public native void BackwardFrom(int start);
  public native void BackwardTo(int end);

  /**
   * \brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  public native void Reshape();

  public native float ForwardBackward(@Const @ByRef FloatBlobVector bottom);

  /** \brief Updates the network weights based on the diff values computed. */
  public native void Update();
  /**
   * \brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  public native void ShareWeights();

  /**
   * \brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  public native void ShareTrainedLayersWith(@Const FloatNet other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * \brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  public native void CopyTrainedLayersFrom(@Const @ByRef NetParameter param);
  public native void CopyTrainedLayersFrom(@StdString BytePointer trained_filename);
  public native void CopyTrainedLayersFrom(@StdString String trained_filename);
  public native void CopyTrainedLayersFromBinaryProto(@StdString BytePointer trained_filename);
  public native void CopyTrainedLayersFromBinaryProto(@StdString String trained_filename);
  public native void CopyTrainedLayersFromHDF5(@StdString BytePointer trained_filename);
  public native void CopyTrainedLayersFromHDF5(@StdString String trained_filename);
  /** \brief Writes the net to a proto. */
  public native void ToProto(NetParameter param, @Cast("bool") boolean write_diff/*=false*/);
  public native void ToProto(NetParameter param);
  /** \brief Writes the net to an HDF5 file. */
  public native void ToHDF5(@StdString BytePointer filename, @Cast("bool") boolean write_diff/*=false*/);
  public native void ToHDF5(@StdString BytePointer filename);
  public native void ToHDF5(@StdString String filename, @Cast("bool") boolean write_diff/*=false*/);
  public native void ToHDF5(@StdString String filename);

  /** \brief returns the network name. */
  public native @StdString BytePointer name();
  /** \brief returns the layer names */
  public native @Const @ByRef StringVector layer_names();
  /** \brief returns the blob names */
  public native @Const @ByRef StringVector blob_names();
  /** \brief returns the blobs */
  public native @Const @ByRef FloatBlobSharedVector blobs();
  /** \brief returns the layers */
  public native @Const @ByRef FloatLayerSharedVector layers();
  /** \brief returns the phase: TRAIN or TEST */
  public native @Cast("caffe::Phase") int phase();
  /**
   * \brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  public native @Const @ByRef FloatBlobVectorVector bottom_vecs();
  /**
   * \brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  public native @Const @ByRef FloatBlobVectorVector top_vecs();
  public native @Const @ByRef BoolVectorVector bottom_need_backward();
  public native @StdVector FloatPointer blob_loss_weights();
  public native @Const @ByRef BoolVector layer_need_backward();
  /** \brief returns the parameters */
  public native @Const @ByRef FloatBlobSharedVector params();
  public native @Const @ByRef FloatBlobVector learnable_params();
  /** \brief returns the learnable parameter learning rate multipliers */
  public native @StdVector FloatPointer params_lr();
  public native @Const @ByRef BoolVector has_params_lr();
  /** \brief returns the learnable parameter decay multipliers */
  public native @StdVector FloatPointer params_weight_decay();
  public native @Const @ByRef BoolVector has_params_decay();
  public native @Const @ByRef StringIntMap param_names_index();
  public native @StdVector IntPointer param_owners();
  /** \brief Input and output blob numbers */
  public native int num_inputs();
  public native int num_outputs();
  public native @Const @ByRef FloatBlobVector input_blobs();
  public native @Const @ByRef FloatBlobVector output_blobs();
  public native @StdVector IntPointer input_blob_indices();
  public native @StdVector IntPointer output_blob_indices();
  public native @Cast("bool") boolean has_blob(@StdString BytePointer blob_name);
  public native @Cast("bool") boolean has_blob(@StdString String blob_name);
  public native @Const @SharedPtr @ByVal FloatBlob blob_by_name(@StdString BytePointer blob_name);
  public native @Const @SharedPtr @ByVal FloatBlob blob_by_name(@StdString String blob_name);
  public native @Cast("bool") boolean has_layer(@StdString BytePointer layer_name);
  public native @Cast("bool") boolean has_layer(@StdString String layer_name);
  public FloatLayer layer_by_name(BytePointer layer_name) { return layer_by_name(FloatLayer.class, layer_name); }
  public FloatLayer layer_by_name(String layer_name) { return layer_by_name(FloatLayer.class, layer_name); };
  public native @Const @Cast({"", "boost::shared_ptr<caffe::Layer<float> >"}) @SharedPtr @ByVal <L extends FloatLayer> L layer_by_name(Class<L> cls, @StdString BytePointer layer_name);
  public native @Const @Cast({"", "boost::shared_ptr<caffe::Layer<float> >"}) @SharedPtr @ByVal <L extends FloatLayer> L layer_by_name(Class<L> cls, @StdString String layer_name);

  public native void set_debug_info(@Cast("const bool") boolean value);

  // Helpers for Init.
  /**
   * \brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  public static native void FilterNet(@Const @ByRef NetParameter param,
        NetParameter param_filtered);
  /** \brief return whether NetState state meets NetStateRule rule */
  public static native @Cast("bool") boolean StateMeetsRule(@Const @ByRef NetState state, @Const @ByRef NetStateRule rule,
        @StdString BytePointer layer_name);
  public static native @Cast("bool") boolean StateMeetsRule(@Const @ByRef NetState state, @Const @ByRef NetStateRule rule,
        @StdString String layer_name);
}
@Name("caffe::Net<double>") @NoOffset public static class DoubleNet extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleNet(Pointer p) { super(p); }

  public DoubleNet(@Const @ByRef NetParameter param, @Const DoubleNet root_net/*=NULL*/) { super((Pointer)null); allocate(param, root_net); }
  private native void allocate(@Const @ByRef NetParameter param, @Const DoubleNet root_net/*=NULL*/);
  public DoubleNet(@Const @ByRef NetParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef NetParameter param);
  public DoubleNet(@StdString BytePointer param_file, @Cast("caffe::Phase") int phase,
        @Const DoubleNet root_net/*=NULL*/) { super((Pointer)null); allocate(param_file, phase, root_net); }
  private native void allocate(@StdString BytePointer param_file, @Cast("caffe::Phase") int phase,
        @Const DoubleNet root_net/*=NULL*/);
  public DoubleNet(@StdString BytePointer param_file, @Cast("caffe::Phase") int phase) { super((Pointer)null); allocate(param_file, phase); }
  private native void allocate(@StdString BytePointer param_file, @Cast("caffe::Phase") int phase);
  public DoubleNet(@StdString String param_file, @Cast("caffe::Phase") int phase,
        @Const DoubleNet root_net/*=NULL*/) { super((Pointer)null); allocate(param_file, phase, root_net); }
  private native void allocate(@StdString String param_file, @Cast("caffe::Phase") int phase,
        @Const DoubleNet root_net/*=NULL*/);
  public DoubleNet(@StdString String param_file, @Cast("caffe::Phase") int phase) { super((Pointer)null); allocate(param_file, phase); }
  private native void allocate(@StdString String param_file, @Cast("caffe::Phase") int phase);

  /** \brief Initialize a network with a NetParameter. */
  public native void Init(@Const @ByRef NetParameter param);

  /**
   * \brief Run Forward with the input Blob%s already fed separately.
   *
   * You can get the input blobs using input_blobs().
   */
  public native @Const @ByRef DoubleBlobVector ForwardPrefilled(DoublePointer loss/*=NULL*/);
  public native @Const @ByRef DoubleBlobVector ForwardPrefilled();
  public native @Const @ByRef DoubleBlobVector ForwardPrefilled(DoubleBuffer loss/*=NULL*/);
  public native @Const @ByRef DoubleBlobVector ForwardPrefilled(double[] loss/*=NULL*/);

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  public native double ForwardFromTo(int start, int end);
  public native double ForwardFrom(int start);
  public native double ForwardTo(int end);
  /** \brief Run forward using a set of bottom blobs, and return the result. */
  public native @Const @ByRef DoubleBlobVector Forward(@Const @ByRef DoubleBlobVector bottom,
        DoublePointer loss/*=NULL*/);
  public native @Const @ByRef DoubleBlobVector Forward(@Const @ByRef DoubleBlobVector bottom);
  public native @Const @ByRef DoubleBlobVector Forward(@Const @ByRef DoubleBlobVector bottom,
        DoubleBuffer loss/*=NULL*/);
  public native @Const @ByRef DoubleBlobVector Forward(@Const @ByRef DoubleBlobVector bottom,
        double[] loss/*=NULL*/);
  /**
   * \brief Run forward using a serialized BlobProtoVector and return the
   *        result as a serialized BlobProtoVector
   */
  public native @StdString BytePointer Forward(@StdString BytePointer input_blob_protos, DoublePointer loss/*=NULL*/);
  public native @StdString BytePointer Forward(@StdString BytePointer input_blob_protos);
  public native @StdString String Forward(@StdString String input_blob_protos, DoubleBuffer loss/*=NULL*/);
  public native @StdString String Forward(@StdString String input_blob_protos);
  public native @StdString BytePointer Forward(@StdString BytePointer input_blob_protos, double[] loss/*=NULL*/);
  public native @StdString String Forward(@StdString String input_blob_protos, DoublePointer loss/*=NULL*/);
  public native @StdString BytePointer Forward(@StdString BytePointer input_blob_protos, DoubleBuffer loss/*=NULL*/);
  public native @StdString String Forward(@StdString String input_blob_protos, double[] loss/*=NULL*/);

  /**
   * \brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  public native void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  public native void Backward();
  public native void BackwardFromTo(int start, int end);
  public native void BackwardFrom(int start);
  public native void BackwardTo(int end);

  /**
   * \brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  public native void Reshape();

  public native double ForwardBackward(@Const @ByRef DoubleBlobVector bottom);

  /** \brief Updates the network weights based on the diff values computed. */
  public native void Update();
  /**
   * \brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  public native void ShareWeights();

  /**
   * \brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  public native void ShareTrainedLayersWith(@Const DoubleNet other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * \brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  public native void CopyTrainedLayersFrom(@Const @ByRef NetParameter param);
  public native void CopyTrainedLayersFrom(@StdString BytePointer trained_filename);
  public native void CopyTrainedLayersFrom(@StdString String trained_filename);
  public native void CopyTrainedLayersFromBinaryProto(@StdString BytePointer trained_filename);
  public native void CopyTrainedLayersFromBinaryProto(@StdString String trained_filename);
  public native void CopyTrainedLayersFromHDF5(@StdString BytePointer trained_filename);
  public native void CopyTrainedLayersFromHDF5(@StdString String trained_filename);
  /** \brief Writes the net to a proto. */
  public native void ToProto(NetParameter param, @Cast("bool") boolean write_diff/*=false*/);
  public native void ToProto(NetParameter param);
  /** \brief Writes the net to an HDF5 file. */
  public native void ToHDF5(@StdString BytePointer filename, @Cast("bool") boolean write_diff/*=false*/);
  public native void ToHDF5(@StdString BytePointer filename);
  public native void ToHDF5(@StdString String filename, @Cast("bool") boolean write_diff/*=false*/);
  public native void ToHDF5(@StdString String filename);

  /** \brief returns the network name. */
  public native @StdString BytePointer name();
  /** \brief returns the layer names */
  public native @Const @ByRef StringVector layer_names();
  /** \brief returns the blob names */
  public native @Const @ByRef StringVector blob_names();
  /** \brief returns the blobs */
  public native @Const @ByRef DoubleBlobSharedVector blobs();
  /** \brief returns the layers */
  public native @Const @ByRef DoubleLayerSharedVector layers();
  /** \brief returns the phase: TRAIN or TEST */
  public native @Cast("caffe::Phase") int phase();
  /**
   * \brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  public native @Const @ByRef DoubleBlobVectorVector bottom_vecs();
  /**
   * \brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  public native @Const @ByRef DoubleBlobVectorVector top_vecs();
  public native @Const @ByRef BoolVectorVector bottom_need_backward();
  public native @StdVector DoublePointer blob_loss_weights();
  public native @Const @ByRef BoolVector layer_need_backward();
  /** \brief returns the parameters */
  public native @Const @ByRef DoubleBlobSharedVector params();
  public native @Const @ByRef DoubleBlobVector learnable_params();
  /** \brief returns the learnable parameter learning rate multipliers */
  public native @StdVector FloatPointer params_lr();
  public native @Const @ByRef BoolVector has_params_lr();
  /** \brief returns the learnable parameter decay multipliers */
  public native @StdVector FloatPointer params_weight_decay();
  public native @Const @ByRef BoolVector has_params_decay();
  public native @Const @ByRef StringIntMap param_names_index();
  public native @StdVector IntPointer param_owners();
  /** \brief Input and output blob numbers */
  public native int num_inputs();
  public native int num_outputs();
  public native @Const @ByRef DoubleBlobVector input_blobs();
  public native @Const @ByRef DoubleBlobVector output_blobs();
  public native @StdVector IntPointer input_blob_indices();
  public native @StdVector IntPointer output_blob_indices();
  public native @Cast("bool") boolean has_blob(@StdString BytePointer blob_name);
  public native @Cast("bool") boolean has_blob(@StdString String blob_name);
  public native @Const @SharedPtr @ByVal DoubleBlob blob_by_name(@StdString BytePointer blob_name);
  public native @Const @SharedPtr @ByVal DoubleBlob blob_by_name(@StdString String blob_name);
  public native @Cast("bool") boolean has_layer(@StdString BytePointer layer_name);
  public native @Cast("bool") boolean has_layer(@StdString String layer_name);
  public DoubleLayer layer_by_name(BytePointer layer_name) { return layer_by_name(DoubleLayer.class, layer_name); }
  public DoubleLayer layer_by_name(String layer_name) { return layer_by_name(DoubleLayer.class, layer_name); };
  public native @Const @Cast({"", "boost::shared_ptr<caffe::Layer<double> >"}) @SharedPtr @ByVal <L extends DoubleLayer> L layer_by_name(Class<L> cls, @StdString BytePointer layer_name);
  public native @Const @Cast({"", "boost::shared_ptr<caffe::Layer<double> >"}) @SharedPtr @ByVal <L extends DoubleLayer> L layer_by_name(Class<L> cls, @StdString String layer_name);

  public native void set_debug_info(@Cast("const bool") boolean value);

  // Helpers for Init.
  /**
   * \brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  public static native void FilterNet(@Const @ByRef NetParameter param,
        NetParameter param_filtered);
  /** \brief return whether NetState state meets NetStateRule rule */
  public static native @Cast("bool") boolean StateMeetsRule(@Const @ByRef NetState state, @Const @ByRef NetStateRule rule,
        @StdString BytePointer layer_name);
  public static native @Cast("bool") boolean StateMeetsRule(@Const @ByRef NetState state, @Const @ByRef NetStateRule rule,
        @StdString String layer_name);
}


  // namespace caffe

// #endif  // CAFFE_NET_HPP_


// Parsed from caffe/parallel.hpp

// #ifndef CAFFE_PARALLEL_HPP_
// #define CAFFE_PARALLEL_HPP_

// #include <boost/date_time/posix_time/posix_time.hpp>

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/internal_thread.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/solver.hpp"
// #include "caffe/syncedmem.hpp"
// #include "caffe/util/blocking_queue.hpp"

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.

// Params stored in GPU memory.

@Namespace("caffe") @NoOffset public static class DevicePair extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DevicePair(Pointer p) { super(p); }

  public DevicePair(int parent, int device) { super((Pointer)null); allocate(parent, device); }
  private native void allocate(int parent, int device);
  public native int parent();
  public native int device();

  // Group GPUs in pairs, by proximity depending on machine's topology
  public static native void compute(@StdVector IntPointer devices, @StdVector DevicePair pairs);
  public static native void compute(@StdVector IntBuffer devices, @StdVector DevicePair pairs);
  public static native void compute(@StdVector int[] devices, @StdVector DevicePair pairs);
}

// Synchronous data parallelism using map-reduce between local GPUs.

  // namespace caffe

// #endif


// Parsed from caffe/solver.hpp

// #ifndef CAFFE_SOLVER_HPP_
// #define CAFFE_SOLVER_HPP_
// #include <boost/function.hpp>
// #include <string>
// #include <vector>

// #include "caffe/net.hpp"
// #include "caffe/solver_factory.hpp"

/**
  * \brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
    /** enum caffe::SolverAction::Enum */
    public static final int
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2;  // Take a snapshot, and keep training.
  

/**
 * \brief Type of a function that returns a Solver Action enumeration.
 */

/**
 * \brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
@Name("caffe::Solver<float>") @NoOffset public static class FloatSolver extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSolver(Pointer p) { super(p); }

  public native void Init(@Const @ByRef SolverParameter param);
  public native void InitTrainNet();
  public native void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  public native void SetActionFunction(@ByVal ActionCallback func);
  public native @Cast("caffe::SolverAction::Enum") int GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  public native void Solve(@Cast("const char*") BytePointer resume_file/*=NULL*/);
  public native void Solve();
  public native void Solve(String resume_file/*=NULL*/);
  public native void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  public native void Restore(@Cast("const char*") BytePointer resume_file);
  public native void Restore(String resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  public native void Snapshot();
  public native @Const @ByRef SolverParameter param();
  public native @SharedPtr @ByVal FloatNet net();
  public native @Const @ByRef FloatNetSharedVector test_nets();
  public native int iter();

  // Invoked at specific points during an iteration
  public static class Callback extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public Callback(Pointer p) { super(p); }
  
  }
  public native @Const @ByRef FloatCallbackVector callbacks();
  public native void add_callback(Callback value);

  public native void CheckSnapshotWritePermissions();
  /**
   * \brief Returns the solver type.
   */
  public native @Cast("const char*") BytePointer type();
}
@Name("caffe::Solver<double>") @NoOffset public static class DoubleSolver extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSolver(Pointer p) { super(p); }

  public native void Init(@Const @ByRef SolverParameter param);
  public native void InitTrainNet();
  public native void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  public native void SetActionFunction(@ByVal ActionCallback func);
  public native @Cast("caffe::SolverAction::Enum") int GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  public native void Solve(@Cast("const char*") BytePointer resume_file/*=NULL*/);
  public native void Solve();
  public native void Solve(String resume_file/*=NULL*/);
  public native void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  public native void Restore(@Cast("const char*") BytePointer resume_file);
  public native void Restore(String resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  public native void Snapshot();
  public native @Const @ByRef SolverParameter param();
  public native @SharedPtr @ByVal DoubleNet net();
  public native @Const @ByRef DoubleNetSharedVector test_nets();
  public native int iter();

  // Invoked at specific points during an iteration
  public static class Callback extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public Callback(Pointer p) { super(p); }
  
  }
  public native @Const @ByRef DoubleCallbackVector callbacks();
  public native void add_callback(Callback value);

  public native void CheckSnapshotWritePermissions();
  /**
   * \brief Returns the solver type.
   */
  public native @Cast("const char*") BytePointer type();
}

/**
 * \brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
 */
@Name("caffe::WorkerSolver<float>") public static class FloatWorkerSolver extends FloatSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatWorkerSolver(Pointer p) { super(p); }

  public FloatWorkerSolver(@Const @ByRef SolverParameter param,
        @Const FloatSolver root_solver/*=NULL*/) { super((Pointer)null); allocate(param, root_solver); }
  private native void allocate(@Const @ByRef SolverParameter param,
        @Const FloatSolver root_solver/*=NULL*/);
  public FloatWorkerSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
}
@Name("caffe::WorkerSolver<double>") public static class DoubleWorkerSolver extends DoubleSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleWorkerSolver(Pointer p) { super(p); }

  public DoubleWorkerSolver(@Const @ByRef SolverParameter param,
        @Const DoubleSolver root_solver/*=NULL*/) { super((Pointer)null); allocate(param, root_solver); }
  private native void allocate(@Const @ByRef SolverParameter param,
        @Const DoubleSolver root_solver/*=NULL*/);
  public DoubleWorkerSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
}

  // namespace caffe

// #endif  // CAFFE_SOLVER_HPP_


// Parsed from caffe/solver_factory.hpp

/**
 * \brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

// #ifndef CAFFE_SOLVER_FACTORY_H_
// #define CAFFE_SOLVER_FACTORY_H_

// #include <map>
// #include <string>
// #include <vector>

// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"

@Name("caffe::SolverRegistry<float>") public static class FloatSolverRegistry extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSolverRegistry(Pointer p) { super(p); }

  public static class Creator extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Creator(Pointer p) { super(p); }
      protected Creator() { allocate(); }
      private native void allocate();
      public native FloatSolver call(@Const @ByRef SolverParameter arg0);
  }

  public static native @Cast("caffe::SolverRegistry<float>::CreatorRegistry*") @ByRef FloatRegistry Registry();

  // Adds a creator.
  public static native void AddCreator(@StdString BytePointer type, Creator creator);
  public static native void AddCreator(@StdString String type, Creator creator);

  // Get a solver using a SolverParameter.
  public static native FloatSolver CreateSolver(@Const @ByRef SolverParameter param);

  public static native @ByVal StringVector SolverTypeList();
}

@Name("caffe::SolverRegistry<double>") public static class DoubleSolverRegistry extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSolverRegistry(Pointer p) { super(p); }

  public static class Creator extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Creator(Pointer p) { super(p); }
      protected Creator() { allocate(); }
      private native void allocate();
      public native DoubleSolver call(@Const @ByRef SolverParameter arg0);
  }

  public static native @Cast("caffe::SolverRegistry<double>::CreatorRegistry*") @ByRef FloatRegistry Registry();

  // Adds a creator.
  public static native void AddCreator(@StdString BytePointer type, Creator creator);
  public static native void AddCreator(@StdString String type, Creator creator);

  // Get a solver using a SolverParameter.
  public static native DoubleSolver CreateSolver(@Const @ByRef SolverParameter param);

  public static native @ByVal StringVector SolverTypeList();
}


@Name("caffe::SolverRegisterer<float>") public static class FloatSolverRegisterer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSolverRegisterer(Pointer p) { super(p); }

  public static class Creator_SolverParameter extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Creator_SolverParameter(Pointer p) { super(p); }
      protected Creator_SolverParameter() { allocate(); }
      private native void allocate();
      public native FloatSolver call(@Const @ByRef SolverParameter arg0);
  }
  public FloatSolverRegisterer(@StdString BytePointer type,
        Creator_SolverParameter creator) { super((Pointer)null); allocate(type, creator); }
  private native void allocate(@StdString BytePointer type,
        Creator_SolverParameter creator);
  public FloatSolverRegisterer(@StdString String type,
        Creator_SolverParameter creator) { super((Pointer)null); allocate(type, creator); }
  private native void allocate(@StdString String type,
        Creator_SolverParameter creator);
}


@Name("caffe::SolverRegisterer<double>") public static class DoubleSolverRegisterer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSolverRegisterer(Pointer p) { super(p); }

  public static class Creator_SolverParameter extends FunctionPointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public    Creator_SolverParameter(Pointer p) { super(p); }
      protected Creator_SolverParameter() { allocate(); }
      private native void allocate();
      public native DoubleSolver call(@Const @ByRef SolverParameter arg0);
  }
  public DoubleSolverRegisterer(@StdString BytePointer type,
        Creator_SolverParameter creator) { super((Pointer)null); allocate(type, creator); }
  private native void allocate(@StdString BytePointer type,
        Creator_SolverParameter creator);
  public DoubleSolverRegisterer(@StdString String type,
        Creator_SolverParameter creator) { super((Pointer)null); allocate(type, creator); }
  private native void allocate(@StdString String type,
        Creator_SolverParameter creator);
}


// #define REGISTER_SOLVER_CREATOR(type, creator)
//   static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);
//   static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   

// #define REGISTER_SOLVER_CLASS(type)
//   template <typename Dtype>
//   Solver<Dtype>* Creator_##type##Solver(
//       const SolverParameter& param)
//   {
//     return new type##Solver<Dtype>(param);
//   }
//   REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

  // namespace caffe

// #endif  // CAFFE_SOLVER_FACTORY_H_


// Parsed from caffe/sgd_solvers.hpp

// #ifndef CAFFE_SGD_SOLVERS_HPP_
// #define CAFFE_SGD_SOLVERS_HPP_

// #include <string>
// #include <vector>

// #include "caffe/solver.hpp"

/**
 * \brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
@Name("caffe::SGDSolver<float>") @NoOffset public static class FloatSGDSolver extends FloatSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatSGDSolver(Pointer p) { super(p); }

  public FloatSGDSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public FloatSGDSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public FloatSGDSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();

  public native @Const @ByRef FloatBlobSharedVector history();
}
@Name("caffe::SGDSolver<double>") @NoOffset public static class DoubleSGDSolver extends DoubleSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleSGDSolver(Pointer p) { super(p); }

  public DoubleSGDSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public DoubleSGDSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public DoubleSGDSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();

  public native @Const @ByRef DoubleBlobSharedVector history();
}

@Name("caffe::NesterovSolver<float>") public static class FloatNesterovSolver extends FloatSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatNesterovSolver(Pointer p) { super(p); }

  public FloatNesterovSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public FloatNesterovSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public FloatNesterovSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}

@Name("caffe::NesterovSolver<double>") public static class DoubleNesterovSolver extends DoubleSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleNesterovSolver(Pointer p) { super(p); }

  public DoubleNesterovSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public DoubleNesterovSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public DoubleNesterovSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}

@Name("caffe::AdaGradSolver<float>") public static class FloatAdaGradSolver extends FloatSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatAdaGradSolver(Pointer p) { super(p); }

  public FloatAdaGradSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public FloatAdaGradSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public FloatAdaGradSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}

@Name("caffe::AdaGradSolver<double>") public static class DoubleAdaGradSolver extends DoubleSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleAdaGradSolver(Pointer p) { super(p); }

  public DoubleAdaGradSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public DoubleAdaGradSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public DoubleAdaGradSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}


@Name("caffe::RMSPropSolver<float>") public static class FloatRMSPropSolver extends FloatSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatRMSPropSolver(Pointer p) { super(p); }

  public FloatRMSPropSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public FloatRMSPropSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public FloatRMSPropSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}


@Name("caffe::RMSPropSolver<double>") public static class DoubleRMSPropSolver extends DoubleSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleRMSPropSolver(Pointer p) { super(p); }

  public DoubleRMSPropSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public DoubleRMSPropSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public DoubleRMSPropSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}

@Name("caffe::AdaDeltaSolver<float>") public static class FloatAdaDeltaSolver extends FloatSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatAdaDeltaSolver(Pointer p) { super(p); }

  public FloatAdaDeltaSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public FloatAdaDeltaSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public FloatAdaDeltaSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}

@Name("caffe::AdaDeltaSolver<double>") public static class DoubleAdaDeltaSolver extends DoubleSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleAdaDeltaSolver(Pointer p) { super(p); }

  public DoubleAdaDeltaSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public DoubleAdaDeltaSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public DoubleAdaDeltaSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}

/**
 * \brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
@Name("caffe::AdamSolver<float>") public static class FloatAdamSolver extends FloatSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatAdamSolver(Pointer p) { super(p); }

  public FloatAdamSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public FloatAdamSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public FloatAdamSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}
@Name("caffe::AdamSolver<double>") public static class DoubleAdamSolver extends DoubleSGDSolver {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleAdamSolver(Pointer p) { super(p); }

  public DoubleAdamSolver(@Const @ByRef SolverParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef SolverParameter param);
  public DoubleAdamSolver(@StdString BytePointer param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString BytePointer param_file);
  public DoubleAdamSolver(@StdString String param_file) { super((Pointer)null); allocate(param_file); }
  private native void allocate(@StdString String param_file);
  public native @Cast("const char*") BytePointer type();
}

  // namespace caffe

// #endif  // CAFFE_SGD_SOLVERS_HPP_


// Parsed from caffe/layers/base_conv_layer.hpp

// #ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
// #define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/im2col.hpp"

/**
 * \brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
@Name("caffe::BaseConvolutionLayer<float>") @NoOffset public static class FloatBaseConvolutionLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatBaseConvolutionLayer(Pointer p) { super(p); }

  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native int MinBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual public native @Cast("bool") boolean EqualNumBottomTopBlobs();
  @Virtual(true) protected native @Cast("bool") boolean reverse_dimensions();
  @Virtual(true) protected native void compute_output_shape();
}
@Name("caffe::BaseConvolutionLayer<double>") @NoOffset public static class DoubleBaseConvolutionLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleBaseConvolutionLayer(Pointer p) { super(p); }

  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native int MinBottomBlobs();
  @Virtual public native int MinTopBlobs();
  @Virtual public native @Cast("bool") boolean EqualNumBottomTopBlobs();
  @Virtual(true) protected native @Cast("bool") boolean reverse_dimensions();
  @Virtual(true) protected native void compute_output_shape();
}

  // namespace caffe

// #endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_


// Parsed from caffe/layers/conv_layer.hpp

// #ifndef CAFFE_CONV_LAYER_HPP_
// #define CAFFE_CONV_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/base_conv_layer.hpp"

/**
 * \brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
@Name("caffe::ConvolutionLayer<float>") public static class FloatConvolutionLayer extends FloatBaseConvolutionLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatConvolutionLayer(Pointer p) { super(p); }

  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group \f$ \geq 1 \f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  public FloatConvolutionLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native @Cast("bool") boolean reverse_dimensions();
  @Virtual protected native void compute_output_shape();
}
@Name("caffe::ConvolutionLayer<double>") public static class DoubleConvolutionLayer extends DoubleBaseConvolutionLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleConvolutionLayer(Pointer p) { super(p); }

  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group \f$ \geq 1 \f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  public DoubleConvolutionLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native @Cast("bool") boolean reverse_dimensions();
  @Virtual protected native void compute_output_shape();
}

  // namespace caffe

// #endif  // CAFFE_CONV_LAYER_HPP_


// Parsed from caffe/layers/deconv_layer.hpp

// #ifndef CAFFE_DECONV_LAYER_HPP_
// #define CAFFE_DECONV_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/base_conv_layer.hpp"

/**
 * \brief Convolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and convolution parameters in the
 *        opposite sense as ConvolutionLayer.
 *
 *   ConvolutionLayer computes each output value by dotting an input window with
 *   a filter; DeconvolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
 *   reversed. DeconvolutionLayer reuses ConvolutionParameter for its
 *   parameters, but they take the opposite sense as in ConvolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
@Name("caffe::DeconvolutionLayer<float>") public static class FloatDeconvolutionLayer extends FloatBaseConvolutionLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatDeconvolutionLayer(Pointer p) { super(p); }

  public FloatDeconvolutionLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native @Cast("bool") boolean reverse_dimensions();
  @Virtual protected native void compute_output_shape();
}
@Name("caffe::DeconvolutionLayer<double>") public static class DoubleDeconvolutionLayer extends DoubleBaseConvolutionLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleDeconvolutionLayer(Pointer p) { super(p); }

  public DoubleDeconvolutionLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native @Cast("bool") boolean reverse_dimensions();
  @Virtual protected native void compute_output_shape();
}

  // namespace caffe

// #endif  // CAFFE_DECONV_LAYER_HPP_


// Parsed from caffe/layers/cudnn_conv_layer.hpp

// #ifndef CAFFE_CUDNN_CONV_LAYER_HPP_
// #define CAFFE_CUDNN_CONV_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/conv_layer.hpp"

// #ifdef USE_CUDNN
// #endif

  // namespace caffe

// #endif  // CAFFE_CUDNN_CONV_LAYER_HPP_


// Parsed from caffe/layers/im2col_layer.hpp

// #ifndef CAFFE_IM2COL_LAYER_HPP_
// #define CAFFE_IM2COL_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief A helper for image operations that rearranges image regions into
 *        column vectors.  Used by ConvolutionLayer to perform convolution
 *        by matrix multiplication.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::Im2colLayer<float>") @NoOffset public static class FloatIm2colLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatIm2colLayer(Pointer p) { super(p); }

  public FloatIm2colLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::Im2colLayer<double>") @NoOffset public static class DoubleIm2colLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleIm2colLayer(Pointer p) { super(p); }

  public DoubleIm2colLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_IM2COL_LAYER_HPP_


// Parsed from caffe/layers/lrn_layer.hpp

// #ifndef CAFFE_LRN_LAYER_HPP_
// #define CAFFE_LRN_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/eltwise_layer.hpp"
// #include "caffe/layers/pooling_layer.hpp"
// #include "caffe/layers/power_layer.hpp"
// #include "caffe/layers/split_layer.hpp"

/**
 * \brief Normalize the input in a local region across or within feature maps.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::LRNLayer<float>") @NoOffset public static class FloatLRNLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatLRNLayer(Pointer p) { super(p); }

  public FloatLRNLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);

  @Virtual protected native void CrossChannelForward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void CrossChannelForward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void WithinChannelForward(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void CrossChannelBackward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void CrossChannelBackward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void WithinChannelBackward(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::LRNLayer<double>") @NoOffset public static class DoubleLRNLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoubleLRNLayer(Pointer p) { super(p); }

  public DoubleLRNLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int ExactNumTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);

  @Virtual protected native void CrossChannelForward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void CrossChannelForward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void WithinChannelForward(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void CrossChannelBackward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void CrossChannelBackward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void WithinChannelBackward(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_LRN_LAYER_HPP_


// Parsed from caffe/layers/cudnn_lrn_layer.hpp

// #ifndef CAFFE_CUDNN_LRN_LAYER_HPP_
// #define CAFFE_CUDNN_LRN_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/lrn_layer.hpp"

// #ifdef USE_CUDNN
// #endif

  // namespace caffe

// #endif  // CAFFE_CUDNN_LRN_LAYER_HPP_


// Parsed from caffe/layers/cudnn_lcn_layer.hpp

// #ifndef CAFFE_CUDNN_LCN_LAYER_HPP_
// #define CAFFE_CUDNN_LCN_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/lrn_layer.hpp"
// #include "caffe/layers/power_layer.hpp"

// #ifdef USE_CUDNN
// #endif

  // namespace caffe

// #endif  // CAFFE_CUDNN_LCN_LAYER_HPP_


// Parsed from caffe/layers/pooling_layer.hpp

// #ifndef CAFFE_POOLING_LAYER_HPP_
// #define CAFFE_POOLING_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
@Name("caffe::PoolingLayer<float>") @NoOffset public static class FloatPoolingLayer extends FloatLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FloatPoolingLayer(Pointer p) { super(p); }

  public FloatPoolingLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  @Virtual public native int MaxTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef FloatBlobVector bottom,
        @Const @ByRef FloatBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef FloatBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef FloatBlobVector bottom);
}
@Name("caffe::PoolingLayer<double>") @NoOffset public static class DoublePoolingLayer extends DoubleLayer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DoublePoolingLayer(Pointer p) { super(p); }

  public DoublePoolingLayer(@Const @ByRef LayerParameter param) { super((Pointer)null); allocate(param); }
  private native void allocate(@Const @ByRef LayerParameter param);
  @Virtual public native void LayerSetUp(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual public native void Reshape(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);

  @Virtual public native @Cast("const char*") BytePointer type();
  @Virtual public native int ExactNumBottomBlobs();
  @Virtual public native int MinTopBlobs();
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  @Virtual public native int MaxTopBlobs();
  @Virtual protected native void Forward_cpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Forward_gpu(@Const @ByRef DoubleBlobVector bottom,
        @Const @ByRef DoubleBlobVector top);
  @Virtual protected native void Backward_cpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
  @Virtual protected native void Backward_gpu(@Const @ByRef DoubleBlobVector top,
        @Const @ByRef BoolVector propagate_down, @Const @ByRef DoubleBlobVector bottom);
}

  // namespace caffe

// #endif  // CAFFE_POOLING_LAYER_HPP_


// Parsed from caffe/layers/cudnn_pooling_layer.hpp

// #ifndef CAFFE_CUDNN_POOLING_LAYER_HPP_
// #define CAFFE_CUDNN_POOLING_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/pooling_layer.hpp"

// #ifdef USE_CUDNN
// #endif

  // namespace caffe

// #endif  // CAFFE_CUDNN_POOLING_LAYER_HPP_


// Parsed from caffe/layers/spp_layer.hpp

// #ifndef CAFFE_SPP_LAYER_HPP_
// #define CAFFE_SPP_LAYER_HPP_

// #include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"

/**
 * \brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */

  // namespace caffe

// #endif  // CAFFE_SPP_LAYER_HPP_


// Parsed from caffe/util/benchmark.hpp

// #ifndef CAFFE_UTIL_BENCHMARK_H_
// #define CAFFE_UTIL_BENCHMARK_H_

// #include <boost/date_time/posix_time/posix_time.hpp>

// #include "caffe/util/device_alternate.hpp"

@Namespace("caffe") @NoOffset public static class Timer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Timer(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Timer(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public Timer position(int position) {
        return (Timer)super.position(position);
    }

  public Timer() { super((Pointer)null); allocate(); }
  private native void allocate();
  public native void Start();
  public native void Stop();
  public native float MilliSeconds();
  public native float MicroSeconds();
  public native float Seconds();

  public native @Cast("bool") boolean initted();
  public native @Cast("bool") boolean running();
  public native @Cast("bool") boolean has_run_at_least_once();
}

@Namespace("caffe") public static class CPUTimer extends Timer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CPUTimer(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public CPUTimer(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public CPUTimer position(int position) {
        return (CPUTimer)super.position(position);
    }

  public CPUTimer() { super((Pointer)null); allocate(); }
  private native void allocate();
  public native void Start();
  public native void Stop();
  public native float MilliSeconds();
  public native float MicroSeconds();
}

  // namespace caffe

// #endif   // CAFFE_UTIL_BENCHMARK_H_


// Parsed from caffe/util/db.hpp

// #ifndef CAFFE_UTIL_DB_HPP
// #define CAFFE_UTIL_DB_HPP

// #include <string>

// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"

/** enum caffe::db::Mode */
public static final int READ = 0, WRITE = 1, NEW = 2;

@Namespace("caffe::db") public static class Cursor extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Cursor(Pointer p) { super(p); }

  public native void SeekToFirst();
  public native void Next();
  public native @StdString BytePointer key();
  public native @StdString BytePointer value();
  public native @Cast("bool") boolean valid();
}

@Namespace("caffe::db") public static class Transaction extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Transaction(Pointer p) { super(p); }

  public native void Put(@StdString BytePointer key, @StdString BytePointer value);
  public native void Put(@StdString String key, @StdString String value);
  public native void Commit();
}

@Namespace("caffe::db") public static class DB extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DB(Pointer p) { super(p); }

  public native void Open(@StdString BytePointer source, @Cast("caffe::db::Mode") int mode);
  public native void Open(@StdString String source, @Cast("caffe::db::Mode") int mode);
  public native void Close();
  public native Cursor NewCursor();
  public native Transaction NewTransaction();
}

@Namespace("caffe::db") public static native DB GetDB(@Cast("caffe::DataParameter::DB") int backend);
@Namespace("caffe::db") public static native DB GetDB(@StdString BytePointer backend);
@Namespace("caffe::db") public static native DB GetDB(@StdString String backend);

  // namespace db
  // namespace caffe

// #endif  // CAFFE_UTIL_DB_HPP


// Parsed from caffe/util/db_leveldb.hpp

// #ifdef USE_LEVELDB
// #ifndef CAFFE_UTIL_DB_LEVELDB_HPP
// #define CAFFE_UTIL_DB_LEVELDB_HPP

// #include <string>

// #include "leveldb/db.h"
// #include "leveldb/write_batch.h"

// #include "caffe/util/db.hpp"

@Namespace("caffe::db") @NoOffset public static class LevelDBCursor extends Cursor {
    static { Loader.load(); }

  public LevelDBCursor(@Cast("leveldb::Iterator*") Pointer iter) { super((Pointer)null); allocate(iter); }
  private native void allocate(@Cast("leveldb::Iterator*") Pointer iter);
  public native void SeekToFirst();
  public native void Next();
  public native @StdString BytePointer key();
  public native @StdString BytePointer value();
  public native @Cast("bool") boolean valid();
}

@Namespace("caffe::db") @NoOffset public static class LevelDBTransaction extends Transaction {
    static { Loader.load(); }

  public LevelDBTransaction(@Cast("leveldb::DB*") Pointer db) { super((Pointer)null); allocate(db); }
  private native void allocate(@Cast("leveldb::DB*") Pointer db);
  public native void Put(@StdString BytePointer key, @StdString BytePointer value);
  public native void Put(@StdString String key, @StdString String value);
  public native void Commit();
}

@Namespace("caffe::db") @NoOffset public static class LevelDB extends DB {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LevelDB(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LevelDB(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public LevelDB position(int position) {
        return (LevelDB)super.position(position);
    }

  public LevelDB() { super((Pointer)null); allocate(); }
  private native void allocate();
  public native void Open(@StdString BytePointer source, @Cast("caffe::db::Mode") int mode);
  public native void Open(@StdString String source, @Cast("caffe::db::Mode") int mode);
  public native void Close();
  public native LevelDBCursor NewCursor();
  public native LevelDBTransaction NewTransaction();
}


  // namespace db
  // namespace caffe

// #endif  // CAFFE_UTIL_DB_LEVELDB_HPP
// #endif  // USE_LEVELDB


// Parsed from caffe/util/db_lmdb.hpp

// #ifdef USE_LMDB
// #ifndef CAFFE_UTIL_DB_LMDB_HPP
// #define CAFFE_UTIL_DB_LMDB_HPP

// #include <string>

// #include "lmdb.h"

// #include "caffe/util/db.hpp"

@Namespace("caffe::db") public static native void MDB_CHECK(int mdb_status);

@Namespace("caffe::db") @NoOffset public static class LMDBCursor extends Cursor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LMDBCursor(Pointer p) { super(p); }

  public LMDBCursor(@Cast("MDB_txn*") Pointer mdb_txn, @Cast("MDB_cursor*") Pointer mdb_cursor) { super((Pointer)null); allocate(mdb_txn, mdb_cursor); }
  private native void allocate(@Cast("MDB_txn*") Pointer mdb_txn, @Cast("MDB_cursor*") Pointer mdb_cursor);
  public native void SeekToFirst();
  public native void Next();
  public native @StdString BytePointer key();
  public native @StdString BytePointer value();
  public native @Cast("bool") boolean valid();
}

@Namespace("caffe::db") @NoOffset public static class LMDBTransaction extends Transaction {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LMDBTransaction(Pointer p) { super(p); }

  public LMDBTransaction(@Cast("MDB_dbi*") Pointer mdb_dbi, @Cast("MDB_txn*") Pointer mdb_txn) { super((Pointer)null); allocate(mdb_dbi, mdb_txn); }
  private native void allocate(@Cast("MDB_dbi*") Pointer mdb_dbi, @Cast("MDB_txn*") Pointer mdb_txn);
  public native void Put(@StdString BytePointer key, @StdString BytePointer value);
  public native void Put(@StdString String key, @StdString String value);
  public native void Commit();
}

@Namespace("caffe::db") @NoOffset public static class LMDB extends DB {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LMDB(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public LMDB(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public LMDB position(int position) {
        return (LMDB)super.position(position);
    }

  public LMDB() { super((Pointer)null); allocate(); }
  private native void allocate();
  public native void Open(@StdString BytePointer source, @Cast("caffe::db::Mode") int mode);
  public native void Open(@StdString String source, @Cast("caffe::db::Mode") int mode);
  public native void Close();
  public native LMDBCursor NewCursor();
  public native LMDBTransaction NewTransaction();
}

  // namespace db
  // namespace caffe

// #endif  // CAFFE_UTIL_DB_LMDB_HPP
// #endif  // USE_LMDB


// Parsed from caffe/util/io.hpp

// #ifndef CAFFE_UTIL_IO_H_
// #define CAFFE_UTIL_IO_H_

// #include <boost/filesystem.hpp>
// #include <iomanip>
// #include <iostream>  // NOLINT(readability/streams)
// #include <string>

// #include "google/protobuf/message.h"

// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/format.hpp"

// #ifndef CAFFE_TMP_DIR_RETRIES
public static final int CAFFE_TMP_DIR_RETRIES = 100;
// #endif

@Namespace("caffe") public static native void MakeTempDir(@StdString @Cast({"char*", "std::string*"}) BytePointer temp_dirname);

@Namespace("caffe") public static native void MakeTempFilename(@StdString @Cast({"char*", "std::string*"}) BytePointer temp_filename);

@Namespace("caffe") public static native @Cast("bool") boolean ReadProtoFromTextFile(@Cast("const char*") BytePointer filename, @Cast("google::protobuf::Message*") Pointer proto);
@Namespace("caffe") public static native @Cast("bool") boolean ReadProtoFromTextFile(String filename, @Cast("google::protobuf::Message*") Pointer proto);

@Namespace("caffe") public static native void ReadProtoFromTextFileOrDie(@Cast("const char*") BytePointer filename, @Cast("google::protobuf::Message*") Pointer proto);
@Namespace("caffe") public static native void ReadProtoFromTextFileOrDie(String filename, @Cast("google::protobuf::Message*") Pointer proto);

@Namespace("caffe") public static native void WriteProtoToTextFile(@Cast("const google::protobuf::Message*") @ByRef Pointer proto, @Cast("const char*") BytePointer filename);
@Namespace("caffe") public static native void WriteProtoToTextFile(@Cast("const google::protobuf::Message*") @ByRef Pointer proto, String filename);

@Namespace("caffe") public static native @Cast("bool") boolean ReadProtoFromBinaryFile(@Cast("const char*") BytePointer filename, @Cast("google::protobuf::Message*") Pointer proto);
@Namespace("caffe") public static native @Cast("bool") boolean ReadProtoFromBinaryFile(String filename, @Cast("google::protobuf::Message*") Pointer proto);

@Namespace("caffe") public static native void ReadProtoFromBinaryFileOrDie(@Cast("const char*") BytePointer filename, @Cast("google::protobuf::Message*") Pointer proto);
@Namespace("caffe") public static native void ReadProtoFromBinaryFileOrDie(String filename, @Cast("google::protobuf::Message*") Pointer proto);


@Namespace("caffe") public static native void WriteProtoToBinaryFile(@Cast("const google::protobuf::Message*") @ByRef Pointer proto, @Cast("const char*") BytePointer filename);
@Namespace("caffe") public static native void WriteProtoToBinaryFile(@Cast("const google::protobuf::Message*") @ByRef Pointer proto, String filename);

@Namespace("caffe") public static native @Cast("bool") boolean ReadFileToDatum(@StdString BytePointer filename, int label, Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean ReadFileToDatum(@StdString String filename, int label, Datum datum);

@Namespace("caffe") public static native @Cast("bool") boolean ReadFileToDatum(@StdString BytePointer filename, Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean ReadFileToDatum(@StdString String filename, Datum datum);

@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString BytePointer filename, int label,
    int height, int width, @Cast("const bool") boolean is_color,
    @StdString BytePointer encoding, Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString String filename, int label,
    int height, int width, @Cast("const bool") boolean is_color,
    @StdString String encoding, Datum datum);

@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString BytePointer filename, int label,
    int height, int width, @Cast("const bool") boolean is_color, Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString String filename, int label,
    int height, int width, @Cast("const bool") boolean is_color, Datum datum);

@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString BytePointer filename, int label,
    int height, int width, Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString String filename, int label,
    int height, int width, Datum datum);

@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString BytePointer filename, int label,
    @Cast("const bool") boolean is_color, Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString String filename, int label,
    @Cast("const bool") boolean is_color, Datum datum);

@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString BytePointer filename, int label,
    Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString String filename, int label,
    Datum datum);

@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString BytePointer filename, int label,
    @StdString BytePointer encoding, Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean ReadImageToDatum(@StdString String filename, int label,
    @StdString String encoding, Datum datum);

@Namespace("caffe") public static native @Cast("bool") boolean DecodeDatumNative(Datum datum);
@Namespace("caffe") public static native @Cast("bool") boolean DecodeDatum(Datum datum, @Cast("bool") boolean is_color);

// #ifdef USE_OPENCV
@Namespace("caffe") public static native @ByVal Mat ReadImageToCVMat(@StdString BytePointer filename,
    int height, int width, @Cast("const bool") boolean is_color);
@Namespace("caffe") public static native @ByVal Mat ReadImageToCVMat(@StdString String filename,
    int height, int width, @Cast("const bool") boolean is_color);

@Namespace("caffe") public static native @ByVal Mat ReadImageToCVMat(@StdString BytePointer filename,
    int height, int width);
@Namespace("caffe") public static native @ByVal Mat ReadImageToCVMat(@StdString String filename,
    int height, int width);

@Namespace("caffe") public static native @ByVal Mat ReadImageToCVMat(@StdString BytePointer filename,
    @Cast("const bool") boolean is_color);
@Namespace("caffe") public static native @ByVal Mat ReadImageToCVMat(@StdString String filename,
    @Cast("const bool") boolean is_color);

@Namespace("caffe") public static native @ByVal Mat ReadImageToCVMat(@StdString BytePointer filename);
@Namespace("caffe") public static native @ByVal Mat ReadImageToCVMat(@StdString String filename);

@Namespace("caffe") public static native @ByVal Mat DecodeDatumToCVMatNative(@Const @ByRef Datum datum);
@Namespace("caffe") public static native @ByVal Mat DecodeDatumToCVMat(@Const @ByRef Datum datum, @Cast("bool") boolean is_color);

@Namespace("caffe") public static native void CVMatToDatum(@Const @ByRef Mat cv_img, Datum datum);
// #endif  // USE_OPENCV

  // namespace caffe

// #endif   // CAFFE_UTIL_IO_H_


// Parsed from caffe/util/rng.hpp

// #ifndef CAFFE_RNG_CPP_HPP_
// #define CAFFE_RNG_CPP_HPP_

// #include <algorithm>
// #include <iterator>

// #include "boost/random/mersenne_twister.hpp"
// #include "boost/random/uniform_int.hpp"

// #include "caffe/common.hpp"

@Namespace("caffe") public static native @Cast("caffe::rng_t*") Pointer caffe_rng();

// Fisher–Yates algorithm
  // namespace caffe

// #endif  // CAFFE_RNG_HPP_


// Parsed from caffe/util/im2col.hpp

// #ifndef _CAFFE_UTIL_IM2COL_HPP_
// #define _CAFFE_UTIL_IM2COL_HPP_

@Namespace("caffe") public static native @Name("im2col_nd_cpu<float>") void im2col_nd_cpu_float(@Const FloatPointer data_im, int num_spatial_axes,
    @Const IntPointer im_shape, @Const IntPointer col_shape,
    @Const IntPointer kernel_shape, @Const IntPointer pad, @Const IntPointer stride,
    FloatPointer data_col);
@Namespace("caffe") public static native @Name("im2col_nd_cpu<float>") void im2col_nd_cpu_float(@Const FloatBuffer data_im, int num_spatial_axes,
    @Const IntBuffer im_shape, @Const IntBuffer col_shape,
    @Const IntBuffer kernel_shape, @Const IntBuffer pad, @Const IntBuffer stride,
    FloatBuffer data_col);
@Namespace("caffe") public static native @Name("im2col_nd_cpu<float>") void im2col_nd_cpu_float(@Const float[] data_im, int num_spatial_axes,
    @Const int[] im_shape, @Const int[] col_shape,
    @Const int[] kernel_shape, @Const int[] pad, @Const int[] stride,
    float[] data_col);

@Namespace("caffe") public static native @Name("im2col_nd_cpu<double>") void im2col_nd_cpu_double(@Const DoublePointer data_im, int num_spatial_axes,
    @Const IntPointer im_shape, @Const IntPointer col_shape,
    @Const IntPointer kernel_shape, @Const IntPointer pad, @Const IntPointer stride,
    DoublePointer data_col);
@Namespace("caffe") public static native @Name("im2col_nd_cpu<double>") void im2col_nd_cpu_double(@Const DoubleBuffer data_im, int num_spatial_axes,
    @Const IntBuffer im_shape, @Const IntBuffer col_shape,
    @Const IntBuffer kernel_shape, @Const IntBuffer pad, @Const IntBuffer stride,
    DoubleBuffer data_col);
@Namespace("caffe") public static native @Name("im2col_nd_cpu<double>") void im2col_nd_cpu_double(@Const double[] data_im, int num_spatial_axes,
    @Const int[] im_shape, @Const int[] col_shape,
    @Const int[] kernel_shape, @Const int[] pad, @Const int[] stride,
    double[] data_col);

@Namespace("caffe") public static native @Name("im2col_cpu<float>") void im2col_cpu_float(@Const FloatPointer data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, FloatPointer data_col);
@Namespace("caffe") public static native @Name("im2col_cpu<float>") void im2col_cpu_float(@Const FloatBuffer data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, FloatBuffer data_col);
@Namespace("caffe") public static native @Name("im2col_cpu<float>") void im2col_cpu_float(@Const float[] data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, float[] data_col);

@Namespace("caffe") public static native @Name("im2col_cpu<double>") void im2col_cpu_double(@Const DoublePointer data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, DoublePointer data_col);
@Namespace("caffe") public static native @Name("im2col_cpu<double>") void im2col_cpu_double(@Const DoubleBuffer data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, DoubleBuffer data_col);
@Namespace("caffe") public static native @Name("im2col_cpu<double>") void im2col_cpu_double(@Const double[] data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, double[] data_col);

@Namespace("caffe") public static native @Name("col2im_nd_cpu<float>") void col2im_nd_cpu_float(@Const FloatPointer data_col, int num_spatial_axes,
    @Const IntPointer im_shape, @Const IntPointer col_shape,
    @Const IntPointer kernel_shape, @Const IntPointer pad, @Const IntPointer stride,
    FloatPointer data_im);
@Namespace("caffe") public static native @Name("col2im_nd_cpu<float>") void col2im_nd_cpu_float(@Const FloatBuffer data_col, int num_spatial_axes,
    @Const IntBuffer im_shape, @Const IntBuffer col_shape,
    @Const IntBuffer kernel_shape, @Const IntBuffer pad, @Const IntBuffer stride,
    FloatBuffer data_im);
@Namespace("caffe") public static native @Name("col2im_nd_cpu<float>") void col2im_nd_cpu_float(@Const float[] data_col, int num_spatial_axes,
    @Const int[] im_shape, @Const int[] col_shape,
    @Const int[] kernel_shape, @Const int[] pad, @Const int[] stride,
    float[] data_im);

@Namespace("caffe") public static native @Name("col2im_nd_cpu<double>") void col2im_nd_cpu_double(@Const DoublePointer data_col, int num_spatial_axes,
    @Const IntPointer im_shape, @Const IntPointer col_shape,
    @Const IntPointer kernel_shape, @Const IntPointer pad, @Const IntPointer stride,
    DoublePointer data_im);
@Namespace("caffe") public static native @Name("col2im_nd_cpu<double>") void col2im_nd_cpu_double(@Const DoubleBuffer data_col, int num_spatial_axes,
    @Const IntBuffer im_shape, @Const IntBuffer col_shape,
    @Const IntBuffer kernel_shape, @Const IntBuffer pad, @Const IntBuffer stride,
    DoubleBuffer data_im);
@Namespace("caffe") public static native @Name("col2im_nd_cpu<double>") void col2im_nd_cpu_double(@Const double[] data_col, int num_spatial_axes,
    @Const int[] im_shape, @Const int[] col_shape,
    @Const int[] kernel_shape, @Const int[] pad, @Const int[] stride,
    double[] data_im);

@Namespace("caffe") public static native @Name("col2im_cpu<float>") void col2im_cpu_float(@Const FloatPointer data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, FloatPointer data_im);
@Namespace("caffe") public static native @Name("col2im_cpu<float>") void col2im_cpu_float(@Const FloatBuffer data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, FloatBuffer data_im);
@Namespace("caffe") public static native @Name("col2im_cpu<float>") void col2im_cpu_float(@Const float[] data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, float[] data_im);

@Namespace("caffe") public static native @Name("col2im_cpu<double>") void col2im_cpu_double(@Const DoublePointer data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, DoublePointer data_im);
@Namespace("caffe") public static native @Name("col2im_cpu<double>") void col2im_cpu_double(@Const DoubleBuffer data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, DoubleBuffer data_im);
@Namespace("caffe") public static native @Name("col2im_cpu<double>") void col2im_cpu_double(@Const double[] data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, double[] data_im);

@Namespace("caffe") public static native @Name("im2col_nd_gpu<float>") void im2col_nd_gpu_float(@Const FloatPointer data_im, int num_spatial_axes,
    int col_size, @Const IntPointer im_shape, @Const IntPointer col_shape,
    @Const IntPointer kernel_shape, @Const IntPointer pad, @Const IntPointer stride,
    FloatPointer data_col);
@Namespace("caffe") public static native @Name("im2col_nd_gpu<float>") void im2col_nd_gpu_float(@Const FloatBuffer data_im, int num_spatial_axes,
    int col_size, @Const IntBuffer im_shape, @Const IntBuffer col_shape,
    @Const IntBuffer kernel_shape, @Const IntBuffer pad, @Const IntBuffer stride,
    FloatBuffer data_col);
@Namespace("caffe") public static native @Name("im2col_nd_gpu<float>") void im2col_nd_gpu_float(@Const float[] data_im, int num_spatial_axes,
    int col_size, @Const int[] im_shape, @Const int[] col_shape,
    @Const int[] kernel_shape, @Const int[] pad, @Const int[] stride,
    float[] data_col);

@Namespace("caffe") public static native @Name("im2col_nd_gpu<double>") void im2col_nd_gpu_double(@Const DoublePointer data_im, int num_spatial_axes,
    int col_size, @Const IntPointer im_shape, @Const IntPointer col_shape,
    @Const IntPointer kernel_shape, @Const IntPointer pad, @Const IntPointer stride,
    DoublePointer data_col);
@Namespace("caffe") public static native @Name("im2col_nd_gpu<double>") void im2col_nd_gpu_double(@Const DoubleBuffer data_im, int num_spatial_axes,
    int col_size, @Const IntBuffer im_shape, @Const IntBuffer col_shape,
    @Const IntBuffer kernel_shape, @Const IntBuffer pad, @Const IntBuffer stride,
    DoubleBuffer data_col);
@Namespace("caffe") public static native @Name("im2col_nd_gpu<double>") void im2col_nd_gpu_double(@Const double[] data_im, int num_spatial_axes,
    int col_size, @Const int[] im_shape, @Const int[] col_shape,
    @Const int[] kernel_shape, @Const int[] pad, @Const int[] stride,
    double[] data_col);

@Namespace("caffe") public static native @Name("im2col_gpu<float>") void im2col_gpu_float(@Const FloatPointer data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, FloatPointer data_col);
@Namespace("caffe") public static native @Name("im2col_gpu<float>") void im2col_gpu_float(@Const FloatBuffer data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, FloatBuffer data_col);
@Namespace("caffe") public static native @Name("im2col_gpu<float>") void im2col_gpu_float(@Const float[] data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, float[] data_col);

@Namespace("caffe") public static native @Name("im2col_gpu<double>") void im2col_gpu_double(@Const DoublePointer data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, DoublePointer data_col);
@Namespace("caffe") public static native @Name("im2col_gpu<double>") void im2col_gpu_double(@Const DoubleBuffer data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, DoubleBuffer data_col);
@Namespace("caffe") public static native @Name("im2col_gpu<double>") void im2col_gpu_double(@Const double[] data_im, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, double[] data_col);

@Namespace("caffe") public static native @Name("col2im_nd_gpu<float>") void col2im_nd_gpu_float(@Const FloatPointer data_col, int num_spatial_axes,
    int im_size, @Const IntPointer im_shape, @Const IntPointer col_shape,
    @Const IntPointer kernel_shape, @Const IntPointer pad, @Const IntPointer stride,
    FloatPointer data_im);
@Namespace("caffe") public static native @Name("col2im_nd_gpu<float>") void col2im_nd_gpu_float(@Const FloatBuffer data_col, int num_spatial_axes,
    int im_size, @Const IntBuffer im_shape, @Const IntBuffer col_shape,
    @Const IntBuffer kernel_shape, @Const IntBuffer pad, @Const IntBuffer stride,
    FloatBuffer data_im);
@Namespace("caffe") public static native @Name("col2im_nd_gpu<float>") void col2im_nd_gpu_float(@Const float[] data_col, int num_spatial_axes,
    int im_size, @Const int[] im_shape, @Const int[] col_shape,
    @Const int[] kernel_shape, @Const int[] pad, @Const int[] stride,
    float[] data_im);

@Namespace("caffe") public static native @Name("col2im_nd_gpu<double>") void col2im_nd_gpu_double(@Const DoublePointer data_col, int num_spatial_axes,
    int im_size, @Const IntPointer im_shape, @Const IntPointer col_shape,
    @Const IntPointer kernel_shape, @Const IntPointer pad, @Const IntPointer stride,
    DoublePointer data_im);
@Namespace("caffe") public static native @Name("col2im_nd_gpu<double>") void col2im_nd_gpu_double(@Const DoubleBuffer data_col, int num_spatial_axes,
    int im_size, @Const IntBuffer im_shape, @Const IntBuffer col_shape,
    @Const IntBuffer kernel_shape, @Const IntBuffer pad, @Const IntBuffer stride,
    DoubleBuffer data_im);
@Namespace("caffe") public static native @Name("col2im_nd_gpu<double>") void col2im_nd_gpu_double(@Const double[] data_col, int num_spatial_axes,
    int im_size, @Const int[] im_shape, @Const int[] col_shape,
    @Const int[] kernel_shape, @Const int[] pad, @Const int[] stride,
    double[] data_im);

@Namespace("caffe") public static native @Name("col2im_gpu<float>") void col2im_gpu_float(@Const FloatPointer data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, FloatPointer data_im);
@Namespace("caffe") public static native @Name("col2im_gpu<float>") void col2im_gpu_float(@Const FloatBuffer data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, FloatBuffer data_im);
@Namespace("caffe") public static native @Name("col2im_gpu<float>") void col2im_gpu_float(@Const float[] data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, float[] data_im);

@Namespace("caffe") public static native @Name("col2im_gpu<double>") void col2im_gpu_double(@Const DoublePointer data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, DoublePointer data_im);
@Namespace("caffe") public static native @Name("col2im_gpu<double>") void col2im_gpu_double(@Const DoubleBuffer data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, DoubleBuffer data_im);
@Namespace("caffe") public static native @Name("col2im_gpu<double>") void col2im_gpu_double(@Const double[] data_col, int channels,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h,
    int stride_w, double[] data_im);

  // namespace caffe

// #endif  // CAFFE_UTIL_IM2COL_HPP_


// Parsed from caffe/util/insert_splits.hpp

// #ifndef _CAFFE_UTIL_INSERT_SPLITS_HPP_
// #define _CAFFE_UTIL_INSERT_SPLITS_HPP_

// #include <string>

// #include "caffe/proto/caffe.pb.h"

// Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
@Namespace("caffe") public static native void InsertSplits(@Const @ByRef NetParameter param, NetParameter param_split);

@Namespace("caffe") public static native void ConfigureSplitLayer(@StdString BytePointer layer_name, @StdString BytePointer blob_name,
    int blob_idx, int split_count, float loss_weight,
    LayerParameter split_layer_param);
@Namespace("caffe") public static native void ConfigureSplitLayer(@StdString String layer_name, @StdString String blob_name,
    int blob_idx, int split_count, float loss_weight,
    LayerParameter split_layer_param);

@Namespace("caffe") public static native @StdString BytePointer SplitLayerName(@StdString BytePointer layer_name, @StdString BytePointer blob_name,
    int blob_idx);
@Namespace("caffe") public static native @StdString String SplitLayerName(@StdString String layer_name, @StdString String blob_name,
    int blob_idx);

@Namespace("caffe") public static native @StdString BytePointer SplitBlobName(@StdString BytePointer layer_name, @StdString BytePointer blob_name,
    int blob_idx, int split_idx);
@Namespace("caffe") public static native @StdString String SplitBlobName(@StdString String layer_name, @StdString String blob_name,
    int blob_idx, int split_idx);

  // namespace caffe

// #endif  // CAFFE_UTIL_INSERT_SPLITS_HPP_


// Parsed from caffe/util/mkl_alternate.hpp

// #ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
// #define CAFFE_UTIL_MKL_ALTERNATE_H_

// #ifdef USE_MKL

// #include <mkl.h>

// #else  // If use MKL, simply include the MKL header
// #include <cblas.h>
// #include <math.h>

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
// #define DEFINE_VSL_UNARY_FUNC(name, operation)
//   template<typename Dtype>
//   void v##name(const int n, const Dtype* a, Dtype* y) {
//     CHECK_GT(n, 0); CHECK(a); CHECK(y);
//     for (int i = 0; i < n; ++i) { operation; }
//   }
//   inline void vs##name(
//     const int n, const float* a, float* y) {
//     v##name<float>(n, a, y);
//   }
//   inline void vd##name(
//       const int n, const double* a, double* y) {
//     v##name<double>(n, a, y);
//   }
  public static native void vsSqr(
      int n, @Const FloatPointer a, FloatPointer y);
  public static native void vsSqr(
      int n, @Const FloatBuffer a, FloatBuffer y);
  public static native void vsSqr(
      int n, @Const float[] a, float[] y);
  public static native void vdSqr(
        int n, @Const DoublePointer a, DoublePointer y);
  public static native void vdSqr(
        int n, @Const DoubleBuffer a, DoubleBuffer y);
  public static native void vdSqr(
        int n, @Const double[] a, double[] y);
  public static native void vsExp(
      int n, @Const FloatPointer a, FloatPointer y);
  public static native void vsExp(
      int n, @Const FloatBuffer a, FloatBuffer y);
  public static native void vsExp(
      int n, @Const float[] a, float[] y);
  public static native void vdExp(
        int n, @Const DoublePointer a, DoublePointer y);
  public static native void vdExp(
        int n, @Const DoubleBuffer a, DoubleBuffer y);
  public static native void vdExp(
        int n, @Const double[] a, double[] y);
  public static native void vsLn(
      int n, @Const FloatPointer a, FloatPointer y);
  public static native void vsLn(
      int n, @Const FloatBuffer a, FloatBuffer y);
  public static native void vsLn(
      int n, @Const float[] a, float[] y);
  public static native void vdLn(
        int n, @Const DoublePointer a, DoublePointer y);
  public static native void vdLn(
        int n, @Const DoubleBuffer a, DoubleBuffer y);
  public static native void vdLn(
        int n, @Const double[] a, double[] y);
  public static native void vsAbs(
      int n, @Const FloatPointer a, FloatPointer y);
  public static native void vsAbs(
      int n, @Const FloatBuffer a, FloatBuffer y);
  public static native void vsAbs(
      int n, @Const float[] a, float[] y);
  public static native void vdAbs(
        int n, @Const DoublePointer a, DoublePointer y);
  public static native void vdAbs(
        int n, @Const DoubleBuffer a, DoubleBuffer y);
  public static native void vdAbs(
        int n, @Const double[] a, double[] y);

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
// #define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation)
//   template<typename Dtype>
//   void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) {
//     CHECK_GT(n, 0); CHECK(a); CHECK(y);
//     for (int i = 0; i < n; ++i) { operation; }
//   }
//   inline void vs##name(
//     const int n, const float* a, const float b, float* y) {
//     v##name<float>(n, a, b, y);
//   }
//   inline void vd##name(
//       const int n, const double* a, const float b, double* y) {
//     v##name<double>(n, a, b, y);
//   }
  public static native void vsPowx(
      int n, @Const FloatPointer a, float b, FloatPointer y);
  public static native void vsPowx(
      int n, @Const FloatBuffer a, float b, FloatBuffer y);
  public static native void vsPowx(
      int n, @Const float[] a, float b, float[] y);
  public static native void vdPowx(
        int n, @Const DoublePointer a, float b, DoublePointer y);
  public static native void vdPowx(
        int n, @Const DoubleBuffer a, float b, DoubleBuffer y);
  public static native void vdPowx(
        int n, @Const double[] a, float b, double[] y);

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
// #define DEFINE_VSL_BINARY_FUNC(name, operation)
//   template<typename Dtype>
//   void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
//     CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y);
//     for (int i = 0; i < n; ++i) { operation; }
//   }
//   inline void vs##name(
//     const int n, const float* a, const float* b, float* y) {
//     v##name<float>(n, a, b, y);
//   }
//   inline void vd##name(
//       const int n, const double* a, const double* b, double* y) {
//     v##name<double>(n, a, b, y);
//   }
  public static native void vsAdd(
      int n, @Const FloatPointer a, @Const FloatPointer b, FloatPointer y);
  public static native void vsAdd(
      int n, @Const FloatBuffer a, @Const FloatBuffer b, FloatBuffer y);
  public static native void vsAdd(
      int n, @Const float[] a, @Const float[] b, float[] y);
  public static native void vdAdd(
        int n, @Const DoublePointer a, @Const DoublePointer b, DoublePointer y);
  public static native void vdAdd(
        int n, @Const DoubleBuffer a, @Const DoubleBuffer b, DoubleBuffer y);
  public static native void vdAdd(
        int n, @Const double[] a, @Const double[] b, double[] y);
  public static native void vsSub(
      int n, @Const FloatPointer a, @Const FloatPointer b, FloatPointer y);
  public static native void vsSub(
      int n, @Const FloatBuffer a, @Const FloatBuffer b, FloatBuffer y);
  public static native void vsSub(
      int n, @Const float[] a, @Const float[] b, float[] y);
  public static native void vdSub(
        int n, @Const DoublePointer a, @Const DoublePointer b, DoublePointer y);
  public static native void vdSub(
        int n, @Const DoubleBuffer a, @Const DoubleBuffer b, DoubleBuffer y);
  public static native void vdSub(
        int n, @Const double[] a, @Const double[] b, double[] y);
  public static native void vsMul(
      int n, @Const FloatPointer a, @Const FloatPointer b, FloatPointer y);
  public static native void vsMul(
      int n, @Const FloatBuffer a, @Const FloatBuffer b, FloatBuffer y);
  public static native void vsMul(
      int n, @Const float[] a, @Const float[] b, float[] y);
  public static native void vdMul(
        int n, @Const DoublePointer a, @Const DoublePointer b, DoublePointer y);
  public static native void vdMul(
        int n, @Const DoubleBuffer a, @Const DoubleBuffer b, DoubleBuffer y);
  public static native void vdMul(
        int n, @Const double[] a, @Const double[] b, double[] y);
  public static native void vsDiv(
      int n, @Const FloatPointer a, @Const FloatPointer b, FloatPointer y);
  public static native void vsDiv(
      int n, @Const FloatBuffer a, @Const FloatBuffer b, FloatBuffer y);
  public static native void vsDiv(
      int n, @Const float[] a, @Const float[] b, float[] y);
  public static native void vdDiv(
        int n, @Const DoublePointer a, @Const DoublePointer b, DoublePointer y);
  public static native void vdDiv(
        int n, @Const DoubleBuffer a, @Const DoubleBuffer b, DoubleBuffer y);
  public static native void vdDiv(
        int n, @Const double[] a, @Const double[] b, double[] y);

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
public static native void cblas_saxpby(int N, float alpha, @Const FloatPointer X,
                         int incX, float beta, FloatPointer Y,
                         int incY);
public static native void cblas_saxpby(int N, float alpha, @Const FloatBuffer X,
                         int incX, float beta, FloatBuffer Y,
                         int incY);
public static native void cblas_saxpby(int N, float alpha, @Const float[] X,
                         int incX, float beta, float[] Y,
                         int incY);
public static native void cblas_daxpby(int N, double alpha, @Const DoublePointer X,
                         int incX, double beta, DoublePointer Y,
                         int incY);
public static native void cblas_daxpby(int N, double alpha, @Const DoubleBuffer X,
                         int incX, double beta, DoubleBuffer Y,
                         int incY);
public static native void cblas_daxpby(int N, double alpha, @Const double[] X,
                         int incX, double beta, double[] Y,
                         int incY);

// #endif  // USE_MKL
// #endif  // CAFFE_UTIL_MKL_ALTERNATE_H_


// Parsed from caffe/util/upgrade_proto.hpp

// #ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
// #define CAFFE_UTIL_UPGRADE_PROTO_H_

// #include <string>

// #include "caffe/proto/caffe.pb.h"

// Return true iff the net is not the current version.
@Namespace("caffe") public static native @Cast("bool") boolean NetNeedsUpgrade(@Const @ByRef NetParameter net_param);

// Check for deprecations and upgrade the NetParameter as needed.
@Namespace("caffe") public static native @Cast("bool") boolean UpgradeNetAsNeeded(@StdString BytePointer param_file, NetParameter param);
@Namespace("caffe") public static native @Cast("bool") boolean UpgradeNetAsNeeded(@StdString String param_file, NetParameter param);

// Read parameters from a file into a NetParameter proto message.
@Namespace("caffe") public static native void ReadNetParamsFromTextFileOrDie(@StdString BytePointer param_file,
                                    NetParameter param);
@Namespace("caffe") public static native void ReadNetParamsFromTextFileOrDie(@StdString String param_file,
                                    NetParameter param);
@Namespace("caffe") public static native void ReadNetParamsFromBinaryFileOrDie(@StdString BytePointer param_file,
                                      NetParameter param);
@Namespace("caffe") public static native void ReadNetParamsFromBinaryFileOrDie(@StdString String param_file,
                                      NetParameter param);

// Return true iff any layer contains parameters specified using
// deprecated V0LayerParameter.
@Namespace("caffe") public static native @Cast("bool") boolean NetNeedsV0ToV1Upgrade(@Const @ByRef NetParameter net_param);

// Perform all necessary transformations to upgrade a V0NetParameter into a
// NetParameter (including upgrading padding layers and LayerParameters).
@Namespace("caffe") public static native @Cast("bool") boolean UpgradeV0Net(@Const @ByRef NetParameter v0_net_param, NetParameter net_param);

// Upgrade NetParameter with padding layers to pad-aware conv layers.
// For any padding layer, remove it and put its pad parameter in any layers
// taking its top blob as input.
// Error if any of these above layers are not-conv layers.
@Namespace("caffe") public static native void UpgradeV0PaddingLayers(@Const @ByRef NetParameter param,
                            NetParameter param_upgraded_pad);

// Upgrade a single V0LayerConnection to the V1LayerParameter format.
@Namespace("caffe") public static native @Cast("bool") boolean UpgradeV0LayerParameter(@Const @ByRef V1LayerParameter v0_layer_connection,
                             V1LayerParameter layer_param);

@Namespace("caffe") public static native @Cast("caffe::V1LayerParameter_LayerType") int UpgradeV0LayerType(@StdString BytePointer type);
@Namespace("caffe") public static native @Cast("caffe::V1LayerParameter_LayerType") int UpgradeV0LayerType(@StdString String type);

// Return true iff any layer contains deprecated data transformation parameters.
@Namespace("caffe") public static native @Cast("bool") boolean NetNeedsDataUpgrade(@Const @ByRef NetParameter net_param);

// Perform all necessary transformations to upgrade old transformation fields
// into a TransformationParameter.
@Namespace("caffe") public static native void UpgradeNetDataTransformation(NetParameter net_param);

// Return true iff the Net contains any layers specified as V1LayerParameters.
@Namespace("caffe") public static native @Cast("bool") boolean NetNeedsV1ToV2Upgrade(@Const @ByRef NetParameter net_param);

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
@Namespace("caffe") public static native @Cast("bool") boolean UpgradeV1Net(@Const @ByRef NetParameter v1_net_param, NetParameter net_param);

@Namespace("caffe") public static native @Cast("bool") boolean UpgradeV1LayerParameter(@Const @ByRef V1LayerParameter v1_layer_param,
                             LayerParameter layer_param);

@Namespace("caffe") public static native @Cast("const char*") BytePointer UpgradeV1LayerType(@Cast("const caffe::V1LayerParameter_LayerType") int type);

// Return true iff the solver contains any old solver_type specified as enums
@Namespace("caffe") public static native @Cast("bool") boolean SolverNeedsTypeUpgrade(@Const @ByRef SolverParameter solver_param);

@Namespace("caffe") public static native @Cast("bool") boolean UpgradeSolverType(SolverParameter solver_param);

// Check for deprecations and upgrade the SolverParameter as needed.
@Namespace("caffe") public static native @Cast("bool") boolean UpgradeSolverAsNeeded(@StdString BytePointer param_file, SolverParameter param);
@Namespace("caffe") public static native @Cast("bool") boolean UpgradeSolverAsNeeded(@StdString String param_file, SolverParameter param);

// Read parameters from a file into a SolverParameter proto message.
@Namespace("caffe") public static native void ReadSolverParamsFromTextFileOrDie(@StdString BytePointer param_file,
                                       SolverParameter param);
@Namespace("caffe") public static native void ReadSolverParamsFromTextFileOrDie(@StdString String param_file,
                                       SolverParameter param);

  // namespace caffe

// #endif   // CAFFE_UTIL_UPGRADE_PROTO_H_


// Parsed from caffe/util/cudnn.hpp

// #ifndef CAFFE_UTIL_CUDNN_H_
// #define CAFFE_UTIL_CUDNN_H_
// #ifdef USE_CUDNN

// #endif  // USE_CUDNN
// #endif  // CAFFE_UTIL_CUDNN_H_


}
