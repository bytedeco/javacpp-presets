// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class tensorflow extends org.bytedeco.javacpp.helper.tensorflow {
    static { Loader.load(); }

@Name("tensorflow::gtl::InlinedVector<tensorflow::DataType,4>") public static class DataTypeVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DataTypeVector(Pointer p) { super(p); }
    public DataTypeVector()       { allocate();  }
    private native void allocate();
    public native @Name("operator=") @ByRef DataTypeVector put(@ByRef DataTypeVector x);

    public native long size();

    @Index public native @Cast("tensorflow::DataType") int get(@Cast("size_t") long i);
    public native DataTypeVector put(@Cast("size_t") long i, int value);
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

@Name("std::vector<tensorflow::Tensor>") public static class TensorVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorVector(Pointer p) { super(p); }
    public TensorVector(Tensor ... array) { this(array.length); put(array); }
    public TensorVector()       { allocate();  }
    public TensorVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef TensorVector put(@ByRef TensorVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef Tensor get(@Cast("size_t") long i);
    public native TensorVector put(@Cast("size_t") long i, Tensor value);

    public TensorVector put(Tensor ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<tensorflow::TensorShape>") public static class TensorShapeVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorShapeVector(Pointer p) { super(p); }
    public TensorShapeVector(TensorShape ... array) { this(array.length); put(array); }
    public TensorShapeVector()       { allocate();  }
    public TensorShapeVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef TensorShapeVector put(@ByRef TensorShapeVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef TensorShape get(@Cast("size_t") long i);
    public native TensorShapeVector put(@Cast("size_t") long i, TensorShape value);

    public TensorShapeVector put(TensorShape ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<tensorflow::NodeBuilder::NodeOut>") public static class NodeOutVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NodeOutVector(Pointer p) { super(p); }
    public NodeOutVector(NodeBuilder.NodeOut ... array) { this(array.length); put(array); }
    public NodeOutVector()       { allocate();  }
    public NodeOutVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef NodeOutVector put(@ByRef NodeOutVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @ByRef NodeBuilder.NodeOut get(@Cast("size_t") long i);
    public native NodeOutVector put(@Cast("size_t") long i, NodeBuilder.NodeOut value);

    public NodeOutVector put(NodeBuilder.NodeOut ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<tensorflow::Node*>") public static class NodeVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NodeVector(Pointer p) { super(p); }
    public NodeVector(Node ... array) { this(array.length); put(array); }
    public NodeVector()       { allocate();  }
    public NodeVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef NodeVector put(@ByRef NodeVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native Node get(@Cast("size_t") long i);
    public native NodeVector put(@Cast("size_t") long i, Node value);

    public NodeVector put(Node ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}

@Name("std::vector<std::pair<std::string,tensorflow::Tensor> >") public static class StringTensorPairVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringTensorPairVector(Pointer p) { super(p); }
    public StringTensorPairVector(BytePointer[] firstValue, Tensor[] secondValue) { this(Math.min(firstValue.length, secondValue.length)); put(firstValue, secondValue); }
    public StringTensorPairVector(String[] firstValue, Tensor[] secondValue) { this(Math.min(firstValue.length, secondValue.length)); put(firstValue, secondValue); }
    public StringTensorPairVector()       { allocate();  }
    public StringTensorPairVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef StringTensorPairVector put(@ByRef StringTensorPairVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @StdString BytePointer first(@Cast("size_t") long i); public native StringTensorPairVector first(@Cast("size_t") long i, BytePointer first);
    @Index public native @ByRef Tensor second(@Cast("size_t") long i);  public native StringTensorPairVector second(@Cast("size_t") long i, Tensor second);
    @MemberSetter @Index public native StringTensorPairVector first(@Cast("size_t") long i, @StdString String first);

    public StringTensorPairVector put(BytePointer[] firstValue, Tensor[] secondValue) {
        for (int i = 0; i < firstValue.length && i < secondValue.length; i++) {
            first(i, firstValue[i]);
            second(i, secondValue[i]);
        }
        return this;
    }

    public StringTensorPairVector put(String[] firstValue, Tensor[] secondValue) {
        for (int i = 0; i < firstValue.length && i < secondValue.length; i++) {
            first(i, firstValue[i]);
            second(i, secondValue[i]);
        }
        return this;
    }
}

@NoOffset @Name("std::pair<tensorflow::EdgeSet::const_iterator,bool>") public static class EdgeSetBoolPair extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EdgeSetBoolPair(Pointer p) { super(p); }
    public EdgeSetBoolPair(EdgeSetIterator firstValue, boolean secondValue) { this(); put(firstValue, secondValue); }
    public EdgeSetBoolPair()       { allocate();  }
    private native void allocate();
    public native @Name("operator=") @ByRef EdgeSetBoolPair put(@ByRef EdgeSetBoolPair x);


    @MemberGetter public native @ByRef EdgeSetIterator first(); public native EdgeSetBoolPair first(EdgeSetIterator first);
    @MemberGetter public native @Cast("bool") boolean second();  public native EdgeSetBoolPair second(boolean second);

    public EdgeSetBoolPair put(EdgeSetIterator firstValue, boolean secondValue) {
        first(firstValue);
        second(secondValue);
        return this;
    }
}

// Parsed from tensorflow/core/platform/default/integral_types.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_
// #define TENSORFLOW_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_

  // namespace tensorflow

// #endif  // TENSORFLOW_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_


// Parsed from tensorflow/core/framework/numeric_types.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_
// #define TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_

// #include <complex>

// #include "tensorflow/core/platform/port.h"

// Single precision complex.

  // end namespace tensorflow

// #endif  // TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_


// Parsed from tensorflow/core/platform/init_main.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PLATFORM_INIT_MAIN_H_
// #define TENSORFLOW_PLATFORM_INIT_MAIN_H_

// Platform-specific initialization routine that may be invoked by a
// main() program that uses TensorFlow.
//
// Default implementation does nothing.
@Namespace("tensorflow::port") public static native void InitMain(@Cast("const char*") BytePointer usage, IntPointer argc, @Cast("char***") PointerPointer argv);
@Namespace("tensorflow::port") public static native void InitMain(String usage, IntBuffer argc, @Cast("char***") PointerPointer argv);
@Namespace("tensorflow::port") public static native void InitMain(@Cast("const char*") BytePointer usage, int[] argc, @Cast("char***") PointerPointer argv);
@Namespace("tensorflow::port") public static native void InitMain(String usage, IntPointer argc, @Cast("char***") PointerPointer argv);
@Namespace("tensorflow::port") public static native void InitMain(@Cast("const char*") BytePointer usage, IntBuffer argc, @Cast("char***") PointerPointer argv);
@Namespace("tensorflow::port") public static native void InitMain(String usage, int[] argc, @Cast("char***") PointerPointer argv);

  // namespace port
  // namespace tensorflow

// #endif  // TENSORFLOW_PLATFORM_INIT_MAIN_H_


// Parsed from tensorflow/core/platform/port.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PLATFORM_PORT_H_
// #define TENSORFLOW_PLATFORM_PORT_H_

// #include <string>
// #include <vector>

// #if !defined(PLATFORM_POSIX) && !defined(PLATFORM_GOOGLE) &&
//     !defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID)

// Choose which platform we are on.
// #if defined(ANDROID) || defined(__ANDROID__)
// #define PLATFORM_POSIX_ANDROID
// #elif defined(__APPLE__)
// #define PLATFORM_POSIX
// #else
// If no platform specified, use:
// #define PLATFORM_POSIX
// #endif

// #endif

// Define tensorflow::string to refer to appropriate platform specific type.
// #if defined(PLATFORM_GOOGLE)
// #else
// #endif
  // namespace tensorflow
/** enum tensorflow::ConditionResult */
public static final int kCond_Timeout = 0, kCond_MaybeNotified = 1;
  // namespace tensorflow

// Include appropriate platform-dependent implementations of mutex etc.
// #if defined(PLATFORM_GOOGLE)
// #include "tensorflow/core/platform/google/dynamic_annotations.h"
// #include "tensorflow/core/platform/google/integral_types.h"
// #include "tensorflow/core/platform/google/mutex.h"
// #elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||
//     defined(PLATFORM_GOOGLE_ANDROID)
// #include "tensorflow/core/platform/default/dynamic_annotations.h"
// #include "tensorflow/core/platform/default/integral_types.h"
// #include "tensorflow/core/platform/default/mutex.h"
// #else
// #error Define the appropriate PLATFORM_<foo> macro for this platform
// #endif

@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::uint8") byte kuint8max();
public static final byte kuint8max = kuint8max();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::uint16") short kuint16max();
public static final short kuint16max = kuint16max();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::uint32") int kuint32max();
public static final int kuint32max = kuint32max();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::uint64") long kuint64max();
public static final long kuint64max = kuint64max();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::int8") byte kint8min();
public static final byte kint8min = kint8min();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::int8") byte kint8max();
public static final byte kint8max = kint8max();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::int16") short kint16min();
public static final short kint16min = kint16min();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::int16") short kint16max();
public static final short kint16max = kint16max();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::int32") int kint32min();
public static final int kint32min = kint32min();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::int32") int kint32max();
public static final int kint32max = kint32max();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::int64") long kint64min();
public static final long kint64min = kint64min();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::int64") long kint64max();
public static final long kint64max = kint64max();

// A typedef for a uint64 used as a short fingerprint.

// The mutex library included above defines:
//   class mutex;
//   class mutex_lock;
//   class condition_variable;
// It also defines the following:

// Like "cv->wait(*mu)", except that it only waits for up to "ms" milliseconds.
//
// Returns kCond_Timeout if the timeout expired without this
// thread noticing a signal on the condition variable.  Otherwise may
// return either kCond_Timeout or kCond_MaybeNotified
@Namespace("tensorflow") public static native @Cast("tensorflow::ConditionResult") int WaitForMilliseconds(@Cast("tensorflow::mutex_lock*") Pointer mu, @Cast("tensorflow::condition_variable*") Pointer cv,
                                    @Cast("tensorflow::int64") long ms);
  // namespace tensorflow

// TODO(jeff,sanjay): Make portable
@Namespace("tensorflow::port") @MemberGetter public static native @Cast("const bool") boolean kLittleEndian();
public static final boolean kLittleEndian = kLittleEndian();

// TODO(jeff,sanjay): Find appropriate places for all the code below.
// Possible places for any particular item below:
//  (a) Here, so it gets reimplemented on every platform
//  (b) Env
//  (c) config.h (auto-generated by autotools?)
//  (d) macros.h
//  ...

// Return the hostname of the machine on which this process is running
@Namespace("tensorflow::port") public static native @StdString BytePointer Hostname();

// Returns an estimate of the number of schedulable CPUs for this
// process.  Usually, it's constant throughout the lifetime of a
// process, but it might change if the underlying cluster management
// software can change it dynamically.
@Namespace("tensorflow::port") public static native int NumSchedulableCPUs();

// Some platforms require that filenames be of a certain form when
// used for logging.  This function is invoked to allow platforms to
// adjust the filename used for logging appropriately, if necessary
// (most ports can just do nothing).  If any changes are necessary, the
// implementation should mutate "*filename" appropriately.
@Namespace("tensorflow::port") public static native void AdjustFilenameForLogging(@StdString @Cast({"char*", "std::string*"}) BytePointer filename);

// Aligned allocation/deallocation
@Namespace("tensorflow::port") public static native Pointer aligned_malloc(@Cast("size_t") long size, int minimum_alignment);
@Namespace("tensorflow::port") public static native void aligned_free(Pointer aligned_memory);

// Prefetching support
//
// Defined behavior on some of the uarchs:
// PREFETCH_HINT_T0:
//   prefetch to all levels of the hierarchy (except on p4: prefetch to L2)
// PREFETCH_HINT_NTA:
//   p4: fetch to L2, but limit to 1 way (out of the 8 ways)
//   core: skip L2, go directly to L1
//   k8 rev E and later: skip L2, can go to either of the 2-ways in L1
/** enum tensorflow::port::PrefetchHint */
public static final int
  PREFETCH_HINT_T0 = 3,  // More temporal locality
  PREFETCH_HINT_T1 = 2,
  PREFETCH_HINT_T2 = 1,  // Less temporal locality
  PREFETCH_HINT_NTA = 0;  // No temporal locality

// Snappy compression/decompression support
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_Compress(@Cast("const char*") BytePointer input, @Cast("size_t") long length, @StdString @Cast({"char*", "std::string*"}) BytePointer output);
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_Compress(String input, @Cast("size_t") long length, @StdString @Cast({"char*", "std::string*"}) BytePointer output);

@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_GetUncompressedLength(@Cast("const char*") BytePointer input, @Cast("size_t") long length,
                                  @Cast("size_t*") SizeTPointer result);
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_GetUncompressedLength(String input, @Cast("size_t") long length,
                                  @Cast("size_t*") SizeTPointer result);
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_Uncompress(@Cast("const char*") BytePointer input, @Cast("size_t") long length, @Cast("char*") BytePointer output);
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_Uncompress(String input, @Cast("size_t") long length, @Cast("char*") ByteBuffer output);
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_Uncompress(@Cast("const char*") BytePointer input, @Cast("size_t") long length, @Cast("char*") byte[] output);
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_Uncompress(String input, @Cast("size_t") long length, @Cast("char*") BytePointer output);
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_Uncompress(@Cast("const char*") BytePointer input, @Cast("size_t") long length, @Cast("char*") ByteBuffer output);
@Namespace("tensorflow::port") public static native @Cast("bool") boolean Snappy_Uncompress(String input, @Cast("size_t") long length, @Cast("char*") byte[] output);

// #if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
public static final int LANG_CXX11 = 1;
// #endif

// Compiler attributes
// #if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
// #define TF_ATTRIBUTE_NORETURN __attribute__((noreturn))
// #define TF_ATTRIBUTE_NOINLINE __attribute__((noinline))
// #define TF_ATTRIBUTE_UNUSED __attribute__((unused))
// #define TF_ATTRIBUTE_COLD __attribute__((cold))
// #define TF_PACKED __attribute__((packed))
// #define TF_MUST_USE_RESULT __attribute__((warn_unused_result))
// #define TF_PRINTF_ATTRIBUTE(string_index, first_to_check)
//   __attribute__((__format__(__printf__, string_index, first_to_check)))
// #define TF_SCANF_ATTRIBUTE(string_index, first_to_check)
//   __attribute__((__format__(__scanf__, string_index, first_to_check)))

// #else
// Non-GCC equivalents
// #define TF_ATTRIBUTE_NORETURN
// #define TF_ATTRIBUTE_NOINLINE
// #define TF_ATTRIBUTE_UNUSED
// #define TF_ATTRIBUTE_COLD
// #define TF_MUST_USE_RESULT
// #define TF_PACKED
// #define TF_PRINTF_ATTRIBUTE(string_index, first_to_check)
// #define TF_SCANF_ATTRIBUTE(string_index, first_to_check)
// #endif

// GCC can be told that a certain branch is not likely to be taken (for
// instance, a CHECK failure), and use that information in static analysis.
// Giving it this information can help it optimize for the common case in
// the absence of better information (ie. -fprofile-arcs).
//
// #if defined(COMPILER_GCC3)
// #define TF_PREDICT_FALSE(x) (__builtin_expect(x, 0))
// #define TF_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
// #else
// #define TF_PREDICT_FALSE(x) x
// #define TF_PREDICT_TRUE(x) x
// #endif

// ---------------------------------------------------------------------------
// Inline implementations of some performance-critical methods
// ---------------------------------------------------------------------------

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
// #define TF_DISALLOW_COPY_AND_ASSIGN(TypeName)
//   TypeName(const TypeName&) = delete;
//   void operator=(const TypeName&) = delete

// The TF_ARRAYSIZE(arr) macro returns the # of elements in an array arr.
//
// The expression TF_ARRAYSIZE(a) is a compile-time constant of type
// size_t.
// #define TF_ARRAYSIZE(a)
//   ((sizeof(a) / sizeof(*(a))) /
//    static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

// #if defined(__clang__) && defined(LANG_CXX11) && defined(__has_warning)
// #if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
// #define TF_FALLTHROUGH_INTENDED [[clang::fallthrough]]  // NOLINT
// #endif
// #endif

// #ifndef TF_FALLTHROUGH_INTENDED
// #define TF_FALLTHROUGH_INTENDED
//   do {
//   } while (0)
// #endif

  // namespace port
  // namespace tensorflow

// #endif  // TENSORFLOW_PLATFORM_PORT_H_


// Parsed from tensorflow/core/lib/core/error_codes.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/lib/core/error_codes.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/generated_enum_reflection.h>
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow::error") public static native void protobuf_AddDesc_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto();
@Namespace("tensorflow::error") public static native void protobuf_AssignDesc_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto();
@Namespace("tensorflow::error") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto();


/** enum tensorflow::error::Code */
public static final int
  OK = 0,
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  UNAUTHENTICATED = 16,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
  DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ = 20,
  Code_INT_MIN_SENTINEL_DO_NOT_USE_ =kint32min,
  Code_INT_MAX_SENTINEL_DO_NOT_USE_ =kint32max;
@Namespace("tensorflow::error") public static native @Cast("bool") boolean Code_IsValid(int value);
@Namespace("tensorflow::error") @MemberGetter public static native @Cast("const tensorflow::error::Code") int Code_MIN();
@Namespace("tensorflow::error") @MemberGetter public static native @Cast("const tensorflow::error::Code") int Code_MAX();
@Namespace("tensorflow::error") @MemberGetter public static native int Code_ARRAYSIZE();

@Namespace("tensorflow::error") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer Code_descriptor();
@Namespace("tensorflow::error") public static native @StdString BytePointer Code_Name(@Cast("tensorflow::error::Code") int value);
@Namespace("tensorflow::error") public static native @Cast("bool") boolean Code_Parse(
    @StdString BytePointer name, @Cast("tensorflow::error::Code*") IntPointer value);
@Namespace("tensorflow::error") public static native @Cast("bool") boolean Code_Parse(
    @StdString String name, @Cast("tensorflow::error::Code*") IntBuffer value);
@Namespace("tensorflow::error") public static native @Cast("bool") boolean Code_Parse(
    @StdString BytePointer name, @Cast("tensorflow::error::Code*") int[] value);
@Namespace("tensorflow::error") public static native @Cast("bool") boolean Code_Parse(
    @StdString String name, @Cast("tensorflow::error::Code*") IntPointer value);
@Namespace("tensorflow::error") public static native @Cast("bool") boolean Code_Parse(
    @StdString BytePointer name, @Cast("tensorflow::error::Code*") IntBuffer value);
@Namespace("tensorflow::error") public static native @Cast("bool") boolean Code_Parse(
    @StdString String name, @Cast("tensorflow::error::Code*") int[] value);
// ===================================================================


// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

  // namespace error
  // namespace tensorflow

// #ifndef SWIG
// #endif  // SWIG

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto__INCLUDED


// Parsed from tensorflow/core/platform/logging.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PLATFORM_LOGGING_H_
// #define TENSORFLOW_PLATFORM_LOGGING_H_

// #include "tensorflow/core/platform/port.h"  // To pick up PLATFORM_define

// #if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID)
// #include "base/logging.h"
// #else
// #include "tensorflow/core/platform/default/logging.h"
// #endif

// #endif  // TENSORFLOW_PLATFORM_LOGGING_H_


// Parsed from tensorflow/core/public/status.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PUBLIC_STATUS_H_
// #define TENSORFLOW_PUBLIC_STATUS_H_

// #include <iosfwd>
// #include <string>
// #include "tensorflow/core/lib/core/error_codes.pb.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/platform/logging.h"

@Namespace("tensorflow") @NoOffset public static class Status extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Status(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Status(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public Status position(int position) {
        return (Status)super.position(position);
    }

  /** Create a success status. */
  public Status() { super((Pointer)null); allocate(); }
  private native void allocate();

  /** \brief Create a status with the specified error code and msg as a
   *  human-readable string containing more detailed information. */
  public Status(@Cast("tensorflow::error::Code") int code, @StringPiece BytePointer msg) { super((Pointer)null); allocate(code, msg); }
  private native void allocate(@Cast("tensorflow::error::Code") int code, @StringPiece BytePointer msg);
  public Status(@Cast("tensorflow::error::Code") int code, @StringPiece String msg) { super((Pointer)null); allocate(code, msg); }
  private native void allocate(@Cast("tensorflow::error::Code") int code, @StringPiece String msg);

  /** Copy the specified status. */
  public Status(@Const @ByRef Status s) { super((Pointer)null); allocate(s); }
  private native void allocate(@Const @ByRef Status s);
  public native @Name("operator =") void put(@Const @ByRef Status s);

  public static native @ByVal Status OK();

  /** Returns true iff the status indicates success. */
  public native @Cast("bool") boolean ok();

  public native @Cast("tensorflow::error::Code") int code();

  public native @StdString BytePointer error_message();

  public native @Cast("bool") @Name("operator ==") boolean equals(@Const @ByRef Status x);
  
  ///
  public native @Cast("bool") @Name("operator !=") boolean notEquals(@Const @ByRef Status x);

  /** \brief If {@code ok()}, stores {@code new_status} into {@code *this}.  If {@code !ok()},
   *  preserves the current status, but may augment with additional
   *  information about {@code new_status}.
   * 
   *  Convenient way of keeping track of the first error encountered.
   *  Instead of:
   *    {@code if (overall_status.ok()) overall_status = new_status}
   *  Use:
   *    {@code overall_status.Update(new_status);} */
  public native void Update(@Const @ByRef Status new_status);

  /** \brief Return a string representation of this status suitable for
   *  printing. Returns the string {@code "OK"} for success. */
  public native @StdString BytePointer ToString();
}









@Namespace("tensorflow") public static native @Cast("std::ostream*") @ByRef @Name("operator <<") Pointer shiftLeft(@Cast("std::ostream*") @ByRef Pointer os, @Const @ByRef Status x);

public static native void TF_CHECK_OK(@ByVal Status val);
public static native void TF_QCHECK_OK(@ByVal Status val);

  // namespace tensorflow

// #endif  // TENSORFLOW_PUBLIC_STATUS_H_


// Parsed from tensorflow/core/platform/protobuf.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PLATFORM_PROTOBUF_H_
// #define TENSORFLOW_PLATFORM_PROTOBUF_H_

// Import whatever namespace protobuf comes from into the
// ::tensorflow::protobuf namespace.
//
// TensorFlow code should use the ::tensorflow::protobuf namespace to
// refer to all protobuf APIs.

// #include "tensorflow/core/platform/port.h"
// #if defined(PLATFORM_GOOGLE)
// #include "tensorflow/core/platform/google/protobuf.h"
// #elif defined(PLATFORM_GOOGLE_ANDROID)
// #include "tensorflow/core/platform/google/protobuf_android.h"
// #else
// #include "tensorflow/core/platform/default/protobuf.h"
// #endif
// Parses a protocol buffer contained in a string in the binary wire format.
// Returns true on success. Note: Unlike protobuf's builtin ParseFromString,
// this function has no size restrictions on the total size of the encoded
// protocol buffer.
@Namespace("tensorflow") public static native @Cast("bool") boolean ParseProtoUnlimited(@Cast("tensorflow::protobuf::Message*") Pointer proto, @StdString BytePointer serialized);
@Namespace("tensorflow") public static native @Cast("bool") boolean ParseProtoUnlimited(@Cast("tensorflow::protobuf::Message*") Pointer proto, @StdString String serialized);
@Namespace("tensorflow") public static native @Cast("bool") boolean ParseProtoUnlimited(@Cast("tensorflow::protobuf::Message*") Pointer proto, @Const Pointer serialized,
                         @Cast("size_t") long size);
  // namespace tensorflow

// #endif  // TENSORFLOW_PLATFORM_PROTOBUF_H_


// Parsed from tensorflow/core/public/env.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PUBLIC_ENV_H_
// #define TENSORFLOW_PUBLIC_ENV_H_

// #include <stdint.h>
// #include <string>
// #include <vector>
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/platform/port.h"
// #include "tensorflow/core/platform/protobuf.h"
// #include "tensorflow/core/public/status.h"

/** \brief An interface used by the tensorflow implementation to
 *  access operating system functionality like the filesystem etc.
 * 
 *  Callers may wish to provide a custom Env object to get fine grain
 *  control.
 * 
 *  All Env implementations are safe for concurrent access from
 *  multiple threads without any external synchronization. */
@Namespace("tensorflow") public static class Env extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Env(Pointer p) { super(p); }


  /** \brief Returns a default environment suitable for the current operating
   *  system.
   * 
   *  Sophisticated users may wish to provide their own Env
   *  implementation instead of relying on this default environment.
   * 
   *  The result of Default() belongs to this library and must never be deleted. */
  
  ///
  public static native Env Default();

  /** \brief Creates a brand new random access read-only file with the
   *  specified name.
   <p>
   *  On success, stores a pointer to the new file in
   *  *result and returns OK.  On failure stores NULL in *result and
   *  returns non-OK.  If the file does not exist, returns a non-OK
   *  status.
   * 
   *  The returned file may be concurrently accessed by multiple threads. */
  
  ///
  ///
  public native @ByVal Status NewRandomAccessFile(@StdString BytePointer fname,
                                       @Cast("tensorflow::RandomAccessFile**") PointerPointer result);
  public native @ByVal Status NewRandomAccessFile(@StdString BytePointer fname,
                                       @ByPtrPtr RandomAccessFile result);
  public native @ByVal Status NewRandomAccessFile(@StdString String fname,
                                       @ByPtrPtr RandomAccessFile result);

  /** \brief Creates an object that writes to a new file with the specified
   *  name.
   * 
   *  Deletes any existing file with the same name and creates a
   *  new file.  On success, stores a pointer to the new file in
   *  *result and returns OK.  On failure stores NULL in *result and
   *  returns non-OK.
   * 
   *  The returned file will only be accessed by one thread at a time. */
  
  ///
  ///
  public native @ByVal Status NewWritableFile(@StdString BytePointer fname,
                                   @Cast("tensorflow::WritableFile**") PointerPointer result);
  public native @ByVal Status NewWritableFile(@StdString BytePointer fname,
                                   @ByPtrPtr WritableFile result);
  public native @ByVal Status NewWritableFile(@StdString String fname,
                                   @ByPtrPtr WritableFile result);

  /** \brief Creates an object that either appends to an existing file, or
   *  writes to a new file (if the file does not exist to begin with).
   * 
   *  On success, stores a pointer to the new file in *result and
   *  returns OK.  On failure stores NULL in *result and returns
   *  non-OK.
   * 
   *  The returned file will only be accessed by one thread at a time. */
  public native @ByVal Status NewAppendableFile(@StdString BytePointer fname,
                                     @Cast("tensorflow::WritableFile**") PointerPointer result);
  public native @ByVal Status NewAppendableFile(@StdString BytePointer fname,
                                     @ByPtrPtr WritableFile result);
  public native @ByVal Status NewAppendableFile(@StdString String fname,
                                     @ByPtrPtr WritableFile result);

  /** Returns true iff the named file exists. */
  
  ///
  public native @Cast("bool") boolean FileExists(@StdString BytePointer fname);
  public native @Cast("bool") boolean FileExists(@StdString String fname);

  /** \brief Stores in *result the names of the children of the specified
   *  directory. The names are relative to "dir".
   * 
   *  Original contents of *results are dropped. */
  public native @ByVal Status GetChildren(@StdString BytePointer dir,
                               StringVector result);
  public native @ByVal Status GetChildren(@StdString String dir,
                               StringVector result);

  /** Deletes the named file. */
  public native @ByVal Status DeleteFile(@StdString BytePointer fname);
  public native @ByVal Status DeleteFile(@StdString String fname);

  /** Creates the specified directory. */
  public native @ByVal Status CreateDir(@StdString BytePointer dirname);
  public native @ByVal Status CreateDir(@StdString String dirname);

  /** Deletes the specified directory. */
  public native @ByVal Status DeleteDir(@StdString BytePointer dirname);
  public native @ByVal Status DeleteDir(@StdString String dirname);

  /** Stores the size of {@code fname} in {@code *file_size}. */
  public native @ByVal Status GetFileSize(@StdString BytePointer fname, @Cast("tensorflow::uint64*") LongPointer file_size);
  public native @ByVal Status GetFileSize(@StdString String fname, @Cast("tensorflow::uint64*") LongBuffer file_size);
  public native @ByVal Status GetFileSize(@StdString BytePointer fname, @Cast("tensorflow::uint64*") long[] file_size);
  public native @ByVal Status GetFileSize(@StdString String fname, @Cast("tensorflow::uint64*") LongPointer file_size);
  public native @ByVal Status GetFileSize(@StdString BytePointer fname, @Cast("tensorflow::uint64*") LongBuffer file_size);
  public native @ByVal Status GetFileSize(@StdString String fname, @Cast("tensorflow::uint64*") long[] file_size);

  /** \brief Renames file src to target. If target already exists, it will be
   *  replaced. */
  public native @ByVal Status RenameFile(@StdString BytePointer src, @StdString BytePointer target);
  public native @ByVal Status RenameFile(@StdString String src, @StdString String target);

  // TODO(jeff,sanjay): Add back thread/thread-pool support if needed.
  // TODO(jeff,sanjay): if needed, tighten spec so relative to epoch, or
  // provide a routine to get the absolute time.

  /** \brief Returns the number of micro-seconds since some fixed point in
   *  time. Only useful for computing deltas of time. */
  public native @Cast("tensorflow::uint64") long NowMicros();

  /** Sleeps/delays the thread for the prescribed number of micro-seconds. */
  
  ///
  public native void SleepForMicroseconds(int micros);

  /** \brief Returns a new thread that is running fn() and is identified
   *  (for debugging/performance-analysis) by "name".
   * 
   *  Caller takes ownership of the result and must delete it eventually
   *  (the deletion will block until fn() stops running). */
  public native Thread StartThread(@Const @ByRef ThreadOptions thread_options,
                                @StdString BytePointer name,
                                @ByVal Fn fn);
  public native Thread StartThread(@Const @ByRef ThreadOptions thread_options,
                                @StdString String name,
                                @ByVal Fn fn);
}

/** A file abstraction for randomly reading the contents of a file. */
@Namespace("tensorflow") public static class RandomAccessFile extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public RandomAccessFile(Pointer p) { super(p); }


  /** \brief Reads up to {@code n} bytes from the file starting at {@code offset}.
   * 
   *  {@code scratch[0..n-1]} may be written by this routine.  Sets {@code *result}
   *  to the data that was read (including if fewer than {@code n} bytes were
   *  successfully read).  May set {@code *result} to point at data in
   *  {@code scratch[0..n-1]}, so {@code scratch[0..n-1]} must be live when
   *  {@code *result} is used.
   * 
   *  On OK returned status: {@code n} bytes have been stored in {@code *result}.
   *  On non-OK returned status: {@code [0..n]} bytes have been stored in {@code *result}.
   * 
   *  Returns {@code OUT_OF_RANGE} if fewer than n bytes were stored in {@code *result}
   *  because of EOF.
   * 
   *  Safe for concurrent use by multiple threads. */
  public native @ByVal Status Read(@Cast("tensorflow::uint64") long offset, @Cast("size_t") long n, @StringPiece BytePointer result,
                        @Cast("char*") BytePointer scratch);
  public native @ByVal Status Read(@Cast("tensorflow::uint64") long offset, @Cast("size_t") long n, @StringPiece BytePointer result,
                        @Cast("char*") ByteBuffer scratch);
  public native @ByVal Status Read(@Cast("tensorflow::uint64") long offset, @Cast("size_t") long n, @StringPiece BytePointer result,
                        @Cast("char*") byte[] scratch);
}

/** \brief A file abstraction for sequential writing.
 * 
 *  The implementation must provide buffering since callers may append
 *  small fragments at a time to the file. */
@Namespace("tensorflow") public static class WritableFile extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public WritableFile(Pointer p) { super(p); }


  public native @ByVal Status Append(@StringPiece BytePointer data);
  public native @ByVal Status Append(@StringPiece String data);
  public native @ByVal Status Close();
  public native @ByVal Status Flush();
  public native @ByVal Status Sync();
}

/** \brief An implementation of Env that forwards all calls to another Env.
 * 
 *  May be useful to clients who wish to override just part of the
 *  functionality of another Env. */
@Namespace("tensorflow") @NoOffset public static class EnvWrapper extends Env {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EnvWrapper(Pointer p) { super(p); }

  /** Initializes an EnvWrapper that delegates all calls to *t */
  public EnvWrapper(Env t) { super((Pointer)null); allocate(t); }
  private native void allocate(Env t);

  /** Returns the target to which this Env forwards all calls */
  public native Env target();

  // The following text is boilerplate that forwards all methods to target()
  public native @ByVal Status NewRandomAccessFile(@StdString BytePointer f, @Cast("tensorflow::RandomAccessFile**") PointerPointer r);
  public native @ByVal Status NewRandomAccessFile(@StdString BytePointer f, @ByPtrPtr RandomAccessFile r);
  public native @ByVal Status NewRandomAccessFile(@StdString String f, @ByPtrPtr RandomAccessFile r);
  public native @ByVal Status NewWritableFile(@StdString BytePointer f, @Cast("tensorflow::WritableFile**") PointerPointer r);
  public native @ByVal Status NewWritableFile(@StdString BytePointer f, @ByPtrPtr WritableFile r);
  public native @ByVal Status NewWritableFile(@StdString String f, @ByPtrPtr WritableFile r);
  public native @ByVal Status NewAppendableFile(@StdString BytePointer f, @Cast("tensorflow::WritableFile**") PointerPointer r);
  public native @ByVal Status NewAppendableFile(@StdString BytePointer f, @ByPtrPtr WritableFile r);
  public native @ByVal Status NewAppendableFile(@StdString String f, @ByPtrPtr WritableFile r);
  public native @Cast("bool") boolean FileExists(@StdString BytePointer f);
  public native @Cast("bool") boolean FileExists(@StdString String f);
  public native @ByVal Status GetChildren(@StdString BytePointer dir, StringVector r);
  public native @ByVal Status GetChildren(@StdString String dir, StringVector r);
  public native @ByVal Status DeleteFile(@StdString BytePointer f);
  public native @ByVal Status DeleteFile(@StdString String f);
  public native @ByVal Status CreateDir(@StdString BytePointer d);
  public native @ByVal Status CreateDir(@StdString String d);
  public native @ByVal Status DeleteDir(@StdString BytePointer d);
  public native @ByVal Status DeleteDir(@StdString String d);
  public native @ByVal Status GetFileSize(@StdString BytePointer f, @Cast("tensorflow::uint64*") LongPointer s);
  public native @ByVal Status GetFileSize(@StdString String f, @Cast("tensorflow::uint64*") LongBuffer s);
  public native @ByVal Status GetFileSize(@StdString BytePointer f, @Cast("tensorflow::uint64*") long[] s);
  public native @ByVal Status GetFileSize(@StdString String f, @Cast("tensorflow::uint64*") LongPointer s);
  public native @ByVal Status GetFileSize(@StdString BytePointer f, @Cast("tensorflow::uint64*") LongBuffer s);
  public native @ByVal Status GetFileSize(@StdString String f, @Cast("tensorflow::uint64*") long[] s);
  public native @ByVal Status RenameFile(@StdString BytePointer s, @StdString BytePointer t);
  public native @ByVal Status RenameFile(@StdString String s, @StdString String t);
  public native @Cast("tensorflow::uint64") long NowMicros();
  public native void SleepForMicroseconds(int micros);
  public native Thread StartThread(@Const @ByRef ThreadOptions thread_options, @StdString BytePointer name,
                        @ByVal Fn fn);
  public native Thread StartThread(@Const @ByRef ThreadOptions thread_options, @StdString String name,
                        @ByVal Fn fn);
}

@Namespace("tensorflow") public static class Thread extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Thread(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public Thread(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public Thread position(int position) {
        return (Thread)super.position(position);
    }

  public Thread() { super((Pointer)null); allocate(); }
  private native void allocate();

  /** Blocks until the thread of control stops running. */
}

/** \brief Options to configure a Thread.
 * 
 *  Note that the options are all hints, and the
 *  underlying implementation may choose to ignore it. */
@Namespace("tensorflow") public static class ThreadOptions extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ThreadOptions() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ThreadOptions(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ThreadOptions(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public ThreadOptions position(int position) {
        return (ThreadOptions)super.position(position);
    }

  /** Thread stack size to use (in bytes). */
  public native @Cast("size_t") long stack_size(); public native ThreadOptions stack_size(long stack_size);  // 0: use system default value
  /** Guard area size to use near thread stacks to use (in bytes) */
  public native @Cast("size_t") long guard_size(); public native ThreadOptions guard_size(long guard_size);  // 0: use system default value
}

/** A utility routine: reads contents of named file into {@code *data} */
@Namespace("tensorflow") public static native @ByVal Status ReadFileToString(Env env, @StdString BytePointer fname, @StdString @Cast({"char*", "std::string*"}) BytePointer data);
@Namespace("tensorflow") public static native @ByVal Status ReadFileToString(Env env, @StdString String fname, @StdString @Cast({"char*", "std::string*"}) BytePointer data);

/** A utility routine: write contents of {@code data} to file named {@code fname}
 *  (overwriting existing contents, if any). */
@Namespace("tensorflow") public static native @ByVal Status WriteStringToFile(Env env, @StdString BytePointer fname,
                         @StringPiece BytePointer data);
@Namespace("tensorflow") public static native @ByVal Status WriteStringToFile(Env env, @StdString String fname,
                         @StringPiece String data);

/** Reads contents of named file and parse as binary encoded proto data
 *  and store into {@code *proto}. */
@Namespace("tensorflow") public static native @ByVal Status ReadBinaryProto(Env env, @StdString BytePointer fname,
                       @Cast("tensorflow::protobuf::MessageLite*") Pointer proto);
@Namespace("tensorflow") public static native @ByVal Status ReadBinaryProto(Env env, @StdString String fname,
                       @Cast("tensorflow::protobuf::MessageLite*") Pointer proto);

  // namespace tensorflow

// #endif  // TENSORFLOW_PUBLIC_ENV_H_


// Parsed from tensorflow/core/framework/config.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/config.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2fconfig_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2fconfig_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/map.h>
// #include <google/protobuf/map_field_inl.h>
// #include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2fconfig_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2fconfig_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2fconfig_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class GPUOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GPUOptions(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GPUOptions(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public GPUOptions position(int position) {
        return (GPUOptions)super.position(position);
    }

  public GPUOptions() { super((Pointer)null); allocate(); }
  private native void allocate();

  public GPUOptions(@Const @ByRef GPUOptions from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef GPUOptions from);

  public native @ByRef @Name("operator =") GPUOptions put(@Const @ByRef GPUOptions from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef GPUOptions default_instance();

  public native void Swap(GPUOptions other);

  // implements Message ----------------------------------------------

  public native GPUOptions New();

  public native GPUOptions New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef GPUOptions from);
  public native void MergeFrom(@Const @ByRef GPUOptions from);
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

  // optional double per_process_gpu_memory_fraction = 1;
  public native void clear_per_process_gpu_memory_fraction();
  @MemberGetter public static native int kPerProcessGpuMemoryFractionFieldNumber();
  public static final int kPerProcessGpuMemoryFractionFieldNumber = kPerProcessGpuMemoryFractionFieldNumber();
  public native double per_process_gpu_memory_fraction();
  public native void set_per_process_gpu_memory_fraction(double value);

  // optional string allocator_type = 2;
  public native void clear_allocator_type();
  @MemberGetter public static native int kAllocatorTypeFieldNumber();
  public static final int kAllocatorTypeFieldNumber = kAllocatorTypeFieldNumber();
  public native @StdString BytePointer allocator_type();
  public native void set_allocator_type(@StdString BytePointer value);
  public native void set_allocator_type(@StdString String value);
  public native void set_allocator_type(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_allocator_type(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_allocator_type();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_allocator_type();
  public native void set_allocated_allocator_type(@StdString @Cast({"char*", "std::string*"}) BytePointer allocator_type);
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class ConfigProto extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ConfigProto(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public ConfigProto(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public ConfigProto position(int position) {
        return (ConfigProto)super.position(position);
    }

  public ConfigProto() { super((Pointer)null); allocate(); }
  private native void allocate();

  public ConfigProto(@Const @ByRef ConfigProto from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef ConfigProto from);

  public native @ByRef @Name("operator =") ConfigProto put(@Const @ByRef ConfigProto from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef ConfigProto default_instance();

  public native void Swap(ConfigProto other);

  // implements Message ----------------------------------------------

  public native ConfigProto New();

  public native ConfigProto New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef ConfigProto from);
  public native void MergeFrom(@Const @ByRef ConfigProto from);
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

  // map<string, int32> device_count = 1;
  public native int device_count_size();
  public native void clear_device_count();
  @MemberGetter public static native int kDeviceCountFieldNumber();
  public static final int kDeviceCountFieldNumber = kDeviceCountFieldNumber();

  // optional int32 intra_op_parallelism_threads = 2;
  public native void clear_intra_op_parallelism_threads();
  @MemberGetter public static native int kIntraOpParallelismThreadsFieldNumber();
  public static final int kIntraOpParallelismThreadsFieldNumber = kIntraOpParallelismThreadsFieldNumber();
  public native @Cast("google::protobuf::int32") int intra_op_parallelism_threads();
  public native void set_intra_op_parallelism_threads(@Cast("google::protobuf::int32") int value);

  // optional int32 inter_op_parallelism_threads = 5;
  public native void clear_inter_op_parallelism_threads();
  @MemberGetter public static native int kInterOpParallelismThreadsFieldNumber();
  public static final int kInterOpParallelismThreadsFieldNumber = kInterOpParallelismThreadsFieldNumber();
  public native @Cast("google::protobuf::int32") int inter_op_parallelism_threads();
  public native void set_inter_op_parallelism_threads(@Cast("google::protobuf::int32") int value);

  // optional int32 placement_period = 3;
  public native void clear_placement_period();
  @MemberGetter public static native int kPlacementPeriodFieldNumber();
  public static final int kPlacementPeriodFieldNumber = kPlacementPeriodFieldNumber();
  public native @Cast("google::protobuf::int32") int placement_period();
  public native void set_placement_period(@Cast("google::protobuf::int32") int value);

  // repeated string device_filters = 4;
  public native int device_filters_size();
  public native void clear_device_filters();
  @MemberGetter public static native int kDeviceFiltersFieldNumber();
  public static final int kDeviceFiltersFieldNumber = kDeviceFiltersFieldNumber();
  public native @StdString BytePointer device_filters(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_device_filters(int index);
  public native void set_device_filters(int index, @StdString BytePointer value);
  public native void set_device_filters(int index, @StdString String value);
  public native void set_device_filters(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_device_filters(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_device_filters();
  public native void add_device_filters(@StdString BytePointer value);
  public native void add_device_filters(@StdString String value);
  public native void add_device_filters(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_device_filters(String value, @Cast("size_t") long size);

  // optional .tensorflow.GPUOptions gpu_options = 6;
  public native @Cast("bool") boolean has_gpu_options();
  public native void clear_gpu_options();
  @MemberGetter public static native int kGpuOptionsFieldNumber();
  public static final int kGpuOptionsFieldNumber = kGpuOptionsFieldNumber();
  public native @Const @ByRef GPUOptions gpu_options();
  public native GPUOptions mutable_gpu_options();
  public native GPUOptions release_gpu_options();
  public native void set_allocated_gpu_options(GPUOptions gpu_options);

  // optional bool allow_soft_placement = 7;
  public native void clear_allow_soft_placement();
  @MemberGetter public static native int kAllowSoftPlacementFieldNumber();
  public static final int kAllowSoftPlacementFieldNumber = kAllowSoftPlacementFieldNumber();
  public native @Cast("bool") boolean allow_soft_placement();
  public native void set_allow_soft_placement(@Cast("bool") boolean value);

  // optional bool log_device_placement = 8;
  public native void clear_log_device_placement();
  @MemberGetter public static native int kLogDevicePlacementFieldNumber();
  public static final int kLogDevicePlacementFieldNumber = kLogDevicePlacementFieldNumber();
  public native @Cast("bool") boolean log_device_placement();
  public native void set_log_device_placement(@Cast("bool") boolean value);
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// GPUOptions

// optional double per_process_gpu_memory_fraction = 1;




// optional string allocator_type = 2;









// -------------------------------------------------------------------

// ConfigProto

// map<string, int32> device_count = 1;





// optional int32 intra_op_parallelism_threads = 2;




// optional int32 inter_op_parallelism_threads = 5;




// optional int32 placement_period = 3;




// repeated string device_filters = 4;














// optional .tensorflow.GPUOptions gpu_options = 6;







// optional bool allow_soft_placement = 7;




// optional bool log_device_placement = 8;




// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2fconfig_2eproto__INCLUDED


// Parsed from tensorflow/core/public/session_options.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_
// #define TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_

// #include <string>
// #include "tensorflow/core/framework/config.pb.h"
// #include "tensorflow/core/platform/port.h"

/** Configuration information for a Session. */
@Namespace("tensorflow") @NoOffset public static class SessionOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SessionOptions(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public SessionOptions(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public SessionOptions position(int position) {
        return (SessionOptions)super.position(position);
    }

  /** The environment to use. */
  
  ///
  ///
  ///
  ///
  ///
  public native Env env(); public native SessionOptions env(Env env);

  /** \brief The TensorFlow runtime to connect to.
   * 
   *  If 'target' is empty or unspecified, the local TensorFlow runtime
   *  implementation will be used.  Otherwise, the TensorFlow engine
   *  defined by 'target' will be used to perform all computations.
   * 
   *  "target" can be either a single entry or a comma separated list
   *  of entries. Each entry is a resolvable address of the
   *  following format:
   *    local
   *    ip:port
   *    host:port
   *    ... other system-specific formats to identify tasks and jobs ...
   * 
   *  NOTE: at the moment 'local' maps to an in-process service-based
   *  runtime.
   * 
   *  Upon creation, a single session affines itself to one of the
   *  remote processes, with possible load balancing choices when the
   *  "target" resolves to a list of possible processes.
   * 
   *  If the session disconnects from the remote process during its
   *  lifetime, session calls may fail immediately. */
  public native @StdString BytePointer target(); public native SessionOptions target(BytePointer target);

  /** Configuration options. */
  public native @ByRef ConfigProto config(); public native SessionOptions config(ConfigProto config);

  public SessionOptions() { super((Pointer)null); allocate(); }
  private native void allocate();
}

  // end namespace tensorflow

// #endif  // TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_


// Parsed from tensorflow/core/lib/core/threadpool.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_LIB_CORE_THREADPOOL_H_
// #define TENSORFLOW_LIB_CORE_THREADPOOL_H_

// #include <deque>
// #include <functional>
// #include <thread>
// #include <vector>
// #include "tensorflow/core/platform/port.h"
// #include "tensorflow/core/public/env.h"

@Namespace("tensorflow::thread") @NoOffset public static class ThreadPool extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ThreadPool(Pointer p) { super(p); }

  // Construct a pool that contains "num_threads" threads with specified "name".
  // env->StartThread() is used to create individual threads.
  //
  // REQUIRES: num_threads > 0
  public ThreadPool(Env env, @StdString BytePointer name, int num_threads) { super((Pointer)null); allocate(env, name, num_threads); }
  private native void allocate(Env env, @StdString BytePointer name, int num_threads);
  public ThreadPool(Env env, @StdString String name, int num_threads) { super((Pointer)null); allocate(env, name, num_threads); }
  private native void allocate(Env env, @StdString String name, int num_threads);

  // Construct a pool that contains "num_threads" threads with specified "name".
  // env->StartThread() is used to create individual threads.
  //
  // REQUIRES: num_threads > 0
  public ThreadPool(Env env, @Const @ByRef ThreadOptions thread_options, @StdString BytePointer name,
               int num_threads) { super((Pointer)null); allocate(env, thread_options, name, num_threads); }
  private native void allocate(Env env, @Const @ByRef ThreadOptions thread_options, @StdString BytePointer name,
               int num_threads);
  public ThreadPool(Env env, @Const @ByRef ThreadOptions thread_options, @StdString String name,
               int num_threads) { super((Pointer)null); allocate(env, thread_options, name, num_threads); }
  private native void allocate(Env env, @Const @ByRef ThreadOptions thread_options, @StdString String name,
               int num_threads);

  // Wait until all scheduled work has finished and then destroy the
  // set of threads.

  // Schedule fn() for execution in the pool of threads.
  public native void Schedule(@ByVal Fn fn);

  public native @Cast("bool") boolean HasPendingClosures();
}

  // namespace thread
  // namespace tensorflow

// #endif  // TENSORFLOW_LIB_CORE_THREADPOOL_H_


// Parsed from tensorflow/core/framework/allocation_description.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/allocation_description.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2fallocation_5fdescription_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2fallocation_5fdescription_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2fallocation_5fdescription_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2fallocation_5fdescription_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2fallocation_5fdescription_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class AllocationDescription extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AllocationDescription(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public AllocationDescription(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public AllocationDescription position(int position) {
        return (AllocationDescription)super.position(position);
    }

  public AllocationDescription() { super((Pointer)null); allocate(); }
  private native void allocate();

  public AllocationDescription(@Const @ByRef AllocationDescription from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef AllocationDescription from);

  public native @ByRef @Name("operator =") AllocationDescription put(@Const @ByRef AllocationDescription from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef AllocationDescription default_instance();

  public native void Swap(AllocationDescription other);

  // implements Message ----------------------------------------------

  public native AllocationDescription New();

  public native AllocationDescription New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef AllocationDescription from);
  public native void MergeFrom(@Const @ByRef AllocationDescription from);
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

  // optional int64 requested_bytes = 1;
  public native void clear_requested_bytes();
  @MemberGetter public static native int kRequestedBytesFieldNumber();
  public static final int kRequestedBytesFieldNumber = kRequestedBytesFieldNumber();
  public native @Cast("google::protobuf::int64") long requested_bytes();
  public native void set_requested_bytes(@Cast("google::protobuf::int64") long value);

  // optional int64 allocated_bytes = 2;
  public native void clear_allocated_bytes();
  @MemberGetter public static native int kAllocatedBytesFieldNumber();
  public static final int kAllocatedBytesFieldNumber = kAllocatedBytesFieldNumber();
  public native @Cast("google::protobuf::int64") long allocated_bytes();
  public native void set_allocated_bytes(@Cast("google::protobuf::int64") long value);

  // optional string allocator_name = 3;
  public native void clear_allocator_name();
  @MemberGetter public static native int kAllocatorNameFieldNumber();
  public static final int kAllocatorNameFieldNumber = kAllocatorNameFieldNumber();
  public native @StdString BytePointer allocator_name();
  public native void set_allocator_name(@StdString BytePointer value);
  public native void set_allocator_name(@StdString String value);
  public native void set_allocator_name(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_allocator_name(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_allocator_name();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_allocator_name();
  public native void set_allocated_allocator_name(@StdString @Cast({"char*", "std::string*"}) BytePointer allocator_name);
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// AllocationDescription

// optional int64 requested_bytes = 1;




// optional int64 allocated_bytes = 2;




// optional string allocator_name = 3;









// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2fallocation_5fdescription_2eproto__INCLUDED


// Parsed from tensorflow/core/framework/allocator.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_FRAMEWORK_ALLOCATOR_H_
// #define TENSORFLOW_FRAMEWORK_ALLOCATOR_H_

// #include <stdlib.h>
// #include <unistd.h>

// #include <limits>

// #include "tensorflow/core/platform/logging.h"
// #include "tensorflow/core/platform/port.h"

// Allocator is an abstract interface for allocating and deallocating
// device memory.
@Namespace("tensorflow") public static class Allocator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Allocator(Pointer p) { super(p); }


  // Return a string identifying this allocator
  public native @StdString BytePointer Name();

  // Return an uninitialized block of memory that is "num_bytes" bytes
  // in size.  The returned pointer is guaranteed to be aligned to a
  // multiple of "alignment" bytes.
  // REQUIRES: "alignment" is a power of 2.
  public native Pointer AllocateRaw(@Cast("size_t") long alignment, @Cast("size_t") long num_bytes);

  // Deallocate a block of memory pointer to by "ptr"
  // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
  public native void DeallocateRaw(Pointer ptr);

  // Convenience functions to do typed allocation.  Note that these functions
  // do not invoke C++ constructors or destructors.  May return NULL if the
  // tensor has too many elements to represent in a single allocation.

  // Returns true if this allocator tracks the sizes of allocations.
  // RequestedSize and AllocatedSize must be overridden if
  // TracksAlloctionSizes is overridden to return true.
  public native @Cast("bool") boolean TracksAllocationSizes();

  // Returns the user-requested size of the data allocated at
  // 'ptr'.  Note that the actual buffer allocated might be larger
  // than requested, but this function returns the size requested by
  // the user.
  //
  // REQUIRES: TracksAllocationSizes() is true.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  public native @Cast("size_t") long RequestedSize(Pointer ptr);

  // Returns the allocated size of the buffer at 'ptr' if known,
  // otherwise returns RequestedSize(ptr). AllocatedSize(ptr) is
  // guaranteed to be >= RequestedSize(ptr).
  //
  // REQUIRES: TracksAllocationSizes() is true.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  public native @Cast("size_t") long AllocatedSize(Pointer ptr);

  // TODO(jeff): Maybe provide some interface to give info about
  // current allocation state (total number of bytes available for
  // allocation, number of bytes free on device, etc.)
}

// A tensorflow Op may need access to different kinds of memory that
// are not simply a function of the device to which the Op has been
// assigned.  For example, an Op executing on a GPU may still need
// to allocate CPU RAM for some purpose.  Internal to the tensorflow
// runtime we may choose to allocate CPU ram from special regions
// that have been prepared for higher performance in some use
// contexts, e.g. doing DMA with particular devices.  For these
// reasons, the Device interface does not expose just one memory
// Allocator, but instead provides an accessor that takes a
// specification of the desired memory attributes in order to select
// an Allocator.
//
// NOTE: The upper 8 bits of the value are reserved for
// device-specific uses.  Implementors of a device can interpret these
// upper 8 bits in device-specific ways, and ops implemented for those
// devices are responsible for setting those 8 bits appropriately.
//
// Example use:
//  // Allocator for ordinary device memory:
//  Allocator* a = allocator(AllocatorAttributes());
// ...
//  // Allocator for CPU RAM, regardless of where Op is executing:
//  AllocatorAttributes attr;
//  attr.set_on_host(true);
//  Allocator* a = allocator(attr);
@Namespace("tensorflow") public static class AllocatorAttributes extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public AllocatorAttributes() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public AllocatorAttributes(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AllocatorAttributes(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public AllocatorAttributes position(int position) {
        return (AllocatorAttributes)super.position(position);
    }

  public native void set_on_host(@Cast("bool") boolean v);
  public native @Cast("bool") boolean on_host();
  public native void set_nic_compatible(@Cast("bool") boolean v);
  public native @Cast("bool") boolean nic_compatible();
  public native void set_gpu_compatible(@Cast("bool") boolean v);
  public native @Cast("bool") boolean gpu_compatible();

  public native void Merge(@ByVal AllocatorAttributes other);

  public native @Cast("tensorflow::uint32") int value(); public native AllocatorAttributes value(int value);
}

// Returns a trivial implementation of Allocator which uses the system
// default malloc.
@Namespace("tensorflow") public static native Allocator cpu_allocator();

  // namespace tensorflow

// #endif  // TENSORFLOW_FRAMEWORK_ALLOCATOR_H_


// Parsed from tensorflow/core/framework/tensor_shape.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/tensor_shape.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_5fshape_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_5fshape_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2ftensor_5fshape_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2ftensor_5fshape_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2ftensor_5fshape_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class TensorShapeProto_Dim extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorShapeProto_Dim(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TensorShapeProto_Dim(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TensorShapeProto_Dim position(int position) {
        return (TensorShapeProto_Dim)super.position(position);
    }

  public TensorShapeProto_Dim() { super((Pointer)null); allocate(); }
  private native void allocate();

  public TensorShapeProto_Dim(@Const @ByRef TensorShapeProto_Dim from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef TensorShapeProto_Dim from);

  public native @ByRef @Name("operator =") TensorShapeProto_Dim put(@Const @ByRef TensorShapeProto_Dim from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef TensorShapeProto_Dim default_instance();

  public native void Swap(TensorShapeProto_Dim other);

  // implements Message ----------------------------------------------

  public native TensorShapeProto_Dim New();

  public native TensorShapeProto_Dim New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef TensorShapeProto_Dim from);
  public native void MergeFrom(@Const @ByRef TensorShapeProto_Dim from);
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

  // optional int64 size = 1;
  public native void clear_size();
  @MemberGetter public static native int kSizeFieldNumber();
  public static final int kSizeFieldNumber = kSizeFieldNumber();
  public native @Cast("google::protobuf::int64") long size();
  public native void set_size(@Cast("google::protobuf::int64") long value);

  // optional string name = 2;
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
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class TensorShapeProto extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorShapeProto(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TensorShapeProto(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TensorShapeProto position(int position) {
        return (TensorShapeProto)super.position(position);
    }

  public TensorShapeProto() { super((Pointer)null); allocate(); }
  private native void allocate();

  public TensorShapeProto(@Const @ByRef TensorShapeProto from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef TensorShapeProto from);

  public native @ByRef @Name("operator =") TensorShapeProto put(@Const @ByRef TensorShapeProto from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef TensorShapeProto default_instance();

  public native void Swap(TensorShapeProto other);

  // implements Message ----------------------------------------------

  public native TensorShapeProto New();

  public native TensorShapeProto New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef TensorShapeProto from);
  public native void MergeFrom(@Const @ByRef TensorShapeProto from);
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

  // repeated .tensorflow.TensorShapeProto.Dim dim = 2;
  public native int dim_size();
  public native void clear_dim();
  @MemberGetter public static native int kDimFieldNumber();
  public static final int kDimFieldNumber = kDimFieldNumber();
  public native @Const @ByRef TensorShapeProto_Dim dim(int index);
  public native TensorShapeProto_Dim mutable_dim(int index);
  public native TensorShapeProto_Dim add_dim();
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// TensorShapeProto_Dim

// optional int64 size = 1;




// optional string name = 2;









// -------------------------------------------------------------------

// TensorShapeProto

// repeated .tensorflow.TensorShapeProto.Dim dim = 2;








// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_5fshape_2eproto__INCLUDED


// Parsed from tensorflow/core/framework/types.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/types.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2ftypes_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2ftypes_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/generated_enum_reflection.h>
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2ftypes_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2ftypes_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2ftypes_2eproto();


/** enum tensorflow::DataType */
public static final int
  DT_INVALID = 0,
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,
  DT_QUINT8 = 12,
  DT_QINT32 = 13,
  DT_BFLOAT16 = 14,
  DT_FLOAT_REF = 101,
  DT_DOUBLE_REF = 102,
  DT_INT32_REF = 103,
  DT_UINT8_REF = 104,
  DT_INT16_REF = 105,
  DT_INT8_REF = 106,
  DT_STRING_REF = 107,
  DT_COMPLEX64_REF = 108,
  DT_INT64_REF = 109,
  DT_BOOL_REF = 110,
  DT_QINT8_REF = 111,
  DT_QUINT8_REF = 112,
  DT_QINT32_REF = 113,
  DT_BFLOAT16_REF = 114,
  DataType_INT_MIN_SENTINEL_DO_NOT_USE_ =kint32min,
  DataType_INT_MAX_SENTINEL_DO_NOT_USE_ =kint32max;
@Namespace("tensorflow") public static native @Cast("bool") boolean DataType_IsValid(int value);
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::DataType") int DataType_MIN();
@Namespace("tensorflow") @MemberGetter public static native @Cast("const tensorflow::DataType") int DataType_MAX();
@Namespace("tensorflow") @MemberGetter public static native int DataType_ARRAYSIZE();

@Namespace("tensorflow") public static native @Cast("const google::protobuf::EnumDescriptor*") Pointer DataType_descriptor();
@Namespace("tensorflow") public static native @StdString BytePointer DataType_Name(@Cast("tensorflow::DataType") int value);
@Namespace("tensorflow") public static native @Cast("bool") boolean DataType_Parse(
    @StdString BytePointer name, @Cast("tensorflow::DataType*") IntPointer value);
@Namespace("tensorflow") public static native @Cast("bool") boolean DataType_Parse(
    @StdString String name, @Cast("tensorflow::DataType*") IntPointer value);
// ===================================================================


// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// #ifndef SWIG
// #endif  // SWIG

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2ftypes_2eproto__INCLUDED


// Parsed from tensorflow/core/framework/tensor.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/tensor.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/unknown_field_set.h>
// #include "tensorflow/core/framework/tensor_shape.pb.h"
// #include "tensorflow/core/framework/types.pb.h"
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2ftensor_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2ftensor_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2ftensor_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class TensorProto extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorProto(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TensorProto(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TensorProto position(int position) {
        return (TensorProto)super.position(position);
    }

  public TensorProto() { super((Pointer)null); allocate(); }
  private native void allocate();

  public TensorProto(@Const @ByRef TensorProto from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef TensorProto from);

  public native @ByRef @Name("operator =") TensorProto put(@Const @ByRef TensorProto from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef TensorProto default_instance();

  public native void Swap(TensorProto other);

  // implements Message ----------------------------------------------

  public native TensorProto New();

  public native TensorProto New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef TensorProto from);
  public native void MergeFrom(@Const @ByRef TensorProto from);
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

  // optional .tensorflow.DataType dtype = 1;
  public native void clear_dtype();
  @MemberGetter public static native int kDtypeFieldNumber();
  public static final int kDtypeFieldNumber = kDtypeFieldNumber();
  public native @Cast("tensorflow::DataType") int dtype();
  public native void set_dtype(@Cast("tensorflow::DataType") int value);

  // optional .tensorflow.TensorShapeProto tensor_shape = 2;
  public native @Cast("bool") boolean has_tensor_shape();
  public native void clear_tensor_shape();
  @MemberGetter public static native int kTensorShapeFieldNumber();
  public static final int kTensorShapeFieldNumber = kTensorShapeFieldNumber();
  public native @Const @ByRef TensorShapeProto tensor_shape();
  public native TensorShapeProto mutable_tensor_shape();
  public native TensorShapeProto release_tensor_shape();
  public native void set_allocated_tensor_shape(TensorShapeProto tensor_shape);

  // optional int32 version_number = 3;
  public native void clear_version_number();
  @MemberGetter public static native int kVersionNumberFieldNumber();
  public static final int kVersionNumberFieldNumber = kVersionNumberFieldNumber();
  public native @Cast("google::protobuf::int32") int version_number();
  public native void set_version_number(@Cast("google::protobuf::int32") int value);

  // optional bytes tensor_content = 4;
  public native void clear_tensor_content();
  @MemberGetter public static native int kTensorContentFieldNumber();
  public static final int kTensorContentFieldNumber = kTensorContentFieldNumber();
  public native @StdString BytePointer tensor_content();
  public native void set_tensor_content(@StdString BytePointer value);
  public native void set_tensor_content(@StdString String value);
  public native void set_tensor_content(@Const Pointer value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_tensor_content();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_tensor_content();
  public native void set_allocated_tensor_content(@StdString @Cast({"char*", "std::string*"}) BytePointer tensor_content);

  // repeated float float_val = 5 [packed = true];
  public native int float_val_size();
  public native void clear_float_val();
  @MemberGetter public static native int kFloatValFieldNumber();
  public static final int kFloatValFieldNumber = kFloatValFieldNumber();
  public native float float_val(int index);
  public native void set_float_val(int index, float value);
  public native void add_float_val(float value);

  // repeated double double_val = 6 [packed = true];
  public native int double_val_size();
  public native void clear_double_val();
  @MemberGetter public static native int kDoubleValFieldNumber();
  public static final int kDoubleValFieldNumber = kDoubleValFieldNumber();
  public native double double_val(int index);
  public native void set_double_val(int index, double value);
  public native void add_double_val(double value);

  // repeated int32 int_val = 7 [packed = true];
  public native int int_val_size();
  public native void clear_int_val();
  @MemberGetter public static native int kIntValFieldNumber();
  public static final int kIntValFieldNumber = kIntValFieldNumber();
  public native @Cast("google::protobuf::int32") int int_val(int index);
  public native void set_int_val(int index, @Cast("google::protobuf::int32") int value);
  public native void add_int_val(@Cast("google::protobuf::int32") int value);

  // repeated bytes string_val = 8;
  public native int string_val_size();
  public native void clear_string_val();
  @MemberGetter public static native int kStringValFieldNumber();
  public static final int kStringValFieldNumber = kStringValFieldNumber();
  public native @StdString BytePointer string_val(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_string_val(int index);
  public native void set_string_val(int index, @StdString BytePointer value);
  public native void set_string_val(int index, @StdString String value);
  public native void set_string_val(int index, @Const Pointer value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_string_val();
  public native void add_string_val(@StdString BytePointer value);
  public native void add_string_val(@StdString String value);
  public native void add_string_val(@Const Pointer value, @Cast("size_t") long size);

  // repeated float scomplex_val = 9 [packed = true];
  public native int scomplex_val_size();
  public native void clear_scomplex_val();
  @MemberGetter public static native int kScomplexValFieldNumber();
  public static final int kScomplexValFieldNumber = kScomplexValFieldNumber();
  public native float scomplex_val(int index);
  public native void set_scomplex_val(int index, float value);
  public native void add_scomplex_val(float value);

  // repeated int64 int64_val = 10 [packed = true];
  public native int int64_val_size();
  public native void clear_int64_val();
  @MemberGetter public static native int kInt64ValFieldNumber();
  public static final int kInt64ValFieldNumber = kInt64ValFieldNumber();
  public native @Cast("google::protobuf::int64") long int64_val(int index);
  public native void set_int64_val(int index, @Cast("google::protobuf::int64") long value);
  public native void add_int64_val(@Cast("google::protobuf::int64") long value);

  // repeated bool bool_val = 11 [packed = true];
  public native int bool_val_size();
  public native void clear_bool_val();
  @MemberGetter public static native int kBoolValFieldNumber();
  public static final int kBoolValFieldNumber = kBoolValFieldNumber();
  public native @Cast("bool") boolean bool_val(int index);
  public native void set_bool_val(int index, @Cast("bool") boolean value);
  public native void add_bool_val(@Cast("bool") boolean value);
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// TensorProto

// optional .tensorflow.DataType dtype = 1;




// optional .tensorflow.TensorShapeProto tensor_shape = 2;







// optional int32 version_number = 3;




// optional bytes tensor_content = 4;









// repeated float float_val = 5 [packed = true];








// repeated double double_val = 6 [packed = true];








// repeated int32 int_val = 7 [packed = true];








// repeated bytes string_val = 8;














// repeated float scomplex_val = 9 [packed = true];








// repeated int64 int64_val = 10 [packed = true];








// repeated bool bool_val = 11 [packed = true];








// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_2eproto__INCLUDED


// Parsed from tensorflow/core/framework/tensor_description.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/tensor_description.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_5fdescription_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_5fdescription_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/unknown_field_set.h>
// #include "tensorflow/core/framework/types.pb.h"
// #include "tensorflow/core/framework/tensor_shape.pb.h"
// #include "tensorflow/core/framework/allocation_description.pb.h"
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2ftensor_5fdescription_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2ftensor_5fdescription_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2ftensor_5fdescription_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class TensorDescription extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorDescription(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TensorDescription(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TensorDescription position(int position) {
        return (TensorDescription)super.position(position);
    }

  public TensorDescription() { super((Pointer)null); allocate(); }
  private native void allocate();

  public TensorDescription(@Const @ByRef TensorDescription from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef TensorDescription from);

  public native @ByRef @Name("operator =") TensorDescription put(@Const @ByRef TensorDescription from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef TensorDescription default_instance();

  public native void Swap(TensorDescription other);

  // implements Message ----------------------------------------------

  public native TensorDescription New();

  public native TensorDescription New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef TensorDescription from);
  public native void MergeFrom(@Const @ByRef TensorDescription from);
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

  // optional .tensorflow.DataType dtype = 1;
  public native void clear_dtype();
  @MemberGetter public static native int kDtypeFieldNumber();
  public static final int kDtypeFieldNumber = kDtypeFieldNumber();
  public native @Cast("tensorflow::DataType") int dtype();
  public native void set_dtype(@Cast("tensorflow::DataType") int value);

  // optional .tensorflow.TensorShapeProto shape = 2;
  public native @Cast("bool") boolean has_shape();
  public native void clear_shape();
  @MemberGetter public static native int kShapeFieldNumber();
  public static final int kShapeFieldNumber = kShapeFieldNumber();
  public native @Const @ByRef TensorShapeProto shape();
  public native TensorShapeProto mutable_shape();
  public native TensorShapeProto release_shape();
  public native void set_allocated_shape(TensorShapeProto shape);

  // optional .tensorflow.AllocationDescription allocation_description = 4;
  public native @Cast("bool") boolean has_allocation_description();
  public native void clear_allocation_description();
  @MemberGetter public static native int kAllocationDescriptionFieldNumber();
  public static final int kAllocationDescriptionFieldNumber = kAllocationDescriptionFieldNumber();
  public native @Const @ByRef AllocationDescription allocation_description();
  public native AllocationDescription mutable_allocation_description();
  public native AllocationDescription release_allocation_description();
  public native void set_allocated_allocation_description(AllocationDescription allocation_description);
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// TensorDescription

// optional .tensorflow.DataType dtype = 1;




// optional .tensorflow.TensorShapeProto shape = 2;







// optional .tensorflow.AllocationDescription allocation_description = 4;







// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2ftensor_5fdescription_2eproto__INCLUDED


// Parsed from tensorflow/core/framework/tensor_types.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_
// #define TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// Helper to define Tensor types given that the scalar is of type T.

  // namespace tensorflow
// #endif  // TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_


// Parsed from tensorflow/core/public/tensor_shape.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PUBLIC_TENSOR_SHAPE_H_
// #define TENSORFLOW_PUBLIC_TENSOR_SHAPE_H_

// #include <string>

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "tensorflow/core/framework/tensor_shape.pb.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/lib/gtl/inlined_vector.h"
// #include "tensorflow/core/lib/strings/strcat.h"
// #include "tensorflow/core/platform/logging.h"  // Declared below

/** Manages the dimensions of a Tensor and their sizes. */
@Namespace("tensorflow") @NoOffset public static class TensorShape extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorShape(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TensorShape(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public TensorShape position(int position) {
        return (TensorShape)super.position(position);
    }

  /** \brief Construct a {@code TensorShape} from the provided sizes.
   *  REQUIRES: {@code dim_sizes[i] >= 0} */
  public TensorShape(@Cast("tensorflow::int64*") @ArraySlice LongPointer dim_sizes) { super((Pointer)null); allocate(dim_sizes); }
  private native void allocate(@Cast("tensorflow::int64*") @ArraySlice LongPointer dim_sizes);
  public TensorShape(@Cast("tensorflow::int64*") @ArraySlice LongBuffer dim_sizes) { super((Pointer)null); allocate(dim_sizes); }
  private native void allocate(@Cast("tensorflow::int64*") @ArraySlice LongBuffer dim_sizes);
  public TensorShape(@Cast("tensorflow::int64*") @ArraySlice long... dim_sizes) { super((Pointer)null); allocate(dim_sizes); }
  private native void allocate(@Cast("tensorflow::int64*") @ArraySlice long... dim_sizes);

  /** REQUIRES: {@code IsValid(proto)} */
  public TensorShape(@Const @ByRef TensorShapeProto proto) { super((Pointer)null); allocate(proto); }
  private native void allocate(@Const @ByRef TensorShapeProto proto);

  /** Create a tensor shape with no dimensions and one element, which you can
   *  then call {@code AddDim()} on. */
  public TensorShape() { super((Pointer)null); allocate(); }
  private native void allocate();

  /** Returns {@code true} iff {@code proto} is a valid tensor shape. */
  public static native @Cast("bool") boolean IsValid(@Const @ByRef TensorShapeProto proto);

  /** Clear a tensor shape */
  public native void Clear();

  /** \brief Add a dimension to the end ("inner-most").
   *  REQUIRES: {@code size >= 0} */
  public native void AddDim(@Cast("tensorflow::int64") long size);

  /** Appends all the dimensions from {@code shape}. */
  public native void AppendShape(@Const @ByRef TensorShape shape);

  /** \brief Insert a dimension somewhere in the {@code TensorShape}.
   *  REQUIRES: {@code 0 <= d <= dims()}
   *  REQUIRES: {@code size >= 0} */
  public native void InsertDim(int d, @Cast("tensorflow::int64") long size);

  /** \brief Modifies the size of the dimension {@code d} to be {@code size}
   *  REQUIRES: {@code 0 <= d < dims()}
   *  REQUIRES: {@code size >= 0} */
  public native void set_dim(int d, @Cast("tensorflow::int64") long size);

  /** \brief Removes dimension {@code d} from the {@code TensorShape}.
   *  REQUIRES: {@code 0 <= d < dims()} */
  public native void RemoveDim(int d);

  /** Return the number of dimensions in the tensor. */
  public native int dims();

  /** \brief Returns the number of elements in dimension {@code d}.
   *  REQUIRES: {@code 0 <= d < dims()} */
  // TODO(touts): Rename to `dimension()` to match
  // `Eigen::Tensor::dimension()`?
  public native @Cast("tensorflow::int64") long dim_size(int d);

  /** Returns sizes of all dimensions. */
  
  ///
  public native @Cast("tensorflow::int64*") @ArraySlice LongPointer dim_sizes();

  /** \brief Returns the number of elements in the tensor.
   * 
   *  We use {@code int64} and not {@code size_t} to be compatible with {@code Eigen::Tensor}
   *  which uses {@code ptrdiff_t}. */
  public native @Cast("tensorflow::int64") long num_elements();

  /** Returns true if {@code *this} and {@code b} have the same sizes. Ignores
   *  dimension names. */
  public native @Cast("bool") boolean IsSameSize(@Const @ByRef TensorShape b);
  public native @Cast("bool") @Name("operator ==") boolean equals(@Const @ByRef TensorShape b);

  /** Fill {@code *proto} from {@code *this}. */
  public native void AsProto(TensorShapeProto proto);

  /** Fill {@code *dsizes} from {@code *this}. */

  /** Same as {@code AsEigenDSizes()} but allows for {@code NDIMS > dims()} -- in
   *  which case we pad the rest of the sizes with 1. */

  /** For iterating through the dimensions. */
  public native @ByVal TensorShapeIter begin();
  public native @ByVal TensorShapeIter end();

  /** For error messages. */
  public native @StdString BytePointer DebugString();
  // TODO(vrv): Remove this, this is the same as DebugString().
  public native @StdString BytePointer ShortDebugString();
}

@Namespace("tensorflow") @NoOffset public static class TensorShapeDim extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorShapeDim(Pointer p) { super(p); }

  public TensorShapeDim(@Cast("tensorflow::int64") long s) { super((Pointer)null); allocate(s); }
  private native void allocate(@Cast("tensorflow::int64") long s);
  public native int size(); public native TensorShapeDim size(int size);
}

@Namespace("tensorflow") @NoOffset public static class TensorShapeIter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorShapeIter(Pointer p) { super(p); }

  public TensorShapeIter(@Const TensorShape shape, int d) { super((Pointer)null); allocate(shape, d); }
  private native void allocate(@Const TensorShape shape, int d);
  public native @Cast("bool") @Name("operator ==") boolean equals(@Const @ByRef TensorShapeIter rhs);
  public native @Cast("bool") @Name("operator !=") boolean notEquals(@Const @ByRef TensorShapeIter rhs);
  public native @Name("operator ++") void increment();
  public native @ByVal @Name("operator *") TensorShapeDim multiply();
}

// In some places, allow shape (1,) to be treated as a scalar and shape () to be
// treated as a vector.  This flag is for temporary backwards compatibility
// only, and will be changed to strict within Google around November 15, 2015.
// #if defined(PLATFORM_GOOGLE)
// TODO(irving): Become strict on November 15, 2015.
@Namespace("tensorflow") @MemberGetter public static native @Cast("const bool") boolean kAllowLegacyScalars();
public static final boolean kAllowLegacyScalars = kAllowLegacyScalars();
// #else
// For open source (outside Google), we are strict.
// #endif

/** \brief Static helper routines for {@code TensorShape}. Includes a few common
 *  predicates on a tensor shape. */
@Namespace("tensorflow") public static class TensorShapeUtils extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public TensorShapeUtils() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public TensorShapeUtils(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorShapeUtils(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public TensorShapeUtils position(int position) {
        return (TensorShapeUtils)super.position(position);
    }

  public static native @Cast("bool") boolean IsScalar(@Const @ByRef TensorShape shape);

  public static native @Cast("bool") boolean IsVector(@Const @ByRef TensorShape shape);

  // Allow either scalars or (if allowing legacy scalars) shape (1,).
  public static native @Cast("bool") boolean IsLegacyScalar(@Const @ByRef TensorShape shape);

  // Allow rank 1 or (if allowing legacy scalars) rank 0.
  public static native @Cast("bool") boolean IsLegacyVector(@Const @ByRef TensorShape shape);

  public static native @Cast("bool") boolean IsVectorOrHigher(@Const @ByRef TensorShape shape);

  public static native @Cast("bool") boolean IsMatrix(@Const @ByRef TensorShape shape);

  public static native @Cast("bool") boolean IsMatrixOrHigher(@Const @ByRef TensorShape shape);

  /** \brief Returns a {@code TensorShape} whose dimensions are
   *  {@code dims[0]}, {@code dims[1]}, ..., {@code dims[n-1]}. */

  public static native @StdString BytePointer ShapeListString(@Const @ByRef TensorShapeVector shapes);

  public static native @Cast("bool") boolean StartsWith(@Const @ByRef TensorShape shape0, @Const @ByRef TensorShape shape1);
}

// ----------------------------------------------------------------------------
// Template method implementation details below
// ----------------------------------------------------------------------------





  // namespace tensorflow

// #endif  // TENSORFLOW_PUBLIC_TENSOR_SHAPE_H_


// Parsed from tensorflow/core/public/tensor.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PUBLIC_TENSOR_H_
// #define TENSORFLOW_PUBLIC_TENSOR_H_

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "tensorflow/core/framework/allocation_description.pb.h"
// #include "tensorflow/core/framework/allocator.h"
// #include "tensorflow/core/framework/tensor.pb.h"
// #include "tensorflow/core/framework/tensor_description.pb.h"
// #include "tensorflow/core/framework/tensor_types.h"
// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/framework/types.pb.h"
// #include "tensorflow/core/lib/core/refcount.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/platform/logging.h"
// #include "tensorflow/core/platform/port.h"
// #include "tensorflow/core/public/status.h"
// #include "tensorflow/core/public/tensor_shape.h"  // Forward declaration.
@Namespace("tensorflow") @Opaque public static class TensorCApi extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TensorCApi() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorCApi(Pointer p) { super(p); }
}

/** Represents an n-dimensional array of values. */
@Namespace("tensorflow") @NoOffset public static class Tensor extends AbstractTensor {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Tensor(Pointer p) { super(p); }

  /** Default Tensor constructor. Creates a 1-dimension, 0-element float tensor. */
  
  ///
  public Tensor() { super((Pointer)null); allocate(); }
  private native void allocate();

  /** \brief Creates a Tensor of the given {@code type} and {@code shape}.
   * 
   *  The underlying buffer is allocated using a {@code CPUAllocator}. */
  
  ///
  public Tensor(@Cast("tensorflow::DataType") int type, @Const @ByRef TensorShape shape) { super((Pointer)null); allocate(type, shape); }
  private native void allocate(@Cast("tensorflow::DataType") int type, @Const @ByRef TensorShape shape);

  /** \brief Creates a tensor with the input {@code type} and {@code shape}, using the
   *  allocator {@code a} to allocate the underlying buffer.
   * 
   *  {@code a} must outlive the lifetime of this Tensor. */
  public Tensor(Allocator a, @Cast("tensorflow::DataType") int type, @Const @ByRef TensorShape shape) { super((Pointer)null); allocate(a, type, shape); }
  private native void allocate(Allocator a, @Cast("tensorflow::DataType") int type, @Const @ByRef TensorShape shape);

  /** Creates an uninitialized Tensor of the given data type. */
  public Tensor(@Cast("tensorflow::DataType") int type) { super((Pointer)null); allocate(type); }
  private native void allocate(@Cast("tensorflow::DataType") int type);

  public Tensor(@Const @ByRef Tensor other) { super((Pointer)null); allocate(other); }
  private native void allocate(@Const @ByRef Tensor other);  /** Copy constructor. */

  /** Returns the data type. */
  public native @Cast("tensorflow::DataType") int dtype();

  /** Returns the shape of the tensor. */
  
  ///
  public native @Const @ByRef TensorShape shape();

  /** \brief Convenience accessor for the tensor shape.
   * 
   *  For all shape accessors, see comments for relevant methods of
   *  {@code TensorShape} in {@code tensor_shape.h}. */
  public native int dims();

  /** Convenience accessor for the tensor shape. */
  public native @Cast("tensorflow::int64") long dim_size(int d);

  /** Convenience accessor for the tensor shape. */
  public native @Cast("tensorflow::int64") long NumElements();

  public native @Cast("bool") boolean IsSameSize(@Const @ByRef Tensor b);

  /** Has this Tensor been initialized? */
  public native @Cast("bool") boolean IsInitialized();

  /** Returns the estimated memory usage of this tensor. */
  public native @Cast("size_t") long TotalBytes();

  /** Assign operator. This tensor shares other's underlying storage. */
  
  ///
  public native @ByRef @Name("operator =") Tensor put(@Const @ByRef Tensor other);

  /** \brief Copy the other tensor into this tensor and reshape it.
   * 
   *  This tensor shares other's underlying storage. Returns {@code true}
   *  iff {@code other.shape()} has the same number of elements of the given
   *  {@code shape}. */
  
  ///
  ///
  public native @Cast("bool") boolean CopyFrom(@Const @ByRef Tensor other,
                  @Const @ByRef TensorShape shape);

  /** \brief Slice this tensor along the 1st dimension.
   <p>
   *  I.e., the returned tensor satisfies
   *      returned[i, ...] == this[dim0_start + i, ...].
   *  The returned tensor shares the underlying tensor buffer with this
   *  tensor.
   * 
   *  NOTE: The returned tensor may not satisfies the same alignment
   *  requirement as this tensor depending on the shape. The caller
   *  must check the returned tensor's alignment before calling certain
   *  methods that have alignment requirement (e.g., {@code flat()}, {@code tensor()}).
   * 
   *  REQUIRES: {@code dims()} >= 1
   *  REQUIRES: {@code 0 <= dim0_start <= dim0_limit <= dim_size(0)} */
  public native @ByVal Tensor Slice(@Cast("tensorflow::int64") long dim0_start, @Cast("tensorflow::int64") long dim0_limit);

  /** \brief Parse {@code other} and construct the tensor.
   <p>
   *  Returns {@code true} iff the parsing succeeds. If the parsing fails,
   *  the state of {@code *this} is unchanged. */
  public native @Cast("bool") boolean FromProto(@Const @ByRef TensorProto other);
  
  ///
  public native @Cast("bool") boolean FromProto(Allocator a, @Const @ByRef TensorProto other);

  /** \brief Fills in {@code proto} with {@code *this} tensor's content.
   * 
   *  {@code AsProtoField()} fills in the repeated field for {@code proto.dtype()}, while
   *  {@code AsProtoTensorContent()} encodes the content in {@code proto.tensor_content()}
   *  in a compact form. */
  public native void AsProtoField(TensorProto proto);
  
  ///
  ///
  ///
  ///
  ///
  public native void AsProtoTensorContent(TensorProto proto);

  /** \brief Return the tensor data as an {@code Eigen::Tensor} with the type and
   *  sizes of this {@code Tensor}.
   * 
   *  Use these methods when you know the data type and the number of
   *  dimensions of the Tensor and you want an {@code Eigen::Tensor}
   *  automatically sized to the {@code Tensor} sizes. The implementation check
   *  fails if either type or sizes mismatch.
   * 
   *  Example:
   * 
   *  <pre>{@code c++
   * 
   *      typedef float T;
   *      Tensor my_mat(...built with Shape{rows: 3, cols: 5}...);
   *      auto mat = my_mat.matrix<T>();    // 2D Eigen::Tensor, 3 x 5.
   *      auto mat = my_mat.tensor<T, 2>(); // 2D Eigen::Tensor, 3 x 5.
   *      auto vec = my_mat.vec<T>();       // CHECK fails as my_mat is 2D.
   *      auto vec = my_mat.tensor<T, 3>(); // CHECK fails as my_mat is 2D.
   *      auto mat = my_mat.matrix<int32>();// CHECK fails as type mismatch.
   * 
   *  }</pre> */

  /** \brief Return the tensor data as an {@code Eigen::Tensor} of the data type and a
   *  specified shape.
   * 
   *  These methods allow you to access the data with the dimensions
   *  and sizes of your choice.  You do not need to know the number of
   *  dimensions of the Tensor to call them.  However, they {@code CHECK} that
   *  the type matches and the dimensions requested creates an
   *  {@code Eigen::Tensor} with the same number of elements as the tensor.
   * 
   *  Example:
   * 
   *  <pre>{@code c++
   * 
   *      typedef float T;
   *      Tensor my_ten(...built with Shape{planes: 4, rows: 3, cols: 5}...);
   *      // 1D Eigen::Tensor, size 60:
   *      auto flat = my_ten.flat<T>();
   *      // 2D Eigen::Tensor 12 x 5:
   *      auto inner = my_ten.flat_inner_dims<T>();
   *      // 2D Eigen::Tensor 4 x 15:
   *      auto outer = my_ten.shaped<T, 2>({4, 15});
   *      // CHECK fails, bad num elements:
   *      auto outer = my_ten.shaped<T, 2>({4, 8});
   *      // 3D Eigen::Tensor 6 x 5 x 2:
   *      auto weird = my_ten.shaped<T, 3>({6, 5, 2});
   *      // CHECK fails, type mismatch:
   *      auto bad   = my_ten.flat<int32>();
   * 
   *  }</pre> */

  /** Returns the data as an Eigen::Tensor with 2 dimensions, collapsing all
   *  Tensor dimensions but the last one into the first dimension of the result. */

  /** Returns the data as an Eigen::Tensor with 2 dimensions, collapsing all
   *  Tensor dimensions but the first one into the last dimension of the result. */

  /** \brief Return the Tensor data as a {@code TensorMap} of fixed size 1:
   *  {@code TensorMap<TensorFixedSize<T, 1>>}.
   <p>
   *  Using {@code scalar()} allows the compiler to perform optimizations as
   *  the size of the tensor is known at compile time. */

  /** Const versions of all the methods above. */

  /** Render the first {@code max_entries} values in {@code *this} into a string. */
  public native @StdString BytePointer SummarizeValue(@Cast("tensorflow::int64") long max_entries);

  /** A human-readable summary of the tensor suitable for debugging. */
  public native @StdString BytePointer DebugString();

  /** Fill in the {@code TensorDescription} proto with metadata about the
   *  tensor that is useful for monitoring and debugging. */
  
  ///
  ///
  ///
  public native void FillDescription(TensorDescription description);

  /** \brief Returns a {@code StringPiece} mapping the current tensor's buffer.
   * 
   *  The returned {@code StringPiece} may point to memory location on devices
   *  that the CPU cannot address directly.
   * 
   *  NOTE: The underlying tensor buffer is refcounted, so the lifetime
   *  of the contents mapped by the {@code StringPiece} matches the lifetime of
   *  the buffer; callers should arrange to make sure the buffer does
   *  not get destroyed while the {@code StringPiece} is still used.
   * 
   *  REQUIRES: {@code DataTypeCanUseMemcpy(dtype())}. */
  public native @StringPiece BytePointer tensor_data();
}

// Implementation details

// Interface to access the raw ref-counted data buffer.
@Namespace("tensorflow") public static class TensorBuffer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorBuffer(Pointer p) { super(p); }


  // data() points to a memory region of size() bytes.
  public native Pointer data();
  public native @Cast("size_t") long size();

  // If this TensorBuffer is sub-buffer of another TensorBuffer,
  // returns that TensorBuffer. Otherwise, returns this.
  public native TensorBuffer root_buffer();

  // Fill metadata about the allocation into the proto.
  public native void FillAllocationDescription(
        AllocationDescription proto);
}

@Namespace("tensorflow") public static native void CheckEigenAlignment(@Const Pointer ptr);



















  // namespace tensorflow

// #endif  // TENSORFLOW_PUBLIC_TENSOR_H_


// Parsed from tensorflow/core/framework/attr_value.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/attr_value.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2fattr_5fvalue_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2fattr_5fvalue_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/map.h>
// #include <google/protobuf/map_field_inl.h>
// #include <google/protobuf/unknown_field_set.h>
// #include "tensorflow/core/framework/tensor.pb.h"
// #include "tensorflow/core/framework/tensor_shape.pb.h"
// #include "tensorflow/core/framework/types.pb.h"
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2fattr_5fvalue_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2fattr_5fvalue_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2fattr_5fvalue_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class AttrValue_ListValue extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AttrValue_ListValue(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public AttrValue_ListValue(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public AttrValue_ListValue position(int position) {
        return (AttrValue_ListValue)super.position(position);
    }

  public AttrValue_ListValue() { super((Pointer)null); allocate(); }
  private native void allocate();

  public AttrValue_ListValue(@Const @ByRef AttrValue_ListValue from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef AttrValue_ListValue from);

  public native @ByRef @Name("operator =") AttrValue_ListValue put(@Const @ByRef AttrValue_ListValue from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef AttrValue_ListValue default_instance();

  public native void Swap(AttrValue_ListValue other);

  // implements Message ----------------------------------------------

  public native AttrValue_ListValue New();

  public native AttrValue_ListValue New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef AttrValue_ListValue from);
  public native void MergeFrom(@Const @ByRef AttrValue_ListValue from);
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

  // repeated bytes s = 2;
  public native int s_size();
  public native void clear_s();
  @MemberGetter public static native int kSFieldNumber();
  public static final int kSFieldNumber = kSFieldNumber();
  public native @StdString BytePointer s(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_s(int index);
  public native void set_s(int index, @StdString BytePointer value);
  public native void set_s(int index, @StdString String value);
  public native void set_s(int index, @Const Pointer value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_s();
  public native void add_s(@StdString BytePointer value);
  public native void add_s(@StdString String value);
  public native void add_s(@Const Pointer value, @Cast("size_t") long size);

  // repeated int64 i = 3 [packed = true];
  public native int i_size();
  public native void clear_i();
  @MemberGetter public static native int kIFieldNumber();
  public static final int kIFieldNumber = kIFieldNumber();
  public native @Cast("google::protobuf::int64") long i(int index);
  public native void set_i(int index, @Cast("google::protobuf::int64") long value);
  public native void add_i(@Cast("google::protobuf::int64") long value);

  // repeated float f = 4 [packed = true];
  public native int f_size();
  public native void clear_f();
  @MemberGetter public static native int kFFieldNumber();
  public static final int kFFieldNumber = kFFieldNumber();
  public native float f(int index);
  public native void set_f(int index, float value);
  public native void add_f(float value);

  // repeated bool b = 5 [packed = true];
  public native int b_size();
  public native void clear_b();
  @MemberGetter public static native int kBFieldNumber();
  public static final int kBFieldNumber = kBFieldNumber();
  public native @Cast("bool") boolean b(int index);
  public native void set_b(int index, @Cast("bool") boolean value);
  public native void add_b(@Cast("bool") boolean value);

  // repeated .tensorflow.DataType type = 6 [packed = true];
  public native int type_size();
  public native void clear_type();
  @MemberGetter public static native int kTypeFieldNumber();
  public static final int kTypeFieldNumber = kTypeFieldNumber();
  public native @Cast("tensorflow::DataType") int type(int index);
  public native void set_type(int index, @Cast("tensorflow::DataType") int value);
  public native void add_type(@Cast("tensorflow::DataType") int value);

  // repeated .tensorflow.TensorShapeProto shape = 7;
  public native int shape_size();
  public native void clear_shape();
  @MemberGetter public static native int kShapeFieldNumber();
  public static final int kShapeFieldNumber = kShapeFieldNumber();
  public native @Const @ByRef TensorShapeProto shape(int index);
  public native TensorShapeProto mutable_shape(int index);
  public native TensorShapeProto add_shape();

  // repeated .tensorflow.TensorProto tensor = 8;
  public native int tensor_size();
  public native void clear_tensor();
  @MemberGetter public static native int kTensorFieldNumber();
  public static final int kTensorFieldNumber = kTensorFieldNumber();
  public native @Const @ByRef TensorProto tensor(int index);
  public native TensorProto mutable_tensor(int index);
  public native TensorProto add_tensor();
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class AttrValue extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AttrValue(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public AttrValue(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public AttrValue position(int position) {
        return (AttrValue)super.position(position);
    }

  public AttrValue() { super((Pointer)null); allocate(); }
  private native void allocate();

  public AttrValue(@Const @ByRef AttrValue from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef AttrValue from);

  public native @ByRef @Name("operator =") AttrValue put(@Const @ByRef AttrValue from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef AttrValue default_instance();

  /** enum tensorflow::AttrValue::ValueCase */
  public static final int
    kS = 2,
    kI = 3,
    kF = 4,
    kB = 5,
    kType = 6,
    kShape = 7,
    kTensor = 8,
    kList = 1,
    kFunc = 10,
    kPlaceholder = 9,
    VALUE_NOT_SET = 0;

  public native void Swap(AttrValue other);

  // implements Message ----------------------------------------------

  public native AttrValue New();

  public native AttrValue New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef AttrValue from);
  public native void MergeFrom(@Const @ByRef AttrValue from);
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
  public native void clear_s();
  @MemberGetter public static native int kSFieldNumber();
  public static final int kSFieldNumber = kSFieldNumber();
  public native @StdString BytePointer s();
  public native void set_s(@StdString BytePointer value);
  public native void set_s(@StdString String value);
  public native void set_s(@Const Pointer value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_s();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_s();
  public native void set_allocated_s(@StdString @Cast({"char*", "std::string*"}) BytePointer s);
  public native void clear_i();
  @MemberGetter public static native int kIFieldNumber();
  public static final int kIFieldNumber = kIFieldNumber();
  public native @Cast("google::protobuf::int64") long i();
  public native void set_i(@Cast("google::protobuf::int64") long value);
  public native void clear_f();
  @MemberGetter public static native int kFFieldNumber();
  public static final int kFFieldNumber = kFFieldNumber();
  public native float f();
  public native void set_f(float value);
  public native void clear_b();
  @MemberGetter public static native int kBFieldNumber();
  public static final int kBFieldNumber = kBFieldNumber();
  public native @Cast("bool") boolean b();
  public native void set_b(@Cast("bool") boolean value);
  public native void clear_type();
  @MemberGetter public static native int kTypeFieldNumber();
  public static final int kTypeFieldNumber = kTypeFieldNumber();
  public native @Cast("tensorflow::DataType") int type();
  public native void set_type(@Cast("tensorflow::DataType") int value);

  // optional .tensorflow.TensorShapeProto shape = 7;
  public native @Cast("bool") boolean has_shape();
  public native void clear_shape();
  @MemberGetter public static native int kShapeFieldNumber();
  public static final int kShapeFieldNumber = kShapeFieldNumber();
  public native @Const @ByRef TensorShapeProto shape();
  public native TensorShapeProto mutable_shape();
  public native TensorShapeProto release_shape();
  public native void set_allocated_shape(TensorShapeProto shape);

  // optional .tensorflow.TensorProto tensor = 8;
  public native @Cast("bool") boolean has_tensor();
  public native void clear_tensor();
  @MemberGetter public static native int kTensorFieldNumber();
  public static final int kTensorFieldNumber = kTensorFieldNumber();
  public native @Const @ByRef TensorProto tensor();
  public native TensorProto mutable_tensor();
  public native TensorProto release_tensor();
  public native void set_allocated_tensor(TensorProto tensor);

  // optional .tensorflow.AttrValue.ListValue list = 1;
  public native @Cast("bool") boolean has_list();
  public native void clear_list();
  @MemberGetter public static native int kListFieldNumber();
  public static final int kListFieldNumber = kListFieldNumber();
  public native @Const @ByRef AttrValue_ListValue list();
  public native AttrValue_ListValue mutable_list();
  public native AttrValue_ListValue release_list();
  public native void set_allocated_list(AttrValue_ListValue list);

  // optional .tensorflow.NameAttrList func = 10;
  public native @Cast("bool") boolean has_func();
  public native void clear_func();
  @MemberGetter public static native int kFuncFieldNumber();
  public static final int kFuncFieldNumber = kFuncFieldNumber();
  public native @Const @ByRef NameAttrList func();
  public native NameAttrList mutable_func();
  public native NameAttrList release_func();
  public native void set_allocated_func(NameAttrList func);
  public native void clear_placeholder();
  @MemberGetter public static native int kPlaceholderFieldNumber();
  public static final int kPlaceholderFieldNumber = kPlaceholderFieldNumber();
  public native @StdString BytePointer placeholder();
  public native void set_placeholder(@StdString BytePointer value);
  public native void set_placeholder(@StdString String value);
  public native void set_placeholder(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_placeholder(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_placeholder();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_placeholder();
  public native void set_allocated_placeholder(@StdString @Cast({"char*", "std::string*"}) BytePointer placeholder);

  public native @Cast("tensorflow::AttrValue::ValueCase") int value_case();
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class NameAttrList extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NameAttrList(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NameAttrList(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public NameAttrList position(int position) {
        return (NameAttrList)super.position(position);
    }

  public NameAttrList() { super((Pointer)null); allocate(); }
  private native void allocate();

  public NameAttrList(@Const @ByRef NameAttrList from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef NameAttrList from);

  public native @ByRef @Name("operator =") NameAttrList put(@Const @ByRef NameAttrList from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef NameAttrList default_instance();

  public native void Swap(NameAttrList other);

  // implements Message ----------------------------------------------

  public native NameAttrList New();

  public native NameAttrList New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef NameAttrList from);
  public native void MergeFrom(@Const @ByRef NameAttrList from);
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

  // map<string, .tensorflow.AttrValue> attr = 2;
  public native int attr_size();
  public native void clear_attr();
  @MemberGetter public static native int kAttrFieldNumber();
  public static final int kAttrFieldNumber = kAttrFieldNumber();
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// AttrValue_ListValue

// repeated bytes s = 2;














// repeated int64 i = 3 [packed = true];








// repeated float f = 4 [packed = true];








// repeated bool b = 5 [packed = true];








// repeated .tensorflow.DataType type = 6 [packed = true];








// repeated .tensorflow.TensorShapeProto shape = 7;








// repeated .tensorflow.TensorProto tensor = 8;








// -------------------------------------------------------------------

// AttrValue

// optional bytes s = 2;











// optional int64 i = 3;






// optional float f = 4;






// optional bool b = 5;






// optional .tensorflow.DataType type = 6;






// optional .tensorflow.TensorShapeProto shape = 7;








// optional .tensorflow.TensorProto tensor = 8;








// optional .tensorflow.AttrValue.ListValue list = 1;








// optional .tensorflow.NameAttrList func = 10;








// optional string placeholder = 9;














// -------------------------------------------------------------------

// NameAttrList

// optional string name = 1;









// map<string, .tensorflow.AttrValue> attr = 2;





// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2fattr_5fvalue_2eproto__INCLUDED


// Parsed from tensorflow/core/framework/op_def.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/op_def.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2fop_5fdef_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2fop_5fdef_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/unknown_field_set.h>
// #include "tensorflow/core/framework/attr_value.pb.h"
// #include "tensorflow/core/framework/types.pb.h"
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2fop_5fdef_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2fop_5fdef_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2fop_5fdef_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class OpDef_ArgDef extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpDef_ArgDef(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OpDef_ArgDef(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OpDef_ArgDef position(int position) {
        return (OpDef_ArgDef)super.position(position);
    }

  public OpDef_ArgDef() { super((Pointer)null); allocate(); }
  private native void allocate();

  public OpDef_ArgDef(@Const @ByRef OpDef_ArgDef from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef OpDef_ArgDef from);

  public native @ByRef @Name("operator =") OpDef_ArgDef put(@Const @ByRef OpDef_ArgDef from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef OpDef_ArgDef default_instance();

  public native void Swap(OpDef_ArgDef other);

  // implements Message ----------------------------------------------

  public native OpDef_ArgDef New();

  public native OpDef_ArgDef New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef OpDef_ArgDef from);
  public native void MergeFrom(@Const @ByRef OpDef_ArgDef from);
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

  // optional string description = 2;
  public native void clear_description();
  @MemberGetter public static native int kDescriptionFieldNumber();
  public static final int kDescriptionFieldNumber = kDescriptionFieldNumber();
  public native @StdString BytePointer description();
  public native void set_description(@StdString BytePointer value);
  public native void set_description(@StdString String value);
  public native void set_description(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_description(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_description();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_description();
  public native void set_allocated_description(@StdString @Cast({"char*", "std::string*"}) BytePointer description);

  // optional .tensorflow.DataType type = 3;
  public native void clear_type();
  @MemberGetter public static native int kTypeFieldNumber();
  public static final int kTypeFieldNumber = kTypeFieldNumber();
  public native @Cast("tensorflow::DataType") int type();
  public native void set_type(@Cast("tensorflow::DataType") int value);

  // optional string type_attr = 4;
  public native void clear_type_attr();
  @MemberGetter public static native int kTypeAttrFieldNumber();
  public static final int kTypeAttrFieldNumber = kTypeAttrFieldNumber();
  public native @StdString BytePointer type_attr();
  public native void set_type_attr(@StdString BytePointer value);
  public native void set_type_attr(@StdString String value);
  public native void set_type_attr(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_type_attr(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_type_attr();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_type_attr();
  public native void set_allocated_type_attr(@StdString @Cast({"char*", "std::string*"}) BytePointer type_attr);

  // optional string number_attr = 5;
  public native void clear_number_attr();
  @MemberGetter public static native int kNumberAttrFieldNumber();
  public static final int kNumberAttrFieldNumber = kNumberAttrFieldNumber();
  public native @StdString BytePointer number_attr();
  public native void set_number_attr(@StdString BytePointer value);
  public native void set_number_attr(@StdString String value);
  public native void set_number_attr(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_number_attr(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_number_attr();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_number_attr();
  public native void set_allocated_number_attr(@StdString @Cast({"char*", "std::string*"}) BytePointer number_attr);

  // optional string type_list_attr = 6;
  public native void clear_type_list_attr();
  @MemberGetter public static native int kTypeListAttrFieldNumber();
  public static final int kTypeListAttrFieldNumber = kTypeListAttrFieldNumber();
  public native @StdString BytePointer type_list_attr();
  public native void set_type_list_attr(@StdString BytePointer value);
  public native void set_type_list_attr(@StdString String value);
  public native void set_type_list_attr(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_type_list_attr(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_type_list_attr();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_type_list_attr();
  public native void set_allocated_type_list_attr(@StdString @Cast({"char*", "std::string*"}) BytePointer type_list_attr);

  // optional bool is_ref = 16;
  public native void clear_is_ref();
  @MemberGetter public static native int kIsRefFieldNumber();
  public static final int kIsRefFieldNumber = kIsRefFieldNumber();
  public native @Cast("bool") boolean is_ref();
  public native void set_is_ref(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class OpDef_AttrDef extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpDef_AttrDef(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OpDef_AttrDef(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OpDef_AttrDef position(int position) {
        return (OpDef_AttrDef)super.position(position);
    }

  public OpDef_AttrDef() { super((Pointer)null); allocate(); }
  private native void allocate();

  public OpDef_AttrDef(@Const @ByRef OpDef_AttrDef from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef OpDef_AttrDef from);

  public native @ByRef @Name("operator =") OpDef_AttrDef put(@Const @ByRef OpDef_AttrDef from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef OpDef_AttrDef default_instance();

  public native void Swap(OpDef_AttrDef other);

  // implements Message ----------------------------------------------

  public native OpDef_AttrDef New();

  public native OpDef_AttrDef New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef OpDef_AttrDef from);
  public native void MergeFrom(@Const @ByRef OpDef_AttrDef from);
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

  // optional .tensorflow.AttrValue default_value = 3;
  public native @Cast("bool") boolean has_default_value();
  public native void clear_default_value();
  @MemberGetter public static native int kDefaultValueFieldNumber();
  public static final int kDefaultValueFieldNumber = kDefaultValueFieldNumber();
  public native @Const @ByRef AttrValue default_value();
  public native AttrValue mutable_default_value();
  public native AttrValue release_default_value();
  public native void set_allocated_default_value(AttrValue default_value);

  // optional string description = 4;
  public native void clear_description();
  @MemberGetter public static native int kDescriptionFieldNumber();
  public static final int kDescriptionFieldNumber = kDescriptionFieldNumber();
  public native @StdString BytePointer description();
  public native void set_description(@StdString BytePointer value);
  public native void set_description(@StdString String value);
  public native void set_description(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_description(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_description();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_description();
  public native void set_allocated_description(@StdString @Cast({"char*", "std::string*"}) BytePointer description);

  // optional bool has_minimum = 5;
  public native void clear_has_minimum();
  @MemberGetter public static native int kHasMinimumFieldNumber();
  public static final int kHasMinimumFieldNumber = kHasMinimumFieldNumber();
  public native @Cast("bool") boolean has_minimum();
  public native void set_has_minimum(@Cast("bool") boolean value);

  // optional int64 minimum = 6;
  public native void clear_minimum();
  @MemberGetter public static native int kMinimumFieldNumber();
  public static final int kMinimumFieldNumber = kMinimumFieldNumber();
  public native @Cast("google::protobuf::int64") long minimum();
  public native void set_minimum(@Cast("google::protobuf::int64") long value);

  // optional .tensorflow.AttrValue allowed_values = 7;
  public native @Cast("bool") boolean has_allowed_values();
  public native void clear_allowed_values();
  @MemberGetter public static native int kAllowedValuesFieldNumber();
  public static final int kAllowedValuesFieldNumber = kAllowedValuesFieldNumber();
  public native @Const @ByRef AttrValue allowed_values();
  public native AttrValue mutable_allowed_values();
  public native AttrValue release_allowed_values();
  public native void set_allocated_allowed_values(AttrValue allowed_values);
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class OpDef extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpDef(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OpDef(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OpDef position(int position) {
        return (OpDef)super.position(position);
    }

  public OpDef() { super((Pointer)null); allocate(); }
  private native void allocate();

  public OpDef(@Const @ByRef OpDef from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef OpDef from);

  public native @ByRef @Name("operator =") OpDef put(@Const @ByRef OpDef from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef OpDef default_instance();

  public native void Swap(OpDef other);

  // implements Message ----------------------------------------------

  public native OpDef New();

  public native OpDef New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef OpDef from);
  public native void MergeFrom(@Const @ByRef OpDef from);
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

  // repeated .tensorflow.OpDef.ArgDef input_arg = 2;
  public native int input_arg_size();
  public native void clear_input_arg();
  @MemberGetter public static native int kInputArgFieldNumber();
  public static final int kInputArgFieldNumber = kInputArgFieldNumber();
  public native @Const @ByRef OpDef_ArgDef input_arg(int index);
  public native OpDef_ArgDef mutable_input_arg(int index);
  public native OpDef_ArgDef add_input_arg();

  // repeated .tensorflow.OpDef.ArgDef output_arg = 3;
  public native int output_arg_size();
  public native void clear_output_arg();
  @MemberGetter public static native int kOutputArgFieldNumber();
  public static final int kOutputArgFieldNumber = kOutputArgFieldNumber();
  public native @Const @ByRef OpDef_ArgDef output_arg(int index);
  public native OpDef_ArgDef mutable_output_arg(int index);
  public native OpDef_ArgDef add_output_arg();

  // repeated .tensorflow.OpDef.AttrDef attr = 4;
  public native int attr_size();
  public native void clear_attr();
  @MemberGetter public static native int kAttrFieldNumber();
  public static final int kAttrFieldNumber = kAttrFieldNumber();
  public native @Const @ByRef OpDef_AttrDef attr(int index);
  public native OpDef_AttrDef mutable_attr(int index);
  public native OpDef_AttrDef add_attr();

  // optional string summary = 5;
  public native void clear_summary();
  @MemberGetter public static native int kSummaryFieldNumber();
  public static final int kSummaryFieldNumber = kSummaryFieldNumber();
  public native @StdString BytePointer summary();
  public native void set_summary(@StdString BytePointer value);
  public native void set_summary(@StdString String value);
  public native void set_summary(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_summary(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_summary();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_summary();
  public native void set_allocated_summary(@StdString @Cast({"char*", "std::string*"}) BytePointer summary);

  // optional string description = 6;
  public native void clear_description();
  @MemberGetter public static native int kDescriptionFieldNumber();
  public static final int kDescriptionFieldNumber = kDescriptionFieldNumber();
  public native @StdString BytePointer description();
  public native void set_description(@StdString BytePointer value);
  public native void set_description(@StdString String value);
  public native void set_description(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_description(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_description();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_description();
  public native void set_allocated_description(@StdString @Cast({"char*", "std::string*"}) BytePointer description);

  // optional bool is_commutative = 18;
  public native void clear_is_commutative();
  @MemberGetter public static native int kIsCommutativeFieldNumber();
  public static final int kIsCommutativeFieldNumber = kIsCommutativeFieldNumber();
  public native @Cast("bool") boolean is_commutative();
  public native void set_is_commutative(@Cast("bool") boolean value);

  // optional bool is_aggregate = 16;
  public native void clear_is_aggregate();
  @MemberGetter public static native int kIsAggregateFieldNumber();
  public static final int kIsAggregateFieldNumber = kIsAggregateFieldNumber();
  public native @Cast("bool") boolean is_aggregate();
  public native void set_is_aggregate(@Cast("bool") boolean value);

  // optional bool is_stateful = 17;
  public native void clear_is_stateful();
  @MemberGetter public static native int kIsStatefulFieldNumber();
  public static final int kIsStatefulFieldNumber = kIsStatefulFieldNumber();
  public native @Cast("bool") boolean is_stateful();
  public native void set_is_stateful(@Cast("bool") boolean value);

  // optional bool allows_uninitialized_input = 19;
  public native void clear_allows_uninitialized_input();
  @MemberGetter public static native int kAllowsUninitializedInputFieldNumber();
  public static final int kAllowsUninitializedInputFieldNumber = kAllowsUninitializedInputFieldNumber();
  public native @Cast("bool") boolean allows_uninitialized_input();
  public native void set_allows_uninitialized_input(@Cast("bool") boolean value);
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class OpList extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpList(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OpList(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OpList position(int position) {
        return (OpList)super.position(position);
    }

  public OpList() { super((Pointer)null); allocate(); }
  private native void allocate();

  public OpList(@Const @ByRef OpList from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef OpList from);

  public native @ByRef @Name("operator =") OpList put(@Const @ByRef OpList from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef OpList default_instance();

  public native void Swap(OpList other);

  // implements Message ----------------------------------------------

  public native OpList New();

  public native OpList New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef OpList from);
  public native void MergeFrom(@Const @ByRef OpList from);
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

  // repeated .tensorflow.OpDef op = 1;
  public native int op_size();
  public native void clear_op();
  @MemberGetter public static native int kOpFieldNumber();
  public static final int kOpFieldNumber = kOpFieldNumber();
  public native @Const @ByRef OpDef op(int index);
  public native OpDef mutable_op(int index);
  public native OpDef add_op();
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// OpDef_ArgDef

// optional string name = 1;









// optional string description = 2;









// optional .tensorflow.DataType type = 3;




// optional string type_attr = 4;









// optional string number_attr = 5;









// optional string type_list_attr = 6;









// optional bool is_ref = 16;




// -------------------------------------------------------------------

// OpDef_AttrDef

// optional string name = 1;









// optional string type = 2;









// optional .tensorflow.AttrValue default_value = 3;







// optional string description = 4;









// optional bool has_minimum = 5;




// optional int64 minimum = 6;




// optional .tensorflow.AttrValue allowed_values = 7;







// -------------------------------------------------------------------

// OpDef

// optional string name = 1;









// repeated .tensorflow.OpDef.ArgDef input_arg = 2;








// repeated .tensorflow.OpDef.ArgDef output_arg = 3;








// repeated .tensorflow.OpDef.AttrDef attr = 4;








// optional string summary = 5;









// optional string description = 6;









// optional bool is_commutative = 18;




// optional bool is_aggregate = 16;




// optional bool is_stateful = 17;




// optional bool allows_uninitialized_input = 19;




// -------------------------------------------------------------------

// OpList

// repeated .tensorflow.OpDef op = 1;








// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2fop_5fdef_2eproto__INCLUDED


// Parsed from tensorflow/core/framework/function.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/function.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2ffunction_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2ffunction_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/map.h>
// #include <google/protobuf/map_field_inl.h>
// #include <google/protobuf/unknown_field_set.h>
// #include "tensorflow/core/framework/attr_value.pb.h"
// #include "tensorflow/core/framework/op_def.pb.h"
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2ffunction_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2ffunction_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2ffunction_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class FunctionDefLibrary extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FunctionDefLibrary(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FunctionDefLibrary(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FunctionDefLibrary position(int position) {
        return (FunctionDefLibrary)super.position(position);
    }

  public FunctionDefLibrary() { super((Pointer)null); allocate(); }
  private native void allocate();

  public FunctionDefLibrary(@Const @ByRef FunctionDefLibrary from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef FunctionDefLibrary from);

  public native @ByRef @Name("operator =") FunctionDefLibrary put(@Const @ByRef FunctionDefLibrary from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef FunctionDefLibrary default_instance();

  public native void Swap(FunctionDefLibrary other);

  // implements Message ----------------------------------------------

  public native FunctionDefLibrary New();

  public native FunctionDefLibrary New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef FunctionDefLibrary from);
  public native void MergeFrom(@Const @ByRef FunctionDefLibrary from);
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

  // repeated .tensorflow.FunctionDef function = 1;
  public native int function_size();
  public native void clear_function();
  @MemberGetter public static native int kFunctionFieldNumber();
  public static final int kFunctionFieldNumber = kFunctionFieldNumber();
  public native @Const @ByRef FunctionDef function(int index);
  public native FunctionDef mutable_function(int index);
  public native FunctionDef add_function();
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class FunctionDef_Node extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FunctionDef_Node(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FunctionDef_Node(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FunctionDef_Node position(int position) {
        return (FunctionDef_Node)super.position(position);
    }

  public FunctionDef_Node() { super((Pointer)null); allocate(); }
  private native void allocate();

  public FunctionDef_Node(@Const @ByRef FunctionDef_Node from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef FunctionDef_Node from);

  public native @ByRef @Name("operator =") FunctionDef_Node put(@Const @ByRef FunctionDef_Node from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef FunctionDef_Node default_instance();

  public native void Swap(FunctionDef_Node other);

  // implements Message ----------------------------------------------

  public native FunctionDef_Node New();

  public native FunctionDef_Node New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef FunctionDef_Node from);
  public native void MergeFrom(@Const @ByRef FunctionDef_Node from);
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

  // repeated string ret = 1;
  public native int ret_size();
  public native void clear_ret();
  @MemberGetter public static native int kRetFieldNumber();
  public static final int kRetFieldNumber = kRetFieldNumber();
  public native @StdString BytePointer ret(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_ret(int index);
  public native void set_ret(int index, @StdString BytePointer value);
  public native void set_ret(int index, @StdString String value);
  public native void set_ret(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_ret(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_ret();
  public native void add_ret(@StdString BytePointer value);
  public native void add_ret(@StdString String value);
  public native void add_ret(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_ret(String value, @Cast("size_t") long size);

  // optional string op = 2;
  public native void clear_op();
  @MemberGetter public static native int kOpFieldNumber();
  public static final int kOpFieldNumber = kOpFieldNumber();
  public native @StdString BytePointer op();
  public native void set_op(@StdString BytePointer value);
  public native void set_op(@StdString String value);
  public native void set_op(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_op(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_op();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_op();
  public native void set_allocated_op(@StdString @Cast({"char*", "std::string*"}) BytePointer op);

  // repeated string arg = 3;
  public native int arg_size();
  public native void clear_arg();
  @MemberGetter public static native int kArgFieldNumber();
  public static final int kArgFieldNumber = kArgFieldNumber();
  public native @StdString BytePointer arg(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_arg(int index);
  public native void set_arg(int index, @StdString BytePointer value);
  public native void set_arg(int index, @StdString String value);
  public native void set_arg(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_arg(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_arg();
  public native void add_arg(@StdString BytePointer value);
  public native void add_arg(@StdString String value);
  public native void add_arg(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_arg(String value, @Cast("size_t") long size);

  // repeated string dep = 4;
  public native int dep_size();
  public native void clear_dep();
  @MemberGetter public static native int kDepFieldNumber();
  public static final int kDepFieldNumber = kDepFieldNumber();
  public native @StdString BytePointer dep(int index);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_dep(int index);
  public native void set_dep(int index, @StdString BytePointer value);
  public native void set_dep(int index, @StdString String value);
  public native void set_dep(int index, @Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_dep(int index, String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer add_dep();
  public native void add_dep(@StdString BytePointer value);
  public native void add_dep(@StdString String value);
  public native void add_dep(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void add_dep(String value, @Cast("size_t") long size);

  // map<string, .tensorflow.AttrValue> attr = 5;
  public native int attr_size();
  public native void clear_attr();
  @MemberGetter public static native int kAttrFieldNumber();
  public static final int kAttrFieldNumber = kAttrFieldNumber();
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class FunctionDef extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FunctionDef(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public FunctionDef(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public FunctionDef position(int position) {
        return (FunctionDef)super.position(position);
    }

  public FunctionDef() { super((Pointer)null); allocate(); }
  private native void allocate();

  public FunctionDef(@Const @ByRef FunctionDef from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef FunctionDef from);

  public native @ByRef @Name("operator =") FunctionDef put(@Const @ByRef FunctionDef from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef FunctionDef default_instance();

  public native void Swap(FunctionDef other);

  // implements Message ----------------------------------------------

  public native FunctionDef New();

  public native FunctionDef New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef FunctionDef from);
  public native void MergeFrom(@Const @ByRef FunctionDef from);
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

  // optional .tensorflow.OpDef signature = 1;
  public native @Cast("bool") boolean has_signature();
  public native void clear_signature();
  @MemberGetter public static native int kSignatureFieldNumber();
  public static final int kSignatureFieldNumber = kSignatureFieldNumber();
  public native @Const @ByRef OpDef signature();
  public native OpDef mutable_signature();
  public native OpDef release_signature();
  public native void set_allocated_signature(OpDef signature);

  // repeated .tensorflow.FunctionDef.Node node = 2;
  public native int node_size();
  public native void clear_node();
  @MemberGetter public static native int kNodeFieldNumber();
  public static final int kNodeFieldNumber = kNodeFieldNumber();
  public native @Const @ByRef FunctionDef_Node node(int index);
  public native FunctionDef_Node mutable_node(int index);
  public native FunctionDef_Node add_node();
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// FunctionDefLibrary

// repeated .tensorflow.FunctionDef function = 1;








// -------------------------------------------------------------------

// FunctionDef_Node

// repeated string ret = 1;














// optional string op = 2;









// repeated string arg = 3;














// repeated string dep = 4;














// map<string, .tensorflow.AttrValue> attr = 5;





// -------------------------------------------------------------------

// FunctionDef

// optional .tensorflow.OpDef signature = 1;







// repeated .tensorflow.FunctionDef.Node node = 2;








// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2ffunction_2eproto__INCLUDED


// Parsed from tensorflow/core/framework/graph.pb.h

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/graph.proto

// #ifndef PROTOBUF_tensorflow_2fcore_2fframework_2fgraph_2eproto__INCLUDED
// #define PROTOBUF_tensorflow_2fcore_2fframework_2fgraph_2eproto__INCLUDED

// #include <string>

// #include <google/protobuf/stubs/common.h>

// #if GOOGLE_PROTOBUF_VERSION < 3000000
// #error This file was generated by a newer version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please update
// #error your headers.
// #endif
// #if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
// #error This file was generated by an older version of protoc which is
// #error incompatible with your Protocol Buffer headers.  Please
// #error regenerate this file with a newer version of protoc.
// #endif

// #include <google/protobuf/arena.h>
// #include <google/protobuf/arenastring.h>
// #include <google/protobuf/generated_message_util.h>
// #include <google/protobuf/metadata.h>
// #include <google/protobuf/message.h>
// #include <google/protobuf/repeated_field.h>
// #include <google/protobuf/extension_set.h>
// #include <google/protobuf/map.h>
// #include <google/protobuf/map_field_inl.h>
// #include <google/protobuf/unknown_field_set.h>
// #include "tensorflow/core/framework/attr_value.pb.h"
// #include "tensorflow/core/framework/function.pb.h"
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
@Namespace("tensorflow") public static native void protobuf_AddDesc_tensorflow_2fcore_2fframework_2fgraph_2eproto();
@Namespace("tensorflow") public static native void protobuf_AssignDesc_tensorflow_2fcore_2fframework_2fgraph_2eproto();
@Namespace("tensorflow") public static native void protobuf_ShutdownFile_tensorflow_2fcore_2fframework_2fgraph_2eproto();

// ===================================================================

@Namespace("tensorflow") @NoOffset public static class GraphDef extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GraphDef(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public GraphDef(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public GraphDef position(int position) {
        return (GraphDef)super.position(position);
    }

  public GraphDef() { super((Pointer)null); allocate(); }
  private native void allocate();

  public GraphDef(@Const @ByRef GraphDef from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef GraphDef from);

  public native @ByRef @Name("operator =") GraphDef put(@Const @ByRef GraphDef from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef GraphDef default_instance();

  public native void Swap(GraphDef other);

  // implements Message ----------------------------------------------

  public native GraphDef New();

  public native GraphDef New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef GraphDef from);
  public native void MergeFrom(@Const @ByRef GraphDef from);
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

  // repeated .tensorflow.NodeDef node = 1;
  public native int node_size();
  public native void clear_node();
  @MemberGetter public static native int kNodeFieldNumber();
  public static final int kNodeFieldNumber = kNodeFieldNumber();
  public native @Const @ByRef NodeDef node(int index);
  public native NodeDef mutable_node(int index);
  public native NodeDef add_node();

  // optional .tensorflow.FunctionDefLibrary library = 2;
  public native @Cast("bool") boolean has_library();
  public native void clear_library();
  @MemberGetter public static native int kLibraryFieldNumber();
  public static final int kLibraryFieldNumber = kLibraryFieldNumber();
  public native @Const @ByRef FunctionDefLibrary library();
  public native FunctionDefLibrary mutable_library();
  public native FunctionDefLibrary release_library();
  public native void set_allocated_library(FunctionDefLibrary library);
}
// -------------------------------------------------------------------

@Namespace("tensorflow") @NoOffset public static class NodeDef extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NodeDef(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public NodeDef(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public NodeDef position(int position) {
        return (NodeDef)super.position(position);
    }

  public NodeDef() { super((Pointer)null); allocate(); }
  private native void allocate();

  public NodeDef(@Const @ByRef NodeDef from) { super((Pointer)null); allocate(from); }
  private native void allocate(@Const @ByRef NodeDef from);

  public native @ByRef @Name("operator =") NodeDef put(@Const @ByRef NodeDef from);

  public static native @Cast("const google::protobuf::Descriptor*") Pointer descriptor();
  public static native @Const @ByRef NodeDef default_instance();

  public native void Swap(NodeDef other);

  // implements Message ----------------------------------------------

  public native NodeDef New();

  public native NodeDef New(@Cast("google::protobuf::Arena*") Pointer arena);
  public native void CopyFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void MergeFrom(@Cast("const google::protobuf::Message*") @ByRef Pointer from);
  public native void CopyFrom(@Const @ByRef NodeDef from);
  public native void MergeFrom(@Const @ByRef NodeDef from);
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

  // optional string op = 2;
  public native void clear_op();
  @MemberGetter public static native int kOpFieldNumber();
  public static final int kOpFieldNumber = kOpFieldNumber();
  public native @StdString BytePointer op();
  public native void set_op(@StdString BytePointer value);
  public native void set_op(@StdString String value);
  public native void set_op(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_op(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_op();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_op();
  public native void set_allocated_op(@StdString @Cast({"char*", "std::string*"}) BytePointer op);

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

  // optional string device = 4;
  public native void clear_device();
  @MemberGetter public static native int kDeviceFieldNumber();
  public static final int kDeviceFieldNumber = kDeviceFieldNumber();
  public native @StdString BytePointer device();
  public native void set_device(@StdString BytePointer value);
  public native void set_device(@StdString String value);
  public native void set_device(@Cast("const char*") BytePointer value, @Cast("size_t") long size);
  public native void set_device(String value, @Cast("size_t") long size);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_device();
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer release_device();
  public native void set_allocated_device(@StdString @Cast({"char*", "std::string*"}) BytePointer device);

  // map<string, .tensorflow.AttrValue> attr = 5;
  public native int attr_size();
  public native void clear_attr();
  @MemberGetter public static native int kAttrFieldNumber();
  public static final int kAttrFieldNumber = kAttrFieldNumber();
}
// ===================================================================


// ===================================================================

// #if !PROTOBUF_INLINE_NOT_IN_HEADERS
// GraphDef

// repeated .tensorflow.NodeDef node = 1;








// optional .tensorflow.FunctionDefLibrary library = 2;







// -------------------------------------------------------------------

// NodeDef

// optional string name = 1;









// optional string op = 2;









// repeated string input = 3;














// optional string device = 4;









// map<string, .tensorflow.AttrValue> attr = 5;





// #endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

// #endif  // PROTOBUF_tensorflow_2fcore_2fframework_2fgraph_2eproto__INCLUDED


// Parsed from tensorflow/core/public/session.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_PUBLIC_SESSION_H_
// #define TENSORFLOW_PUBLIC_SESSION_H_

// #include <string>
// #include <vector>

// #include "tensorflow/core/framework/device_attributes.pb.h"
// #include "tensorflow/core/framework/graph.pb.h"
// #include "tensorflow/core/public/env.h"
// #include "tensorflow/core/public/session_options.h"
// #include "tensorflow/core/public/status.h"
// #include "tensorflow/core/public/tensor.h"

/** \brief A Session instance lets a caller drive a TensorFlow graph
 *  computation.
 * 
 *  When a Session is created with a given target, a new Session object
 *  is bound to the universe of resources specified by that target.
 *  Those resources are available to this session to perform
 *  computation described in the GraphDef.  After extending the session
 *  with a graph, the caller uses the Run() API to perform the
 *  computation and potentially fetch outputs as Tensors.
 * 
 *  Example:
 * 
 *  <pre>{@code c++
 * 
 *      tensorflow::GraphDef graph;
 *      // ... Create or load graph into "graph".
 * 
 *      // This example uses the default options which connects
 *      // to a local runtime.
 *      tensorflow::SessionOptions options;
 *      std::unique_ptr<tensorflow::Session>
 *      session(tensorflow::NewSession(options));
 * 
 *      // Create the session with this graph.
 *      tensorflow::Status s = session->Create(graph);
 *      if (!s.ok()) { ... }
 * 
 *      // Run the graph and fetch the first output of the "output"
 *      // operation, and also run to but do not return anything
 *      // for the "update_state" operation.
 *      std::vector<tensorflow::Tensor> outputs;
 *      s = session->Run({}, {"output:0"}, {"update_state"}, &outputs);
 *      if (!s.ok()) { ... }
 * 
 *      // Map the output as a flattened float tensor, and do something
 *      // with it.
 *      auto output_tensor = outputs[0].flat<float>();
 *      if (output_tensor(0) > 0.5) { ... }
 * 
 *      // Close the session to release the resources associated with
 *      // this session.
 *      session->Close()
 * 
 *  }</pre>
 * 
 *  A Session allows concurrent calls to Run(), though a Session must
 *  be created / extended by a single thread.
 * 
 *  Only one thread must call Close(), and Close() must only be called
 *  after all other calls to Run() have returned. */
@Namespace("tensorflow") public static class Session extends AbstractSession {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Session(Pointer p) { super(p); }

  /** \brief Create the graph to be used for the session.
   * 
   *  Returns an error if this session has already been created with a
   *  graph. To re-use the session with a different graph, the caller
   *  must Close() the session first. */
  
  ///
  public native @ByVal Status Create(@Const @ByRef GraphDef graph);

  /** \brief Adds operations to the graph that is already registered with the
   *  Session.
   * 
   *  The names of new operations in "graph" must not exist in the
   *  graph that is already registered. */
  
  ///
  ///
  ///
  ///
  public native @ByVal Status Extend(@Const @ByRef GraphDef graph);

  /** \brief Runs the graph with the provided input tensors and fills
   *  {@code outputs} for the endpoints specified in {@code output_tensor_names}.
   *  Runs to but does not return Tensors for the nodes in
   *  {@code target_node_names}.
   * 
   *  The order of tensors in {@code outputs} will match the order provided
   *  by {@code output_tensor_names}.
   * 
   *  If {@code Run} returns {@code OK()}, then {@code outputs->size()} will be equal to
   *  {@code output_tensor_names.size()}.  If {@code Run} does not return {@code OK()}, the
   *  state of {@code outputs} is undefined.
   * 
   *  REQUIRES: The name of each Tensor of the input or output must
   *  match a "Tensor endpoint" in the {@code GraphDef} passed to {@code Create()}.
   * 
   *  REQUIRES: outputs is not nullptr if {@code output_tensor_names} is non-empty. */
  
  ///
  public native @ByVal Status Run(@Const @ByRef StringTensorPairVector inputs,
                       @Const @ByRef StringVector output_tensor_names,
                       @Const @ByRef StringVector target_node_names,
                       TensorVector outputs);

  /** \brief Closes this session.
   * 
   *  Closing a session releases the resources used by this session
   *  on the TensorFlow runtime (specified during session creation by
   *  the {@code SessionOptions::target} field). */
  public native @ByVal Status Close();

  /** Calls {@link tensorflow#NewSession(SessionOptions)} and registers a deallocator. */
  public Session(SessionOptions options) { super(options); }
}

/** \brief Create a new session with the given options.
 * 
 *  If a new {@code Session} object could not be created, this function will
 *  return nullptr. */

///
@Namespace("tensorflow") public static native Session NewSession(@Const @ByRef SessionOptions options);

/** \brief Create a new session with the given options.
 * 
 *  If session creation succeeds, the new {@code Session} will be stored in
 *  {@code *out_session}, the caller will take ownership of the returned
 *  {@code *out_session}, and this function will return {@code OK()}. Otherwise, this
 *  function will return an error status. */
@Namespace("tensorflow") public static native @ByVal Status NewSession(@Const @ByRef SessionOptions options, @Cast("tensorflow::Session**") PointerPointer out_session);
@Namespace("tensorflow") public static native @ByVal Status NewSession(@Const @ByRef SessionOptions options, @ByPtrPtr Session out_session);

  // end namespace tensorflow

// #endif  // TENSORFLOW_PUBLIC_SESSION_H_


// Parsed from tensorflow/core/public/tensor_c_api.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// TODO(jeff,sanjay): Rename to tensorflow/public/c_api.h
// #ifndef TENSORFLOW_PUBLIC_TENSOR_C_API_H_
// #define TENSORFLOW_PUBLIC_TENSOR_C_API_H_

// #include <stddef.h>

// --------------------------------------------------------------------------
// C API for TensorFlow.
//
// The API leans towards simplicity and uniformity instead of convenience
// since most usage will be by language specific wrappers.
//
// Conventions:
// * We use the prefix TF_ for everything in the API.
// * Objects are always passed around as pointers to opaque structs
//   and these structs are allocated/deallocated via the API.
// * TF_Status holds error information.  It is an object type
//   and threfore is passed around as a pointer to an opaque
//   struct as mentioned above.
// * Every call that has a TF_Status* argument clears it on success
//   and fills it with error info on failure.
//
// Questions left to address:
// * Might need to add stride info to TF_Tensor?
// * Might at some point need a way for callers to provide their own Env.
// * Should we remove the TF_Status arg from TF_AddProto calls and only
//   report errors later (e.g., on Run call).
// * Should dimensions be unsigned instead of signed?
// * Maybe add TF_TensorShape that encapsulates dimension info.
//
// Design decisions made:
// * Backing store for tensor memory has an associated deallocation
//   function.  This deallocation function will point to client code
//   for tensors populated by the client.  So the client can do things
//   like shadowing a numpy array.
// * We do not provide TF_OK since it is not strictly necessary and we
//   are not optimizing for convenience.
// * We make assumption that one session has one graph.  This should be
//   fine since we have the ability to run sub-graphs.
// * We are not providing TF_AddNode/TF_AddNodes to better support
//   languages/platforms where proto is not available.  This is because
//   we can just point authors of bindings at the .proto file and the
//   proto serialization spec and they can do the right thing for
//   their language.
// * We could allow NULL for some arguments (e.g., NULL options arg).
//   However since convenience is not a primary goal, we don't do this.
// * Devices are not in this API.  Instead, they are created/used internally
//   and the API just provides high level controls over the number of
//   devices of each type.

// #ifdef __cplusplus
// #endif

// --------------------------------------------------------------------------
// TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The enum values here are identical to corresponding values in types.proto.
/** enum TF_DataType */
public static final int
  TF_FLOAT = 1,
  TF_DOUBLE = 2,
  TF_INT32 = 3,  // Int32 tensors are always in 'host' memory.
  TF_UINT8 = 4,
  TF_INT16 = 5,
  TF_INT8 = 6,
  TF_STRING = 7,
  TF_COMPLEX = 8,  // Single-precision complex
  TF_INT64 = 9,
  TF_BOOL = 10,
  TF_QINT8 = 11,     // Quantized int8
  TF_QUINT8 = 12,    // Quantized uint8
  TF_QINT32 = 13,    // Quantized int32
  TF_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.

// --------------------------------------------------------------------------
// TF_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
/** enum TF_Code */
public static final int
  TF_OK = 0,
  TF_CANCELLED = 1,
  TF_UNKNOWN = 2,
  TF_INVALID_ARGUMENT = 3,
  TF_DEADLINE_EXCEEDED = 4,
  TF_NOT_FOUND = 5,
  TF_ALREADY_EXISTS = 6,
  TF_PERMISSION_DENIED = 7,
  TF_UNAUTHENTICATED = 16,
  TF_RESOURCE_EXHAUSTED = 8,
  TF_FAILED_PRECONDITION = 9,
  TF_ABORTED = 10,
  TF_OUT_OF_RANGE = 11,
  TF_UNIMPLEMENTED = 12,
  TF_INTERNAL = 13,
  TF_UNAVAILABLE = 14,
  TF_DATA_LOSS = 15;

// --------------------------------------------------------------------------
// TF_Status holds error information.  It either has an OK code, or
// else an error code with an associated error message.
@Opaque public static class TF_Status extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TF_Status() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TF_Status(Pointer p) { super(p); }
}

// Return a new status object.
public static native TF_Status TF_NewStatus();

// Delete a previously created status object.
public static native void TF_DeleteStatus(TF_Status arg0);

// Record <code, msg> in *s.  Any previous information is lost.
// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
public static native void TF_SetStatus(TF_Status s, @Cast("TF_Code") int code, @Cast("const char*") BytePointer msg);
public static native void TF_SetStatus(TF_Status s, @Cast("TF_Code") int code, String msg);

// Return the code record in *s.
public static native @Cast("TF_Code") int TF_GetCode(@Const TF_Status s);

// Return a pointer to the error message in *s.  The return value
// points to memory that is only usable until the next mutation to *s.
// Always returns an empty string if TF_GetCode(s) is TF_OK.
public static native @Cast("const char*") BytePointer TF_Message(@Const TF_Status s);

// --------------------------------------------------------------------------
// TF_Tensor holds a multi-dimensional array of elements of a single data type.
// For all types other than TF_STRING, the data buffer stores elements
// in row major order.  E.g. if data is treated as a vector of TF_DataType:
//
//   element 0:   index (0, ..., 0)
//   element 1:   index (0, ..., 1)
//   ...
//
// TODO(jeff,sanjay): Define format for TF_STRING tensors.  Perhaps:
//   start_offset: array[uint64]
//   data:         byte[...]
//
//   String length is encoded (varint?) starting at data[start_offset[i]]
//   String contents follow immediately after string length.

@Opaque public static class TF_Tensor extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TF_Tensor() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TF_Tensor(Pointer p) { super(p); }
}

// Return a new tensor that holds the bytes data[0,len-1].
//
// The data will be deallocated by a subsequent call to TF_DeleteTensor via:
//      (*deallocator_fn)(data, len, deallocator_arg)
// Clients can provide a custom deallocator function so they can pass in
// memory managed by something like numpy.
public static class Deallocator_Pointer_long_Pointer extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Deallocator_Pointer_long_Pointer(Pointer p) { super(p); }
    protected Deallocator_Pointer_long_Pointer() { allocate(); }
    private native void allocate();
    public native void call(Pointer data, @Cast("size_t") long len,
                                                   Pointer arg);
}
public static native TF_Tensor TF_NewTensor(@Cast("TF_DataType") int arg0, @Cast("long long*") LongPointer dims, int num_dims,
                               Pointer data, @Cast("size_t") long len,
                               Deallocator_Pointer_long_Pointer deallocator,
                               Pointer deallocator_arg);
public static native TF_Tensor TF_NewTensor(@Cast("TF_DataType") int arg0, @Cast("long long*") LongBuffer dims, int num_dims,
                               Pointer data, @Cast("size_t") long len,
                               Deallocator_Pointer_long_Pointer deallocator,
                               Pointer deallocator_arg);
public static native TF_Tensor TF_NewTensor(@Cast("TF_DataType") int arg0, @Cast("long long*") long[] dims, int num_dims,
                               Pointer data, @Cast("size_t") long len,
                               Deallocator_Pointer_long_Pointer deallocator,
                               Pointer deallocator_arg);

// Destroy a tensor.
public static native void TF_DeleteTensor(TF_Tensor arg0);

// Return the type of a tensor element.
public static native @Cast("TF_DataType") int TF_TensorType(@Const TF_Tensor arg0);

// Return the number of dimensions that the tensor has.
public static native int TF_NumDims(@Const TF_Tensor arg0);

// Return the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
public static native @Cast("long long") long TF_Dim(@Const TF_Tensor tensor, int dim_index);

// Return the size of the underlying data in bytes.
public static native @Cast("size_t") long TF_TensorByteSize(@Const TF_Tensor arg0);

// Return a pointer to the underlying data buffer.
public static native Pointer TF_TensorData(@Const TF_Tensor arg0);

// --------------------------------------------------------------------------
// TF_SessionOptions holds options that can be passed during session creation.
@Opaque public static class TF_SessionOptions extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TF_SessionOptions() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TF_SessionOptions(Pointer p) { super(p); }
}

// Return a new options object.
public static native TF_SessionOptions TF_NewSessionOptions();

// Set the target in TF_SessionOptions.options.
// target can be empty, a single entry, or a comma separated list of entries.
// Each entry is in one of the following formats :
// "local"
// ip:port
// host:port
public static native void TF_SetTarget(TF_SessionOptions options, @Cast("const char*") BytePointer target);
public static native void TF_SetTarget(TF_SessionOptions options, String target);

// Set the config in TF_SessionOptions.options.
// config should be a serialized brain.ConfigProto proto.
// If config was not parsed successfully as a ConfigProto, record the
// error information in *status.
public static native void TF_SetConfig(TF_SessionOptions options, @Const Pointer proto,
                         @Cast("size_t") long proto_len, TF_Status status);

// Destroy an options object.
public static native void TF_DeleteSessionOptions(TF_SessionOptions arg0);

// TODO(jeff,sanjay):
// - export functions to set Config fields

// --------------------------------------------------------------------------
// TF_Session manages a single graph and execution.
@Opaque public static class TF_Session extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TF_Session() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TF_Session(Pointer p) { super(p); }
}

// Return a new execution session, or NULL on error.
public static native TF_Session TF_NewSession(@Const TF_SessionOptions arg0, TF_Status status);

// Close a session.
public static native void TF_CloseSession(TF_Session arg0, TF_Status status);

// Destroy a session.  Even if error information is recorded in *status,
// this call discards all resources associated with the session.
public static native void TF_DeleteSession(TF_Session arg0, TF_Status status);

// Treat the bytes proto[0,proto_len-1] as a serialized GraphDef and
// add the nodes in that GraphDef to the graph for the session.
public static native void TF_ExtendGraph(TF_Session arg0, @Const Pointer proto, @Cast("size_t") long proto_len,
                           TF_Status arg3);

// Run the graph associated with the session starting with the
// supplied inputs (inputs[0,ninputs-1]).  Regardless of success or
// failure, inputs[] become the property of the implementation (the
// implementation will eventually call TF_DeleteTensor on each input).
//
// On success, the tensors corresponding to output_names[0,noutputs-1]
// are placed in outputs[].  and these outputs[] become the property
// of the caller (the caller must eventually call TF_DeleteTensor on
// them).
//
// On failure, outputs[] contains nulls.
public static native void TF_Run(TF_Session arg0,
                   @Cast("const char**") PointerPointer input_names, @Cast("TF_Tensor**") PointerPointer inputs, int ninputs,
                   @Cast("const char**") PointerPointer output_tensor_names, @Cast("TF_Tensor**") PointerPointer outputs,
                   int noutputs,
                   @Cast("const char**") PointerPointer target_node_names, int ntargets,
                   TF_Status arg9);
public static native void TF_Run(TF_Session arg0,
                   @Cast("const char**") @ByPtrPtr BytePointer input_names, @ByPtrPtr TF_Tensor inputs, int ninputs,
                   @Cast("const char**") @ByPtrPtr BytePointer output_tensor_names, @ByPtrPtr TF_Tensor outputs,
                   int noutputs,
                   @Cast("const char**") @ByPtrPtr BytePointer target_node_names, int ntargets,
                   TF_Status arg9);
public static native void TF_Run(TF_Session arg0,
                   @Cast("const char**") @ByPtrPtr ByteBuffer input_names, @ByPtrPtr TF_Tensor inputs, int ninputs,
                   @Cast("const char**") @ByPtrPtr ByteBuffer output_tensor_names, @ByPtrPtr TF_Tensor outputs,
                   int noutputs,
                   @Cast("const char**") @ByPtrPtr ByteBuffer target_node_names, int ntargets,
                   TF_Status arg9);
public static native void TF_Run(TF_Session arg0,
                   @Cast("const char**") @ByPtrPtr byte[] input_names, @ByPtrPtr TF_Tensor inputs, int ninputs,
                   @Cast("const char**") @ByPtrPtr byte[] output_tensor_names, @ByPtrPtr TF_Tensor outputs,
                   int noutputs,
                   @Cast("const char**") @ByPtrPtr byte[] target_node_names, int ntargets,
                   TF_Status arg9);

// #ifdef __cplusplus /* end extern "C" */
// #endif

// #endif  // TENSORFLOW_PUBLIC_TENSOR_C_API_H_


// Parsed from tensorflow/core/framework/op_def_builder.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Class and associated machinery for specifying an Op's OpDef for Op
// registration.

// #ifndef TENSORFLOW_FRAMEWORK_OP_DEF_BUILDER_H_
// #define TENSORFLOW_FRAMEWORK_OP_DEF_BUILDER_H_

// #include <string>
// #include <vector>
// #include "tensorflow/core/framework/op_def.pb.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/public/status.h"

// Builder class passed to the REGISTER_OP() macro.
@Namespace("tensorflow") @NoOffset public static class OpDefBuilder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpDefBuilder(Pointer p) { super(p); }

  // Constructs an OpDef with just the name field set.
  public OpDefBuilder(@StringPiece BytePointer op_name) { super((Pointer)null); allocate(op_name); }
  private native void allocate(@StringPiece BytePointer op_name);
  public OpDefBuilder(@StringPiece String op_name) { super((Pointer)null); allocate(op_name); }
  private native void allocate(@StringPiece String op_name);

  // Adds an attr to this OpDefBuilder (and returns *this). The spec has
  // format "<name>:<type>" or "<name>:<type>=<default>"
  // where <name> matches regexp [a-zA-Z][a-zA-Z0-9_]*
  // (by convention only using capital letters for attrs that can be inferred)
  // <type> can be:
  //   "string", "int", "float", "bool", "type", "shape", or "tensor"
  //   "numbertype", "realnumbertype", "quantizedtype", "{int32,int64}"
  //       (meaning "type" with a restriction on valid values)
  //   "{\"foo\", \"bar\n baz\"}", or "{'foo', 'bar\n baz'}"
  //       (meaning "string" with a restriction on valid values)
  //   "list(string)", ..., "list(tensor)", "list(numbertype)", ...
  //       (meaning lists of the above types)
  //   "int >= 2" (meaning "int" with a restriction on valid values)
  //   "list(string) >= 2", "list(int) >= 2"
  //       (meaning "list(string)" / "list(int)" with length at least 2)
  // <default>, if included, should use the Proto text format
  // of <type>.  For lists use [a, b, c] format.
  //
  // Note that any attr specifying the length of an input or output will
  // get a default minimum of 1 unless the >= # syntax is used.
  //
  // TODO(josh11b): Perhaps support restrictions and defaults as optional
  // extra arguments to Attr() instead of encoding them in the spec string.
  // TODO(josh11b): Would like to have better dtype handling for tensor attrs:
  // * Ability to say the type of an input/output matches the type of
  //   the tensor.
  // * Ability to restrict the type of the tensor like the existing
  //   restrictions for type attrs.
  // Perhaps by linking the type of the tensor to a type attr?
  public native @ByRef OpDefBuilder Attr(@StringPiece BytePointer spec);
  public native @ByRef OpDefBuilder Attr(@StringPiece String spec);

  // Adds an input or ouput to this OpDefBuilder (and returns *this).
  // The spec has form "<name>:<type-expr>" or "<name>:Ref(<type-expr>)"
  // where <name> matches regexp [a-z][a-z0-9_]* and <type-expr> can be:
  // * For a single tensor: <type>
  // * For a sequence of tensors with the same type: <number>*<type>
  // * For a sequence of tensors with different types: <type-list>
  // Where:
  //   <type> is either one of "float", "int32", "string", ...
  //                 or the name of an attr (see above) with type "type".
  //   <number> is the name of an attr with type "int".
  //   <type-list> is the name of an attr with type "list(type)".
  // TODO(josh11b): Indicate Ref() via an optional argument instead of
  // in the spec?
  // TODO(josh11b): SparseInput() and SparseOutput() matching the Python
  // handling?
  public native @ByRef OpDefBuilder Input(@StringPiece BytePointer spec);
  public native @ByRef OpDefBuilder Input(@StringPiece String spec);
  public native @ByRef OpDefBuilder Output(@StringPiece BytePointer spec);
  public native @ByRef OpDefBuilder Output(@StringPiece String spec);

  // Turns on the indicated boolean flag in this OpDefBuilder (and
  // returns *this).
  public native @ByRef OpDefBuilder SetIsCommutative();
  public native @ByRef OpDefBuilder SetIsAggregate();
  public native @ByRef OpDefBuilder SetIsStateful();
  public native @ByRef OpDefBuilder SetAllowsUninitializedInput();

  // Adds docs to this OpDefBuilder (and returns *this).
  // Docs have the format:
  //   <1-line summary>
  //   <rest of the description>
  //   <name>: <description of name>
  //   <name>: <description of name>
  //     <if long, indent the description on subsequent lines>
  // Where <name> is the name of an attr, input, or output.  Please
  // wrap docs at 72 columns so that it may be indented in the
  // generated output.  For tensor inputs or outputs (not attrs), you
  // may start the description with an "=" (like name:= <description>)
  // to suppress the automatically-generated type documentation in
  // generated output.
  public native @ByRef OpDefBuilder Doc(@StringPiece BytePointer text);
  public native @ByRef OpDefBuilder Doc(@StringPiece String text);

  // Sets *op_def to the requested OpDef, or returns an error.
  // Must be called after all of the above methods.
  // Note that OpDefBuilder only reports parsing errors.  You should also
  // call ValidateOpDef() to detect other problems.
  public native @ByVal Status Finalize(OpDef op_def);
}

  // namespace tensorflow

// #endif  // TENSORFLOW_FRAMEWORK_OP_DEF_BUILDER_H_


// Parsed from tensorflow/core/framework/op_def_util.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// TODO(josh11b): Probably not needed for OpKernel authors, so doesn't
// need to be as publicly accessible as other files in framework/.

// #ifndef TENSORFLOW_FRAMEWORK_OP_DEF_UTIL_H_
// #define TENSORFLOW_FRAMEWORK_OP_DEF_UTIL_H_

// #include <string>
// #include "tensorflow/core/framework/op_def.pb.h"
// #include "tensorflow/core/public/status.h"

// Performs a consistency check across the fields of the op_def.
@Namespace("tensorflow") public static native @ByVal Status ValidateOpDef(@Const @ByRef OpDef op_def);

// Validates that attr_value satisfies the type and constraints from attr.
// REQUIRES: attr has already been validated.
@Namespace("tensorflow") public static native @ByVal Status ValidateAttrValue(@Const @ByRef AttrValue attr_value,
                         @Cast("const tensorflow::OpDef::AttrDef*") @ByRef OpDef_AttrDef attr);

// The following search through op_def for an attr with the indicated name.
// Returns nullptr if no such attr is found.
@Namespace("tensorflow") public static native @Cast("const tensorflow::OpDef::AttrDef*") OpDef_AttrDef FindAttr(@StringPiece BytePointer name, @Const @ByRef OpDef op_def);
@Namespace("tensorflow") public static native @Cast("const tensorflow::OpDef::AttrDef*") OpDef_AttrDef FindAttr(@StringPiece String name, @Const @ByRef OpDef op_def);
@Namespace("tensorflow") public static native @Cast("tensorflow::OpDef::AttrDef*") OpDef_AttrDef FindAttrMutable(@StringPiece BytePointer name, OpDef op_def);
@Namespace("tensorflow") public static native @Cast("tensorflow::OpDef::AttrDef*") OpDef_AttrDef FindAttrMutable(@StringPiece String name, OpDef op_def);

// Produce a human-readable version of an op_def that is more concise
// than a text-format proto.  Excludes descriptions.
@Namespace("tensorflow") public static native @StdString BytePointer SummarizeOpDef(@Const @ByRef OpDef op_def);

// Returns an error if new_op is not backwards-compatible with (more
// accepting than) old_op.
// REQUIRES: old_op and new_op must pass validation.
@Namespace("tensorflow") public static native @ByVal Status OpDefCompatible(@Const @ByRef OpDef old_op, @Const @ByRef OpDef new_op);

// Returns an error if any attr in penultimate_op that is not in old_op
// has a different default value in new_op.  In general it is not safe
// to change the default for an attr that has been added to an op.
@Namespace("tensorflow") public static native @ByVal Status OpDefAddedDefaultsUnchanged(@Const @ByRef OpDef old_op,
                                   @Const @ByRef OpDef penultimate_op,
                                   @Const @ByRef OpDef new_op);

  // namespace tensorflow

// #endif  // TENSORFLOW_FRAMEWORK_OP_DEF_UTIL_H_


// Parsed from tensorflow/core/framework/op.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_FRAMEWORK_OP_H_
// #define TENSORFLOW_FRAMEWORK_OP_H_

// #include <functional>
// #include <unordered_map>

// #include "tensorflow/core/framework/op_def.pb.h"
// #include "tensorflow/core/framework/op_def_builder.h"
// #include "tensorflow/core/framework/op_def_util.h"
// #include "tensorflow/core/lib/strings/str_util.h"
// #include "tensorflow/core/lib/strings/strcat.h"
// #include "tensorflow/core/platform/logging.h"
// #include "tensorflow/core/platform/port.h"
// #include "tensorflow/core/platform/thread_annotations.h"
// #include "tensorflow/core/public/status.h"

// Users that want to look up an OpDef by type name should take an
// OpRegistryInterface.  Functions accepting a
// (const) OpRegistryInterface* may call LookUp() from multiple threads.
@Namespace("tensorflow") public static class OpRegistryInterface extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpRegistryInterface(Pointer p) { super(p); }


  // Returns nullptr and sets *status if no OpDef is registered under that
  // name, otherwise returns the registered OpDef.
  // Caller must not delete the returned pointer.
  public native @Const OpDef LookUp(@StdString BytePointer op_type_name,
                                Status status);
  public native @Const OpDef LookUp(@StdString String op_type_name,
                                Status status);
}

// The standard implementation of OpRegistryInterface, along with a
// global singleton used for registering OpDefs via the REGISTER
// macros below.  Thread-safe.
//
// Example registration:
//   OpRegistry::Global()->Register([]()->OpDef{
//     OpDef def;
//     // Populate def here.
//     return def;
//   });
@Namespace("tensorflow") @NoOffset public static class OpRegistry extends OpRegistryInterface {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpRegistry(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public OpRegistry(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public OpRegistry position(int position) {
        return (OpRegistry)super.position(position);
    }

  public OpRegistry() { super((Pointer)null); allocate(); }
  private native void allocate();

  // Calls func() and registers the returned OpDef.  Since Register()
  // is normally called during program initialization (before main()),
  // we defer calling func() until the first call to LookUp() or
  // Export() (if one of those has already been called, func() is
  // called immediately).
  public native void Register(@ByVal OpDefFunc func);

  public native @Const OpDef LookUp(@StdString BytePointer op_type_name,
                        Status status);
  public native @Const OpDef LookUp(@StdString String op_type_name,
                        Status status);

  // Fills *ops with all registered OpDefss (except those with names
  // starting with '_' if include_internal == false).
  public native void Export(@Cast("bool") boolean include_internal, OpList ops);

  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  public native @StdString BytePointer DebugString(@Cast("bool") boolean include_internal);

  // A singleton available at startup.
  public static native OpRegistry Global();
}

// Support for defining the OpDef (specifying the semantics of the Op and how
// it should be created) and registering it in the OpRegistry::Global()
// registry.  Usage:
//
// REGISTER_OP("my_op_name")
//     .Attr("<name>:<type>")
//     .Attr("<name>:<type>=<default>")
//     .Input("<name>:<type-expr>")
//     .Input("<name>:Ref(<type-expr>)")
//     .Output("<name>:<type-expr>")
//     .Doc(R"(
// <1-line summary>
// <rest of the description (potentially many lines)>
// <name-of-attr-input-or-output>: <description of name>
// <name-of-attr-input-or-output>: <description of name;
//   if long, indent the description on subsequent lines>
// )");
//
// Note: .Doc() should be last.
// For details, see the OpDefBuilder class in op_def_builder.h.
// To call OpRegistry::Global()->Register(...), used by the
// REGISTER_OP macro below.
@Namespace("tensorflow::register_op") public static native @ByRef OpDefBuilder RegisterOp(@StringPiece BytePointer name);
@Namespace("tensorflow::register_op") public static native @ByRef OpDefBuilder RegisterOp(@StringPiece String name);
  // namespace register_op

// #define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
// #define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
// #define REGISTER_OP_UNIQ(ctr, name)
//   static ::tensorflow::OpDefBuilder& register_op##ctr TF_ATTRIBUTE_UNUSED =
//       ::tensorflow::register_op::RegisterOp(name)

  // namespace tensorflow

// #endif  // TENSORFLOW_FRAMEWORK_OP_H_


// Parsed from tensorflow/core/framework/types.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_FRAMEWORK_TYPES_H_
// #define TENSORFLOW_FRAMEWORK_TYPES_H_

// #include <map>
// #include <set>
// #include <string>

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// Disable clang-format to prevent 'FixedPoint' header from being included
// before 'Tensor' header on which it depends.
// clang-format off
// #include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
// clang-format on
// #include "tensorflow/core/framework/bfloat16.h"
// #include "tensorflow/core/framework/numeric_types.h"
// #include "tensorflow/core/framework/types.pb.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/lib/gtl/inlined_vector.h"
// #include "tensorflow/core/platform/logging.h"
// #include "tensorflow/core/platform/port.h"

// MemoryType is used to describe whether input or output Tensors of
// an OpKernel should reside in "Host memory" (e.g., CPU memory) or
// "Device" Memory (CPU memory for CPU devices, GPU memory for GPU
// devices).
/** enum tensorflow::MemoryType */
public static final int
  DEVICE_MEMORY = 0,
  HOST_MEMORY = 1;

// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
@Namespace("tensorflow") @NoOffset public static class DeviceType extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DeviceType(Pointer p) { super(p); }

  public DeviceType(@Cast("const char*") BytePointer type) { super((Pointer)null); allocate(type); }
  private native void allocate(@Cast("const char*") BytePointer type);
  public DeviceType(String type) { super((Pointer)null); allocate(type); }
  private native void allocate(String type);

  public native @Cast("const char*") BytePointer type();

  public native @Cast("bool") @Name("operator <") boolean lessThan(@Const @ByRef DeviceType other);
  public native @Cast("bool") @Name("operator ==") boolean equals(@Const @ByRef DeviceType other);
  public native @Cast("bool") @Name("operator !=") boolean notEquals(@Const @ByRef DeviceType other);
}
@Namespace("tensorflow") public static native @Cast("std::ostream*") @ByRef @Name("operator <<") Pointer shiftLeft(@Cast("std::ostream*") @ByRef Pointer os, @Const @ByRef DeviceType d);

// Convenient constants that can be passed to a DeviceType constructor
@Namespace("tensorflow") @MemberGetter public static native @Cast("const char*") BytePointer DEVICE_CPU();  // "CPU"
@Namespace("tensorflow") @MemberGetter public static native @Cast("const char*") BytePointer DEVICE_GPU();  // "GPU"

// Convert the enums to strings for errors:
@Namespace("tensorflow") public static native @StdString BytePointer DataTypeString(@Cast("tensorflow::DataType") int dtype);
@Namespace("tensorflow") public static native @StdString BytePointer DeviceTypeString(@ByVal DeviceType device_type);
@Namespace("tensorflow") public static native @StdString BytePointer DataTypeSliceString(@Const @ByVal DataTypeVector dtypes);
@Namespace("tensorflow") public static native @StdString BytePointer DataTypeVectorString(@Const @ByRef DataTypeVector dtypes);

// If "sp" names a valid type, store it in "*dt" and return true.  Otherwise,
// return false.
@Namespace("tensorflow") public static native @Cast("bool") boolean DataTypeFromString(@StringPiece BytePointer sp, @Cast("tensorflow::DataType*") IntPointer dt);
@Namespace("tensorflow") public static native @Cast("bool") boolean DataTypeFromString(@StringPiece String sp, @Cast("tensorflow::DataType*") IntPointer dt);

// DT_FLOAT + kDataTypeRefOffset == DT_FLOAT_REF, etc.
/** enum tensorflow:: */
public static final int kDataTypeRefOffset = 100;
@Namespace("tensorflow") public static native @Cast("bool") boolean IsRefType(@Cast("tensorflow::DataType") int dtype);
@Namespace("tensorflow") public static native @Cast("tensorflow::DataType") int MakeRefType(@Cast("tensorflow::DataType") int dtype);
@Namespace("tensorflow") public static native @Cast("tensorflow::DataType") int RemoveRefType(@Cast("tensorflow::DataType") int dtype);
@Namespace("tensorflow") public static native @Cast("tensorflow::DataType") int BaseType(@Cast("tensorflow::DataType") int dtype);

// Returns true if the actual type is the same as or ref of the expected type.
@Namespace("tensorflow") public static native @Cast("bool") boolean TypesCompatible(@Cast("tensorflow::DataType") int expected, @Cast("tensorflow::DataType") int actual);

// Does not include _ref types.
@Namespace("tensorflow") public static native @ByVal DataTypeVector AllTypes();

// Return the list of all numeric types.
// NOTE: On Android, we only include the float and int32 types for now.
@Namespace("tensorflow") public static native @ByVal DataTypeVector RealNumberTypes();  // Types that support '<' and '>'.
@Namespace("tensorflow") public static native @ByVal DataTypeVector NumberTypes();      // Includes complex and quantized types.

@Namespace("tensorflow") public static native @ByVal DataTypeVector QuantizedTypes();
@Namespace("tensorflow") public static native @ByVal DataTypeVector RealAndQuantizedTypes();  // Types that support '<' and
                                         // '>', including quantized
                                         // types

// Validates type T for whether it is a supported DataType.

// DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.  // Specializations below

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.  // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
// #define MATCH_TYPE_AND_ENUM(TYPE, ENUM)
//   template <>
//   struct DataTypeToEnum<TYPE> {
//     static DataType v() { return ENUM; }
//     static DataType ref() { return MakeRefType(ENUM); }
//     static constexpr DataType value = ENUM;
//   };
//   template <>
//   struct IsValidDataType<TYPE> {
//     static constexpr bool value = true;
//   };
//   template <>
//   struct EnumToDataType<ENUM> {
//     typedef TYPE Type;
//   }

// We use Eigen's QInt implementations for our quantized int types.

@Name("tensorflow::DataTypeToEnum<float>") public static class DataTypeToEnum extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public DataTypeToEnum() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public DataTypeToEnum(int size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DataTypeToEnum(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(int size);
    @Override public DataTypeToEnum position(int position) {
        return (DataTypeToEnum)super.position(position);
    }

    public static native @Cast("tensorflow::DataType") int v();
    public static native @Cast("tensorflow::DataType") int ref();
    @MemberGetter public static native @Cast("const tensorflow::DataType") int value();
    public static final int value = value();
  }
  @Name("tensorflow::IsValidDataType<float>") public static class IsValidDataType extends Pointer {
      static { Loader.load(); }
      /** Default native constructor. */
      public IsValidDataType() { super((Pointer)null); allocate(); }
      /** Native array allocator. Access with {@link Pointer#position(int)}. */
      public IsValidDataType(int size) { super((Pointer)null); allocateArray(size); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public IsValidDataType(Pointer p) { super(p); }
      private native void allocate();
      private native void allocateArray(int size);
      @Override public IsValidDataType position(int position) {
          return (IsValidDataType)super.position(position);
      }
  
    @MemberGetter public static native @Cast("const bool") boolean value();
    public static final boolean value = value();
  }
  @Name("tensorflow::EnumToDataType<tensorflow::DT_FLOAT>") public static class EnumToDataType extends Pointer {
      static { Loader.load(); }
      /** Default native constructor. */
      public EnumToDataType() { super((Pointer)null); allocate(); }
      /** Native array allocator. Access with {@link Pointer#position(int)}. */
      public EnumToDataType(int size) { super((Pointer)null); allocateArray(size); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public EnumToDataType(Pointer p) { super(p); }
      private native void allocate();
      private native void allocateArray(int size);
      @Override public EnumToDataType position(int position) {
          return (EnumToDataType)super.position(position);
      }
  
  }

// #undef MATCH_TYPE_AND_ENUM

@Namespace("tensorflow") public static native @Cast("bool") boolean DataTypeCanUseMemcpy(@Cast("tensorflow::DataType") int dt);

@Namespace("tensorflow") public static native @Cast("bool") boolean DataTypeIsQuantized(@Cast("tensorflow::DataType") int dt);

  // namespace tensorflow

// #endif  // TENSORFLOW_FRAMEWORK_TYPES_H_


// Parsed from tensorflow/core/graph/edgeset.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_GRAPH_EDGESET_H_
// #define TENSORFLOW_GRAPH_EDGESET_H_

// #include <stddef.h>
// #include <set>
// #include "tensorflow/core/platform/port.h"

// #include "tensorflow/core/platform/logging.h"

// An unordered set of edges.  Uses very little memory for small sets.
// Unlike std::set, EdgeSet does NOT allow mutations during iteration.
@Namespace("tensorflow") @NoOffset public static class EdgeSet extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EdgeSet(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public EdgeSet(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public EdgeSet position(int position) {
        return (EdgeSet)super.position(position);
    }

  public EdgeSet() { super((Pointer)null); allocate(); }
  private native void allocate();

  @Name("const_iterator") @Opaque public static class EdgeSetIterator extends Pointer {
      /** Empty constructor. Calls {@code super((Pointer)null)}. */
      public EdgeSetIterator() { super((Pointer)null); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public EdgeSetIterator(Pointer p) { super(p); }
  }

  public native @Cast("bool") boolean empty();
  public native @Cast("tensorflow::EdgeSet::size_type") long size();
  public native void clear();
  public native @ByVal EdgeSetBoolPair insert(@Cast("tensorflow::EdgeSet::value_type") Edge value);
  public native @Cast("tensorflow::EdgeSet::size_type") long erase(@Cast("tensorflow::EdgeSet::key_type") Edge key);

  // Caller is not allowed to mutate the EdgeSet while iterating.
  public native @ByVal EdgeSetIterator begin();
  public native @ByVal EdgeSetIterator end();
}

@Name("tensorflow::EdgeSet::const_iterator") @NoOffset public static class EdgeSetIterator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EdgeSetIterator(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public EdgeSetIterator(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public EdgeSetIterator position(int position) {
        return (EdgeSetIterator)super.position(position);
    }


  public EdgeSetIterator() { super((Pointer)null); allocate(); }
  private native void allocate();

  public native @ByRef @Name("operator ++") EdgeSetIterator increment();
  public native @ByVal @Name("operator ++") EdgeSetIterator increment(int arg0);
  public native @Cast("const tensorflow::EdgeSet::value_type*") @Name("operator ->") PointerPointer access();
  public native @Cast("tensorflow::EdgeSet::value_type") @Name("operator *") Edge multiply();
  public native @Cast("bool") @Name("operator ==") boolean equals(@Const @ByRef EdgeSetIterator other);
  public native @Cast("bool") @Name("operator !=") boolean notEquals(@Const @ByRef EdgeSetIterator other);
}



















// gcc's set and multiset always use const_iterator since it will otherwise
// allow modification of keys.


// gcc's set and multiset always use const_iterator since it will otherwise
// allow modification of keys.




  // namespace tensorflow

// #endif  // TENSORFLOW_GRAPH_EDGESET_H_


// Parsed from tensorflow/core/lib/gtl/iterator_range.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This provides a very simple, boring adaptor for a begin and end iterator
// into a range type. This should be used to build range views that work well
// with range based for loops and range based constructors.
//
// Note that code here follows more standards-based coding conventions as it
// is mirroring proposed interfaces for standardization.
//
// Converted from chandlerc@'s code to Google style by joshl@.

// #ifndef TENSORFLOW_LIB_GTL_ITERATOR_RANGE_H_
// #define TENSORFLOW_LIB_GTL_ITERATOR_RANGE_H_

// #include <utility>

// A range adaptor for a pair of iterators.
//
// This just wraps two iterators into a range-compatible interface. Nothing
// fancy at all.
@Name("tensorflow::gtl::iterator_range<tensorflow::NeighborIter>") @NoOffset public static class NeighborIterRange extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NeighborIterRange(Pointer p) { super(p); }

  
  public NeighborIterRange(@ByVal NeighborIter begin_iterator, @ByVal NeighborIter end_iterator) { super((Pointer)null); allocate(begin_iterator, end_iterator); }
  private native void allocate(@ByVal NeighborIter begin_iterator, @ByVal NeighborIter end_iterator);

  public native @ByVal NeighborIter begin();
  public native @ByVal NeighborIter end();
}
@Name("tensorflow::gtl::iterator_range<tensorflow::NodeIter>") @NoOffset public static class NodeIterRange extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NodeIterRange(Pointer p) { super(p); }

  
  public NodeIterRange(@ByVal NodeIter begin_iterator, @ByVal NodeIter end_iterator) { super((Pointer)null); allocate(begin_iterator, end_iterator); }
  private native void allocate(@ByVal NodeIter begin_iterator, @ByVal NodeIter end_iterator);

  public native @ByVal NodeIter begin();
  public native @ByVal NodeIter end();
}

// Convenience function for iterating over sub-ranges.
//
// This provides a bit of syntactic sugar to make using sub-ranges
// in for loops a bit easier. Analogous to std::make_pair().

  // namespace gtl
  // namespace tensorflow

// #endif  // TENSORFLOW_LIB_GTL_ITERATOR_RANGE_H_


// Parsed from tensorflow/core/graph/graph.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A Graph describes a set of computations that are to be
// performed, as well as the dependencies between those
// compuations. The basic model is a DAG (directed acyclic graph) with
// * internal nodes representing computational operations to be performed;
// * edges represent dependencies, indicating the target may only be
//   executed once the source has completed; and
// * predefined "source" (start) and "sink" (finish) nodes -- the source
//   should be the only node that doesn't depend on anything, and the sink
//   should be the only node that nothing depends on.
//
// Note: Node ids are intended to be relatively dense in the
// 0..max_id range, but there may be gaps since ids won't be reused.
//
// Note: Some dependencies between operations are due to one operation
// consuming the output of another. In fact operations can produce
// multiple outputs and consume multiple inputs, and some
// optimizations will care about which specific outputs are connected
// to which specific inputs.  We therefore represent data dependency
// between output O of layer A and input I of layer B using
// "input index" and "output index" labels per edge.

// #ifndef TENSORFLOW_GRAPH_GRAPH_H_
// #define TENSORFLOW_GRAPH_GRAPH_H_

// #include <functional>
// #include <string>
// #include <vector>
// #include "tensorflow/core/framework/graph.pb.h"
// #include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/edgeset.h"
// #include "tensorflow/core/lib/core/arena.h"
// #include "tensorflow/core/lib/core/refcount.h"
// #include "tensorflow/core/lib/gtl/iterator_range.h"
// #include "tensorflow/core/platform/logging.h"
// #include "tensorflow/core/platform/port.h"
// #include "tensorflow/core/public/status.h"
@Namespace("tensorflow") @Opaque public static class EdgeSetTest extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public EdgeSetTest() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EdgeSetTest(Pointer p) { super(p); }
}  // Declared below      // Declared below

@Namespace("tensorflow") @NoOffset public static class Node extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Node(Pointer p) { super(p); }

  public native @StdString BytePointer DebugString();
  public native int id();
  public native int cost_id();
  public native @StdString BytePointer name();
  public native @StdString BytePointer type_string();
  public native @Const @ByRef NodeDef def();
  public native @Const @ByRef OpDef op_def();

  // input and output types
  public native int num_inputs();
  public native @Cast("tensorflow::DataType") int input_type(int i);
  public native @Const @ByRef DataTypeVector input_types();

  public native int num_outputs();
  public native @Cast("tensorflow::DataType") int output_type(int o);
  public native @Const @ByRef DataTypeVector output_types();

  // This gives the device the runtime has assigned this node to.  If
  // you want the device the user requested, use def().device() instead.
  // TODO(josh11b): Validate that the assigned_device, if not empty:
  // fully specifies a device, and satisfies def().device().
  // TODO(josh11b): Move device_name outside of Node into a NodeId->DeviceName
  // map.
  public native @StdString BytePointer assigned_device_name();
  public native void set_assigned_device_name(@StdString BytePointer device_name);
  public native void set_assigned_device_name(@StdString String device_name);

  // Get the neighboring nodes via edges either in or out of this node.
  public native @ByVal NeighborIterRange in_nodes();
  public native @ByVal NeighborIterRange out_nodes();
  public native @Const @ByRef EdgeSet in_edges();
  public native @Const @ByRef EdgeSet out_edges();

  // Node type helpers.
  public native @Cast("bool") boolean IsSource();
  public native @Cast("bool") boolean IsSink();
  // Anything other than the special Source & Sink nodes.
  public native @Cast("bool") boolean IsOp();
}

@Namespace("tensorflow") @NoOffset public static class Edge extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Edge(Pointer p) { super(p); }

  public native Node src();
  public native Node dst();
  public native int id();

  // Return the number of the source output that produces the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  public native int src_output();

  // Return the number of the destination input that consumes the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  public native int dst_input();

  // Return true iff this is an edge that indicates a control-flow
  // (as opposed to a data-flow) dependency.
  public native @Cast("bool") boolean IsControlEdge();
}

// Thread compatible but not thread safe.
@Namespace("tensorflow") @NoOffset public static class Graph extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Graph(Pointer p) { super(p); }

  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in registry.
  public Graph(@Const OpRegistryInterface registry) { super((Pointer)null); allocate(registry); }
  private native void allocate(@Const OpRegistryInterface registry);

  @MemberGetter public static native int kControlSlot();
  public static final int kControlSlot = kControlSlot();

  // Adds a new node to this graph, and returns it. Infers the Op and
  // input/output types for the node. *this owns the returned instance.
  // Returns nullptr and sets *status on error.
  public native Node AddNode(@Const @ByRef NodeDef node_def, Status status);

  // Copies *node, which may belong to another graph, to a new node,
  // which is returned.  Does not copy any edges.  *this owns the
  // returned instance.
  public native Node CopyNode(Node node);

  // Remove a node from this graph, including all edges from or to it.
  // *node should not be accessed after calling this function.
  // REQUIRES: node->IsOp()
  public native void RemoveNode(Node node);

  // Add an edge that connects the xth output of "source" to the yth input
  // of "dest".
  public native @Const Edge AddEdge(Node source, int x, Node dest, int y);

  // Add a control-edge (no data flows along this edge) that
  // connects "source" to "dest".
  public native @Const Edge AddControlEdge(Node source, Node dest);

  // Removes edge from the graph.
  // REQUIRES: The edge must exist.
  public native void RemoveEdge(@Const Edge edge);

  // Returns one more than the maximum id assigned to any node.
  public native int num_node_ids();

  // Serialize to a GraphDef.
  public native void ToGraphDef(GraphDef graph_def);

  // Generate new node name with the specified prefix that is unique
  // across this graph.
  public native @StdString BytePointer NewName(@StringPiece BytePointer prefix);
  public native @StdString String NewName(@StringPiece String prefix);

  // Access to the list of all nodes.  Example usage:
  //   for (Node* node : graph.nodes()) { ... }
  public native @ByVal NodeIterRange nodes();

  // Returns the node associated with an id, or nullptr if no node
  // with that id (the node with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_node_ids().
  public native Node FindNodeId(int id);

  // Returns one more than the maximum id assigned to any edge.
  public native int num_edge_ids();

  // Returns the Edge associated with an id, or nullptr if no edge
  // with that id (the node with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_node_ids().
  public native @Const Edge FindEdgeId(int id);

  // Access to the set of all edges.  Example usage:
  //   for (const Edge* e : graph.edges()) { ... }
  public native @Const @ByRef EdgeSet edges();

  // The pre-defined nodes.
  /** enum tensorflow::Graph:: */
  public static final int kSourceId = 0, kSinkId = 1;
  public native Node source_node();
  public native Node sink_node();

  public native @Const OpRegistryInterface op_registry();
}

// TODO(josh11b): We may want to support keeping an index on various
// node/edge attributes in a graph, particularly node names.

// Helper routines

@Namespace("tensorflow") public static native @Cast("bool") boolean IsSwitch(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsMerge(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsEnter(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsExit(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsNextIteration(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsLoopCond(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsControlTrigger(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsSend(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsRecv(@Const Node node);

// True for Nodes that mediate the transfer of values between processes.
@Namespace("tensorflow") public static native @Cast("bool") boolean IsTransferNode(@Const Node n);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsConstant(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsVariable(@Const Node node);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsIdentity(@Const Node node);

// Returns true iff 'n' is a control flow node.
@Namespace("tensorflow") public static native @Cast("bool") boolean IsControlFlow(@Const Node n);

@Namespace("tensorflow") public static native @Cast("bool") boolean IsHostMemoryPreserving(@Const Node node);

// Iterator for stepping through the nodes of a graph.
@Namespace("tensorflow") @NoOffset public static class NodeIter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NodeIter(Pointer p) { super(p); }

  public NodeIter(@Const Graph graph, int id) { super((Pointer)null); allocate(graph, id); }
  private native void allocate(@Const Graph graph, int id);
  public native @Cast("bool") @Name("operator ==") boolean equals(@Const @ByRef NodeIter rhs);
  public native @Cast("bool") @Name("operator !=") boolean notEquals(@Const @ByRef NodeIter rhs);
  public native @Name("operator ++") void increment();
  public native @Name("operator *") Node multiply();
  public native @Name("operator ->") Node access();
}

// Iterator for stepping through the neighbors of a node.
@Namespace("tensorflow") @NoOffset public static class NeighborIter extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NeighborIter(Pointer p) { super(p); }

  public NeighborIter(@ByVal EdgeSetIterator iter, @Cast("bool") boolean incoming) { super((Pointer)null); allocate(iter, incoming); }
  private native void allocate(@ByVal EdgeSetIterator iter, @Cast("bool") boolean incoming);
  public native @Cast("bool") @Name("operator ==") boolean equals(@Const @ByRef NeighborIter rhs);
  public native @Cast("bool") @Name("operator !=") boolean notEquals(@Const @ByRef NeighborIter rhs);
  public native @Name("operator ++") void increment();
  public native @Name("operator *") Node multiply();
  public native @Name("operator ->") Node access();
}

// IMPLEMENTATION DETAILS, PLEASE IGNORE



























  // namespace tensorflow

// #endif  // TENSORFLOW_GRAPH_GRAPH_H_


// Parsed from tensorflow/core/graph/node_builder.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_GRAPH_NODE_BUILDER_H_
// #define TENSORFLOW_GRAPH_NODE_BUILDER_H_

// #include <vector>
// #include "tensorflow/core/framework/node_def_builder.h"
// #include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/framework/op_def.pb.h"
// #include "tensorflow/core/graph/graph.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/status.h"

// This is a helper for creating a Node and adding it to a Graph.
// Internally, it uses a NodeDefBuilder to automatically set attrs
// that can be inferred from the inputs, and use default values
// (where they exist) for unspecified attrs.  Example usage:
//
//  Node* node;
//  Status status = NodeBuilder(node_name, op_name)
//                           .Input(...)
//                           .Attr(...)
//                           .Finalize(&graph, &node);
//  if (!status.ok()) return status;
//  // Use node here.
@Namespace("tensorflow") @NoOffset public static class NodeBuilder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NodeBuilder(Pointer p) { super(p); }

  // For specifying the output of a Node to provide to one of the Input()
  // functions below.  It supports both regular inputs (where you are
  // connecting to an existing Node*), and inputs from outside the graph
  // (or haven't been added to the graph yet, like back edges, where
  // you don't have a Node*). Both types can be mixed, e.g. in an
  // ArraySlice.
  @NoOffset public static class NodeOut extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public NodeOut(Pointer p) { super(p); }
      /** Native array allocator. Access with {@link Pointer#position(int)}. */
      public NodeOut(int size) { super((Pointer)null); allocateArray(size); }
      private native void allocateArray(int size);
      @Override public NodeOut position(int position) {
          return (NodeOut)super.position(position);
      }
  
    // For referencing an existing Node.
    public NodeOut(Node n, int i/*=0*/) { super((Pointer)null); allocate(n, i); }
    private native void allocate(Node n, int i/*=0*/);
    public NodeOut(Node n) { super((Pointer)null); allocate(n); }
    private native void allocate(Node n);

    // For referencing Nodes not in the graph being built. It is
    // useful when preparing a graph for ExtendSession or creating a
    // back edge to a node that hasn't been added to the graph yet,
    // but will be.
    public NodeOut(@StdString BytePointer name, int i, @Cast("tensorflow::DataType") int t) { super((Pointer)null); allocate(name, i, t); }
    private native void allocate(@StdString BytePointer name, int i, @Cast("tensorflow::DataType") int t);
    public NodeOut(@StdString String name, int i, @Cast("tensorflow::DataType") int t) { super((Pointer)null); allocate(name, i, t); }
    private native void allocate(@StdString String name, int i, @Cast("tensorflow::DataType") int t);

    // Default constructor for std::vector<NodeOut>.
    public NodeOut() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native Node node(); public native NodeOut node(Node node);
    // error is set to true if:
    // * the NodeOut was default constructed and never overwritten,
    // * a nullptr Node* was passed to the NodeOut constructor, or
    // * an out-of-range index was passed to the NodeOut constructor.
    public native @Cast("bool") boolean error(); public native NodeOut error(boolean error);
    public native @StdString BytePointer name(); public native NodeOut name(BytePointer name);
    public native int index(); public native NodeOut index(int index);
    public native @Cast("tensorflow::DataType") int dt(); public native NodeOut dt(int dt);
  }

  // Specify the name and the Op (either via an OpDef or the name of
  // the Op plus a registry) for the Node.  Other fields are
  // specified by calling the methods below.
  // REQUIRES: The OpDef must satisfy ValidateOpDef().
  public NodeBuilder(@StdString BytePointer name, @StdString BytePointer op_name,
                @Const OpRegistryInterface op_registry/*=tensorflow::OpRegistry::Global()*/) { super((Pointer)null); allocate(name, op_name, op_registry); }
  private native void allocate(@StdString BytePointer name, @StdString BytePointer op_name,
                @Const OpRegistryInterface op_registry/*=tensorflow::OpRegistry::Global()*/);
  public NodeBuilder(@StdString BytePointer name, @StdString BytePointer op_name) { super((Pointer)null); allocate(name, op_name); }
  private native void allocate(@StdString BytePointer name, @StdString BytePointer op_name);
  public NodeBuilder(@StdString String name, @StdString String op_name,
                @Const OpRegistryInterface op_registry/*=tensorflow::OpRegistry::Global()*/) { super((Pointer)null); allocate(name, op_name, op_registry); }
  private native void allocate(@StdString String name, @StdString String op_name,
                @Const OpRegistryInterface op_registry/*=tensorflow::OpRegistry::Global()*/);
  public NodeBuilder(@StdString String name, @StdString String op_name) { super((Pointer)null); allocate(name, op_name); }
  private native void allocate(@StdString String name, @StdString String op_name);
  public NodeBuilder(@StdString BytePointer name, @Const OpDef op_def) { super((Pointer)null); allocate(name, op_def); }
  private native void allocate(@StdString BytePointer name, @Const OpDef op_def);
  public NodeBuilder(@StdString String name, @Const OpDef op_def) { super((Pointer)null); allocate(name, op_def); }
  private native void allocate(@StdString String name, @Const OpDef op_def);

  // You must call one Input() function per input_arg in the Op,
  // *and in the same order as the input_args appear in the OpDef.*

  // For inputs that take a single tensor.
  public native @ByRef NodeBuilder Input(Node src_node, int src_index/*=0*/);
  public native @ByRef NodeBuilder Input(Node src_node);
  public native @ByRef NodeBuilder Input(@ByVal NodeOut src);

  // For inputs that take a list of tensors.

  // Require that this node run after src_node(s).
  public native @ByRef NodeBuilder ControlInput(Node src_node);
  public native @ByRef NodeBuilder ControlInputs(@ByVal NodeVector src_nodes);

  // Sets the "requested device spec" in the NodeDef (not the
  // "assigned device" in the Node).
  public native @ByRef NodeBuilder Device(@StdString BytePointer device_spec);
  public native @ByRef NodeBuilder Device(@StdString String device_spec);

  // Set the value of an attr.  attr_name must match the name of one of
  // attrs defined by the Op, and value must have the corresponding type
  // (see SetAttrValue() in ../framework/attr_value_util.h for legal
  // types for value).  Note that attrs will be set automatically if
  // they can be determined by the inputs.

  // Validates the described node and adds it to *graph, adding edges
  // for all (non-back) inputs.  If created_node is not nullptr,
  // *created_node will be set to the new node (or nullptr on error).
  public native @ByVal Status Finalize(Graph graph, @Cast("tensorflow::Node**") PointerPointer created_node);
  public native @ByVal Status Finalize(Graph graph, @ByPtrPtr Node created_node);
}

// IMPLEMENTATION -------------------------------------------------------------





  // namespace tensorflow

// #endif  // TENSORFLOW_GRAPH_NODE_BUILDER_H_


// Parsed from tensorflow/core/graph/graph_def_builder.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_GRAPH_GRAPH_DEF_BUILDER_H_
// #define TENSORFLOW_GRAPH_GRAPH_DEF_BUILDER_H_

// #include <vector>
// #include "tensorflow/core/framework/graph.pb.h"
// #include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/graph/graph.h"
// #include "tensorflow/core/graph/node_builder.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/status.h"

// Given a function like:
//   namespace ops {
//   Node* Identity(NodeOut input, const GraphDefBuilder::Options& opts) {
//     if (opts.HaveError()) return nullptr;
//     static const string kOpName = "Identity";
//     NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
//                              opts.op_registry());
//     node_builder.Input(input);
//     return opts.FinalizeBuilder(&node_builder);
//   }
//   }  // namspace ops
//
//   // Or, alternatively:
//   namespace ops {
//   Node* Identity(NodeOut input, const GraphDefBuilder::Options& opts) {
//     static const string kOpName = "Identity";
//     return UnaryOp(kOpName, input, opts);
//   }
//   }  // namspace ops
//
// You call it like:
//   GraphDefBuilder b;
//   using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
//   Node* a = Const(7, b.opts());
//   // Note: WithName() returns a copy, opts is unchanged.
//   Node* b = Const(5, b.opts().WithName("control-input"));
//   Node* c = Identity(a, b.opts().WithControlInput(b));
//   GraphDef graph_def;
//   Status status = b.ToGraphDef(&graph_def);
//   if (!status.ok()) { /* Handle error */ }
//
// In tests you can skip the status handling via:
//   GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
//   ...
//   b.ToGraphDef(&graph_def);

@Namespace("tensorflow") @NoOffset public static class GraphDefBuilder extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GraphDefBuilder(Pointer p) { super(p); }

  // Options for adding a Node to a Graph.
  @NoOffset public static class Options extends Pointer {
      static { Loader.load(); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public Options(Pointer p) { super(p); }
  
    // Sets the Graph (that Nodes will be added to) and the status.  The
    // status may be set to nullptr, in which case errors cause CHECK
    // failures.  The graph and status must outlive *this.
    public Options(Graph graph, Status status) { super((Pointer)null); allocate(graph, status); }
    private native void allocate(Graph graph, Status status);

    // Methods for setting options.  These are const methods: they
    // return a copy of *this with the option set.
    public native @ByVal Options WithName(@StringPiece BytePointer name);
    public native @ByVal Options WithName(@StringPiece String name);
    public native @ByVal Options WithDevice(@StringPiece BytePointer device);
    public native @ByVal Options WithDevice(@StringPiece String device);
    public native @ByVal Options WithControlInput(Node control_input);
    public native @ByVal Options WithControlInputs(@ByVal NodeVector control_inputs);

    // Override the default value for an optional attr.
    // Note: overload needed to allow {...} expressions for value.

    // Methods for using options from a function that creates a Node.

    // Returns true if the status associated with *this has an error.
    // Use this to skip processing that may depend on prior results.
    public native @Cast("bool") boolean HaveError();

    // Given the Op type name, return a name for a node of that type.
    // Uses the value set in WithName() if that has been called.  Otherwise,
    // returns a name built out of the Op type name.
    public native @StdString BytePointer GetNameForOp(@StringPiece BytePointer op);
    public native @StdString String GetNameForOp(@StringPiece String op);

    // Sets the device, adds control inputs, adds attrs, and calls Finalize().
    // If Finalize returns an error, it is saved and this function returns
    // nullptr.
    public native Node FinalizeBuilder(NodeBuilder builder);

    // Updates the associated status, if any, or calls TF_CHECK_OK if none.
    public native void UpdateStatus(@Const @ByRef Status status);

    // Accessor
    public native @Const OpRegistryInterface op_registry();
  }

  // Start building a new graph.
  public GraphDefBuilder(
        @Const OpRegistryInterface op_registry/*=tensorflow::OpRegistry::Global()*/) { super((Pointer)null); allocate(op_registry); }
  private native void allocate(
        @Const OpRegistryInterface op_registry/*=tensorflow::OpRegistry::Global()*/);
  public GraphDefBuilder() { super((Pointer)null); allocate(); }
  private native void allocate();

  // For use in tests, where you want to fail immediately on error instead
  // of checking the status at the end.
  /** enum tensorflow::GraphDefBuilder::TestFailImmediatelyType */
  public static final int kFailImmediately = 0;
  public GraphDefBuilder(
        @Cast("tensorflow::GraphDefBuilder::TestFailImmediatelyType") int arg0,
        @Const OpRegistryInterface op_registry/*=tensorflow::OpRegistry::Global()*/) { super((Pointer)null); allocate(arg0, op_registry); }
  private native void allocate(
        @Cast("tensorflow::GraphDefBuilder::TestFailImmediatelyType") int arg0,
        @Const OpRegistryInterface op_registry/*=tensorflow::OpRegistry::Global()*/);
  public GraphDefBuilder(
        @Cast("tensorflow::GraphDefBuilder::TestFailImmediatelyType") int arg0) { super((Pointer)null); allocate(arg0); }
  private native void allocate(
        @Cast("tensorflow::GraphDefBuilder::TestFailImmediatelyType") int arg0);

  // Gets the Options with the associated Graph and Status.
  public native @Const @ByRef Options opts();

  // Once all the nodes have been added, call this to get whether it was
  // successful, and if so fill *graph_def.
  public native @ByVal Status ToGraphDef(GraphDef graph_def);

  // Like ToGraphDef(), but converts to a Graph (using the default
  // GraphConstructorOptions).
  // TODO(josh11b): Make this faster; right now it converts
  // Graph->GraphDef->Graph.  This cleans up the graph (e.g. adds
  // edges from the source and to the sink node, resolves back edges
  // by name), and makes sure the resulting graph is valid.
  public native @ByVal Status ToGraph(Graph graph);
}

// A NodeOut may either be a regular input or back input.  Regular
// inputs are specified via either a Node* or a Node* and an output
// index.  Back inputs are specified by a node name, output index, and
// output type.

// For adding an Op with no inputs to a GraphDefBuilder.
@Namespace("tensorflow::ops") public static native Node SourceOp(@StdString BytePointer op_name, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SourceOp(@StdString String op_name, @Const @ByRef GraphDefBuilder.Options opts);

// For adding an Op with one input to a GraphDefBuilder.
@Namespace("tensorflow::ops") public static native Node UnaryOp(@StdString BytePointer op_name, @ByVal NodeBuilder.NodeOut input,
              @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node UnaryOp(@StdString String op_name, Node input,
              @Const @ByRef GraphDefBuilder.Options opts);

// For adding an Op with two inputs to a GraphDefBuilder.
@Namespace("tensorflow::ops") public static native Node BinaryOp(@StdString BytePointer op_name, @ByVal NodeBuilder.NodeOut a, @ByVal NodeBuilder.NodeOut b,
               @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BinaryOp(@StdString String op_name, Node a, Node b,
               @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_GRAPH_GRAPH_DEF_BUILDER_H_


// Parsed from tensorflow/core/graph/default_device.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_
// #define TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_

// #include <string>

// #include "tensorflow/core/framework/graph.pb.h"

// Sets the default device for all nodes in graph_def to "device",
// only if not already set.
@Namespace("tensorflow::graph") public static native void SetDefaultDevice(@StdString BytePointer device, GraphDef graph_def);
@Namespace("tensorflow::graph") public static native void SetDefaultDevice(@StdString String device, GraphDef graph_def);

  // namespace graph
  // namespace tensorflow

// #endif  // TENSORFLOW_GRAPH_DEFAULT_DEVICE_H_


// Parsed from tensorflow/cc/ops/standard_ops.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #include this file to get access to the standard set of C++ graph
// definition libraries.

// #ifndef TENSORFLOW_CC_OPS_STANDARD_OPS_H_
// #define TENSORFLOW_CC_OPS_STANDARD_OPS_H_

// #include "tensorflow/cc/ops/array_ops.h"
// #include "tensorflow/cc/ops/attention_ops.h"
// #include "tensorflow/cc/ops/const_op.h"
// #include "tensorflow/cc/ops/data_flow_ops.h"
// #include "tensorflow/cc/ops/image_ops.h"
// #include "tensorflow/cc/ops/io_ops.h"
// #include "tensorflow/cc/ops/linalg_ops.h"
// #include "tensorflow/cc/ops/logging_ops.h"
// #include "tensorflow/cc/ops/math_ops.h"
// #include "tensorflow/cc/ops/nn_ops.h"
// #include "tensorflow/cc/ops/parsing_ops.h"
// #include "tensorflow/cc/ops/random_ops.h"
// #include "tensorflow/cc/ops/sparse_ops.h"
// #include "tensorflow/cc/ops/state_ops.h"
// #include "tensorflow/cc/ops/string_ops.h"
// #include "tensorflow/cc/ops/summary_ops.h"
// #include "tensorflow/cc/ops/training_ops.h"
// #include "tensorflow/cc/ops/user_ops.h"

// #endif  // TENSORFLOW_CC_OPS_STANDARD_OPS_H_


// Parsed from tensorflow/cc/ops/const_op.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_CC_OPS_CONST_OP_H_
// #define TENSORFLOW_CC_OPS_CONST_OP_H_

// #include "tensorflow/core/framework/tensor.pb.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"

// If a shape is specified, you may either provide the same number of values,
// or a single value and that value will be duplicated to fill out the Tensor.
// #define DECLARE_CONST(TYPE)
//   Node* Const(TYPE s, const GraphDefBuilder::Options& options); /* Scalar */
//   Node* Const(gtl::ArraySlice<TYPE> v,
//               const GraphDefBuilder::Options& options); /* Vector */
//   Node* Const(gtl::ArraySlice<TYPE> t, const TensorShape& shape,
//               const GraphDefBuilder::Options& options); /* Tensor */
//   inline Node* Const(std::initializer_list<TYPE> v, /* Vector using {...} */
//                      const GraphDefBuilder::Options& options) {
//     return Const(gtl::ArraySlice<TYPE>(v), options);
//   }
//   inline Node* Const(std::initializer_list<TYPE> t, /* Tensor using {...} */
//                      const TensorShape& shape,
//                      const GraphDefBuilder::Options& options) {
//     return Const(gtl::ArraySlice<TYPE>(t), shape, options);
//   }

@Namespace("tensorflow::ops") public static native Node Const(float s, @Const @ByRef GraphDefBuilder.Options options); /* Scalar */
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice FloatPointer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice FloatBuffer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice float[] v,
                @Const @ByRef GraphDefBuilder.Options options); /* Vector */
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice FloatPointer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice FloatBuffer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice float[] t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options); /* Tensor */
@Namespace("tensorflow::ops") public static native Node Const(double s, @Const @ByRef GraphDefBuilder.Options options); /* Scalar */
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice DoublePointer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice DoubleBuffer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice double[] v,
                @Const @ByRef GraphDefBuilder.Options options); /* Vector */
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice DoublePointer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice DoubleBuffer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@ArraySlice double[] t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options); /* Tensor */
@Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int32") int s, @Const @ByRef GraphDefBuilder.Options options); /* Scalar */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int32*") @ArraySlice IntPointer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int32*") @ArraySlice IntBuffer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int32*") @ArraySlice int[] v,
                @Const @ByRef GraphDefBuilder.Options options); /* Vector */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int32*") @ArraySlice IntPointer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int32*") @ArraySlice IntBuffer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int32*") @ArraySlice int[] t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options); /* Tensor */
@Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::uint8") byte s, @Const @ByRef GraphDefBuilder.Options options); /* Scalar */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::uint8*") @ArraySlice BytePointer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::uint8*") @ArraySlice ByteBuffer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::uint8*") @ArraySlice byte[] v,
                @Const @ByRef GraphDefBuilder.Options options); /* Vector */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::uint8*") @ArraySlice BytePointer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::uint8*") @ArraySlice ByteBuffer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::uint8*") @ArraySlice byte[] t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options); /* Tensor */
@Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int16") short s, @Const @ByRef GraphDefBuilder.Options options); /* Scalar */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int16*") @ArraySlice ShortPointer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int16*") @ArraySlice ShortBuffer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int16*") @ArraySlice short[] v,
                @Const @ByRef GraphDefBuilder.Options options); /* Vector */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int16*") @ArraySlice ShortPointer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int16*") @ArraySlice ShortBuffer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int16*") @ArraySlice short[] t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options); /* Tensor */ /* Scalar */ /* Vector */ /* Tensor */ /* Scalar */ /* Vector */ /* Tensor */
@Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int64") long s, @Const @ByRef GraphDefBuilder.Options options); /* Scalar */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int64*") @ArraySlice LongPointer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int64*") @ArraySlice LongBuffer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int64*") @ArraySlice long[] v,
                @Const @ByRef GraphDefBuilder.Options options); /* Vector */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int64*") @ArraySlice LongPointer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int64*") @ArraySlice LongBuffer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("tensorflow::int64*") @ArraySlice long[] t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options); /* Tensor */
@Namespace("tensorflow::ops") public static native Node Const(@Cast("bool") boolean s, @Const @ByRef GraphDefBuilder.Options options); /* Scalar */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("bool*") @ArraySlice BoolPointer v,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("bool*") @ArraySlice boolean[] v,
                @Const @ByRef GraphDefBuilder.Options options); /* Vector */
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("bool*") @ArraySlice BoolPointer t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options);
  @Namespace("tensorflow::ops") public static native Node Const(@Cast("bool*") @ArraySlice boolean[] t, @Const @ByRef TensorShape shape,
                @Const @ByRef GraphDefBuilder.Options options); /* Tensor */

// #undef DECLARE_CONST

// String
@Namespace("tensorflow::ops") public static native Node Const(@Cast({"", "tensorflow::StringPiece&"}) @StringPiece String s, @Const @ByRef GraphDefBuilder.Options options);

// A Tensor of any type.
@Namespace("tensorflow::ops") public static native Node Const(@Const @ByRef Tensor t, @Const @ByRef GraphDefBuilder.Options options);
@Namespace("tensorflow::ops") public static native Node Const(@Const @ByRef TensorProto proto, @Const @ByRef GraphDefBuilder.Options options);

// TODO(josh11b): Support other types (e.g. quantized ints, float16).

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_CONST_OP_H_


// Parsed from tensorflow/cc/ops/cc_op_gen.h

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #ifndef TENSORFLOW_CC_OPS_CC_OP_GEN_H_
// #define TENSORFLOW_CC_OPS_CC_OP_GEN_H_

// #include "tensorflow/core/framework/op_def.pb.h"

// Result is written to files dot_h and dot_cc.
@Namespace("tensorflow") public static native void WriteCCOps(@Const @ByRef OpList ops, @StdString BytePointer dot_h_fname,
                @StdString BytePointer dot_cc_fname);
@Namespace("tensorflow") public static native void WriteCCOps(@Const @ByRef OpList ops, @StdString String dot_h_fname,
                @StdString String dot_cc_fname);

  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_CC_OP_GEN_H_


// Parsed from tensorflow/cc/ops/array_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_ARRAY_OPS_H_
// #define TENSORFLOW_CC_OPS_ARRAY_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
//
// This is typically used by gradient computations for a broadcasting operation.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * r0
// * r1
@Namespace("tensorflow::ops") public static native Node BroadcastGradientArgs(@ByVal NodeBuilder.NodeOut s0, @ByVal NodeBuilder.NodeOut s1, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BroadcastGradientArgs(Node s0, Node s1, @Const @ByRef GraphDefBuilder.Options opts);

// Checks a tensor for NaN and Inf values.
//
// When run, reports an `InvalidArgument` error if `tensor` has any values
// that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.
//
// Arguments:
// * message: Prefix of the error message.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node CheckNumerics(@ByVal NodeBuilder.NodeOut tensor, @StringPiece BytePointer message, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node CheckNumerics(Node tensor, @StringPiece String message, @Const @ByRef GraphDefBuilder.Options opts);

// Concatenates tensors along one dimension.
//
// Arguments:
// * concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
// range [0, rank(values)).
// * values: The `N` Tensors to concatenate. Their ranks and types must match,
// and their sizes must match in all dimensions except `concat_dim`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A `Tensor` with the concatenation of values stacked along the
// `concat_dim` dimension.  This tensor's shape matches that of `values` except
// in `concat_dim` where it has the sum of the sizes.
@Namespace("tensorflow::ops") public static native Node Concat(@ByVal NodeBuilder.NodeOut concat_dim, @ByVal NodeOutVector values, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Concat(Node concat_dim, @ByVal NodeOutVector values, @Const @ByRef GraphDefBuilder.Options opts);

// Returns a constant tensor.
//
// Arguments:
// * value: Attr `value` is the tensor to return.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Const(@Const @ByRef Tensor value, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);

// Returns a diagonal tensor with a given diagonal values.
//
// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
// everything else padded with zeros. The diagonal is computed as follows:
//
// Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
// rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
//
// `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
//
// For example:
//
// ```prettyprint
// # 'diagonal' is [1, 2, 3, 4]
// tf.diag(diagonal) ==> [[1, 0, 0, 0]
//                        [0, 2, 0, 0]
//                        [0, 0, 3, 0]
//                        [0, 0, 0, 4]]
// ```
//
// Arguments:
// * diagonal: Rank k tensor where k is at most 3.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Diag(@ByVal NodeBuilder.NodeOut diagonal, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Diag(Node diagonal, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the (possibly normalized) Levenshtein Edit Distance.
//
// The inputs are variable-length sequences provided by SparseTensors
//   (hypothesis_indices, hypothesis_values, hypothesis_shape)
// and
//   (truth_indices, truth_values, truth_shape).
//
// The inputs are:
//
// Arguments:
// * hypothesis_indices: The indices of the hypothesis list SparseTensor.
// This is an N x R int64 matrix.
// * hypothesis_values: The values of the hypothesis list SparseTensor.
// This is an N-length vector.
// * hypothesis_shape: The shape of the hypothesis list SparseTensor.
// This is an R-length vector.
// * truth_indices: The indices of the truth list SparseTensor.
// This is an M x R int64 matrix.
// * truth_values: The values of the truth list SparseTensor.
// This is an M-length vector.
// * truth_shape: truth indices, vector.
// * opts:
//   .WithAttr("normalize", bool): Defaults to true.
//     boolean (if true, edit distances are normalized by length of truth).
//
// The output is:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A dense float tensor with rank R - 1.
//
// For the example input:
//
//     // hypothesis represents a 2x1 matrix with variable-length values:
//     //   (0,0) = ["a"]
//     //   (1,0) = ["b"]
//     hypothesis_indices = [[0, 0, 0],
//                           [1, 0, 0]]
//     hypothesis_values = ["a", "b"]
//     hypothesis_shape = [2, 1, 1]
//
//     // truth represents a 2x2 matrix with variable-length values:
//     //   (0,0) = []
//     //   (0,1) = ["a"]
//     //   (1,0) = ["b", "c"]
//     //   (1,1) = ["a"]
//     truth_indices = [[0, 1, 0],
//                      [1, 0, 0],
//                      [1, 0, 1],
//                      [1, 1, 0]]
//     truth_values = ["a", "b", "c", "a"]
//     truth_shape = [2, 2, 2]
//     normalize = true
//
// The output will be:
//
//     // output is a 2x2 matrix with edit distances normalized by truth lengths.
//     output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
//               [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
@Namespace("tensorflow::ops") public static native Node EditDistance(@ByVal NodeBuilder.NodeOut hypothesis_indices, @ByVal NodeBuilder.NodeOut hypothesis_values,
                   @ByVal NodeBuilder.NodeOut hypothesis_shape, @ByVal NodeBuilder.NodeOut truth_indices, @ByVal NodeBuilder.NodeOut truth_values, @ByVal NodeBuilder.NodeOut truth_shape, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node EditDistance(Node hypothesis_indices, Node hypothesis_values,
                   Node hypothesis_shape, Node truth_indices, Node truth_values, Node truth_shape, @Const @ByRef GraphDefBuilder.Options opts);

// Inserts a dimension of 1 into a tensor's shape.
//
// Given a tensor `input`, this operation inserts a dimension of 1 at the
// dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
// zero; if you specify a negative number for `dim` it is counted backward from
// the end.
//
// This operation is useful if you want to add a batch dimension to a single
// element. For example, if you have a single image of shape `[height, width,
// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
// which will make the shape `[1, height, width, channels]`.
//
// Other examples:
//
// ```prettyprint
// # 't' is a tensor of shape [2]
// shape(expand_dims(t, 0)) ==> [1, 2]
// shape(expand_dims(t, 1)) ==> [2, 1]
// shape(expand_dims(t, -1)) ==> [2, 1]
//
// # 't2' is a tensor of shape [2, 3, 5]
// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
// ```
//
// This operation requires that:
//
// `-1-input.dims() <= dim <= input.dims()`
//
// This operation is related to `squeeze()`, which removes dimensions of
// size 1.
//
// Arguments:
// * dim: 0-D (scalar). Specifies the dimension index at which to
// expand the shape of `input`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Contains the same data as `input`, but its shape has an additional
// dimension of size 1 added.
@Namespace("tensorflow::ops") public static native Node ExpandDims(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut dim, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ExpandDims(Node input, Node dim, @Const @ByRef GraphDefBuilder.Options opts);

// Creates a tensor filled with a scalar value.
//
// This operation creates a tensor of shape `dims` and fills it with `value`.
//
// For example:
//
// ```prettyprint
// # output tensor shape needs to be [2, 3]
// # so 'dims' is [2, 3]
// fill(dims, 9) ==> [[9, 9, 9]
//                    [9, 9, 9]]
// ```
//
// Arguments:
// * dims: 1-D. Represents the shape of the output tensor.
// * value: 0-D (scalar). Value to fill the returned tensor.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Fill(@ByVal NodeBuilder.NodeOut dims, @ByVal NodeBuilder.NodeOut value, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Fill(Node dims, Node value, @Const @ByRef GraphDefBuilder.Options opts);

// Gather slices from `params` according to `indices`.
//
// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
//
//     # Scalar indices
//     output[:, ..., :] = params[indices, :, ... :]
//
//     # Vector indices
//     output[i, :, ..., :] = params[indices[i], :, ... :]
//
//     # Higher rank indices
//     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
//
// If `indices` is a permutation and `len(indices) == params.shape[0]` then
// this operation will permute `params` accordingly.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/Gather.png" alt>
// </div>
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Gather(@ByVal NodeBuilder.NodeOut params, @ByVal NodeBuilder.NodeOut indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Gather(Node params, Node indices, @Const @ByRef GraphDefBuilder.Options opts);

// Return a tensor with the same shape and contents as the input tensor or value.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Identity(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Identity(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the inverse permutation of a tensor.
//
// This operation computes the inverse of an index permutation. It takes a 1-D
// integer tensor `x`, which represents the indices of a zero-based array, and
// swaps each value with its index position. In other words, for an ouput tensor
// `y` and an input tensor `x`, this operation computes the following:
//
// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
//
// The values must include 0. There can be no duplicate values or negative values.
//
// For example:
//
// ```prettyprint
// # tensor `x` is [3, 4, 0, 2, 1]
// invert_permutation(x) ==> [2, 4, 3, 0, 1]
// ```
//
// Arguments:
// * x: 1-D.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 1-D.
@Namespace("tensorflow::ops") public static native Node InvertPermutation(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node InvertPermutation(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the difference between two lists of numbers or strings.
//
// Given a list `x` and a list `y`, this operation returns a list `out` that
// represents all values that are in `x` but not in `y`. The returned list `out`
// is sorted in the same order that the numbers appear in `x` (duplicates are
// preserved). This operation also returns a list `idx` that represents the
// position of each `out` element in `x`. In other words:
//
// `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
//
// For example, given this input:
//
// ```prettyprint
// x = [1, 2, 3, 4, 5, 6]
// y = [1, 3, 5]
// ```
//
// This operation would return:
//
// ```prettyprint
// out ==> [2, 4, 6]
// idx ==> [1, 3, 5]
// ```
//
// Arguments:
// * x: 1-D. Values to keep.
// * y: 1-D. Values to remove.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * out: 1-D. Values present in `x` but not in `y`.
// * idx: 1-D. Positions of `x` values preserved in `out`.
@Namespace("tensorflow::ops") public static native Node ListDiff(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ListDiff(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
//
// Packs the `N` tensors in `values` into a tensor with rank one higher than each
// tensor in `values` and shape `[N] + values[0].shape`. The output satisfies
// `output[i, ...] = values[i][...]`.
//
// This is the opposite of `unpack`.
//
// Arguments:
// * values: Must be of same shape and type.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The packed tensor.
@Namespace("tensorflow::ops") public static native Node Pack(@ByVal NodeOutVector values, @Const @ByRef GraphDefBuilder.Options opts);

// Pads a tensor with zeros.
//
// This operation pads a `input` with zeros according to the `paddings` you
// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many zeros to add before the contents of `input` in that dimension, and
// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
// in that dimension.
//
// The padded size of each dimension D of the output is:
//
// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
//
// For example:
//
// ```prettyprint
// # 't' is [[1, 1], [2, 2]]
// # 'paddings' is [[1, 1]], [2, 2]]
// # rank of 't' is 2
// pad(t, paddings) ==> [[0, 0, 0, 0, 0]
//                       [0, 0, 0, 0, 0]
//                       [0, 1, 1, 0, 0]
//                      [[0, 2, 2, 0, 0]
//                       [0, 0, 0, 0, 0]]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Pad(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut paddings, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Pad(Node input, Node paddings, @Const @ByRef GraphDefBuilder.Options opts);

// A placeholder op for a value that will be fed into the computation.
//
// N.B. This operation will fail with an error if it is executed. It is
// intended as a way to represent a value that will always be fed, and to
// provide attrs that enable the fed value to be checked at runtime.
//
// Arguments:
// * dtype: The type of elements in the tensor.
// * shape: (Optional) The shape of the tensor. If the shape has 0 dimensions, the
// shape is unconstrained.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A placeholder tensor that must be replaced using the feed mechanism.
@Namespace("tensorflow::ops") public static native Node Placeholder(@Cast("tensorflow::DataType") int dtype, @ByVal TensorShape shape, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the rank of a tensor.
//
// This operation returns an integer representing the rank of `input`.
//
// For example:
//
// ```prettyprint
// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
// # shape of tensor 't' is [2, 2, 3]
// rank(t) ==> 3
// ```
//
// **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
// of a tensor is the number of indices required to uniquely select each element
// of the tensor. Rank is also known as "order", "degree", or "ndims."
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Rank(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Rank(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Return the same ref tensor as the input ref tensor.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node RefIdentity(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node RefIdentity(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Reshapes a tensor.
//
// Given `tensor`, this operation returns a tensor that has the same values
// as `tensor` with shape `shape`.
//
// If one component of `shape` is the special value -1, the size of that dimension
// is computed so that the total size remains constant.  In particular, a `shape`
// of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
//
// If `shape` is 1-D or higher, then the operation returns a tensor with shape
// `shape` filled with the values of `tensor`. In this case, the number of elements
// implied by `shape` must be the same as the number of elements in `tensor`.
//
// For example:
//
// ```prettyprint
// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
// # tensor 't' has shape [9]
// reshape(t, [3, 3]) ==> [[1, 2, 3]
//                         [4, 5, 6]
//                         [7, 8, 9]]
//
// # tensor 't' is [[[1, 1], [2, 2]]
// #                [[3, 3], [4, 4]]]
// # tensor 't' has shape [2, 2, 2]
// reshape(t, [2, 4]) ==> [[1, 1, 2, 2]
//                         [3, 3, 4, 4]]
//
// # tensor 't' is [[[1, 1, 1],
// #                 [2, 2, 2]],
// #                [[3, 3, 3],
// #                 [4, 4, 4]],
// #                [[5, 5, 5],
// #                 [6, 6, 6]]]
// # tensor 't' has shape [3, 2, 3]
// # pass '[-1]' to flatten 't'
// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
// # -1 can also be used with higher dimensional shapes
// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
//                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
//
// # tensor 't' is [7]
// # shape `[]` reshapes to a scalar
// reshape(t, []) ==> 7
// ```
//
// Arguments:
// * shape: Defines the shape of the output tensor.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Reshape(@ByVal NodeBuilder.NodeOut tensor, @ByVal NodeBuilder.NodeOut shape, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Reshape(Node tensor, Node shape, @Const @ByRef GraphDefBuilder.Options opts);

// Reverses specific dimensions of a tensor.
//
// Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
// of `tensor`, this operation reverses each dimension i of `tensor` where
// `dims[i]` is `True`.
//
// `tensor` can have up to 8 dimensions. The number of dimensions
// of `tensor` must equal the number of elements in `dims`. In other words:
//
// `rank(tensor) = size(dims)`
//
// For example:
//
// ```prettyprint
// # tensor 't' is [[[[ 0,  1,  2,  3],
// #                  [ 4,  5,  6,  7],
// #                  [ 8,  9, 10, 11]],
// #                 [[12, 13, 14, 15],
// #                  [16, 17, 18, 19],
// #                  [20, 21, 22, 23]]]]
// # tensor 't' shape is [1, 2, 3, 4]
//
// # 'dims' is [False, False, False, True]
// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
//                         [ 7,  6,  5,  4],
//                         [ 11, 10, 9, 8]],
//                        [[15, 14, 13, 12],
//                         [19, 18, 17, 16],
//                         [23, 22, 21, 20]]]]
//
// # 'dims' is [False, True, False, False]
// reverse(t, dims) ==> [[[[12, 13, 14, 15],
//                         [16, 17, 18, 19],
//                         [20, 21, 22, 23]
//                        [[ 0,  1,  2,  3],
//                         [ 4,  5,  6,  7],
//                         [ 8,  9, 10, 11]]]]
//
// # 'dims' is [False, False, True, False]
// reverse(t, dims) ==> [[[[8, 9, 10, 11],
//                         [4, 5, 6, 7],
//                         [0, 1, 2, 3]]
//                        [[20, 21, 22, 23],
//                         [16, 17, 18, 19],
//                         [12, 13, 14, 15]]]]
// ```
//
// Arguments:
// * tensor: Up to 8-D.
// * dims: 1-D. The dimensions to reverse.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same shape as `tensor`.
@Namespace("tensorflow::ops") public static native Node Reverse(@ByVal NodeBuilder.NodeOut tensor, @ByVal NodeBuilder.NodeOut dims, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Reverse(Node tensor, Node dims, @Const @ByRef GraphDefBuilder.Options opts);

// Reverses variable length slices.
//
// This op first slices `input` along the dimension `batch_dim`, and for each
// slice `i`, reverses the first `seq_lengths[i]` elements along
// the dimension `seq_dim`.
//
// The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
// and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
//
// The output slice `i` along dimension `batch_dim` is then given by input
// slice `i`, with the first `seq_lengths[i]` slices along dimension
// `seq_dim` reversed.
//
// For example:
//
// ```prettyprint
// # Given this:
// batch_dim = 0
// seq_dim = 1
// input.dims = (4, 8, ...)
// seq_lengths = [7, 2, 3, 5]
//
// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
// output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
// output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
// output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
// output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
//
// # while entries past seq_lens are copied through:
// output[0, 7:, :, ...] = input[0, 7:, :, ...]
// output[1, 2:, :, ...] = input[1, 2:, :, ...]
// output[2, 3:, :, ...] = input[2, 3:, :, ...]
// output[3, 2:, :, ...] = input[3, 2:, :, ...]
// ```
//
// In contrast, if:
// ```prettyprint
// # Given this:
// batch_dim = 2
// seq_dim = 0
// input.dims = (8, ?, 4, ...)
// seq_lengths = [7, 2, 3, 5]
//
// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
// output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
// output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
// output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
// output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
//
// # while entries past seq_lens are copied through:
// output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
// output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
// output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
// output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
// ```
//
// Arguments:
// * input: The input to reverse.
// * seq_lengths: 1-D with length `input.dims(0)` and
// `max(seq_lengths) < input.dims(seq_dim)`
// * seq_dim: The dimension which is partially reversed.
// * opts:
//   .WithAttr("batch_dim", int64): Defaults to 0.
//     The dimension along which reversal is performed.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The partially reversed input. It has the same shape as `input`.
@Namespace("tensorflow::ops") public static native Node ReverseSequence(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut seq_lengths, @Cast("tensorflow::int64") long seq_dim, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReverseSequence(Node input, Node seq_lengths, @Cast("tensorflow::int64") long seq_dim, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the shape of a tensor.
//
// This operation returns a 1-D integer tensor representing the shape of `input`.
//
// For example:
//
// ```prettyprint
// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
// shape(t) ==> [2, 2, 3]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Shape(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Shape(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the size of a tensor.
//
// This operation returns an integer representing the number of elements in
// `input`.
//
// For example:
//
// ```prettyprint
// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
// size(t) ==> 12
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Size(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Size(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Return a slice from 'input'.
//
// The output tensor is a tensor with dimensions described by 'size'
// whose values are extracted from 'input' starting at the offsets in
// 'begin'.
//
// *Requirements*:
//   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
//
// Arguments:
// * begin: begin[i] specifies the offset into the 'i'th dimension of
// 'input' to slice from.
// * size: size[i] specifies the number of elements of the 'i'th dimension
// of 'input' to slice. If size[i] is -1, all remaining elements in dimension
// i are included in the slice (i.e. this is equivalent to setting
// size[i] = input.dim_size(i) - begin[i]).
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Slice(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut begin, @ByVal NodeBuilder.NodeOut size, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Slice(Node input, Node begin, Node size, @Const @ByRef GraphDefBuilder.Options opts);

// Splits a tensor into `num_split` tensors along one dimension.
//
// Arguments:
// * split_dim: 0-D.  The dimension along which to split.  Must be in the range
// `[0, rank(value))`.
// * value: The tensor to split.
// * num_split: The number of ways to split.  Must evenly divide
// `value.shape[split_dim]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// They are identically shaped tensors, whose shape matches that of `value`
// except along `split_dim`, where their sizes are
// `values.shape[split_dim] / num_split`.
@Namespace("tensorflow::ops") public static native Node Split(@ByVal NodeBuilder.NodeOut split_dim, @ByVal NodeBuilder.NodeOut value, @Cast("tensorflow::int64") long num_split, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Split(Node split_dim, Node value, @Cast("tensorflow::int64") long num_split, @Const @ByRef GraphDefBuilder.Options opts);

// Removes dimensions of size 1 from the shape of a tensor.
//
// Given a tensor `input`, this operation returns a tensor of the same type with
// all dimensions of size 1 removed. If you don't want to remove all size 1
// dimensions, you can remove specific size 1 dimensions by specifying
// `squeeze_dims`.
//
// For example:
//
// ```prettyprint
// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
// shape(squeeze(t)) ==> [2, 3]
// ```
//
// Or, to remove specific size 1 dimensions:
//
// ```prettyprint
// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
// ```
//
// Arguments:
// * input: The `input` to squeeze.
// * opts:
//   .WithAttr("squeeze_dims", gtl::ArraySlice<int>): Defaults to [].
//     If specified, only squeezes the dimensions listed. The dimension
// index starts at 0. It is an error to squeeze a dimension that is not 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Contains the same data as `input`, but has one or more dimensions of
// size 1 removed.
@Namespace("tensorflow::ops") public static native Node Squeeze(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Squeeze(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Stops gradient computation.
//
// When executed in a graph, this op outputs its input tensor as-is.
//
// When building ops to compute gradients, this op prevents the contribution of
// its inputs to be taken into account.  Normally, the gradient generator adds ops
// to a graph to compute the derivatives of a specified 'loss' by recursively
// finding out inputs that contributed to its computation.  If you insert this op
// in the graph it inputs are masked from the gradient generator.  They are not
// taken into account for computing gradients.
//
// This is useful any time you want to compute a value with TensorFlow but need
// to pretend that the value was a constant. Some examples include:
//
// *  The *EM* algorithm where the *M-step* should not involve backpropagation
//    through the output of the *E-step*.
// *  Contrastive divergence training of Boltzmann machines where, when
//    differentiating the energy function, the training must not backpropagate
//    through the graph that generated the samples from the model.
// *  Adversarial training, where no backprop should happen through the adversarial
//    example generation process.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node StopGradient(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node StopGradient(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Constructs a tensor by tiling a given tensor.
//
// This operation creates a new tensor by replicating `input` `multiples` times.
// The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
// and the values of `input` are replicated `multiples[i]` times along the 'i'th
// dimension. For example, tiling `[a b c d]` by `[2]` produces
// `[a b c d a b c d]`.
//
// Arguments:
// * input: 1-D or higher.
// * multiples: 1-D. Length must be the same as the number of dimensions in `input`
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Tile(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut multiples, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Tile(Node input, Node multiples, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the gradient of `Tile`.
//
// Since `Tile` takes an input and repeats the input `multiples` times
// along each dimension, `TileGrad` takes in `multiples` and aggregates
// each repeated tile of `input` into `output`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node TileGrad(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut multiples, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node TileGrad(Node input, Node multiples, @Const @ByRef GraphDefBuilder.Options opts);

// Shuffle dimensions of x according to a permutation.
//
// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
//   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Transpose(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut perm, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Transpose(Node x, Node perm, @Const @ByRef GraphDefBuilder.Options opts);

// Finds unique elements in a 1-D tensor.
//
// This operation returns a tensor `y` containing all of the unique elements of `x`
// sorted in the same order that they occur in `x`. This operation also returns a
// tensor `idx` the same size as `x` that contains the index of each value of `x`
// in the unique output `y`. In other words:
//
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
//
// For example:
//
// ```prettyprint
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx = unique(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// ```
//
// Arguments:
// * x: 1-D.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * y: 1-D.
// * idx: 1-D.
@Namespace("tensorflow::ops") public static native Node Unique(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Unique(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Unpacks the outer dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.
//
// Unpacks `num` tensors from `value` by chipping it along the first dimension.
// The i'th tensor in `output` is the slice `value[i, ...]`. Each tensor in
// `output` has shape `value.shape[1:]`.
//
// This is the opposite of `pack`.
//
// Arguments:
// * value: 1-D or higher, with first dimension `num`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The list of tensors unpacked from `value`.
@Namespace("tensorflow::ops") public static native Node Unpack(@ByVal NodeBuilder.NodeOut value, @Cast("tensorflow::int64") long num, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Unpack(Node value, @Cast("tensorflow::int64") long num, @Const @ByRef GraphDefBuilder.Options opts);

// Returns locations of true values in a boolean tensor.
//
// This operation returns the coordinates of true elements in `input`. The
// coordinates are returned in a 2-D tensor where the first dimension (rows)
// represents the number of true elements, and the second dimension (columns)
// represents the coordinates of the true elements. Keep in mind, the shape of
// the output tensor can vary depending on how many true values there are in
// `input`. Indices are output in row-major order.
//
// For example:
//
// ```prettyprint
// # 'input' tensor is [[True, False]
// #                    [True, False]]
// # 'input' has two true values, so output has two coordinates.
// # 'input' has rank of 2, so coordinates have two indices.
// where(input) ==> [[0, 0],
//                   [1, 0]]
//
// # `input` tensor is [[[True, False]
// #                     [True, False]]
// #                    [[False, True]
// #                     [False, True]]
// #                    [[False, False]
// #                     [False, True]]]
// # 'input' has 5 true values, so output has 5 coordinates.
// # 'input' has rank of 3, so coordinates have three indices.
// where(input) ==> [[0, 0, 0],
//                   [0, 1, 0],
//                   [1, 0, 1],
//                   [1, 1, 1],
//                   [2, 1, 1]]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Where(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Where(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Returns a tensor of zeros with the same shape and type as x.
//
// Arguments:
// * x: a tensor of type T.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// a tensor of the same shape and type as x but filled with zeros.
@Namespace("tensorflow::ops") public static native Node ZerosLike(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ZerosLike(Node x, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_ARRAY_OPS_H_


// Parsed from tensorflow/cc/ops/attention_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_ATTENTION_OPS_H_
// #define TENSORFLOW_CC_OPS_ATTENTION_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Extracts a glimpse from the input tensor.
//
// Returns a set of windows called glimpses extracted at location `offsets`
// from the input tensor. If the windows only partially overlaps the inputs, the
// non overlapping areas will be filled with random noise.
//
// The result is a 4-D tensor of shape `[batch_size, glimpse_height,
// glimpse_width, channels]`. The channels and batch dimensions are the same as that
// of the input tensor. The height and width of the output windows are
// specified in the `size` parameter.
//
// The argument `normalized` and `centered` controls how the windows are built:
// * If the coordinates are normalized but not centered, 0.0 and 1.0
//   correspond to the minimum and maximum of each height and width dimension.
// * If the coordinates are both normalized and centered, they range from -1.0 to
//   1.0. The coordinates (-1.0, -1.0) correspond to the upper left corner, the
//   lower right corner is located at  (1.0, 1.0) and the center is at (0, 0).
// * If the coordinates are not normalized they are interpreted as numbers of pixels.
//
// Arguments:
// * input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
// * size: A 1-D tensor of 2 elements containing the size of the glimpses to extract.
// The glimpse height must be specified first, following by the glimpse width.
// * offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing the x, y
// locations of the center of each window.
// * opts:
//   .WithAttr("centered", bool): Defaults to true.
//     indicates if the offset coordinates are centered relative to
// the image, in which case the (0, 0) offset is relative to the center of the
// input images. If false, the (0,0) offset corresponds to the upper left corner
// of the input images.
//   .WithAttr("normalized", bool): Defaults to true.
//     indicates if the offset coordinates are normalized.
//   .WithAttr("uniform_noise", bool): Defaults to true.
//     indicates if the noise should be generated using a
// uniform distribution or a gaussian distribution.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor representing the glimpses `[batch_size, glimpse_height,
// glimpse_width, channels]`.
@Namespace("tensorflow::ops") public static native Node ExtractGlimpse(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut size, @ByVal NodeBuilder.NodeOut offsets, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ExtractGlimpse(Node input, Node size, Node offsets, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_ATTENTION_OPS_H_


// Parsed from tensorflow/cc/ops/data_flow_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_
// #define TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Partitions `data` into `num_partitions` tensors using indices from `partitions`.
//
// For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
// becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
// are placed in `outputs[i]` in lexicographic order of `js`, and the first
// dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
// In detail,
//
//     outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
//
//     outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
//
// `data.shape` must start with `partitions.shape`.
//
// For example:
//
//     # Scalar partitions
//     partitions = 1
//     num_partitions = 2
//     data = [10, 20]
//     outputs[0] = []  # Empty with shape [0, 2]
//     outputs[1] = [[10, 20]]
//
//     # Vector partitions
//     partitions = [0, 0, 1, 1, 0]
//     num_partitions = 2
//     data = [10, 20, 30, 40, 50]
//     outputs[0] = [10, 20, 50]
//     outputs[1] = [30, 40]
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/DynamicPartition.png" alt>
// </div>
//
// Arguments:
// * partitions: Any shape.  Indices in the range `[0, num_partitions)`.
// * num_partitions: The number of partitions to output.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node DynamicPartition(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut partitions, @Cast("tensorflow::int64") long num_partitions,
                       @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node DynamicPartition(Node data, Node partitions, @Cast("tensorflow::int64") long num_partitions,
                       @Const @ByRef GraphDefBuilder.Options opts);

// Interleave the values from the `data` tensors into a single tensor.
//
// Builds a merged tensor such that
//
//     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
//
// For example, if each `indices[m]` is scalar or vector, we have
//
//     # Scalar indices
//     merged[indices[m], ...] = data[m][...]
//
//     # Vector indices
//     merged[indices[m][i], ...] = data[m][i, ...]
//
// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
// `constant`, the output shape is
//
//     merged.shape = [max(indices)] + constant
//
// Values are merged in order, so if an index appears in both `indices[m][i]` and
// `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
// merged result.
//
// For example:
//
//     indices[0] = 6
//     indices[1] = [4, 1]
//     indices[2] = [[5, 2], [0, 3]]
//     data[0] = [61, 62]
//     data[1] = [[41, 42], [11, 12]]
//     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
//     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
//               [51, 52], [61, 62]]
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/DynamicStitch.png" alt>
// </div>
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node DynamicStitch(@ByVal NodeOutVector indices, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);

// A queue that produces elements in first-in first-out order.
//
// Arguments:
// * component_types: The type of each component in a value.
// * opts:
//   .WithAttr("shapes", gtl::ArraySlice<TensorShape>): Defaults to [].
//     The shape of each component in a value. The length of this attr must
// be either 0 or the same as the length of component_types. If the length of
// this attr is 0, the shapes of queue elements are not constrained, and
// only one element may be dequeued at a time.
//   .WithAttr("capacity", int64): Defaults to -1.
//     The upper bound on the number of elements in this queue.
// Negative numbers mean no limit.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this queue is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this queue will be shared under the given name
// across multiple sessions.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to the queue.
@Namespace("tensorflow::ops") public static native Node FIFOQueue(@ByVal DataTypeVector component_types, @Const @ByRef GraphDefBuilder.Options opts);

// Creates a non-initialized hash table.
//
// This op creates a hash table, specifying the type of its keys and values.
// Before using the table you will have to initialize it.  After initialization the
// table will be immutable.
//
// Arguments:
// * key_dtype: Type of the table keys.
// * value_dtype: Type of the table values.
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this table is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this table is shared under the given name across
// multiple sessions.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Handle to a table.
@Namespace("tensorflow::ops") public static native Node HashTable(@Cast("tensorflow::DataType") int key_dtype, @Cast("tensorflow::DataType") int value_dtype, @Const @ByRef GraphDefBuilder.Options opts);

// Table initializer that takes two tensors for keys and values respectively.
//
// Arguments:
// * table_handle: Handle to a table which will be initialized.
// * keys: Keys of type Tkey.
// * values: Values of type Tval. Same shape as `keys`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node InitializeTable(@ByVal NodeBuilder.NodeOut table_handle, @ByVal NodeBuilder.NodeOut keys, @ByVal NodeBuilder.NodeOut values, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node InitializeTable(Node table_handle, Node keys, Node values, @Const @ByRef GraphDefBuilder.Options opts);

// Looks up keys in a table, outputs the corresponding values.
//
// The tensor `keys` must of the same type as the keys of the table.
// The output `values` is of the type of the table values.
//
// The scalar `default_value` is the value output for keys not present in the
// table. It must also be of the same type as the table values.
//
// Arguments:
// * table_handle: Handle to the table.
// * keys: Any shape.  Keys to look up.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same shape as `keys`.  Values found in the table, or `default_values`
// for missing keys.
@Namespace("tensorflow::ops") public static native Node LookupTableFind(@ByVal NodeBuilder.NodeOut table_handle, @ByVal NodeBuilder.NodeOut keys, @ByVal NodeBuilder.NodeOut default_value, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LookupTableFind(Node table_handle, Node keys, Node default_value, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the number of elements in the given table.
//
// Arguments:
// * table_handle: Handle to the table.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Scalar that contains number of elements in the table.
@Namespace("tensorflow::ops") public static native Node LookupTableSize(@ByVal NodeBuilder.NodeOut table_handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LookupTableSize(Node table_handle, @Const @ByRef GraphDefBuilder.Options opts);

// Closes the given queue.
//
// This operation signals that no more elements will be enqueued in the
// given queue. Subsequent Enqueue(Many) operations will fail.
// Subsequent Dequeue(Many) operations will continue to succeed if
// sufficient elements remain in the queue. Subsequent Dequeue(Many)
// operations that would block will fail immediately.
//
// Arguments:
// * handle: The handle to a queue.
// * opts:
//   .WithAttr("cancel_pending_enqueues", bool): Defaults to false.
//     If true, all pending enqueue requests that are
// blocked on the given queue will be cancelled.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node QueueClose(@ByVal NodeBuilder.NodeOut handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node QueueClose(Node handle, @Const @ByRef GraphDefBuilder.Options opts);

// Dequeues a tuple of one or more tensors from the given queue.
//
// This operation has k outputs, where k is the number of components
// in the tuples stored in the given queue, and output i is the ith
// component of the dequeued tuple.
//
// N.B. If the queue is empty, this operation will block until an element
// has been dequeued (or 'timeout_ms' elapses, if specified).
//
// Arguments:
// * handle: The handle to a queue.
// * component_types: The type of each component in a tuple.
// * opts:
//   .WithAttr("timeout_ms", int64): Defaults to -1.
//     If the queue is empty, this operation will block for up to
// timeout_ms milliseconds.
// Note: This option is not supported yet.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// One or more tensors that were dequeued as a tuple.
@Namespace("tensorflow::ops") public static native Node QueueDequeue(@ByVal NodeBuilder.NodeOut handle, @ByVal DataTypeVector component_types, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node QueueDequeue(Node handle, @ByVal DataTypeVector component_types, @Const @ByRef GraphDefBuilder.Options opts);

// Dequeues n tuples of one or more tensors from the given queue.
//
// This operation concatenates queue-element component tensors along the
// 0th dimension to make a single component tensor.  All of the components
// in the dequeued tuple will have size n in the 0th dimension.
//
// This operation has k outputs, where k is the number of components in
// the tuples stored in the given queue, and output i is the ith
// component of the dequeued tuple.
//
// N.B. If the queue is empty, this operation will block until n elements
// have been dequeued (or 'timeout_ms' elapses, if specified).
//
// Arguments:
// * handle: The handle to a queue.
// * n: The number of tuples to dequeue.
// * component_types: The type of each component in a tuple.
// * opts:
//   .WithAttr("timeout_ms", int64): Defaults to -1.
//     If the queue has fewer than n elements, this operation
// will block for up to timeout_ms milliseconds.
// Note: This option is not supported yet.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// One or more tensors that were dequeued as a tuple.
@Namespace("tensorflow::ops") public static native Node QueueDequeueMany(@ByVal NodeBuilder.NodeOut handle, @ByVal NodeBuilder.NodeOut n, @ByVal DataTypeVector component_types, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node QueueDequeueMany(Node handle, Node n, @ByVal DataTypeVector component_types, @Const @ByRef GraphDefBuilder.Options opts);

// Enqueues a tuple of one or more tensors in the given queue.
//
// The components input has k elements, which correspond to the components of
// tuples stored in the given queue.
//
// N.B. If the queue is full, this operation will block until the given
// element has been enqueued (or 'timeout_ms' elapses, if specified).
//
// Arguments:
// * handle: The handle to a queue.
// * components: One or more tensors from which the enqueued tensors should be taken.
// * opts:
//   .WithAttr("timeout_ms", int64): Defaults to -1.
//     If the queue is full, this operation will block for up to
// timeout_ms milliseconds.
// Note: This option is not supported yet.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node QueueEnqueue(@ByVal NodeBuilder.NodeOut handle, @ByVal NodeOutVector components, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node QueueEnqueue(Node handle, @ByVal NodeOutVector components, @Const @ByRef GraphDefBuilder.Options opts);

// Enqueues zero or more tuples of one or more tensors in the given queue.
//
// This operation slices each component tensor along the 0th dimension to
// make multiple queue elements. All of the tuple components must have the
// same size in the 0th dimension.
//
// The components input has k elements, which correspond to the components of
// tuples stored in the given queue.
//
// N.B. If the queue is full, this operation will block until the given
// elements have been enqueued (or 'timeout_ms' elapses, if specified).
//
// Arguments:
// * handle: The handle to a queue.
// * components: One or more tensors from which the enqueued tensors should
// be taken.
// * opts:
//   .WithAttr("timeout_ms", int64): Defaults to -1.
//     If the queue is too full, this operation will block for up
// to timeout_ms milliseconds.
// Note: This option is not supported yet.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node QueueEnqueueMany(@ByVal NodeBuilder.NodeOut handle, @ByVal NodeOutVector components,
                       @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node QueueEnqueueMany(Node handle, @ByVal NodeOutVector components,
                       @Const @ByRef GraphDefBuilder.Options opts);

// Computes the number of elements in the given queue.
//
// Arguments:
// * handle: The handle to a queue.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The number of elements in the given queue.
@Namespace("tensorflow::ops") public static native Node QueueSize(@ByVal NodeBuilder.NodeOut handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node QueueSize(Node handle, @Const @ByRef GraphDefBuilder.Options opts);

// A queue that randomizes the order of elements.
//
// Arguments:
// * component_types: The type of each component in a value.
// * opts:
//   .WithAttr("shapes", gtl::ArraySlice<TensorShape>): Defaults to [].
//     The shape of each component in a value. The length of this attr must
// be either 0 or the same as the length of component_types. If the length of
// this attr is 0, the shapes of queue elements are not constrained, and
// only one element may be dequeued at a time.
//   .WithAttr("capacity", int64): Defaults to -1.
//     The upper bound on the number of elements in this queue.
// Negative numbers mean no limit.
//   .WithAttr("min_after_dequeue", int64): Defaults to 0.
//     Dequeue will block unless there would be this
// many elements after the dequeue or the queue is closed. This
// ensures a minimum level of mixing of elements.
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 is set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, a random seed is used.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this queue is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this queue will be shared under the given name
// across multiple sessions.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to the queue.
@Namespace("tensorflow::ops") public static native Node RandomShuffleQueue(@ByVal DataTypeVector component_types, @Const @ByRef GraphDefBuilder.Options opts);

// A stack that produces elements in first-in last-out order.
//
// Arguments:
// * elem_type: The type of the elements on the stack.
// * opts:
//   .WithAttr("stack_name", StringPiece): Defaults to "".
//     Overrides the name used for the temporary stack resource. Default
// value is the name of the 'Stack' op (which is guaranteed unique).
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to the stack.
@Namespace("tensorflow::ops") public static native Node Stack(@Cast("tensorflow::DataType") int elem_type, @Const @ByRef GraphDefBuilder.Options opts);

// Delete the stack from its resource container.
//
// Arguments:
// * handle: The handle to a stack.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node StackClose(@ByVal NodeBuilder.NodeOut handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node StackClose(Node handle, @Const @ByRef GraphDefBuilder.Options opts);

// Pop the element at the top of the stack.
//
// Arguments:
// * handle: The handle to a stack.
// * elem_type: The type of the elem that is popped.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The tensor that is popped from the top of the stack.
@Namespace("tensorflow::ops") public static native Node StackPop(@ByVal NodeBuilder.NodeOut handle, @Cast("tensorflow::DataType") int elem_type, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node StackPop(Node handle, @Cast("tensorflow::DataType") int elem_type, @Const @ByRef GraphDefBuilder.Options opts);

// Push an element onto the stack.
//
// Arguments:
// * handle: The handle to a stack.
// * elem: The tensor to be pushed onto the stack.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as the input 'elem'.
@Namespace("tensorflow::ops") public static native Node StackPush(@ByVal NodeBuilder.NodeOut handle, @ByVal NodeBuilder.NodeOut elem, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node StackPush(Node handle, Node elem, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_


// Parsed from tensorflow/cc/ops/image_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_IMAGE_OPS_H_
// #define TENSORFLOW_CC_OPS_IMAGE_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Adjust the contrast of one or more images.
//
// `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
// interpreted as `[height, width, channels]`.  The other dimensions only
// represent a collection of images, such as `[batch, height, width, channels].`
//
// Contrast is adjusted independently for each channel of each image.
//
// For each channel, the Op first computes the mean of the image pixels in the
// channel and then adjusts each component of each pixel to
// `(x - mean) * contrast_factor + mean`.
//
// These adjusted values are then clipped to fit in the `[min_value, max_value]`
// interval.
//
// `images: Images to adjust.  At least 3-D.
//
// Arguments:
// * contrast_factor: A float multiplier for adjusting contrast.
// * min_value: Minimum value for clipping the adjusted pixels.
// * max_value: Maximum value for clipping the adjusted pixels.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The constrast-adjusted image or images.
@Namespace("tensorflow::ops") public static native Node AdjustContrast(@ByVal NodeBuilder.NodeOut images, @ByVal NodeBuilder.NodeOut contrast_factor, @ByVal NodeBuilder.NodeOut min_value, @ByVal NodeBuilder.NodeOut max_value, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AdjustContrast(Node images, Node contrast_factor, Node min_value, Node max_value, @Const @ByRef GraphDefBuilder.Options opts);

// Decode a JPEG-encoded image to a uint8 tensor.
//
// The attr `channels` indicates the desired number of color channels for the
// decoded image.
//
// Accepted values are:
//
// *   0: Use the number of channels in the JPEG-encoded image.
// *   1: output a grayscale image.
// *   3: output an RGB image.
//
// If needed, the JPEG-encoded image is transformed to match the requested number
// of color channels.
//
// The attr `ratio` allows downscaling the image by an integer factor during
// decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
// downscaling the image later.
//
// Arguments:
// * contents: 0-D.  The JPEG-encoded image.
// * opts:
//   .WithAttr("channels", int64): Defaults to 0.
//     Number of color channels for the decoded image.
//   .WithAttr("ratio", int64): Defaults to 1.
//     Downscaling ratio.
//   .WithAttr("fancy_upscaling", bool): Defaults to true.
//     If true use a slower but nicer upscaling of the
// chroma planes (yuv420/422 only).
//   .WithAttr("try_recover_truncated", bool): Defaults to false.
//     If true try to recover an image from truncated input.
//   .WithAttr("acceptable_fraction", float): Defaults to 1.
//     The minimum required fraction of lines before a truncated
// input is accepted.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 3-D with shape `[height, width, channels]`..
@Namespace("tensorflow::ops") public static native Node DecodeJpeg(@ByVal NodeBuilder.NodeOut contents, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node DecodeJpeg(Node contents, @Const @ByRef GraphDefBuilder.Options opts);

// Decode a PNG-encoded image to a uint8 tensor.
//
// The attr `channels` indicates the desired number of color channels for the
// decoded image.
//
// Accepted values are:
//
// *   0: Use the number of channels in the PNG-encoded image.
// *   1: output a grayscale image.
// *   3: output an RGB image.
// *   4: output an RGBA image.
//
// If needed, the PNG-encoded image is transformed to match the requested number
// of color channels.
//
// Arguments:
// * contents: 0-D.  The PNG-encoded image.
// * opts:
//   .WithAttr("channels", int64): Defaults to 0.
//     Number of color channels for the decoded image.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 3-D with shape `[height, width, channels]`.
@Namespace("tensorflow::ops") public static native Node DecodePng(@ByVal NodeBuilder.NodeOut contents, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node DecodePng(Node contents, @Const @ByRef GraphDefBuilder.Options opts);

// JPEG-encode an image.
//
// `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
//
// The attr `format` can be used to override the color format of the encoded
// output.  Values can be:
//
// *   `''`: Use a default format based on the number of channels in the image.
// *   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
//     of `image` must be 1.
// *   `rgb`: Output an RGB JPEG image. The `channels` dimension
//     of `image` must be 3.
//
// If `format` is not specified or is the empty string, a default format is picked
// in function of the number of channels in `image`:
//
// *   1: Output a grayscale image.
// *   3: Output an RGB image.
//
// Arguments:
// * image: 3-D with shape `[height, width, channels]`.
// * opts:
//   .WithAttr("format", StringPiece): Defaults to "".
//     Per pixel image format.
//   .WithAttr("quality", int64): Defaults to 95.
//     Quality of the compression from 0 to 100 (higher is better and slower).
//   .WithAttr("progressive", bool): Defaults to false.
//     If True, create a JPEG that loads progressively (coarse to fine).
//   .WithAttr("optimize_size", bool): Defaults to false.
//     If True, spend CPU/RAM to reduce size with no quality change.
//   .WithAttr("chroma_downsampling", bool): Defaults to true.
//     See http://en.wikipedia.org/wiki/Chroma_subsampling.
//   .WithAttr("density_unit", StringPiece): Defaults to "in".
//     Unit used to specify `x_density` and `y_density`:
// pixels per inch (`'in'`) or centimeter (`'cm'`).
//   .WithAttr("x_density", int64): Defaults to 300.
//     Horizontal pixels per density unit.
//   .WithAttr("y_density", int64): Defaults to 300.
//     Vertical pixels per density unit.
//   .WithAttr("xmp_metadata", StringPiece): Defaults to "".
//     If not empty, embed this XMP metadata in the image header.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 0-D. JPEG-encoded image.
@Namespace("tensorflow::ops") public static native Node EncodeJpeg(@ByVal NodeBuilder.NodeOut image, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node EncodeJpeg(Node image, @Const @ByRef GraphDefBuilder.Options opts);

// PNG-encode an image.
//
// `image` is a 3-D uint8 Tensor of shape `[height, width, channels]` where
// `channels` is:
//
// *   1: for grayscale.
// *   3: for RGB.
// *   4: for RGBA.
//
// The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
// default or a value from 0 to 9.  9 is the highest compression level, generating
// the smallest output, but is slower.
//
// Arguments:
// * image: 3-D with shape `[height, width, channels]`.
// * opts:
//   .WithAttr("compression", int64): Defaults to -1.
//     Compression level.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 0-D. PNG-encoded image.
@Namespace("tensorflow::ops") public static native Node EncodePng(@ByVal NodeBuilder.NodeOut image, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node EncodePng(Node image, @Const @ByRef GraphDefBuilder.Options opts);

// Randomly crop `image`.
//
// `size` is a 1-D int64 tensor with 2 elements representing the crop height and
// width.  The values must be non negative.
//
// This Op picks a random location in `image` and crops a `height` by `width`
// rectangle from that location.  The random location is picked so the cropped
// area will fit inside the original image.
//
// Arguments:
// * image: 3-D of shape `[height, width, channels]`.
// * size: 1-D of length 2 containing: `crop_height`, `crop_width`..
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 3-D of shape `[crop_height, crop_width, channels].`
@Namespace("tensorflow::ops") public static native Node RandomCrop(@ByVal NodeBuilder.NodeOut image, @ByVal NodeBuilder.NodeOut size, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node RandomCrop(Node image, Node size, @Const @ByRef GraphDefBuilder.Options opts);

// Resize `images` to `size` using area interpolation.
//
// Input images can be of different types but output images are always float.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
// new size for the images.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[batch, new_height, new_width, channels]`.
@Namespace("tensorflow::ops") public static native Node ResizeArea(@ByVal NodeBuilder.NodeOut images, @ByVal NodeBuilder.NodeOut size, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ResizeArea(Node images, Node size, @Const @ByRef GraphDefBuilder.Options opts);

// Resize `images` to `size` using bicubic interpolation.
//
// Input images can be of different types but output images are always float.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
// new size for the images.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[batch, new_height, new_width, channels]`.
@Namespace("tensorflow::ops") public static native Node ResizeBicubic(@ByVal NodeBuilder.NodeOut images, @ByVal NodeBuilder.NodeOut size, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ResizeBicubic(Node images, Node size, @Const @ByRef GraphDefBuilder.Options opts);

// Resize `images` to `size` using bilinear interpolation.
//
// Input images can be of different types but output images are always float.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
// new size for the images.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[batch, new_height, new_width, channels]`.
@Namespace("tensorflow::ops") public static native Node ResizeBilinear(@ByVal NodeBuilder.NodeOut images, @ByVal NodeBuilder.NodeOut size, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ResizeBilinear(Node images, Node size, @Const @ByRef GraphDefBuilder.Options opts);

// Resize `images` to `size` using nearest neighbor interpolation.
//
// Input images can be of different types but output images are always float.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
// new size for the images.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[batch, new_height, new_width, channels]`.
@Namespace("tensorflow::ops") public static native Node ResizeNearestNeighbor(@ByVal NodeBuilder.NodeOut images, @ByVal NodeBuilder.NodeOut size, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ResizeNearestNeighbor(Node images, Node size, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the gradient of nearest neighbor interpolation.
//
// Arguments:
// * grads: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
// original input size.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
// with respect to the input image.
@Namespace("tensorflow::ops") public static native Node ResizeNearestNeighborGrad(@ByVal NodeBuilder.NodeOut grads, @ByVal NodeBuilder.NodeOut size, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ResizeNearestNeighborGrad(Node grads, Node size, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_IMAGE_OPS_H_


// Parsed from tensorflow/cc/ops/io_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_IO_OPS_H_
// #define TENSORFLOW_CC_OPS_IO_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// A Reader that outputs fixed-length records from a file.
//
// Arguments:
// * opts:
//   .WithAttr("header_bytes", int64): Defaults to 0.
//   .WithAttr("footer_bytes", int64): Defaults to 0.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
@Namespace("tensorflow::ops") public static native Node FixedLengthRecordReader(@Cast("tensorflow::int64") long record_bytes, @Const @ByRef GraphDefBuilder.Options opts);

// A Reader that outputs the queued work as both the key and value.
//
// To use, enqueue strings in a Queue.  ReaderRead will take the front
// work string and output (work, work).
//
// Arguments:
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
@Namespace("tensorflow::ops") public static native Node IdentityReader(@Const @ByRef GraphDefBuilder.Options opts);

// Returns the set of files matching a pattern.
//
// Note that this routine only supports wildcard characters in the
// basename portion of the pattern, not in the directory portion.
//
// Arguments:
// * pattern: A (scalar) shell wildcard pattern.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A vector of matching filenames.
@Namespace("tensorflow::ops") public static native Node MatchingFiles(@ByVal NodeBuilder.NodeOut pattern, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MatchingFiles(Node pattern, @Const @ByRef GraphDefBuilder.Options opts);

// Reads and outputs the entire contents of the input filename.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ReadFile(@ByVal NodeBuilder.NodeOut filename, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReadFile(Node filename, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the number of records this Reader has produced.
//
// This is the same as the number of ReaderRead executions that have
// succeeded.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ReaderNumRecordsProduced(@ByVal NodeBuilder.NodeOut reader_handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReaderNumRecordsProduced(Node reader_handle, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the number of work units this Reader has finished processing.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ReaderNumWorkUnitsCompleted(@ByVal NodeBuilder.NodeOut reader_handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReaderNumWorkUnitsCompleted(Node reader_handle, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the next record (key, value pair) produced by a Reader.
//
// Will dequeue from the input queue if necessary (e.g. when the
// Reader needs to start reading from a new file since it has finished
// with the previous file).
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * queue_handle: Handle to a Queue, with string work items.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * key: A scalar.
// * value: A scalar.
@Namespace("tensorflow::ops") public static native Node ReaderRead(@ByVal NodeBuilder.NodeOut reader_handle, @ByVal NodeBuilder.NodeOut queue_handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReaderRead(Node reader_handle, Node queue_handle, @Const @ByRef GraphDefBuilder.Options opts);

// Restore a Reader to its initial clean state.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ReaderReset(@ByVal NodeBuilder.NodeOut reader_handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReaderReset(Node reader_handle, @Const @ByRef GraphDefBuilder.Options opts);

// Restore a reader to a previously saved state.
//
// Not all Readers support being restored, so this can produce an
// Unimplemented error.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * state: Result of a ReaderSerializeState of a Reader with type
// matching reader_handle.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ReaderRestoreState(@ByVal NodeBuilder.NodeOut reader_handle, @ByVal NodeBuilder.NodeOut state, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReaderRestoreState(Node reader_handle, Node state, @Const @ByRef GraphDefBuilder.Options opts);

// Produce a string tensor that encodes the state of a Reader.
//
// Not all Readers support being serialized, so this can produce an
// Unimplemented error.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ReaderSerializeState(@ByVal NodeBuilder.NodeOut reader_handle, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReaderSerializeState(Node reader_handle, @Const @ByRef GraphDefBuilder.Options opts);

// Restores a tensor from checkpoint files.
//
// Reads a tensor stored in one or several files. If there are several files (for
// instance because a tensor was saved as slices), `file_pattern` may contain
// wildcard symbols (`*` and `?`) in the filename portion only, not in the
// directory portion.
//
// If a `file_pattern` matches several files, `preferred_shard` can be used to hint
// in which file the requested tensor is likely to be found. This op will first
// open the file at index `preferred_shard` in the list of matching files and try
// to restore tensors from that file.  Only if some tensors or tensor slices are
// not found in that first file, then the Op opens all the files. Setting
// `preferred_shard` to match the value passed as the `shard` input
// of a matching `Save` Op may speed up Restore.  This attribute only affects
// performance, not correctness.  The default value -1 means files are processed in
// order.
//
// See also `RestoreSlice`.
//
// Arguments:
// * file_pattern: Must have a single element. The pattern of the files from
// which we read the tensor.
// * tensor_name: Must have a single element. The name of the tensor to be
// restored.
// * dt: The type of the tensor to be restored.
// * opts:
//   .WithAttr("preferred_shard", int64): Defaults to -1.
//     Index of file to open first if multiple files match
// `file_pattern`.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The restored tensor.
@Namespace("tensorflow::ops") public static native Node Restore(@ByVal NodeBuilder.NodeOut file_pattern, @ByVal NodeBuilder.NodeOut tensor_name, @Cast("tensorflow::DataType") int dt, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Restore(Node file_pattern, Node tensor_name, @Cast("tensorflow::DataType") int dt, @Const @ByRef GraphDefBuilder.Options opts);

// Restores a tensor from checkpoint files.
//
// This is like `Restore` except that restored tensor can be listed as filling
// only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
// larger tensor and the slice that the restored tensor covers.
//
// The `shape_and_slice` input has the same format as the
// elements of the `shapes_and_slices` input of the `SaveSlices` op.
//
// Arguments:
// * file_pattern: Must have a single element. The pattern of the files from
// which we read the tensor.
// * tensor_name: Must have a single element. The name of the tensor to be
// restored.
// * shape_and_slice: Scalar. The shapes and slice specifications to use when
// restoring a tensors.
// * dt: The type of the tensor to be restored.
// * opts:
//   .WithAttr("preferred_shard", int64): Defaults to -1.
//     Index of file to open first if multiple files match
// `file_pattern`. See the documentation for `Restore`.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The restored tensor.
@Namespace("tensorflow::ops") public static native Node RestoreSlice(@ByVal NodeBuilder.NodeOut file_pattern, @ByVal NodeBuilder.NodeOut tensor_name, @ByVal NodeBuilder.NodeOut shape_and_slice, @Cast("tensorflow::DataType") int dt, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node RestoreSlice(Node file_pattern, Node tensor_name, Node shape_and_slice, @Cast("tensorflow::DataType") int dt, @Const @ByRef GraphDefBuilder.Options opts);

// Saves the input tensors to disk.
//
// The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
// is written to `filename` with name `tensor_names[i]`.
//
// See also `SaveSlices`.
//
// Arguments:
// * filename: Must have a single element. The name of the file to which we write
// the tensor.
// * tensor_names: Shape `[N]`. The names of the tensors to be saved.
// * data: `N` tensors to save.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Save(@ByVal NodeBuilder.NodeOut filename, @ByVal NodeBuilder.NodeOut tensor_names, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Save(Node filename, Node tensor_names, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);

// Saves input tensors slices to disk.
//
// This is like `Save` except that tensors can be listed in the saved file as being
// a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
// larger tensor and the slice that this tensor covers. `shapes_and_slices` must
// have as many elements as `tensor_names`.
//
// Elements of the `shapes_and_slices` input must either be:
//
// *  The empty string, in which case the corresponding tensor is
//    saved normally.
// *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
//    `dimI` are the dimensions of the larger tensor and `slice-spec`
//    specifies what part is covered by the tensor to save.
//
// `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
// where each `sliceI` is either:
//
// *  The string `-` meaning that the slice covers all indices of this dimension
// *  `start,length` where `start` and `length` are integers.  In that
//    case the slice covers `length` indices starting at `start`.
//
// See also `Save`.
//
// Arguments:
// * filename: Must have a single element. The name of the file to which we write the
// tensor.
// * tensor_names: Shape `[N]`. The names of the tensors to be saved.
// * shapes_and_slices: Shape `[N]`.  The shapes and slice specifications to use when
// saving the tensors.
// * data: `N` tensors to save.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node SaveSlices(@ByVal NodeBuilder.NodeOut filename, @ByVal NodeBuilder.NodeOut tensor_names, @ByVal NodeBuilder.NodeOut shapes_and_slices, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SaveSlices(Node filename, Node tensor_names, Node shapes_and_slices, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);

// Generate a sharded filename. The filename is printf formated as
//
//    %s-%05d-of-%05d, basename, shard, num_shards.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ShardedFilename(@ByVal NodeBuilder.NodeOut basename, @ByVal NodeBuilder.NodeOut shard, @ByVal NodeBuilder.NodeOut num_shards,
                      @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ShardedFilename(Node basename, Node shard, Node num_shards,
                      @Const @ByRef GraphDefBuilder.Options opts);

// Generate a glob pattern matching all sharded file names.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ShardedFilespec(@ByVal NodeBuilder.NodeOut basename, @ByVal NodeBuilder.NodeOut num_shards, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ShardedFilespec(Node basename, Node num_shards, @Const @ByRef GraphDefBuilder.Options opts);

// A Reader that outputs the records from a TensorFlow Records file.
//
// Arguments:
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
@Namespace("tensorflow::ops") public static native Node TFRecordReader(@Const @ByRef GraphDefBuilder.Options opts);

// A Reader that outputs the lines of a file delimited by '\n'.
//
// Arguments:
// * opts:
//   .WithAttr("skip_header_lines", int64): Defaults to 0.
//     Number of lines to skip from the beginning of every file.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
@Namespace("tensorflow::ops") public static native Node TextLineReader(@Const @ByRef GraphDefBuilder.Options opts);

// A Reader that outputs the entire contents of a file as a value.
//
// To use, enqueue filenames in a Queue.  The output of ReaderRead will
// be a filename (key) and the contents of that file (value).
//
// Arguments:
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
@Namespace("tensorflow::ops") public static native Node WholeFileReader(@Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_IO_OPS_H_


// Parsed from tensorflow/cc/ops/linalg_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_LINALG_OPS_H_
// #define TENSORFLOW_CC_OPS_LINALG_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Calculates the Cholesky decomposition of a batch of square matrices.
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices, with the same constraints as the single matrix Cholesky
// decomposition above. The output is a tensor of the same shape as the input
// containing the Cholesky decompositions for all input submatrices `[..., :, :]`.
//
// Arguments:
// * input: Shape is `[..., M, M]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Shape is `[..., M, M]`.
@Namespace("tensorflow::ops") public static native Node BatchCholesky(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BatchCholesky(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Calculates the determinants for a batch of square matrices.
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. The output is a 1-D tensor containing the determinants
// for all input submatrices `[..., :, :]`.
//
// Arguments:
// * input: Shape is `[..., M, M]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Shape is `[...]`.
@Namespace("tensorflow::ops") public static native Node BatchMatrixDeterminant(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BatchMatrixDeterminant(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Calculates the inverse of square invertible matrices.
//
// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
// form square matrices. The output is a tensor of the same shape as the input
// containing the inverse for all input submatrices `[..., :, :]`.
//
// The op uses the Cholesky decomposition if the matrices are symmetric positive
// definite and LU decomposition with partial pivoting otherwise.
//
// If a matrix is not invertible there is no guarantee what the op does. It
// may detect the condition and raise an exception or it may simply return a
// garbage result.
//
// Arguments:
// * input: Shape is `[..., M, M]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Shape is `[..., M, M]`.
@Namespace("tensorflow::ops") public static native Node BatchMatrixInverse(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BatchMatrixInverse(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Calculates the Cholesky decomposition of a square matrix.
//
// The input has to be symmetric and positive definite. Only the lower-triangular
// part of the input will be used for this operation. The upper-triangular part
// will not be read.
//
// The result is the lower-triangular matrix of the Cholesky decomposition of the
// input.
//
// Arguments:
// * input: Shape is `[M, M]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Shape is `[M, M]`.
@Namespace("tensorflow::ops") public static native Node Cholesky(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Cholesky(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Calculates the determinant of a square matrix.
//
// Arguments:
// * input: A tensor of shape `[M, M]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A scalar, equal to the determinant of the input.
@Namespace("tensorflow::ops") public static native Node MatrixDeterminant(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MatrixDeterminant(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Calculates the inverse of a square invertible matrix.
//
// The op uses the Cholesky decomposition if the matrix is symmetric positive
// definite and LU decomposition with partial pivoting otherwise.
//
// If the matrix is not invertible there is no guarantee what the op does. It
// may detect the condition and raise an exception or it may simply return a
// garbage result.
//
// Arguments:
// * input: Shape is `[M, M]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Shape is `[M, M]` containing the matrix inverse of the input.
@Namespace("tensorflow::ops") public static native Node MatrixInverse(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MatrixInverse(Node input, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_LINALG_OPS_H_


// Parsed from tensorflow/cc/ops/logging_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_LOGGING_OPS_H_
// #define TENSORFLOW_CC_OPS_LOGGING_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Asserts that the given condition is true.
//
// If `condition` evaluates to false, print the list of tensors in `data`.
// `summarize` determines how many entries of the tensors to print.
//
// Arguments:
// * condition: The condition to evaluate.
// * data: The tensors to print out when condition is false.
// * opts:
//   .WithAttr("summarize", int64): Defaults to 3.
//     Print this many entries of each tensor.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Assert(@ByVal NodeBuilder.NodeOut condition, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Assert(Node condition, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);

// Prints a list of tensors.
//
// Passes `input` through to `output` and prints `data` when evaluating.
//
// Arguments:
// * input: The tensor passed to `output`
// * data: A list of tensors to print out when op is evaluated.
// * opts:
//   .WithAttr("message", StringPiece): Defaults to "".
//     A string, prefix of the error message.
//   .WithAttr("first_n", int64): Defaults to -1.
//     Only log `first_n` number of times. -1 disables logging.
//   .WithAttr("summarize", int64): Defaults to 3.
//     Only print this many entries of each tensor.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The unmodified `input` tensor
@Namespace("tensorflow::ops") public static native Node Print(@ByVal NodeBuilder.NodeOut input, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Print(Node input, @ByVal NodeOutVector data, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_LOGGING_OPS_H_


// Parsed from tensorflow/cc/ops/math_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_MATH_OPS_H_
// #define TENSORFLOW_CC_OPS_MATH_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Computes the absolute value of a tensor.
//
// Given a tensor `x`, this operation returns a tensor containing the absolute
// value of each element in `x`. For example, if x is an input element and y is
// an output element, this operation computes \\(y = |x|\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Abs(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Abs(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns x + y element-wise.
//
// *NOTE*: Add supports broadcasting. AddN does not.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Add(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Add(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Add all input tensors element wise.
//
// Arguments:
// * inputs: Must all be the same size and shape.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node AddN(@ByVal NodeOutVector inputs, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the "logical and" of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
@Namespace("tensorflow::ops") public static native Node All(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node All(Node input, Node reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the "logical or" of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
@Namespace("tensorflow::ops") public static native Node Any(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Any(Node input, Node reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the index with the largest value across dimensions of a tensor.
//
// Arguments:
// * dimension: int32, 0 <= dimension < rank(input).  Describes which dimension
// of the input Tensor to reduce across. For vectors, use dimension = 0.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ArgMax(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut dimension, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ArgMax(Node input, Node dimension, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the index with the smallest value across dimensions of a tensor.
//
// Arguments:
// * dimension: int32, 0 <= dimension < rank(input).  Describes which dimension
// of the input Tensor to reduce across. For vectors, use dimension = 0.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ArgMin(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut dimension, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ArgMin(Node input, Node dimension, @Const @ByRef GraphDefBuilder.Options opts);

// Multiplies slices of two tensors in batches.
//
// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
// viewed as an element of a batch), and arranges the individual results
// in a single output tensor of the same batch size. Each of the
// individual slices can optionally be adjointed (to adjoint a matrix
// means to transpose and conjugate it) before multiplication by setting
// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
//
// The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
// and `[..., r_y, c_y]`.
//
// The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:
//
//     r_o = c_x if adj_x else r_x
//     c_o = r_y if adj_y else c_y
//
// It is computed as:
//
//     out[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
//
// Arguments:
// * x: 3-D or higher with shape `[..., r_x, c_x]`.
// * y: 3-D or higher with shape `[..., r_y, c_y]`.
// * opts:
//   .WithAttr("adj_x", bool): Defaults to false.
//     If `True`, adjoint the slices of `x`. Defaults to `False`.
//   .WithAttr("adj_y", bool): Defaults to false.
//     If `True`, adjoint the slices of `y`. Defaults to `False`.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 3-D or higher with shape `[..., r_o, c_o]`
@Namespace("tensorflow::ops") public static native Node BatchMatMul(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BatchMatMul(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Cast x of type SrcT to y of DstT.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Cast(@ByVal NodeBuilder.NodeOut x, @Cast("tensorflow::DataType") int DstT, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Cast(Node x, @Cast("tensorflow::DataType") int DstT, @Const @ByRef GraphDefBuilder.Options opts);

// Returns element-wise smallest integer in not less than x.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Ceil(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Ceil(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Converts two real numbers to a complex number.
//
// Given a tensor `real` representing the real part of a complex number, and a
// tensor `imag` representing the imaginary part of a complex number, this
// operation returns complex numbers elementwise of the form \\(a + bj\\), where
// *a* represents the `real` part and *b* represents the `imag` part.
//
// The input tensors `real` and `imag` must have the same shape.
//
// For example:
//
// ```
// # tensor 'real' is [2.25, 3.25]
// # tensor `imag` is [4.75, 5.75]
// tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Complex(@ByVal NodeBuilder.NodeOut real, @ByVal NodeBuilder.NodeOut imag, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Complex(Node real, Node imag, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the complex absolute value of a tensor.
//
// Given a tensor `x` of complex numbers, this operation returns a tensor of type
// `float` that is the absolute value of each element in `x`. All elements in `x`
// must be complex numbers of the form \\(a + bj\\). The absolute value is
// computed as \\( \sqrt{a^2 + b^2}\\).
//
// For example:
//
// ```
// # tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
// tf.complex_abs(x) ==> [5.25594902, 6.60492229]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node ComplexAbs(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ComplexAbs(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the complex conjugate of a complex number.
//
// Given a tensor `in` of complex numbers, this operation returns a tensor of
// complex numbers that are the complex conjugate of each element in `in`. The
// complex numbers in `in` must be of the form \\(a + bj\\), where *a* is the real
// part and *b* is the imaginary part.
//
// The complex conjugate returned by this operation is of the form \\(a - bj\\).
//
// For example:
//
// ```
// # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.conj(in) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Conj(@ByVal NodeBuilder.NodeOut in, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conj(Node in, @Const @ByRef GraphDefBuilder.Options opts);

// Computes cos of x element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Cos(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Cos(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns x / y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Div(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Div(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of (x == y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Equal(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Equal(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Computes exponential of x element-wise.  \\(y = e^x\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Exp(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Exp(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns element-wise largest integer not greater than x.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Floor(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Floor(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of (x > y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Greater(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Greater(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of (x >= y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node GreaterEqual(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node GreaterEqual(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the imaginary part of a complex number.
//
// Given a tensor `in` of complex numbers, this operation returns a tensor of type
// `float` that is the imaginary part of each element in `in`. All elements in `in`
// must be complex numbers of the form \\(a + bj\\), where *a* is the real part
// and *b* is the imaginary part returned by this operation.
//
// For example:
//
// ```
// # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.imag(in) ==> [4.75, 5.75]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Imag(@ByVal NodeBuilder.NodeOut in, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Imag(Node in, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the reciprocal of x element-wise.
//
// I.e., \\(y = 1 / x\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Inv(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Inv(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns which elements of x are finite.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node IsFinite(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node IsFinite(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns which elements of x are Inf.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node IsInf(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node IsInf(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns which elements of x are NaN.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node IsNan(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node IsNan(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of (x < y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Less(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Less(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of (x <= y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node LessEqual(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LessEqual(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Generates values in an interval.
//
// A sequence of `num` evenly-spaced values are generated beginning at `start`.
// If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
// so that the last one is exactly `stop`.
//
// For example:
//
// ```
// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
// ```
//
// Arguments:
// * start: First entry in the range.
// * stop: Last entry in the range.
// * num: Number of values to generate.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 1-D. The generated values.
@Namespace("tensorflow::ops") public static native Node LinSpace(@ByVal NodeBuilder.NodeOut start, @ByVal NodeBuilder.NodeOut stop, @ByVal NodeBuilder.NodeOut num, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LinSpace(Node start, Node stop, Node num, @Const @ByRef GraphDefBuilder.Options opts);

// Computes natural logrithm of x element-wise.
//
// I.e., \\(y = \log_e x\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Log(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Log(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of x AND y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node LogicalAnd(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LogicalAnd(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of NOT x element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node LogicalNot(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LogicalNot(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of x OR y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node LogicalOr(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LogicalOr(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Multiply the matrix "a" by the matrix "b".
//
// The inputs must be two-dimensional matrices and the inner dimension of
// "a" (after being transposed if transpose_a is true) must match the
// outer dimension of "b" (after being transposed if transposed_b is
// true).
//
// *Note*: The default kernel implementation for MatMul on GPUs uses
// cublas.
//
// Arguments:
// * opts:
//   .WithAttr("transpose_a", bool): Defaults to false.
//     If true, "a" is transposed before multiplication.
//   .WithAttr("transpose_b", bool): Defaults to false.
//     If true, "b" is transposed before multiplication.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node MatMul(@ByVal NodeBuilder.NodeOut a, @ByVal NodeBuilder.NodeOut b, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MatMul(Node a, Node b, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the maximum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
@Namespace("tensorflow::ops") public static native Node Max(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Max(Node input, Node reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Maximum(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Maximum(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the mean of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
@Namespace("tensorflow::ops") public static native Node Mean(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Mean(Node input, Node reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the minimum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
@Namespace("tensorflow::ops") public static native Node Min(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Min(Node input, Node reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Minimum(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Minimum(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Returns element-wise remainder of division.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Mod(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Mod(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Returns x * y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Mul(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Mul(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Computes numerical negative value element-wise.
//
// I.e., \\(y = -x\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Neg(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Neg(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the truth value of (x != y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node NotEqual(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node NotEqual(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the power of one value to another.
//
// Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
// corresponding elements in `x` and `y`. For example:
//
// ```
// # tensor 'x' is [[2, 2]], [3, 3]]
// # tensor 'y' is [[8, 16], [2, 3]]
// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Pow(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Pow(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the product of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
@Namespace("tensorflow::ops") public static native Node Prod(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Prod(Node input, Node reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);

// Creates a sequence of integers.
//
// This operation creates a sequence of integers that begins at `start` and
// extends by increments of `delta` up to but not including `limit`.
//
// For example:
//
// ```
// # 'start' is 3
// # 'limit' is 18
// # 'delta' is 3
// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
// ```
//
// Arguments:
// * start: 0-D (scalar). First entry in the sequence.
// * limit: 0-D (scalar). Upper limit of sequence, exclusive.
// * delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 1-D.
@Namespace("tensorflow::ops") public static native Node Range(@ByVal NodeBuilder.NodeOut start, @ByVal NodeBuilder.NodeOut limit, @ByVal NodeBuilder.NodeOut delta, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Range(Node start, Node limit, Node delta, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the real part of a complex number.
//
// Given a tensor `in` of complex numbers, this operation returns a tensor of type
// `float` that is the real part of each element in `in`. All elements in `in`
// must be complex numbers of the form \\(a + bj\\), where *a* is the real part
// returned by this operation and *b* is the imaginary part.
//
// For example:
//
// ```
// # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.real(in) ==> [-2.25, 3.25]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Real(@ByVal NodeBuilder.NodeOut in, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Real(Node in, @Const @ByRef GraphDefBuilder.Options opts);

// Computes reciprocal of square root of x element-wise.
//
// I.e., \\(y = 1 / \sqrt{x}\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Rsqrt(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Rsqrt(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the maximum along segments of a tensor.
//
// Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
// for an explanation of segments.
//
// Computes a tensor such that
// \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
// that `segment_ids[j] == i`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/SegmentMax.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
@Namespace("tensorflow::ops") public static native Node SegmentMax(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut segment_ids, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SegmentMax(Node data, Node segment_ids, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the mean along segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Computes a tensor such that
// \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
// over `j` such that `segment_ids[j] == i` and `N` is the total number of
// values summed.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/SegmentMean.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
@Namespace("tensorflow::ops") public static native Node SegmentMean(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut segment_ids, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SegmentMean(Node data, Node segment_ids, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the minimum along segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Computes a tensor such that
// \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
// that `segment_ids[j] == i`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/SegmentMin.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
@Namespace("tensorflow::ops") public static native Node SegmentMin(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut segment_ids, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SegmentMin(Node data, Node segment_ids, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the product along segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Computes a tensor such that
// \\(output_i = \prod_j data_j\\) where the product is over `j` such
// that `segment_ids[j] == i`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/SegmentProd.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
@Namespace("tensorflow::ops") public static native Node SegmentProd(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut segment_ids, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SegmentProd(Node data, Node segment_ids, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the sum along segments of a tensor.
//
// Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
// for an explanation of segments.
//
// Computes a tensor such that
// \\(output_i = \sum_j data_j\\) where sum is over `j` such
// that `segment_ids[j] == i`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/SegmentSum.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
@Namespace("tensorflow::ops") public static native Node SegmentSum(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut segment_ids, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SegmentSum(Node data, Node segment_ids, @Const @ByRef GraphDefBuilder.Options opts);

// Selects elements from `t` or `e`, depending on `condition`.
//
// The `condition`, `t`, and `e` tensors must all have the same shape,
// and the output will also have that shape. The `condition` tensor acts
// as an element-wise mask that chooses, based on the value at each
// element, whether the corresponding element in the output should be
// taken from `t` (if true) or `e` (if false). For example:
//
// For example:
//
// ```prettyprint
// # 'condition' tensor is [[True, False]
// #                        [True, False]]
// # 't' is [[1, 1],
// #         [1, 1]]
// # 'e' is [[2, 2],
// #         [2, 2]]
// select(condition, t, e) ==> [[1, 2],
//                              [1, 2]]
// ```
//
// Arguments:
// * t: = A `Tensor` with the same shape as `condition`.
// * e: = A `Tensor` with the same type and shape as `t`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A `Tensor` with the same type and shape as `t` and `e`.
@Namespace("tensorflow::ops") public static native Node Select(@ByVal NodeBuilder.NodeOut condition, @ByVal NodeBuilder.NodeOut t, @ByVal NodeBuilder.NodeOut e, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Select(Node condition, Node t, Node e, @Const @ByRef GraphDefBuilder.Options opts);

// Computes sigmoid of `x` element-wise.
//
// Specifically, `y = 1 / (1 + exp(-x))`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Sigmoid(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Sigmoid(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns an element-wise indication of the sign of a number.
//
// y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Sign(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Sign(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Computes sin of x element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Sin(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Sin(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Multiply matrix "a" by matrix "b".
//
// The inputs must be two-dimensional matrices and the inner dimension of "a" must
// match the outer dimension of "b". This op is optimized for the case where at
// least one of "a" or "b" is sparse. The breakeven for using this versus a dense
// matrix multiply on one platform was 30% zero values in the sparse matrix.
//
// Arguments:
// * opts:
//   .WithAttr("transpose_a", bool): Defaults to false.
//   .WithAttr("transpose_b", bool): Defaults to false.
//   .WithAttr("a_is_sparse", bool): Defaults to false.
//   .WithAttr("b_is_sparse", bool): Defaults to false.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node SparseMatMul(@ByVal NodeBuilder.NodeOut a, @ByVal NodeBuilder.NodeOut b, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SparseMatMul(Node a, Node b, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the mean along sparse segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
// dimension, selecting a subset of dimension 0, specified by `indices`.
//
// Arguments:
// * indices: A 1-D tensor. Has same rank as `segment_ids`.
// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
@Namespace("tensorflow::ops") public static native Node SparseSegmentMean(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut indices, @ByVal NodeBuilder.NodeOut segment_ids,
                        @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SparseSegmentMean(Node data, Node indices, Node segment_ids,
                        @Const @ByRef GraphDefBuilder.Options opts);

// Computes gradients for SparseSegmentMean.
//
// Returns tensor "output" with same shape as grad, except for dimension 0 whose
// value is output_dim0.
//
// Arguments:
// * grad: gradient propagated to the SparseSegmentMean op.
// * indices: indices passed to the corresponding SparseSegmentMean op.
// * segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
// * output_dim0: dimension 0 of "data" passed to SparseSegmentMean op.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node SparseSegmentMeanGrad(@ByVal NodeBuilder.NodeOut grad, @ByVal NodeBuilder.NodeOut indices, @ByVal NodeBuilder.NodeOut segment_ids,
                            @ByVal NodeBuilder.NodeOut output_dim0, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SparseSegmentMeanGrad(Node grad, Node indices, Node segment_ids,
                            Node output_dim0, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the sum along sparse segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
// dimension, selecting a subset of dimension 0, specified by `indices`.
//
// For example:
//
// ```prettyprint
// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
//
// # Select two rows, one segment.
// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
//   ==> [[0 0 0 0]]
//
// # Select two rows, two segment.
// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
//   ==> [[ 1  2  3  4]
//        [-1 -2 -3 -4]]
//
// # Select all rows, two segments.
// tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
//   ==> [[0 0 0 0]
//        [5 6 7 8]]
//
// # Which is equivalent to:
// tf.segment_sum(c, tf.constant([0, 0, 1]))
// ```
//
// Arguments:
// * indices: A 1-D tensor. Has same rank as `segment_ids`.
// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
@Namespace("tensorflow::ops") public static native Node SparseSegmentSum(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut indices, @ByVal NodeBuilder.NodeOut segment_ids,
                       @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SparseSegmentSum(Node data, Node indices, Node segment_ids,
                       @Const @ByRef GraphDefBuilder.Options opts);

// Computes square root of x element-wise.
//
// I.e., \\(y = \sqrt{x} = x^{1/2}\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Sqrt(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Sqrt(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Computes square of x element-wise.
//
// I.e., \\(y = x * x = x^2\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Square(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Square(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Returns x - y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Sub(@ByVal NodeBuilder.NodeOut x, @ByVal NodeBuilder.NodeOut y, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Sub(Node x, Node y, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the sum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
@Namespace("tensorflow::ops") public static native Node Sum(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Sum(Node input, Node reduction_indices, @Const @ByRef GraphDefBuilder.Options opts);

// Computes hyperbolic tangent of `x` element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Tanh(@ByVal NodeBuilder.NodeOut x, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Tanh(Node x, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the sum along segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Computes a tensor such that
// \\(output_i = \sum_j data_j\\) where sum is over `j` such
// that `segment_ids[j] == i`. Unlike `SegmentSum`, `segment_ids`
// need not be sorted and need not cover all values in the full
//   range of valid values.
//
// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
//
// `num_segments` should equal the number of distinct segment IDs.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/UnsortedSegmentSum.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `num_segments`.
@Namespace("tensorflow::ops") public static native Node UnsortedSegmentSum(@ByVal NodeBuilder.NodeOut data, @ByVal NodeBuilder.NodeOut segment_ids, @ByVal NodeBuilder.NodeOut num_segments, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node UnsortedSegmentSum(Node data, Node segment_ids, Node num_segments, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_MATH_OPS_H_


// Parsed from tensorflow/cc/ops/nn_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_NN_OPS_H_
// #define TENSORFLOW_CC_OPS_NN_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Performs average pooling on the input.
//
// Each entry in `output` is the mean of the corresponding size `ksize`
// window in `value`.
//
// Arguments:
// * value: 4-D with shape `[batch, height, width, channels]`.
// * ksize: The size of the sliding window for each dimension of `value`.
// * strides: The stride of the sliding window for each dimension of `value`.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The average pooled output tensor.
@Namespace("tensorflow::ops") public static native Node AvgPool(@ByVal NodeBuilder.NodeOut value, @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPool(Node value, @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPool(@ByVal NodeBuilder.NodeOut value, @ArraySlice int[] ksize, @ArraySlice int[] strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPool(Node value, @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPool(@ByVal NodeBuilder.NodeOut value, @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPool(Node value, @ArraySlice int[] ksize, @ArraySlice int[] strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);

// Computes gradients of the average pooling function.
//
// Arguments:
// * orig_input_shape: 1-D.  Shape of the original input to `avg_pool`.
// * grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
// the output of `avg_pool`.
// * ksize: The size of the sliding window for each dimension of the input.
// * strides: The stride of the sliding window for each dimension of the input.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D.  Gradients w.r.t. the input of `avg_pool`.
@Namespace("tensorflow::ops") public static native Node AvgPoolGrad(@ByVal NodeBuilder.NodeOut orig_input_shape, @ByVal NodeBuilder.NodeOut grad, @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides, @StringPiece BytePointer padding,
                  @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPoolGrad(Node orig_input_shape, Node grad, @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides, @StringPiece String padding,
                  @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPoolGrad(@ByVal NodeBuilder.NodeOut orig_input_shape, @ByVal NodeBuilder.NodeOut grad, @ArraySlice int[] ksize, @ArraySlice int[] strides, @StringPiece BytePointer padding,
                  @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPoolGrad(Node orig_input_shape, Node grad, @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides, @StringPiece String padding,
                  @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPoolGrad(@ByVal NodeBuilder.NodeOut orig_input_shape, @ByVal NodeBuilder.NodeOut grad, @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides, @StringPiece BytePointer padding,
                  @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AvgPoolGrad(Node orig_input_shape, Node grad, @ArraySlice int[] ksize, @ArraySlice int[] strides, @StringPiece String padding,
                  @Const @ByRef GraphDefBuilder.Options opts);

// Batch normalization.
//
// Arguments:
// * t: A 4D input Tensor.
// * m: A 1D mean Tensor with size matching the last dimension of t.
// This is the first output from tf.nn.moments,
// or a saved moving average thereof.
// * v: A 1D variance Tensor with size matching the last dimension of t.
// This is the second output from tf.nn.moments,
// or a saved moving average thereof.
// * beta: A 1D beta Tensor with size matching the last dimension of t.
// An offset to be added to the normalized tensor.
// * gamma: A 1D gamma Tensor with size matching the last dimension of t.
// If "scale_after_normalization" is true, this tensor will be multiplied
// with the normalized tensor.
// * variance_epsilon: A small float number to avoid dividing by 0.
// * scale_after_normalization: A bool indicating whether the resulted tensor
// needs to be multiplied with gamma.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node BatchNormWithGlobalNormalization(@ByVal NodeBuilder.NodeOut t, @ByVal NodeBuilder.NodeOut m, @ByVal NodeBuilder.NodeOut v, @ByVal NodeBuilder.NodeOut beta, @ByVal NodeBuilder.NodeOut gamma, float variance_epsilon, @Cast("bool") boolean scale_after_normalization, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BatchNormWithGlobalNormalization(Node t, Node m, Node v, Node beta, Node gamma, float variance_epsilon, @Cast("bool") boolean scale_after_normalization, @Const @ByRef GraphDefBuilder.Options opts);

// Gradients for batch normalization.
//
// Arguments:
// * t: A 4D input Tensor.
// * m: A 1D mean Tensor with size matching the last dimension of t.
// This is the first output from tf.nn.moments,
// or a saved moving average thereof.
// * v: A 1D variance Tensor with size matching the last dimension of t.
// This is the second output from tf.nn.moments,
// or a saved moving average thereof.
// * gamma: A 1D gamma Tensor with size matching the last dimension of t.
// If "scale_after_normalization" is true, this Tensor will be multiplied
// with the normalized Tensor.
// * backprop: 4D backprop Tensor.
// * variance_epsilon: A small float number to avoid dividing by 0.
// * scale_after_normalization: A bool indicating whether the resulted tensor
// needs to be multiplied with gamma.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * dx: 4D backprop tensor for input.
// * dm: 1D backprop tensor for mean.
// * dv: 1D backprop tensor for variance.
// * db: 1D backprop tensor for beta.
// * dg: 1D backprop tensor for gamma.
@Namespace("tensorflow::ops") public static native Node BatchNormWithGlobalNormalizationGrad(@ByVal NodeBuilder.NodeOut t, @ByVal NodeBuilder.NodeOut m, @ByVal NodeBuilder.NodeOut v,
                                           @ByVal NodeBuilder.NodeOut gamma, @ByVal NodeBuilder.NodeOut backprop,
                                           float variance_epsilon, @Cast("bool") boolean scale_after_normalization, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BatchNormWithGlobalNormalizationGrad(Node t, Node m, Node v,
                                           Node gamma, Node backprop,
                                           float variance_epsilon, @Cast("bool") boolean scale_after_normalization, @Const @ByRef GraphDefBuilder.Options opts);

// Adds `bias` to `value`.
//
// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
// Broadcasting is supported, so `value` may have any number of dimensions.
//
// Arguments:
// * value: Any number of dimensions.
// * bias: 1-D with size the last dimension of `value`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Broadcasted sum of `value` and `bias`.
@Namespace("tensorflow::ops") public static native Node BiasAdd(@ByVal NodeBuilder.NodeOut value, @ByVal NodeBuilder.NodeOut bias, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BiasAdd(Node value, Node bias, @Const @ByRef GraphDefBuilder.Options opts);

// Computes a 2-D convolution given 4-D `input` and `filter` tensors.
//
// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
// and a filter / kernel tensor of shape
// `[filter_height, filter_width, in_channels, out_channels]`, this op
// performs the following:
//
// 1. Flattens the filter to a 2-D matrix with shape
//    `[filter_height * filter_width * in_channels, output_channels]`.
// 2. Extracts image patches from the the input tensor to form a *virtual*
//    tensor of shape `[batch, out_height, out_width,
//    filter_height * filter_width * in_channels]`.
// 3. For each patch, right-multiplies the filter matrix and the image patch
//    vector.
//
// In detail,
//
//     output[b, i, j, k] =
//         sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
//                         filter[di, dj, q, k]
//
// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
//
// Arguments:
// * strides: 1-D of length 4.  The stride of the sliding window for each dimension
// of `input`.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithAttr("use_cudnn_on_gpu", bool): Defaults to true.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Conv2D(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut filter, @ArraySlice IntPointer strides,
             @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2D(Node input, Node filter, @ArraySlice IntBuffer strides,
             @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2D(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut filter, @ArraySlice int[] strides,
             @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2D(Node input, Node filter, @ArraySlice IntPointer strides,
             @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2D(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut filter, @ArraySlice IntBuffer strides,
             @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2D(Node input, Node filter, @ArraySlice int[] strides,
             @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the gradients of convolution with respect to the filter.
//
// Arguments:
// * input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
// * filter_sizes: An integer vector representing the tensor shape of `filter`,
// where `filter` is a 4-D
// `[filter_height, filter_width, in_channels, out_channels]` tensor.
// * out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
// Gradients w.r.t. the output of the convolution.
// * strides: The stride of the sliding window for each dimension of the input
// of the convolution.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithAttr("use_cudnn_on_gpu", bool): Defaults to true.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
// the `filter` input of the convolution.
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropFilter(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut filter_sizes, @ByVal NodeBuilder.NodeOut out_backprop, @ArraySlice IntPointer strides,
                           @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropFilter(Node input, Node filter_sizes, Node out_backprop, @ArraySlice IntBuffer strides,
                           @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropFilter(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut filter_sizes, @ByVal NodeBuilder.NodeOut out_backprop, @ArraySlice int[] strides,
                           @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropFilter(Node input, Node filter_sizes, Node out_backprop, @ArraySlice IntPointer strides,
                           @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropFilter(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut filter_sizes, @ByVal NodeBuilder.NodeOut out_backprop, @ArraySlice IntBuffer strides,
                           @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropFilter(Node input, Node filter_sizes, Node out_backprop, @ArraySlice int[] strides,
                           @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);

// Computes the gradients of convolution with respect to the input.
//
// Arguments:
// * input_sizes: An integer vector representing the shape of `input`,
// where `input` is a 4-D `[batch, height, width, channels]` tensor.
// * filter: 4-D with shape
// `[filter_height, filter_width, in_channels, out_channels]`.
// * out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
// Gradients w.r.t. the output of the convolution.
// * strides: The stride of the sliding window for each dimension of the input
// of the convolution.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithAttr("use_cudnn_on_gpu", bool): Defaults to true.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
// w.r.t. the input of the convolution.
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropInput(@ByVal NodeBuilder.NodeOut input_sizes, @ByVal NodeBuilder.NodeOut filter, @ByVal NodeBuilder.NodeOut out_backprop, @ArraySlice IntPointer strides,
                          @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropInput(Node input_sizes, Node filter, Node out_backprop, @ArraySlice IntBuffer strides,
                          @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropInput(@ByVal NodeBuilder.NodeOut input_sizes, @ByVal NodeBuilder.NodeOut filter, @ByVal NodeBuilder.NodeOut out_backprop, @ArraySlice int[] strides,
                          @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropInput(Node input_sizes, Node filter, Node out_backprop, @ArraySlice IntPointer strides,
                          @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropInput(@ByVal NodeBuilder.NodeOut input_sizes, @ByVal NodeBuilder.NodeOut filter, @ByVal NodeBuilder.NodeOut out_backprop, @ArraySlice IntBuffer strides,
                          @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Conv2DBackpropInput(Node input_sizes, Node filter, Node out_backprop, @ArraySlice int[] strides,
                          @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);

// Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.
//
// See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
// ](http://arxiv.org/abs/1511.07289)
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Elu(@ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Elu(Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Computes gradients for the exponential linear (Elu) operation.
//
// Arguments:
// * gradients: The backpropagated gradients to the corresponding Elu operation.
// * outputs: The outputs of the corresponding Elu operation.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The gradients: `gradients * (outputs + 1)` if outputs < 0,
// `gradients` otherwise.
@Namespace("tensorflow::ops") public static native Node EluGrad(@ByVal NodeBuilder.NodeOut gradients, @ByVal NodeBuilder.NodeOut outputs, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node EluGrad(Node gradients, Node outputs, @Const @ByRef GraphDefBuilder.Options opts);

// Says whether the targets are in the top `K` predictions.
//
// This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
// prediction for the target class is among the top `k` predictions among
// all predictions for example `i`. Note that the behavior of `InTopK` differs
// from the `TopK` op in its handling of ties; if multiple classes have the
// same prediction value and straddle the top-`k` boundary, all of those
// classes are considered to be in the top `k`.
//
// More formally, let
//
//   \\(predictions_i\\) be the predictions for all classes for example `i`,
//   \\(targets_i\\) be the target class for example `i`,
//   \\(out_i\\) be the output for example `i`,
//
// $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$
//
// Arguments:
// * predictions: A `batch_size` x `classes` tensor.
// * targets: A `batch_size` vector of class ids.
// * k: Number of top elements to look at for computing precision.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Computed Precision at `k` as a `bool Tensor`.
@Namespace("tensorflow::ops") public static native Node InTopK(@ByVal NodeBuilder.NodeOut predictions, @ByVal NodeBuilder.NodeOut targets, @Cast("tensorflow::int64") long k, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node InTopK(Node predictions, Node targets, @Cast("tensorflow::int64") long k, @Const @ByRef GraphDefBuilder.Options opts);

// L2 Loss.
//
// Computes half the L2 norm of a tensor without the `sqrt`:
//
//     output = sum(t ** 2) / 2
//
// Arguments:
// * t: Typically 2-D, but may have any dimensions.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 0-D.
@Namespace("tensorflow::ops") public static native Node L2Loss(@ByVal NodeBuilder.NodeOut t, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node L2Loss(Node t, @Const @ByRef GraphDefBuilder.Options opts);

// Local Response Normalization.
//
// The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
// dimension), and each vector is normalized independently.  Within a given vector,
// each component is divided by the weighted, squared sum of inputs within
// `depth_radius`.  In detail,
//
//     sqr_sum[a, b, c, d] =
//         sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
//     output = input / (bias + alpha * sqr_sum ** beta)
//
// For details, see [Krizhevsky et al., ImageNet classification with deep
// convolutional neural networks (NIPS 2012)]
// (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
//
// Arguments:
// * input: 4-D.
// * opts:
//   .WithAttr("depth_radius", int64): Defaults to 5.
//     0-D.  Half-width of the 1-D normalization window.
//   .WithAttr("bias", float): Defaults to 1.
//     An offset (usually positive to avoid dividing by 0).
//   .WithAttr("alpha", float): Defaults to 1.
//     A scale factor, usually positive.
//   .WithAttr("beta", float): Defaults to 0.5.
//     An exponent.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node LRN(@ByVal NodeBuilder.NodeOut input, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LRN(Node input, @Const @ByRef GraphDefBuilder.Options opts);

// Gradients for Local Response Normalization.
//
// Arguments:
// * input_grads: 4-D with shape `[batch, height, width, channels]`.
// * input_image: 4-D with shape `[batch, height, width, channels]`.
// * output_image: 4-D with shape `[batch, height, width, channels]`.
// * opts:
//   .WithAttr("depth_radius", int64): Defaults to 5.
//     A depth radius.
//   .WithAttr("bias", float): Defaults to 1.
//     An offset (usually > 0 to avoid dividing by 0).
//   .WithAttr("alpha", float): Defaults to 1.
//     A scale factor, usually positive.
//   .WithAttr("beta", float): Defaults to 0.5.
//     An exponent.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The gradients for LRN.
@Namespace("tensorflow::ops") public static native Node LRNGrad(@ByVal NodeBuilder.NodeOut input_grads, @ByVal NodeBuilder.NodeOut input_image, @ByVal NodeBuilder.NodeOut output_image,
              @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node LRNGrad(Node input_grads, Node input_image, Node output_image,
              @Const @ByRef GraphDefBuilder.Options opts);

// Performs max pooling on the input.
//
// Arguments:
// * input: 4-D input to pool over.
// * ksize: The size of the window for each dimension of the input tensor.
// * strides: The stride of the sliding window for each dimension of the
// input tensor.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The max pooled output tensor.
@Namespace("tensorflow::ops") public static native Node MaxPool(@ByVal NodeBuilder.NodeOut input, @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPool(Node input, @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPool(@ByVal NodeBuilder.NodeOut input, @ArraySlice int[] ksize, @ArraySlice int[] strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPool(Node input, @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPool(@ByVal NodeBuilder.NodeOut input, @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPool(Node input, @ArraySlice int[] ksize, @ArraySlice int[] strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);

// Computes gradients of the maxpooling function.
//
// Arguments:
// * orig_input: The original input tensor.
// * orig_output: The original output tensor.
// * grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
// * ksize: The size of the window for each dimension of the input tensor.
// * strides: The stride of the sliding window for each dimension of the
// input tensor.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Gradients w.r.t. the input to `max_pool`.
@Namespace("tensorflow::ops") public static native Node MaxPoolGrad(@ByVal NodeBuilder.NodeOut orig_input, @ByVal NodeBuilder.NodeOut orig_output, @ByVal NodeBuilder.NodeOut grad,
                  @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides,
                  @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGrad(Node orig_input, Node orig_output, Node grad,
                  @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides,
                  @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGrad(@ByVal NodeBuilder.NodeOut orig_input, @ByVal NodeBuilder.NodeOut orig_output, @ByVal NodeBuilder.NodeOut grad,
                  @ArraySlice int[] ksize, @ArraySlice int[] strides,
                  @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGrad(Node orig_input, Node orig_output, Node grad,
                  @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides,
                  @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGrad(@ByVal NodeBuilder.NodeOut orig_input, @ByVal NodeBuilder.NodeOut orig_output, @ByVal NodeBuilder.NodeOut grad,
                  @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides,
                  @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGrad(Node orig_input, Node orig_output, Node grad,
                  @ArraySlice int[] ksize, @ArraySlice int[] strides,
                  @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);

// Computes gradients of the maxpooling function.
//
// Arguments:
// * input: The original input.
// * grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
// output of `max_pool`.
// * argmax: The indices of the maximum values chosen for each output of `max_pool`.
// * ksize: The size of the window for each dimension of the input tensor.
// * strides: The stride of the sliding window for each dimension of the
// input tensor.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Gradients w.r.t. the input of `max_pool`.
@Namespace("tensorflow::ops") public static native Node MaxPoolGradWithArgmax(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut grad, @ByVal NodeBuilder.NodeOut argmax,
                            @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGradWithArgmax(Node input, Node grad, Node argmax,
                            @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGradWithArgmax(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut grad, @ByVal NodeBuilder.NodeOut argmax,
                            @ArraySlice int[] ksize, @ArraySlice int[] strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGradWithArgmax(Node input, Node grad, Node argmax,
                            @ArraySlice IntPointer ksize, @ArraySlice IntPointer strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGradWithArgmax(@ByVal NodeBuilder.NodeOut input, @ByVal NodeBuilder.NodeOut grad, @ByVal NodeBuilder.NodeOut argmax,
                            @ArraySlice IntBuffer ksize, @ArraySlice IntBuffer strides, @StringPiece BytePointer padding, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolGradWithArgmax(Node input, Node grad, Node argmax,
                            @ArraySlice int[] ksize, @ArraySlice int[] strides, @StringPiece String padding, @Const @ByRef GraphDefBuilder.Options opts);

// Performs max pooling on the input and outputs both max values and indices.
//
// The indices in `argmax` are flattened, so that a maximum value at position
// `[b, y, x, c]` becomes flattened index
// `((b * height + y) * width + x) * channels + c`.
//
// Arguments:
// * input: 4-D with shape `[batch, height, width, channels]`.  Input to pool over.
// * ksize: The size of the window for each dimension of the input tensor.
// * strides: The stride of the sliding window for each dimension of the
// input tensor.
// * padding: The type of padding algorithm to use.
// * opts:
//   .WithAttr("Targmax", DataType): Defaults to DT_INT64.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output: The max pooled output tensor.
// * argmax: 4-D.  The flattened indices of the max values chosen for each output.
@Namespace("tensorflow::ops") public static native Node MaxPoolWithArgmax(@ByVal NodeBuilder.NodeOut input, @ArraySlice IntPointer ksize,
                        @ArraySlice IntPointer strides, @StringPiece BytePointer padding,
                        @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolWithArgmax(Node input, @ArraySlice IntBuffer ksize,
                        @ArraySlice IntBuffer strides, @StringPiece String padding,
                        @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolWithArgmax(@ByVal NodeBuilder.NodeOut input, @ArraySlice int[] ksize,
                        @ArraySlice int[] strides, @StringPiece BytePointer padding,
                        @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolWithArgmax(Node input, @ArraySlice IntPointer ksize,
                        @ArraySlice IntPointer strides, @StringPiece String padding,
                        @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolWithArgmax(@ByVal NodeBuilder.NodeOut input, @ArraySlice IntBuffer ksize,
                        @ArraySlice IntBuffer strides, @StringPiece BytePointer padding,
                        @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node MaxPoolWithArgmax(Node input, @ArraySlice int[] ksize,
                        @ArraySlice int[] strides, @StringPiece String padding,
                        @Const @ByRef GraphDefBuilder.Options opts);

// Computes rectified linear: `max(features, 0)`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Relu(@ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Relu(Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Computes rectified linear 6: `min(max(features, 0), 6)`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Relu6(@ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Relu6(Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Computes rectified linear 6 gradients for a Relu6 operation.
//
// Arguments:
// * gradients: The backpropagated gradients to the corresponding Relu6 operation.
// * features: The features passed as input to the corresponding Relu6 operation.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The gradients:
// `gradients * features * (features > 0) * (features < 6)`.
@Namespace("tensorflow::ops") public static native Node Relu6Grad(@ByVal NodeBuilder.NodeOut gradients, @ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Relu6Grad(Node gradients, Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Computes rectified linear gradients for a Relu operation.
//
// Arguments:
// * gradients: The backpropagated gradients to the corresponding Relu operation.
// * features: The features passed as input to the corresponding Relu operation.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The gradients: `gradients * features * (features > 0)`.
@Namespace("tensorflow::ops") public static native Node ReluGrad(@ByVal NodeBuilder.NodeOut gradients, @ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ReluGrad(Node gradients, Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Computes softmax activations.
//
// For each batch `i` and class `j` we have
//
//     softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
//
// Arguments:
// * logits: 2-D with shape `[batch_size, num_classes]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same shape as `logits`.
@Namespace("tensorflow::ops") public static native Node Softmax(@ByVal NodeBuilder.NodeOut logits, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Softmax(Node logits, @Const @ByRef GraphDefBuilder.Options opts);

// Computes softmax cross entropy cost and gradients to backpropagate.
//
// Inputs are the logits, not probabilities.
//
// Arguments:
// * features: batch_size x num_classes matrix
// * labels: batch_size x num_classes matrix
// The caller must ensure that each batch of labels represents a valid
// probability distribution.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * loss: Per example loss (batch_size vector).
// * backprop: backpropagated gradients (batch_size x num_classes matrix).
@Namespace("tensorflow::ops") public static native Node SoftmaxCrossEntropyWithLogits(@ByVal NodeBuilder.NodeOut features, @ByVal NodeBuilder.NodeOut labels, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SoftmaxCrossEntropyWithLogits(Node features, Node labels, @Const @ByRef GraphDefBuilder.Options opts);

// Computes softplus: `log(exp(features) + 1)`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Softplus(@ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Softplus(Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Computes softplus gradients for a softplus operation.
//
// Arguments:
// * gradients: The backpropagated gradients to the corresponding softplus operation.
// * features: The features passed as input to the corresponding softplus operation.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The gradients: `gradients / (1 + exp(-features))`.
@Namespace("tensorflow::ops") public static native Node SoftplusGrad(@ByVal NodeBuilder.NodeOut gradients, @ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SoftplusGrad(Node gradients, Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Computes softsign: `features / (abs(features) + 1)`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Softsign(@ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Softsign(Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Computes softsign gradients for a softsign operation.
//
// Arguments:
// * gradients: The backpropagated gradients to the corresponding softsign operation.
// * features: The features passed as input to the corresponding softsign operation.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The gradients: `gradients / (1 + abs(-features)) ** 2`.
@Namespace("tensorflow::ops") public static native Node SoftsignGrad(@ByVal NodeBuilder.NodeOut gradients, @ByVal NodeBuilder.NodeOut features, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SoftsignGrad(Node gradients, Node features, @Const @ByRef GraphDefBuilder.Options opts);

// Returns the values and indices of the `k` largest elements for each row.
//
// \\(values_{i, j}\\) represents the j-th largest element in \\(input_i\\).
//
// \\(indices_{i, j}\\) gives the column index of the corresponding element,
// such that \\(input_{i, indices_{i, j}} = values_{i, j}\\). If two
// elements are equal, the lower-index element appears first.
//
// Arguments:
// * input: A `batch_size` x `classes` tensor.
// * k: Number of top elements to look for within each row.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * values: A `batch_size` x `k` tensor with the `k` largest elements for
// each row, sorted in descending order.
// * indices: A `batch_size` x `k` tensor with the index of each value within
// each row.
@Namespace("tensorflow::ops") public static native Node TopK(@ByVal NodeBuilder.NodeOut input, @Cast("tensorflow::int64") long k, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node TopK(Node input, @Cast("tensorflow::int64") long k, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_NN_OPS_H_


// Parsed from tensorflow/cc/ops/parsing_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_PARSING_OPS_H_
// #define TENSORFLOW_CC_OPS_PARSING_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Convert CSV records to tensors. Each column maps to one tensor.
//
// RFC 4180 format is expected for the CSV records.
// (https://tools.ietf.org/html/rfc4180)
// Note that we allow leading and trailing spaces with int or float field.
//
// Arguments:
// * records: Each string is a record/row in the csv and all records should have
// the same format.
// * record_defaults: One tensor per column of the input record, with either a
// scalar default value for that column or empty if the column is required.
// * opts:
//   .WithAttr("field_delim", StringPiece): Defaults to ",".
//     delimiter to separate fields in a record.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Each tensor will have the same shape as records.
@Namespace("tensorflow::ops") public static native Node DecodeCSV(@ByVal NodeBuilder.NodeOut records, @ByVal NodeOutVector record_defaults,
                @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node DecodeCSV(Node records, @ByVal NodeOutVector record_defaults,
                @Const @ByRef GraphDefBuilder.Options opts);

// Reinterpret the bytes of a string as a vector of numbers.
//
// Arguments:
// * bytes: All the elements must have the same length.
// * opts:
//   .WithAttr("little_endian", bool): Defaults to true.
//     Whether the input `bytes` are in little-endian order.
// Ignored for `out_type` values that are stored in a single byte like
// `uint8`.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A Tensor with one more dimension than the input `bytes`.  The
// added dimension will have size equal to the length of the elements
// of `bytes` divided by the number of bytes to represent `out_type`.
@Namespace("tensorflow::ops") public static native Node DecodeRaw(@ByVal NodeBuilder.NodeOut bytes, @Cast("tensorflow::DataType") int out_type, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node DecodeRaw(Node bytes, @Cast("tensorflow::DataType") int out_type, @Const @ByRef GraphDefBuilder.Options opts);

// Transforms a vector of brain.Example protos (as strings) into typed tensors.
//
// Arguments:
// * serialized: A vector containing a batch of binary serialized Example protos.
// * names: A vector containing the names of the serialized protos.
// May contain, for example, table key (descriptive) names for the
// corresponding serialized protos.  These are purely useful for debugging
// purposes, and the presence of values here has no effect on the output.
// May also be an empty vector if no names are available.
// If non-empty, this vector must be the same length as "serialized".
// * sparse_keys: A list of Nsparse string Tensors (scalars).
// The keys expected in the Examples' features associated with sparse values.
// * dense_keys: A list of Ndense string Tensors (scalars).
// The keys expected in the Examples' features associated with dense values.
// * dense_defaults: A list of Ndense Tensors (some may be empty).
// dense_defaults[j] provides default values
// when the example's feature_map lacks dense_key[j].  If an empty Tensor is
// provided for dense_defaults[j], then the Feature dense_keys[j] is required.
// The input type is inferred from dense_defaults[j], even when it's empty.
// If dense_defaults[j] is not empty, its shape must match dense_shapes[j].
// * sparse_types: A list of Nsparse types; the data types of data in each Feature
// given in sparse_keys.
// Currently the ParseExample supports DT_FLOAT (FloatList),
// DT_INT64 (Int64List), and DT_STRING (BytesList).
// * dense_shapes: A list of Ndense shapes; the shapes of data in each Feature
// given in dense_keys.
// The number of elements in the Feature corresponding to dense_key[j]
// must always equal dense_shapes[j].NumEntries().
// If dense_shapes[j] == (D0, D1, ..., DN) then the the shape of output
// Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
// The dense outputs are just the inputs row-stacked by batch.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sparse_indices
// * sparse_values
// * sparse_shapes
// * dense_values
@Namespace("tensorflow::ops") public static native Node ParseExample(@ByVal NodeBuilder.NodeOut serialized, @ByVal NodeBuilder.NodeOut names, @ByVal NodeOutVector sparse_keys, @ByVal NodeOutVector dense_keys,
                   @ByVal NodeOutVector dense_defaults, @ByVal DataTypeVector sparse_types, @ByVal TensorShapeVector dense_shapes,
                   @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ParseExample(Node serialized, Node names, @ByVal NodeOutVector sparse_keys, @ByVal NodeOutVector dense_keys,
                   @ByVal NodeOutVector dense_defaults, @ByVal DataTypeVector sparse_types, @ByVal TensorShapeVector dense_shapes,
                   @Const @ByRef GraphDefBuilder.Options opts);

// Converts each string in the input Tensor to the specified numeric type.
//
// (Note that int32 overflow results in an error while float overflow
// results in a rounded value.)
//
// Arguments:
// * opts:
//   .WithAttr("out_type", DataType): Defaults to DT_FLOAT.
//     The numeric type to interpret each string in string_tensor as.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A Tensor of the same shape as the input `string_tensor`.
@Namespace("tensorflow::ops") public static native Node StringToNumber(@ByVal NodeBuilder.NodeOut string_tensor, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node StringToNumber(Node string_tensor, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_PARSING_OPS_H_


// Parsed from tensorflow/cc/ops/random_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_RANDOM_OPS_H_
// #define TENSORFLOW_CC_OPS_RANDOM_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Randomly shuffles a tensor along its first dimension.
//
//   The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
//   to one and only one `output[i]`. For example, a mapping that might occur for a
//   3x2 tensor is:
//
// ```prettyprint
// [[1, 2],       [[5, 6],
//  [3, 4],  ==>   [1, 2],
//  [5, 6]]        [3, 4]]
// ```
//
// Arguments:
// * value: The tensor to be shuffled.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of same shape and type as `value`, shuffled along its first
// dimension.
@Namespace("tensorflow::ops") public static native Node RandomShuffle(@ByVal NodeBuilder.NodeOut value, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node RandomShuffle(Node value, @Const @ByRef GraphDefBuilder.Options opts);

// Outputs random values from a normal distribution.
//
// The generated values will have mean 0 and standard deviation 1.
//
// Arguments:
// * shape: The shape of the output tensor.
// * dtype: The type of the output.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of the specified shape filled with random normal values.
@Namespace("tensorflow::ops") public static native Node RandomStandardNormal(@ByVal NodeBuilder.NodeOut shape, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node RandomStandardNormal(Node shape, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);

// Outputs random values from a uniform distribution.
//
// The generated values follow a uniform distribution in the range `[0, 1)`. The
// lower bound 0 is included in the range, while the upper bound 1 is excluded.
//
// Arguments:
// * shape: The shape of the output tensor.
// * dtype: The type of the output.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of the specified shape filled with uniform random values.
@Namespace("tensorflow::ops") public static native Node RandomUniform(@ByVal NodeBuilder.NodeOut shape, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node RandomUniform(Node shape, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);

// Outputs random integers from a uniform distribution.
//
// The generated values are uniform integers in the range `[minval, maxval)`.
// The lower bound `minval` is included in the range, while the upper bound
// `maxval` is excluded.
//
// The random integers are slightly biased unless `maxval - minval` is an exact
// power of two.  The bias is small for values of `maxval - minval` significantly
// smaller than the range of the output (either `2^32` or `2^64`).
//
// Arguments:
// * shape: The shape of the output tensor.
// * minval: 0-D.  Inclusive lower bound on the generated integers.
// * maxval: 0-D.  Exclusive upper bound on the generated integers.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of the specified shape filled with uniform random integers.
@Namespace("tensorflow::ops") public static native Node RandomUniformInt(@ByVal NodeBuilder.NodeOut shape, @ByVal NodeBuilder.NodeOut minval, @ByVal NodeBuilder.NodeOut maxval, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node RandomUniformInt(Node shape, Node minval, Node maxval, @Const @ByRef GraphDefBuilder.Options opts);

// Outputs random values from a truncated normal distribution.
//
// The generated values follow a normal distribution with mean 0 and standard
// deviation 1, except that values whose magnitude is more than 2 standard
// deviations from the mean are dropped and re-picked.
//
// Arguments:
// * shape: The shape of the output tensor.
// * dtype: The type of the output.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of the specified shape filled with random truncated normal
// values.
@Namespace("tensorflow::ops") public static native Node TruncatedNormal(@ByVal NodeBuilder.NodeOut shape, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node TruncatedNormal(Node shape, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_RANDOM_OPS_H_


// Parsed from tensorflow/cc/ops/sparse_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_SPARSE_OPS_H_
// #define TENSORFLOW_CC_OPS_SPARSE_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Concatenates a list of `SparseTensor` along the specified dimension.
//
// Concatenation is with respect to the dense versions of these sparse tensors.
// It is assumed that each input is a `SparseTensor` whose elements are ordered
// along increasing dimension number.
//
// All inputs' shapes must match, except for the concat dimension.  The
// `indices`, `values`, and `shapes` lists must have the same length.
//
// The output shape is identical to the inputs', except along the concat
// dimension, where it is the sum of the inputs' sizes along that dimension.
//
// The output elements will be resorted to preserve the sort order along
// increasing dimension number.
//
// This op runs in `O(M log M)` time, where `M` is the total number of non-empty
// values across all inputs. This is due to the need for an internal sort in
// order to concatenate efficiently across an arbitrary dimension.
//
// For example, if `concat_dim = 1` and the inputs are
//
//     sp_inputs[0]: shape = [2, 3]
//     [0, 2]: "a"
//     [1, 0]: "b"
//     [1, 1]: "c"
//
//     sp_inputs[1]: shape = [2, 4]
//     [0, 1]: "d"
//     [0, 2]: "e"
//
// then the output will be
//
//     shape = [2, 7]
//     [0, 2]: "a"
//     [0, 4]: "d"
//     [0, 5]: "e"
//     [1, 0]: "b"
//     [1, 1]: "c"
//
// Graphically this is equivalent to doing
//
//     [    a] concat [  d e  ] = [    a   d e  ]
//     [b c  ]        [       ]   [b c          ]
//
// Arguments:
// * indices: 2-D.  Indices of each input `SparseTensor`.
// * values: 1-D.  Non-empty values of each `SparseTensor`.
// * shapes: 1-D.  Shapes of each `SparseTensor`.
// * concat_dim: Dimension to concatenate along.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
// * output_values: 1-D.  Non-empty values of the concatenated `SparseTensor`.
// * output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
@Namespace("tensorflow::ops") public static native Node SparseConcat(@ByVal NodeOutVector indices, @ByVal NodeOutVector values, @ByVal NodeOutVector shapes, @Cast("tensorflow::int64") long concat_dim,
                   @Const @ByRef GraphDefBuilder.Options opts);

// Reorders a SparseTensor into the canonical, row-major ordering.
//
// Note that by convention, all sparse ops preserve the canonical ordering along
// increasing dimension number. The only time ordering can be violated is during
// manual manipulation of the indices and values vectors to add entries.
//
// Reordering does not affect the shape of the SparseTensor.
//
// If the tensor has rank `R` and `N` non-empty values, `input_indices` has
// shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.
//
// Arguments:
// * input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
// SparseTensor, possibly not in canonical ordering.
// * input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
// * input_shape: 1-D.  Shape of the input SparseTensor.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output_indices: 2-D.  `N x R` matrix with the same indices as input_indices, but
// in canonical row-major ordering.
// * output_values: 1-D.  `N` non-empty values corresponding to `output_indices`.
@Namespace("tensorflow::ops") public static native Node SparseReorder(@ByVal NodeBuilder.NodeOut input_indices, @ByVal NodeBuilder.NodeOut input_values, @ByVal NodeBuilder.NodeOut input_shape, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SparseReorder(Node input_indices, Node input_values, Node input_shape, @Const @ByRef GraphDefBuilder.Options opts);

// Converts a sparse representation into a dense tensor.
//
// Builds an array `dense` with shape `output_shape` such that
//
// ```prettyprint
// # If sparse_indices is scalar
// dense[i] = (i == sparse_indices ? sparse_values : default_value)
//
// # If sparse_indices is a vector, then for each i
// dense[sparse_indices[i]] = sparse_values[i]
//
// # If sparse_indices is an n by d matrix, then for each i in [0, n)
// dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
// ```
//
// All other values in `dense` are set to `default_value`.  If `sparse_values` is a
// scalar, all sparse indices are set to this single value.
//
// Arguments:
// * sparse_indices: 0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
// index where `sparse_values[i]` will be placed.
// * output_shape: 1-D.  Shape of the dense output tensor.
// * sparse_values: 1-D.  Values corresponding to each row of `sparse_indices`,
// or a scalar value to be used for all sparse indices.
// * default_value: Scalar value to set for indices not specified in
// `sparse_indices`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Dense output tensor of shape `output_shape`.
@Namespace("tensorflow::ops") public static native Node SparseToDense(@ByVal NodeBuilder.NodeOut sparse_indices, @ByVal NodeBuilder.NodeOut output_shape, @ByVal NodeBuilder.NodeOut sparse_values, @ByVal NodeBuilder.NodeOut default_value, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SparseToDense(Node sparse_indices, Node output_shape, Node sparse_values, Node default_value, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_SPARSE_OPS_H_


// Parsed from tensorflow/cc/ops/state_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_STATE_OPS_H_
// #define TENSORFLOW_CC_OPS_STATE_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Update 'ref' by assigning 'value' to it.
//
// This operation outputs "ref" after the assignment is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Arguments:
// * ref: Should be from a `Variable` node. May be uninitialized.
// * value: The value to be assigned to the variable.
// * opts:
//   .WithAttr("validate_shape", bool): Defaults to true.
//     If true, the operation will validate that the shape
// of 'value' matches the shape of the Tensor being assigned to.  If false,
// 'ref' will take on the shape of 'value'.
//   .WithAttr("use_locking", bool): Defaults to true.
//     If True, the assignment will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "ref".  Returned as a convenience for operations that want
// to use the new value after the variable has been reset.
@Namespace("tensorflow::ops") public static native Node Assign(@ByVal NodeBuilder.NodeOut ref, @ByVal NodeBuilder.NodeOut value, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node Assign(Node ref, Node value, @Const @ByRef GraphDefBuilder.Options opts);

// Update 'ref' by adding 'value' to it.
//
// This operation outputs "ref" after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * value: The value to be added to the variable.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the addition will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "ref".  Returned as a convenience for operations that want
// to use the new value after the variable has been updated.
@Namespace("tensorflow::ops") public static native Node AssignAdd(@ByVal NodeBuilder.NodeOut ref, @ByVal NodeBuilder.NodeOut value, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AssignAdd(Node ref, Node value, @Const @ByRef GraphDefBuilder.Options opts);

// Update 'ref' by subtracting 'value' from it.
//
// This operation outputs "ref" after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * value: The value to be subtracted to the variable.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the subtraction will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "ref".  Returned as a convenience for operations that want
// to use the new value after the variable has been updated.
@Namespace("tensorflow::ops") public static native Node AssignSub(@ByVal NodeBuilder.NodeOut ref, @ByVal NodeBuilder.NodeOut value, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node AssignSub(Node ref, Node value, @Const @ByRef GraphDefBuilder.Options opts);

// Increments 'ref' until it reaches 'limit'.
//
// This operation outputs "ref" after the update is done.  This makes it
// easier to chain operations that need to use the updated value.
//
// Arguments:
// * ref: Should be from a scalar `Variable` node.
// * limit: If incrementing ref would bring it above limit, instead generates an
// 'OutOfRange' error.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A copy of the input before increment. If nothing else modifies the
// input, the values produced will all be distinct.
@Namespace("tensorflow::ops") public static native Node CountUpTo(@ByVal NodeBuilder.NodeOut ref, @Cast("tensorflow::int64") long limit, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node CountUpTo(Node ref, @Cast("tensorflow::int64") long limit, @Const @ByRef GraphDefBuilder.Options opts);

// Destroys the temporary variable and returns its final value.
//
// Sets output to the value of the Tensor pointed to by 'ref', then destroys
// the temporary variable called 'var_name'.
// All other uses of 'ref' *must* have executed before this op.
// This is typically achieved by chaining the ref through each assign op, or by
// using control dependencies.
//
// Outputs the final value of the tensor pointed to by 'ref'.
//
// Arguments:
// * ref: A reference to the temporary variable tensor.
// * var_name: Name of the temporary variable, usually the name of the matching
// 'TemporaryVariable' op.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node DestroyTemporaryVariable(@ByVal NodeBuilder.NodeOut ref, @StringPiece BytePointer var_name, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node DestroyTemporaryVariable(Node ref, @StringPiece String var_name, @Const @ByRef GraphDefBuilder.Options opts);

// Adds sparse updates to a variable reference.
//
// This operation computes
//
//     # Scalar indices
//     ref[indices, ...] += updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] += updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions add.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/ScatterAdd.png" alt>
// </div>
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * indices: A tensor of indices into the first dimension of `ref`.
// * updates: A tensor of updated values to add to `ref`.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the addition will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as `ref`.  Returned as a convenience for operations that want
// to use the updated values after the update is done.
@Namespace("tensorflow::ops") public static native Node ScatterAdd(@ByVal NodeBuilder.NodeOut ref, @ByVal NodeBuilder.NodeOut indices, @ByVal NodeBuilder.NodeOut updates, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ScatterAdd(Node ref, Node indices, Node updates, @Const @ByRef GraphDefBuilder.Options opts);

// Subtracts sparse updates to a variable reference.
//
//     # Scalar indices
//     ref[indices, ...] -= updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] -= updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their (negated) contributions add.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/ScatterSub.png" alt>
// </div>
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * indices: A tensor of indices into the first dimension of `ref`.
// * updates: A tensor of updated values to subtract from `ref`.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the subtraction will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as `ref`.  Returned as a convenience for operations that want
// to use the updated values after the update is done.
@Namespace("tensorflow::ops") public static native Node ScatterSub(@ByVal NodeBuilder.NodeOut ref, @ByVal NodeBuilder.NodeOut indices, @ByVal NodeBuilder.NodeOut updates, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ScatterSub(Node ref, Node indices, Node updates, @Const @ByRef GraphDefBuilder.Options opts);

// Applies sparse updates to a variable reference.
//
// This operation computes
//
//     # Scalar indices
//     ref[indices, ...] = updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] = updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// If `indices` contains duplicate entries, lexicographically later entries
// override earlier entries.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../images/ScatterUpdate.png" alt>
// </div>
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * indices: A tensor of indices into the first dimension of `ref`.
// * updates: A tensor of updated values to store in `ref`.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to true.
//     If True, the assignment will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as `ref`.  Returned as a convenience for operations that want
// to use the updated values after the update is done.
@Namespace("tensorflow::ops") public static native Node ScatterUpdate(@ByVal NodeBuilder.NodeOut ref, @ByVal NodeBuilder.NodeOut indices, @ByVal NodeBuilder.NodeOut updates, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ScatterUpdate(Node ref, Node indices, Node updates, @Const @ByRef GraphDefBuilder.Options opts);

// Returns a tensor that may be mutated, but only persists within a single step.
//
// This is an experimental op for internal use only and it is possible to use this
// op in unsafe ways.  DO NOT USE unless you fully understand the risks.
//
// It is the caller's responsibility to ensure that 'ref' is eventually passed to a
// matching 'DestroyTemporaryVariable' op after all other uses have completed.
//
// Outputs a ref to the tensor state so it may be read or modified.
//
//   E.g.
//       var = state_ops._temporary_variable([1, 2], types.float_)
//       var_name = var.op.name
//       var = state_ops.assign(var, [[4.0, 5.0]])
//       var = state_ops.assign_add(var, [[6.0, 7.0]])
//       final = state_ops._destroy_temporary_variable(var, var_name=var_name)
//
// Arguments:
// * shape: The shape of the variable tensor.
// * dtype: The type of elements in the variable tensor.
// * opts:
//   .WithAttr("var_name", StringPiece): Defaults to "".
//     Overrides the name used for the temporary variable resource. Default
// value is the name of the 'TemporaryVariable' op (which is guaranteed unique).
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A reference to the variable tensor.
@Namespace("tensorflow::ops") public static native Node TemporaryVariable(@ByVal TensorShape shape, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);

// Holds state in the form of a tensor that persists across steps.
//
// Outputs a ref to the tensor state so it may be read or modified.
// TODO(zhifengc/mrry): Adds a pointer to a more detail document
// about sharing states in tensorflow.
//
// Arguments:
// * shape: The shape of the variable tensor.
// * dtype: The type of elements in the variable tensor.
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this variable is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this variable is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A reference to the variable tensor.
@Namespace("tensorflow::ops") public static native Node Variable(@ByVal TensorShape shape, @Cast("tensorflow::DataType") int dtype, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_STATE_OPS_H_


// Parsed from tensorflow/cc/ops/string_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_STRING_OPS_H_
// #define TENSORFLOW_CC_OPS_STRING_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Converts each string in the input Tensor to its hash mod by a number of buckets.
//
// The hash function is deterministic on the content of the string within the
// process.
//
// Note that the hash function may change from time to time.
//
// Arguments:
// * num_buckets: The number of buckets.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A Tensor of the same shape as the input `string_tensor`.
@Namespace("tensorflow::ops") public static native Node StringToHashBucket(@ByVal NodeBuilder.NodeOut string_tensor, @Cast("tensorflow::int64") long num_buckets, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node StringToHashBucket(Node string_tensor, @Cast("tensorflow::int64") long num_buckets, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_STRING_OPS_H_


// Parsed from tensorflow/cc/ops/summary_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_SUMMARY_OPS_H_
// #define TENSORFLOW_CC_OPS_SUMMARY_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Outputs a `Summary` protocol buffer with a histogram.
//
// The generated
// [`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
// has one summary value containing a histogram for `values`.
//
// This op reports an `OutOfRange` error if any value is not finite.
//
// Arguments:
// * tag: Scalar.  Tag to use for the `Summary.Value`.
// * values: Any shape. Values to use to build the histogram.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Scalar. Serialized `Summary` protocol buffer.
@Namespace("tensorflow::ops") public static native Node HistogramSummary(@ByVal NodeBuilder.NodeOut tag, @ByVal NodeBuilder.NodeOut values, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node HistogramSummary(Node tag, Node values, @Const @ByRef GraphDefBuilder.Options opts);

// Outputs a `Summary` protocol buffer with images.
//
// The summary has up to `max_images` summary values containing images. The
// images are built from `tensor` which must be 4-D with shape `[batch_size,
// height, width, channels]` and where `channels` can be:
//
// *  1: `tensor` is interpreted as Grayscale.
// *  3: `tensor` is interpreted as RGB.
// *  4: `tensor` is interpreted as RGBA.
//
// The images have the same number of channels as the input tensor. Their values
// are normalized, one image at a time, to fit in the range `[0, 255]`.  The
// op uses two different normalization algorithms:
//
// *  If the input values are all positive, they are rescaled so the largest one
//    is 255.
//
// *  If any input value is negative, the values are shifted so input value 0.0
//    is at 127.  They are then rescaled so that either the smallest value is 0,
//    or the largest one is 255.
//
// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
// build the `tag` of the summary values:
//
// *  If `max_images` is 1, the summary value tag is '*tag*/image'.
// *  If `max_images` is greater than 1, the summary value tags are
//    generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.
//
// The `bad_color` argument is the color to use in the generated images for
// non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
// Each element must be in the range `[0, 255]` (It represents the value of a
// pixel in the output image).  Non-finite values in the input tensor are
// replaced by this tensor in the output image.  The default value is the color
// red.
//
// Arguments:
// * tag: Scalar. Used to build the `tag` attribute of the summary values.
// * tensor: 4-D of shape `[batch_size, height, width, channels]` where
// `channels` is 1, 3, or 4.
// * opts:
//   .WithAttr("max_images", int64): Defaults to 3.
//     Max number of batch elements to generate images for.
//   .WithAttr("bad_color", const Tensor&): Defaults to Tensor<type: uint8 shape: [4] values: 255 0 0...>.
//     Color to use for pixels with non-finite values.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Scalar. Serialized `Summary` protocol buffer.
@Namespace("tensorflow::ops") public static native Node ImageSummary(@ByVal NodeBuilder.NodeOut tag, @ByVal NodeBuilder.NodeOut tensor, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ImageSummary(Node tag, Node tensor, @Const @ByRef GraphDefBuilder.Options opts);

// Merges summaries.
//
// This op creates a
// [`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
// protocol buffer that contains the union of all the values in the input
// summaries.
//
// When the Op is run, it reports an `InvalidArgument` error if multiple values
// in the summaries to merge use the same tag.
//
// Arguments:
// * inputs: Can be of any shape.  Each must contain serialized `Summary` protocol
// buffers.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Scalar. Serialized `Summary` protocol buffer.
@Namespace("tensorflow::ops") public static native Node MergeSummary(@ByVal NodeOutVector inputs, @Const @ByRef GraphDefBuilder.Options opts);

// Outputs a `Summary` protocol buffer with scalar values.
//
// The input `tags` and `values` must have the same shape.  The generated summary
// has a summary value for each tag-value pair in `tags` and `values`.
//
// Arguments:
// * tags: 1-D. Tags for the summary.
// * values: 1-D, same size as `tags.  Values for the summary.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Scalar.  Serialized `Summary` protocol buffer.
@Namespace("tensorflow::ops") public static native Node ScalarSummary(@ByVal NodeBuilder.NodeOut tags, @ByVal NodeBuilder.NodeOut values, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ScalarSummary(Node tags, Node values, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_SUMMARY_OPS_H_


// Parsed from tensorflow/cc/ops/training_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_TRAINING_OPS_H_
// #define TENSORFLOW_CC_OPS_TRAINING_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Update '*var' according to the adagrad scheme.
//
// accum += grad * grad
// var -= lr * grad * (1 / sqrt(accum))
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * lr: Scaling factor. Must be a scalar.
// * grad: The gradient.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
@Namespace("tensorflow::ops") public static native Node ApplyAdagrad(@ByVal NodeBuilder.NodeOut var, @ByVal NodeBuilder.NodeOut accum, @ByVal NodeBuilder.NodeOut lr, @ByVal NodeBuilder.NodeOut grad, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ApplyAdagrad(Node var, Node accum, Node lr, Node grad, @Const @ByRef GraphDefBuilder.Options opts);

// Update '*var' according to the Adam algorithm.
//
// lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
// m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
// v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
// variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
//
// Arguments:
// * var: Should be from a Variable().
// * m: Should be from a Variable().
// * v: Should be from a Variable().
// * beta1_power: Must be a scalar.
// * beta2_power: Must be a scalar.
// * lr: Scaling factor. Must be a scalar.
// * beta1: Momentum factor. Must be a scalar.
// * beta2: Momentum factor. Must be a scalar.
// * epsilon: Ridge term. Must be a scalar.
// * grad: The gradient.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var, m, and v tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
@Namespace("tensorflow::ops") public static native Node ApplyAdam(@ByVal NodeBuilder.NodeOut var, @ByVal NodeBuilder.NodeOut m, @ByVal NodeBuilder.NodeOut v, @ByVal NodeBuilder.NodeOut beta1_power, @ByVal NodeBuilder.NodeOut beta2_power, @ByVal NodeBuilder.NodeOut lr, @ByVal NodeBuilder.NodeOut beta1, @ByVal NodeBuilder.NodeOut beta2, @ByVal NodeBuilder.NodeOut epsilon, @ByVal NodeBuilder.NodeOut grad, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ApplyAdam(Node var, Node m, Node v, Node beta1_power, Node beta2_power, Node lr, Node beta1, Node beta2, Node epsilon, Node grad, @Const @ByRef GraphDefBuilder.Options opts);

// Update '*var' by subtracting 'alpha' * 'delta' from it.
//
// Arguments:
// * var: Should be from a Variable().
// * alpha: Scaling factor. Must be a scalar.
// * delta: The change.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the subtraction will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
@Namespace("tensorflow::ops") public static native Node ApplyGradientDescent(@ByVal NodeBuilder.NodeOut var, @ByVal NodeBuilder.NodeOut alpha, @ByVal NodeBuilder.NodeOut delta, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ApplyGradientDescent(Node var, Node alpha, Node delta, @Const @ByRef GraphDefBuilder.Options opts);

// Update '*var' according to the momentum scheme.
//
// accum = accum * momentum + grad
// var -= lr * accum
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * lr: Scaling factor. Must be a scalar.
// * grad: The gradient.
// * momentum: Momentum. Must be a scalar.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
@Namespace("tensorflow::ops") public static native Node ApplyMomentum(@ByVal NodeBuilder.NodeOut var, @ByVal NodeBuilder.NodeOut accum, @ByVal NodeBuilder.NodeOut lr, @ByVal NodeBuilder.NodeOut grad,
                    @ByVal NodeBuilder.NodeOut momentum, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ApplyMomentum(Node var, Node accum, Node lr, Node grad,
                    Node momentum, @Const @ByRef GraphDefBuilder.Options opts);

// Update '*var' according to the RMSProp algorithm.
//
// mean_square = decay * mean_square + (1-decay) * gradient ** 2
// Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
//
// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
// var <- var - mom
//
// Arguments:
// * var: Should be from a Variable().
// * ms: Should be from a Variable().
// * mom: Should be from a Variable().
// * lr: Scaling factor. Must be a scalar.
// * rho: Decay rate. Must be a scalar.
// * epsilon: Ridge term. Must be a scalar.
// * grad: The gradient.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var, m, and v tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
@Namespace("tensorflow::ops") public static native Node ApplyRMSProp(@ByVal NodeBuilder.NodeOut var, @ByVal NodeBuilder.NodeOut ms, @ByVal NodeBuilder.NodeOut mom, @ByVal NodeBuilder.NodeOut lr, @ByVal NodeBuilder.NodeOut rho, @ByVal NodeBuilder.NodeOut momentum, @ByVal NodeBuilder.NodeOut epsilon, @ByVal NodeBuilder.NodeOut grad, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node ApplyRMSProp(Node var, Node ms, Node mom, Node lr, Node rho, Node momentum, Node epsilon, Node grad, @Const @ByRef GraphDefBuilder.Options opts);

// Update relevant entries in '*var' and '*accum' according to the adagrad scheme.
//
// That is for rows we have grad for, we update var and accum as follows:
// accum += grad * grad
// var -= lr * grad * (1 / sqrt(accum))
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * lr: Learning rate. Must be a scalar.
// * grad: The gradient.
// * indices: A vector of indices into the first dimension of var and accum.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
@Namespace("tensorflow::ops") public static native Node SparseApplyAdagrad(@ByVal NodeBuilder.NodeOut var, @ByVal NodeBuilder.NodeOut accum, @ByVal NodeBuilder.NodeOut lr, @ByVal NodeBuilder.NodeOut grad,
                         @ByVal NodeBuilder.NodeOut indices, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SparseApplyAdagrad(Node var, Node accum, Node lr, Node grad,
                         Node indices, @Const @ByRef GraphDefBuilder.Options opts);

// Update relevant entries in '*var' and '*accum' according to the momentum scheme.
//
// That is for rows we have grad for, we update var and accum as follows:
//
// accum = accum * momentum + grad
// var -= lr * accum
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * lr: Learning rate. Must be a scalar.
// * grad: The gradient.
// * indices: A vector of indices into the first dimension of var and accum.
// * momentum: Momentum. Must be a scalar.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
@Namespace("tensorflow::ops") public static native Node SparseApplyMomentum(@ByVal NodeBuilder.NodeOut var, @ByVal NodeBuilder.NodeOut accum, @ByVal NodeBuilder.NodeOut lr, @ByVal NodeBuilder.NodeOut grad,
                          @ByVal NodeBuilder.NodeOut indices, @ByVal NodeBuilder.NodeOut momentum, @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node SparseApplyMomentum(Node var, Node accum, Node lr, Node grad,
                          Node indices, Node momentum, @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_TRAINING_OPS_H_


// Parsed from tensorflow/cc/ops/user_ops.h

// This file is MACHINE GENERATED! Do not edit.

// #ifndef TENSORFLOW_CC_OPS_USER_OPS_H_
// #define TENSORFLOW_CC_OPS_USER_OPS_H_

// #include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/gtl/array_slice.h"
// #include "tensorflow/core/public/tensor.h"
// #include "tensorflow/core/public/tensor_shape.h"

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Output a fact about factorials.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control depencies on the specified Node(s).
//
// Returns a pointer to the created Node.
@Namespace("tensorflow::ops") public static native Node Fact(@Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_CC_OPS_USER_OPS_H_


}
