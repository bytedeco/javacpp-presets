// Targeted by JavaCPP version 1.2-SNAPSHOT

package org.bytedeco.javacpp;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class tensorflow extends org.bytedeco.javacpp.presets.tensorflow {
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
    public StringTensorPairVector()       { allocate();  }
    public StringTensorPairVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator=") @ByRef StringTensorPairVector put(@ByRef StringTensorPairVector x);

    public native long size();
    public native void resize(@Cast("size_t") long n);

    @Index public native @StdString BytePointer first(@Cast("size_t") long i); public native StringTensorPairVector first(@Cast("size_t") long i, BytePointer first);
    @Index public native @ByRef Tensor second(@Cast("size_t") long i);  public native StringTensorPairVector second(@Cast("size_t") long i, Tensor second);
}

@NoOffset @Name("std::pair<tensorflow::EdgeSet::const_iterator,bool>") public static class EdgeSetBoolPair extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EdgeSetBoolPair(Pointer p) { super(p); }
    public EdgeSetBoolPair()       { allocate();  }
    private native void allocate();
    public native @Name("operator=") @ByRef EdgeSetBoolPair put(@ByRef EdgeSetBoolPair x);


    @MemberGetter public native @ByRef EdgeSetIterator first(); public native EdgeSetBoolPair first(EdgeSetIterator first);
    @MemberGetter public native @Cast("bool") boolean second();  public native EdgeSetBoolPair second(boolean second);
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


// Parsed from tensorflow/core/platform/default/dynamic_annotations.h

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

// #ifndef THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_DYNAMIC_ANNOTATIONS_H_
// #define THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_DYNAMIC_ANNOTATIONS_H_

// Do nothing for this platform
// #define TF_ANNOTATE_MEMORY_IS_INITIALIZED(ptr, bytes)
//   do {
//   } while (0)

// #endif  // THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_DYNAMIC_ANNOTATIONS_H_


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


// Parsed from tensorflow/core/lib/core/stringpiece.h

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

// StringPiece is a simple structure containing a pointer into some external
// storage and a size.  The user of a StringPiece must ensure that the slice
// is not used after the corresponding external storage has been
// deallocated.
//
// Multiple threads can invoke const methods on a StringPiece without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same StringPiece must use
// external synchronization.

// #ifndef TENSORFLOW_LIB_CORE_STRINGPIECE_H_
// #define TENSORFLOW_LIB_CORE_STRINGPIECE_H_

// #include <assert.h>
// #include <stddef.h>
// #include <string.h>
// #include <iosfwd>
// #include <string>
// #include "tensorflow/core/platform/port.h"

@Namespace("tensorflow") @NoOffset public static class StringPiece extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringPiece(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(int)}. */
    public StringPiece(int size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(int size);
    @Override public StringPiece position(int position) {
        return (StringPiece)super.position(position);
    }


  // Create an empty slice.
  public StringPiece() { super((Pointer)null); allocate(); }
  private native void allocate();

  // Create a slice that refers to d[0,n-1].
  public StringPiece(@Cast("const char*") BytePointer d, @Cast("size_t") long n) { super((Pointer)null); allocate(d, n); }
  private native void allocate(@Cast("const char*") BytePointer d, @Cast("size_t") long n);
  public StringPiece(String d, @Cast("size_t") long n) { super((Pointer)null); allocate(d, n); }
  private native void allocate(String d, @Cast("size_t") long n);

  // Create a slice that refers to the contents of "s"
  public StringPiece(@StdString BytePointer s) { super((Pointer)null); allocate(s); }
  private native void allocate(@StdString BytePointer s);
  public StringPiece(@StdString String s) { super((Pointer)null); allocate(s); }
  private native void allocate(@StdString String s);

  // Create a slice that refers to s[0,strlen(s)-1]

  public native void set(@Const Pointer data, @Cast("size_t") long len);

  // Return a pointer to the beginning of the referenced data
  public native @Cast("const char*") BytePointer data();

  // Return the length (in bytes) of the referenced data
  public native @Cast("size_t") long size();

  // Return true iff the length of the referenced data is zero
  public native @Cast("bool") boolean empty();
  public native @Cast("tensorflow::StringPiece::iterator") BytePointer begin();
  public native @Cast("tensorflow::StringPiece::iterator") BytePointer end();

  @MemberGetter public static native @Cast("const size_t") long npos();
  public static final long npos = npos();

  // Return the ith byte in the referenced data.
  // REQUIRES: n < size()
  public native @Cast("char") @Name("operator []") byte get(@Cast("size_t") long n);

  // Change this slice to refer to an empty array
  public native void clear();

  // Drop the first "n" bytes from this slice.
  public native void remove_prefix(@Cast("size_t") long n);

  public native void remove_suffix(@Cast("size_t") long n);

  public native @Cast("size_t") long find(@Cast("char") byte c, @Cast("size_t") long pos/*=0*/);
  public native @Cast("size_t") long find(@Cast("char") byte c);
  public native @Cast("size_t") long rfind(@Cast("char") byte c, @Cast("size_t") long pos/*=tensorflow::StringPiece::npos*/);
  public native @Cast("size_t") long rfind(@Cast("char") byte c);
  public native @Cast("bool") boolean contains(@ByVal StringPiece s);

  // Checks whether StringPiece starts with x and if so advances the beginning
  // of it to past the match.  It's basically a shortcut for starts_with
  // followed by remove_prefix.
  public native @Cast("bool") boolean Consume(@ByVal StringPiece x);

  public native @ByVal StringPiece substr(@Cast("size_t") long pos, @Cast("size_t") long n/*=tensorflow::StringPiece::npos*/);
  public native @ByVal StringPiece substr(@Cast("size_t") long pos);

  public static class Hasher extends Pointer {
      static { Loader.load(); }
      /** Default native constructor. */
      public Hasher() { super((Pointer)null); allocate(); }
      /** Native array allocator. Access with {@link Pointer#position(int)}. */
      public Hasher(int size) { super((Pointer)null); allocateArray(size); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public Hasher(Pointer p) { super(p); }
      private native void allocate();
      private native void allocateArray(int size);
      @Override public Hasher position(int position) {
          return (Hasher)super.position(position);
      }
  
    public native @Cast("size_t") @Name("operator ()") long apply(@ByVal StringPiece arg);
  }

  // Return a string that contains the copy of the referenced data.
  public native @StdString BytePointer ToString();

  // Three-way comparison.  Returns value:
  //   <  0 iff "*this" <  "b",
  //   == 0 iff "*this" == "b",
  //   >  0 iff "*this" >  "b"
  public native int compare(@ByVal StringPiece b);

  // Return true iff "x" is a prefix of "*this"
  public native @Cast("bool") boolean starts_with(@ByVal StringPiece x);
  // Return true iff "x" is a suffix of "*this"
  public native @Cast("bool") boolean ends_with(@ByVal StringPiece x);
}

@Namespace("tensorflow") public static native @Cast("bool") @Name("operator ==") boolean equals(@ByVal StringPiece x, @ByVal StringPiece y);

@Namespace("tensorflow") public static native @Cast("bool") @Name("operator !=") boolean notEquals(@ByVal StringPiece x, @ByVal StringPiece y);

@Namespace("tensorflow") public static native @Cast("bool") @Name("operator <") boolean lessThan(@ByVal StringPiece x, @ByVal StringPiece y);
@Namespace("tensorflow") public static native @Cast("bool") @Name("operator >") boolean greaterThan(@ByVal StringPiece x, @ByVal StringPiece y);
@Namespace("tensorflow") public static native @Cast("bool") @Name("operator <=") boolean lessThanEquals(@ByVal StringPiece x, @ByVal StringPiece y);
@Namespace("tensorflow") public static native @Cast("bool") @Name("operator >=") boolean greaterThanEquals(@ByVal StringPiece x, @ByVal StringPiece y);



// allow StringPiece to be logged
@Namespace("tensorflow") public static native @Cast("std::ostream*") @ByRef @Name("operator <<") Pointer shiftLeft(@Cast("std::ostream*") @ByRef Pointer o, @ByVal StringPiece piece);

  // namespace tensorflow

// #endif  // TENSORFLOW_LIB_CORE_STRINGPIECE_H_


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
  public Status(@Cast("tensorflow::error::Code") int code, @ByVal StringPiece msg) { super((Pointer)null); allocate(code, msg); }
  private native void allocate(@Cast("tensorflow::error::Code") int code, @ByVal StringPiece msg);

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

// #define TF_CHECK_OK(val) CHECK_EQ(::tensorflow::Status::OK(), (val))
// #define TF_QCHECK_OK(val) QCHECK_EQ(::tensorflow::Status::OK(), (val))

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

  /** Stores the size of fname in *file_size. */
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


  /** \brief Reads up to "n" bytes from the file starting at "offset".
   * 
   *  "scratch[0..n-1]" may be written by this routine.  Sets "*result"
   *  to the data that was read (including if fewer than "n" bytes were
   *  successfully read).  May set "*result" to point at data in
   *  "scratch[0..n-1]", so "scratch[0..n-1]" must be live when
   *  "*result" is used.
   * 
   *  On OK returned status: "n" bytes have been stored in "*result".
   *  On non-OK returned status: [0..n] bytes have been stored in "*result".
   * 
   *  Returns {@code OUT_OF_RANGE} if fewer than n bytes were stored in "*result"
   *  because of EOF.
   * 
   *  Safe for concurrent use by multiple threads. */
  public native @ByVal Status Read(@Cast("tensorflow::uint64") long offset, @Cast("size_t") long n, StringPiece result,
                        @Cast("char*") BytePointer scratch);
  public native @ByVal Status Read(@Cast("tensorflow::uint64") long offset, @Cast("size_t") long n, StringPiece result,
                        @Cast("char*") ByteBuffer scratch);
  public native @ByVal Status Read(@Cast("tensorflow::uint64") long offset, @Cast("size_t") long n, StringPiece result,
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


  public native @ByVal Status Append(@Const @ByRef StringPiece data);
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

/** A utility routine: reads contents of named file into *data */
@Namespace("tensorflow") public static native @ByVal Status ReadFileToString(Env env, @StdString BytePointer fname, @StdString @Cast({"char*", "std::string*"}) BytePointer data);
@Namespace("tensorflow") public static native @ByVal Status ReadFileToString(Env env, @StdString String fname, @StdString @Cast({"char*", "std::string*"}) BytePointer data);

/** A utility routine: write contents of "data" to file named "fname"
 *  (overwriting existing contents, if any). */
@Namespace("tensorflow") public static native @ByVal Status WriteStringToFile(Env env, @StdString BytePointer fname,
                         @Const @ByRef StringPiece data);
@Namespace("tensorflow") public static native @ByVal Status WriteStringToFile(Env env, @StdString String fname,
                         @Const @ByRef StringPiece data);

/** Reads contents of named file and parse as binary encoded proto data
 *  and store into *proto. */
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
  public TensorShape(@Cast({"", "const std::vector<long long>&"}) @StdVector @ByVal LongPointer dim_sizes) { super((Pointer)null); allocate(dim_sizes); }
  private native void allocate(@Cast({"", "const std::vector<long long>&"}) @StdVector @ByVal LongPointer dim_sizes);
  public TensorShape(@Cast({"", "const std::vector<long long>&"}) @StdVector @ByVal LongBuffer dim_sizes) { super((Pointer)null); allocate(dim_sizes); }
  private native void allocate(@Cast({"", "const std::vector<long long>&"}) @StdVector @ByVal LongBuffer dim_sizes);
  public TensorShape(@Cast({"", "const std::vector<long long>&"}) @StdVector @ByVal long[] dim_sizes) { super((Pointer)null); allocate(dim_sizes); }
  private native void allocate(@Cast({"", "const std::vector<long long>&"}) @StdVector @ByVal long[] dim_sizes);

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
@Namespace("tensorflow") @NoOffset public static class Tensor extends Pointer {
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
  public native @ByVal StringPiece tensor_data();
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
@Namespace("tensorflow") public static class Session extends Pointer {
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
  public OpDefBuilder(@ByVal StringPiece op_name) { super((Pointer)null); allocate(op_name); }
  private native void allocate(@ByVal StringPiece op_name);

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
  public native @ByRef OpDefBuilder Attr(@ByVal StringPiece spec);

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
  public native @ByRef OpDefBuilder Input(@ByVal StringPiece spec);
  public native @ByRef OpDefBuilder Output(@ByVal StringPiece spec);

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
  public native @ByRef OpDefBuilder Doc(@ByVal StringPiece text);

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
@Namespace("tensorflow") public static native @Cast("const tensorflow::OpDef::AttrDef*") OpDef_AttrDef FindAttr(@ByVal StringPiece name, @Const @ByRef OpDef op_def);
@Namespace("tensorflow") public static native @Cast("tensorflow::OpDef::AttrDef*") OpDef_AttrDef FindAttrMutable(@ByVal StringPiece name, OpDef op_def);

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
@Namespace("tensorflow::register_op") public static native @ByRef OpDefBuilder RegisterOp(@ByVal StringPiece name);
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

  public DeviceType(@ByVal StringPiece type) { super((Pointer)null); allocate(type); }
  private native void allocate(@ByVal StringPiece type);

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
@Namespace("tensorflow") public static native @Cast("bool") boolean DataTypeFromString(@ByVal StringPiece sp, @Cast("tensorflow::DataType*") IntPointer dt);

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
  public native @StdString BytePointer NewName(@ByVal StringPiece prefix);

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
  public native @ByRef NodeBuilder Input(@ByVal NodeOutVector src_list);

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
    public native @ByVal Options WithName(@ByVal StringPiece name);
    public native @ByVal Options WithDevice(@ByVal StringPiece device);
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
    public native @StdString BytePointer GetNameForOp(@ByVal StringPiece op);

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
@Namespace("tensorflow::ops") public static native Node UnaryOp(@StdString BytePointer op_name, @ByVal @Cast("tensorflow::ops::NodeOut*") NodeBuilder.NodeOut input,
              @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node UnaryOp(@StdString String op_name, @ByVal @Cast("tensorflow::ops::NodeOut*") NodeBuilder.NodeOut input,
              @Const @ByRef GraphDefBuilder.Options opts);

// For adding an Op with two inputs to a GraphDefBuilder.
@Namespace("tensorflow::ops") public static native Node BinaryOp(@StdString BytePointer op_name, @ByVal @Cast("tensorflow::ops::NodeOut*") NodeBuilder.NodeOut a, @ByVal @Cast("tensorflow::ops::NodeOut*") NodeBuilder.NodeOut b,
               @Const @ByRef GraphDefBuilder.Options opts);
@Namespace("tensorflow::ops") public static native Node BinaryOp(@StdString String op_name, @ByVal @Cast("tensorflow::ops::NodeOut*") NodeBuilder.NodeOut a, @ByVal @Cast("tensorflow::ops::NodeOut*") NodeBuilder.NodeOut b,
               @Const @ByRef GraphDefBuilder.Options opts);

  // namespace ops
  // namespace tensorflow

// #endif  // TENSORFLOW_GRAPH_GRAPH_DEF_BUILDER_H_


}
