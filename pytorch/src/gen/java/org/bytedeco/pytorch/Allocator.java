// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.Module;
import org.bytedeco.javacpp.annotation.Cast;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.javacpp.chrono.*;
import static org.bytedeco.javacpp.global.chrono.*;

import static org.bytedeco.pytorch.global.torch.*;


// Note [raw_allocate/raw_deallocate and Thrust]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Thrust's support for custom allocators requires us to write something
// like this:
//
//  class ThrustAllocator {
//    char* allocate(size_t);
//    void deallocate(char*, size_t);
//  };
//
// This is not good for our unique_ptr based allocator interface, as
// there is no way to get to the context when we free.
//
// However, in some cases the context is exactly the same as
// the data pointer.  In this case, we can support the "raw"
// allocate and deallocate interface.  This is what
// raw_deleter signifies.  By default, it returns a nullptr, which means that
// the raw interface is not implemented.  Be sure to implement it whenever
// possible, or the raw interface will incorrectly reported as unsupported,
// when it is actually possible.

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
@Namespace("c10") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Allocator extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Allocator(Pointer p) { super(p); }


  public native @Name("allocate") @StdMove DataPtr _allocate(@Cast("size_t") long n);

  // Clones an allocation that came from this allocator.
  //
  // To perform the copy, this function calls `copy_data`, which
  // must be implemented by derived classes.
  //
  // Note that this explicitly ignores any context that may have been
  // attached to the input data.
  //
  // Requires: input data was allocated by the same allocator.
  public native @StdMove DataPtr clone(@Const Pointer data, @Cast("std::size_t") long n);

  // Checks if DataPtr has a simple context, not wrapped with any out of the
  // ordinary contexts.
  public native @Cast("bool") boolean is_simple_data_ptr(@StdMove DataPtr data_ptr);

  // If this returns a non nullptr, it means that allocate()
  // is guaranteed to return a unique_ptr with this deleter attached;
  // it means the rawAllocate and rawDeallocate APIs are safe to use.
  // This function MUST always return the same BoundDeleter.
  public native PointerConsumer raw_deleter();
  public native Pointer raw_allocate(@Cast("size_t") long n);
  public native void raw_deallocate(Pointer ptr);

  // Copies data from one allocation to another.
  // Pure virtual, so derived classes must define behavior.
  // Derived class implementation can simply call `default_copy_data`
  // to use `std::memcpy`.
  //
  // Requires: src and dest were allocated by this allocator
  // Requires: src and dest both have length >= count
  public native void copy_data(Pointer dest, @Const Pointer src, @Cast("std::size_t") long count);
}
