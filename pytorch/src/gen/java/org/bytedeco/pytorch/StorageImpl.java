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


// A storage represents the underlying backing data buffer for a
// tensor.  This concept was inherited from the original Torch7
// codebase; we'd kind of like to get rid of the concept
// (see https://github.com/pytorch/pytorch/issues/14797) but
// it's hard work and no one has gotten around to doing it.
//
// NB: storage is supposed to uniquely own a data pointer; e.g.,
// two non-null data pointers alias if and only if they are from
// the same storage.  Technically you can violate this invariant
// (e.g., you can create a non-owning StorageImpl with at::from_blob)
// but a lot of things won't work correctly, including:
//
// - An ordinary deleter on such a storage is wrong, because normal deleters
//   assume unique ownership, but if you have two storages at the same data,
//   that implies there is some sort of shared ownership. So your deleter would
//   have to actually be internally doing some sort of refcount thing
// - Deepcopy in Python side relies on storage equality and not data pointer
//   equality; so if there are two separate storages pointing to the same data,
//   the data will actually get duplicated in that case (one data ptr before,
//   two data ptrs after)
// - Version counts won't work correctly, because we do all VC tracking at the
//   level of storages (unless you explicitly disconnect the VC with detach);
//   mutation because data pointers are the same are totally untracked
@Namespace("c10") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StorageImpl extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StorageImpl(Pointer p) { super(p); }

  @Opaque public static class use_byte_size_t extends Pointer {
      /** Empty constructor. Calls {@code super((Pointer)null)}. */
      public use_byte_size_t() { super((Pointer)null); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public use_byte_size_t(Pointer p) { super(p); }
  }

  public StorageImpl(
        @ByVal use_byte_size_t arg0,
        @ByVal SymInt size_bytes,
        @StdMove DataPtr data_ptr,
        Allocator allocator,
        @Cast("bool") boolean resizable) { super((Pointer)null); allocate(arg0, size_bytes, data_ptr, allocator, resizable); }
  @IntrusivePtr @Name("c10::make_intrusive<c10::StorageImpl>") private native void allocate(
        @ByVal use_byte_size_t arg0,
        @ByVal SymInt size_bytes,
        @StdMove DataPtr data_ptr,
        Allocator allocator,
        @Cast("bool") boolean resizable);

  public StorageImpl(
        @ByVal use_byte_size_t arg0,
        @Const @ByRef SymInt size_bytes,
        Allocator allocator,
        @Cast("bool") boolean resizable) { super((Pointer)null); allocate(arg0, size_bytes, allocator, resizable); }
  @IntrusivePtr @Name("c10::make_intrusive<c10::StorageImpl>") private native void allocate(
        @ByVal use_byte_size_t arg0,
        @Const @ByRef SymInt size_bytes,
        Allocator allocator,
        @Cast("bool") boolean resizable);

  
  
  
  
  

  public native void reset();

  // Destructor doesn't call release_resources because it's
  // unnecessary; don't forget to change that if needed!
  public native void release_resources();

  public native @Cast("size_t") long nbytes();

  public native @ByVal SymInt sym_nbytes();

  // TODO: remove later
  public native void set_nbytes(@Cast("size_t") long size_bytes);

  public native void set_nbytes(@ByVal SymInt size_bytes);

  public native @Cast("bool") boolean resizable();

  public native @StdMove DataPtr data_ptr();

  public native @ByRef DataPtr mutable_data_ptr();

  // Returns the data_ptr. Bypasses all checks.
  public native @ByRef DataPtr _mutable_data_ptr_no_checks();

  // Returns the previous data_ptr
  public native @StdMove DataPtr set_data_ptr(@StdMove DataPtr data_ptr);

  public native void set_data_ptr_noswap(@StdMove DataPtr data_ptr);

  public native @Const Pointer data();

  public native Pointer mutable_data();

  public native @ByVal DeviceType device_type();

  public native Allocator allocator();

  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  public native void set_allocator(Allocator allocator);

  public native @ByVal Device device();

  public native void set_resizable(@Cast("bool") boolean resizable);

  /**
   * Can only be called when use_count is 1
   */
  public native void UniqueStorageShareExternalPointer(
        Pointer src,
        @Cast("size_t") long size_bytes,
        PointerConsumer d/*=nullptr*/);
  public native void UniqueStorageShareExternalPointer(
        Pointer src,
        @Cast("size_t") long size_bytes);

  /**
   * Can only be called when use_count is 1
   */
  public native void UniqueStorageShareExternalPointer(@Cast({"", "c10::DataPtr&&"}) @StdMove DataPtr data_ptr,  @Cast("size_t") long size_bytes);

  // This method can be used only after storage construction and cannot be used
  // to modify storage status
  public native void set_received_cuda(@Cast("bool") boolean received_cuda);

  public native @Cast("bool") boolean received_cuda();

  public native @Cast("c10::impl::PyObjectSlot*") Pointer pyobj_slot();

  public native @ByRef StorageExtraMeta get_extra_meta();

  public native void throw_data_ptr_access_error();

  public native void release_data_and_set_meta_custom_data_ptr_error_msg_(
        @ByVal StringOptional s);

  public native void set_throw_on_mutable_data_ptr();

  public native void set_warn_deprecated_on_mutable_data_ptr();
}
