// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cudart;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cuda.global.cudart.*;


/**
 * External memory handle descriptor
 */
@Properties(inherit = org.bytedeco.cuda.presets.cudart.class)
public class cudaExternalMemoryHandleDesc extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cudaExternalMemoryHandleDesc() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public cudaExternalMemoryHandleDesc(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cudaExternalMemoryHandleDesc(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public cudaExternalMemoryHandleDesc position(long position) {
        return (cudaExternalMemoryHandleDesc)super.position(position);
    }
    @Override public cudaExternalMemoryHandleDesc getPointer(long i) {
        return new cudaExternalMemoryHandleDesc((Pointer)this).offsetAddress(i);
    }

    /**
     * Type of the handle
     */
    public native @Cast("cudaExternalMemoryHandleType") int type(); public native cudaExternalMemoryHandleDesc type(int setter);
        /**
         * File descriptor referencing the memory object. Valid
         * when type is
         * ::cudaExternalMemoryHandleTypeOpaqueFd
         */
        @Name("handle.fd") public native int handle_fd(); public native cudaExternalMemoryHandleDesc handle_fd(int setter);
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalMemoryHandleTypeOpaqueWin32
         * - ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
         * - ::cudaExternalMemoryHandleTypeD3D12Heap 
         * - ::cudaExternalMemoryHandleTypeD3D12Resource
		 * - ::cudaExternalMemoryHandleTypeD3D11Resource
		 * - ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following: 
         * ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
         * ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
         * then 'name' must be NULL.
         */
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            @Name("handle.win32.handle") public native Pointer handle_win32_handle(); public native cudaExternalMemoryHandleDesc handle_win32_handle(Pointer setter);
            /**
             * Name of a valid memory object.
             * Must be NULL if 'handle' is non-NULL.
             */
            @Name("handle.win32.name") public native @Const Pointer handle_win32_name(); public native cudaExternalMemoryHandleDesc handle_win32_name(Pointer setter);
        /**
         * A handle representing NvSciBuf Object. Valid when type
         * is ::cudaExternalMemoryHandleTypeNvSciBuf
         */
        @Name("handle.nvSciBufObject") public native @Const Pointer handle_nvSciBufObject(); public native cudaExternalMemoryHandleDesc handle_nvSciBufObject(Pointer setter);
    /**
     * Size of the memory allocation
     */
    public native @Cast("unsigned long long") long size(); public native cudaExternalMemoryHandleDesc size(long setter);
    /**
     * Flags must either be zero or ::cudaExternalMemoryDedicated
     */
    public native @Cast("unsigned int") int flags(); public native cudaExternalMemoryHandleDesc flags(int setter);
}
