// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.avutil;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.ffmpeg.global.avutil.*;


/**
 * Buffer to print data progressively
 *
 * The string buffer grows as necessary and is always 0-terminated.
 * The content of the string is never accessed, and thus is
 * encoding-agnostic and can even hold binary data.
 *
 * Small buffers are kept in the structure itself, and thus require no
 * memory allocation at all (unless the contents of the buffer is needed
 * after the structure goes out of scope). This is almost as lightweight as
 * declaring a local {@code char buf[512]}.
 *
 * The length of the string can go beyond the allocated size: the buffer is
 * then truncated, but the functions still keep account of the actual total
 * length.
 *
 * In other words, AVBPrint.len can be greater than AVBPrint.size and records
 * the total length of what would have been to the buffer if there had been
 * enough memory.
 *
 * Append operations do not need to be tested for failure: if a memory
 * allocation fails, data stop being appended to the buffer, but the length
 * is still updated. This situation can be tested with
 * av_bprint_is_complete().
 *
 * The AVBPrint.size_max field determines several possible behaviours:
 * - {@code size_max = -1} (= {@code UINT_MAX}) or any large value will let the buffer be
 *   reallocated as necessary, with an amortized linear cost.
 * - {@code size_max = 0} prevents writing anything to the buffer: only the total
 *   length is computed. The write operations can then possibly be repeated in
 *   a buffer with exactly the necessary size
 *   (using {@code size_init = size_max = len + 1}).
 * - {@code size_max = 1} is automatically replaced by the exact size available in the
 *   structure itself, thus ensuring no dynamic memory allocation. The
 *   internal buffer is large enough to hold a reasonable paragraph of text,
 *   such as the current paragraph.
 */

@Properties(inherit = org.bytedeco.ffmpeg.presets.avutil.class)
public class ff_pad_helper_AVBPrint extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ff_pad_helper_AVBPrint() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public ff_pad_helper_AVBPrint(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ff_pad_helper_AVBPrint(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public ff_pad_helper_AVBPrint position(long position) {
        return (ff_pad_helper_AVBPrint)super.position(position);
    }
    @Override public ff_pad_helper_AVBPrint getPointer(long i) {
        return new ff_pad_helper_AVBPrint((Pointer)this).offsetAddress(i);
    }

     /** string so far */
     public native @Cast("char*") BytePointer str(); public native ff_pad_helper_AVBPrint str(BytePointer setter);
    /** length so far */
    public native @Cast("unsigned") int len(); public native ff_pad_helper_AVBPrint len(int setter);
    /** allocated memory */
    public native @Cast("unsigned") int size(); public native ff_pad_helper_AVBPrint size(int setter);
    /** maximum allocated memory */
    public native @Cast("unsigned") int size_max(); public native ff_pad_helper_AVBPrint size_max(int setter);
    public native @Cast("char") byte reserved_internal_buffer(int i); public native ff_pad_helper_AVBPrint reserved_internal_buffer(int i, byte setter);
    @MemberGetter public native @Cast("char*") BytePointer reserved_internal_buffer(); }
