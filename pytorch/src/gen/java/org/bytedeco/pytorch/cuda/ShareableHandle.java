// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch.cuda;

import org.bytedeco.pytorch.Allocator;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.javacpp.chrono.*;
import static org.bytedeco.javacpp.global.chrono.*;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;
import org.bytedeco.cuda.cublas.*;
import static org.bytedeco.cuda.global.cublas.*;
import org.bytedeco.cuda.cudnn.*;
import static org.bytedeco.cuda.global.cudnn.*;
import org.bytedeco.cuda.cusparse.*;
import static org.bytedeco.cuda.global.cusparse.*;
import org.bytedeco.cuda.cusolver.*;
import static org.bytedeco.cuda.global.cusolver.*;
import org.bytedeco.cuda.cupti.*;
import static org.bytedeco.cuda.global.cupti.*;

import static org.bytedeco.pytorch.global.torch_cuda.*;


@Namespace("c10::cuda::CUDACachingAllocator") @Properties(inherit = org.bytedeco.pytorch.presets.torch_cuda.class)
public class ShareableHandle extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ShareableHandle() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public ShareableHandle(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ShareableHandle(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public ShareableHandle position(long position) {
        return (ShareableHandle)super.position(position);
    }
    @Override public ShareableHandle getPointer(long i) {
        return new ShareableHandle((Pointer)this).offsetAddress(i);
    }

  public native @Cast("ptrdiff_t") long offset(); public native ShareableHandle offset(long setter);
  public native @StdString BytePointer handle(); public native ShareableHandle handle(BytePointer setter);
}
