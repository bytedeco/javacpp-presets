// Targeted by JavaCPP version 1.5.9-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.functions.*;
import org.bytedeco.pytorch.Module;
import org.bytedeco.javacpp.annotation.Cast;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.pytorch.global.torch.*;


// this is a reader implemented by std::istream
@Namespace("caffe2::serialize") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class IStreamAdapter extends ReadAdapterInterface {
    static { Loader.load(); }

  
  
  public IStreamAdapter(@Cast("std::istream*") Pointer istream) { super((Pointer)null); allocate(istream); }
  private native void allocate(@Cast("std::istream*") Pointer istream);
  public native @Cast("size_t") long size();
  public native @Cast("size_t") long read(@Cast("uint64_t") long pos, Pointer buf, @Cast("size_t") long n, @Cast("const char*") BytePointer what/*=""*/);
  public native @Cast("size_t") long read(@Cast("uint64_t") long pos, Pointer buf, @Cast("size_t") long n);
  public native @Cast("size_t") long read(@Cast("uint64_t") long pos, Pointer buf, @Cast("size_t") long n, String what/*=""*/);
}