// Targeted by JavaCPP version 1.5.5: DO NOT EDIT THIS FILE

package org.bytedeco.parquet;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.arrow.*;
import static org.bytedeco.arrow.global.arrow.*;

import static org.bytedeco.arrow.global.parquet.*;


@Namespace("parquet") @Properties(inherit = org.bytedeco.arrow.presets.parquet.class)
public class FileInterface extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FileInterface(Pointer p) { super(p); }


  // Close the file
  public native void Close();

  // Return the current position in the file relative to the start
  public native @Cast("int64_t") long Tell();
}
