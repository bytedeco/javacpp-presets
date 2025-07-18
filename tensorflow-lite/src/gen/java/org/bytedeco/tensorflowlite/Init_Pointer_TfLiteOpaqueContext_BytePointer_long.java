// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;


/** Sets the initialization callback for the registration. The function returns
 *  an error upon failure.
 * 
 *  The callback is called to initialize the op from serialized data. The value
 *  passed in the {@code user_data} parameter is the value that was passed to
 *  {@code TfLiteOperatorCreate}.  Please refer {@code init} of {@code TfLiteRegistration}
 *  for the detail.
 *  */
@Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class Init_Pointer_TfLiteOpaqueContext_BytePointer_long extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    Init_Pointer_TfLiteOpaqueContext_BytePointer_long(Pointer p) { super(p); }
    protected Init_Pointer_TfLiteOpaqueContext_BytePointer_long() { allocate(); }
    private native void allocate();
    public native Pointer call(Pointer user_data, TfLiteOpaqueContext context,
                  @Cast("const char*") BytePointer buffer, @Cast("size_t") long length);
}
