// Targeted by JavaCPP version 1.5.8-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;


// Opaque type similar to TfLiteDelegate / TfLiteOpaqueDelegate.
// This is used for cases (e.g. when using "TF Lite with Google Play Services")
// where the TF Lite runtime might be built using a newer (or older)
// version of the TF Lite sources than the app, and hence might have a
// different definition of the TfLiteDelegate type. TF Lite APIs use
// TfLiteOpaqueDelegate rather than TfLiteDelegate when they want to
// refer to a delegate defined with that potentially different version
// of the TfLiteDelegate type.
@Opaque @Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class TfLiteOpaqueDelegateStruct extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TfLiteOpaqueDelegateStruct() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TfLiteOpaqueDelegateStruct(Pointer p) { super(p); }
}