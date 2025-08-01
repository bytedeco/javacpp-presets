// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.tensorflowlite;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;


// TfLite model information and settings of the interpreter.
// Note: This struct does not comply with ABI stability.
@Opaque @Properties(inherit = org.bytedeco.tensorflowlite.presets.tensorflowlite.class)
public class TfLiteTelemetryInterpreterSettings extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TfLiteTelemetryInterpreterSettings() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TfLiteTelemetryInterpreterSettings(Pointer p) { super(p); }
}
