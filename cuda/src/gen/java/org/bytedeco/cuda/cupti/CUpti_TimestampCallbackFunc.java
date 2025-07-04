// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cupti;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.cupti.*;


/**
 * \brief Function type for callback used by CUPTI to request a timestamp
 * to be used in activity records.
 *
 * This callback function signals the CUPTI client that a timestamp needs
 * to be returned. This timestamp would be treated as normalized timestamp
 * to be used for various purposes in CUPTI. For example to store start and
 * end timestamps reported in the CUPTI activity records.
 * The returned timestamp must be in nanoseconds.
 *
 * @see ::cuptiActivityRegisterTimestampCallback
 */
@Convention("CUPTIAPI") @Properties(inherit = org.bytedeco.cuda.presets.cupti.class)
public class CUpti_TimestampCallbackFunc extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CUpti_TimestampCallbackFunc(Pointer p) { super(p); }
    protected CUpti_TimestampCallbackFunc() { allocate(); }
    private native void allocate();
    public native @Cast("uint64_t") long call();
}
