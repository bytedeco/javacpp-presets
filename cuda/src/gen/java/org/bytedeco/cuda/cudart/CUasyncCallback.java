// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cudart;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cuda.global.cudart.*;


/**
 * CUDA async notification callback
 * @param info Information describing what actions to take as a result of this notification.
 * @param userData Pointer to user defined data provided at callback registration.
 * @param callback The callback handle associated with this specific callback.
 */
@Properties(inherit = org.bytedeco.cuda.presets.cudart.class)
public class CUasyncCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CUasyncCallback(Pointer p) { super(p); }
    protected CUasyncCallback() { allocate(); }
    private native void allocate();
    public native void call(CUasyncNotificationInfo info, Pointer userData, CUasyncCallbackEntry_st callback);
}
