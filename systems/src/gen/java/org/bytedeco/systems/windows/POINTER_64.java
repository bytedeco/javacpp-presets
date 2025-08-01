// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;

@Namespace @Name("void") @Opaque @Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class POINTER_64 extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public POINTER_64() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public POINTER_64(Pointer p) { super(p); }
}
