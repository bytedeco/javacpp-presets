// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;

@Opaque @Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btMprSupport_t extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public btMprSupport_t() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btMprSupport_t(Pointer p) { super(p); }
}
