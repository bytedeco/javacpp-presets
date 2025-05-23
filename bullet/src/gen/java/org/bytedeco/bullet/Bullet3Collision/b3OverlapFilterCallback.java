// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.Bullet3Collision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.Bullet3Common.*;
import static org.bytedeco.bullet.global.Bullet3Common.*;

import static org.bytedeco.bullet.global.Bullet3Collision.*;


@Properties(inherit = org.bytedeco.bullet.presets.Bullet3Collision.class)
public class b3OverlapFilterCallback extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public b3OverlapFilterCallback(Pointer p) { super(p); }

	// return true when pairs need collision
	public native @Cast("bool") boolean needBroadphaseCollision(int proxy0, int proxy1);
}
