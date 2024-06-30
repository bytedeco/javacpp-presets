package org.bytedeco.pytorch.chrono;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;

@Name("std::chrono::duration<float>") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FloatDuration extends Pointer {
    public FloatDuration() {  super((Pointer)null); allocate(); };
    private native void allocate();
    public FloatDuration(float r) {  super((Pointer)null); allocate(r); };
    private native void allocate(float r);

    native float count();
    static native @ByVal @Name("zero") FloatDuration zero_();
    static native @ByVal FloatDuration min();
    static native @ByVal FloatDuration max();
}
