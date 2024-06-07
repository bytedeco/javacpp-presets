package org.bytedeco.pytorch.chrono;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;

@Name("std::chrono::milliseconds") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Milliseconds extends Pointer {
    public Milliseconds() {  super((Pointer)null); allocate(); };
    private native void allocate();
    public Milliseconds(long r) {  super((Pointer)null); allocate(r); };
    private native void allocate(long r);

    native long count();
    static native @ByVal @Name("zero") Milliseconds zero_();
    static native @ByVal Milliseconds min();
    static native @ByVal Milliseconds max();
}
