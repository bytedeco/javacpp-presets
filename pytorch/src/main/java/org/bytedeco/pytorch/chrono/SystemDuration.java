package org.bytedeco.pytorch.chrono;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;

@Name("std::chrono::system_clock::duration") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SystemDuration extends Pointer {
    public SystemDuration() {  super((Pointer)null); allocate(); };
    private native void allocate();
    public SystemDuration(long r) {  super((Pointer)null); allocate(r); };
    private native void allocate(long r);

    native long count();
    static native @ByVal @Name("zero") SystemDuration zero_();
    static native @ByVal SystemDuration min();
    static native @ByVal SystemDuration max();}
