package org.bytedeco.pytorch.chrono;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;

@Name("std::chrono::time_point<std::chrono::system_clock>") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TimePoint extends Pointer {
    public TimePoint() {  super((Pointer)null); allocate(); }
    private native void allocate();

    public native @ByVal SystemDuration time_since_epoch();
}
