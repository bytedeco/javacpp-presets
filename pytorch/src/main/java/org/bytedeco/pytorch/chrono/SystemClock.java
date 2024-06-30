package org.bytedeco.pytorch.chrono;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;

@Name("std::chrono::system_clock") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SystemClock extends Pointer {
    public static native @ByVal TimePoint now();
}
