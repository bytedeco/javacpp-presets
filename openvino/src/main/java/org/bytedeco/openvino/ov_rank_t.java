package org.bytedeco.openvino;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Cast;

public class ov_rank_t extends Pointer {
    static { Loader.load(); }
    public ov_rank_t() { super((Pointer)null); }
    public ov_rank_t(Pointer p) { super(p); }
    public native @Cast("int64_t") long value();
    public native ov_rank_t value(long setter);
}
