package org.bytedeco.openvino;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Cast;

public class ov_dimension_t extends Pointer {
    static { Loader.load(); }
    public ov_dimension_t() { super((Pointer)null); }
    public ov_dimension_t(Pointer p) { super(p); }
    public native @Cast("int64_t") long min();
    public native ov_dimension_t min(long setter);
    public native @Cast("int64_t") long max();
    public native ov_dimension_t max(long setter);
}
