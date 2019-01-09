package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.ValueGetter;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = opencv_core_presets.class)
@Name("CvMatND*")
public class CvMatNDArray extends CvArrArray {
    public CvMatNDArray(CvMatND ... array) { this(array.length); put(array); position(0); }
    public CvMatNDArray(long size) { allocateArray(size); }
    public CvMatNDArray(Pointer p) { super(p); }
    private native void allocateArray(long size);

    @Override public CvMatNDArray position(long position) {
        return (CvMatNDArray)super.position(position);
    }
    @Override public CvMatNDArray put(CvArr ... array) {
        return (CvMatNDArray)super.put(array);
    }
    @Override @ValueGetter public native CvMatND get();
    @Override public CvMatNDArray put(CvArr p) {
        if (p instanceof CvMatND) {
            return (CvMatNDArray)super.put(p);
        } else {
            throw new ArrayStoreException(p.getClass().getName());
        }
    }
}
