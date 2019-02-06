package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.ValueGetter;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
@Name("IplImage*")
public class IplImageArray extends CvArrArray {
    public IplImageArray(IplImage ... array) { this(array.length); put(array); position(0); }
    public IplImageArray(long size) { allocateArray(size); }
    public IplImageArray(Pointer p) { super(p); }
    private native void allocateArray(long size);

    @Override public IplImageArray position(long position) {
        return (IplImageArray)super.position(position);
    }
    @Override public IplImageArray put(CvArr ... array) {
        return (IplImageArray)super.put(array);
    }
    @Override @ValueGetter public native IplImage get();
    @Override public IplImageArray put(CvArr p) {
        if (p instanceof IplImage) {
            return (IplImageArray)super.put(p);
        } else {
            throw new ArrayStoreException(p.getClass().getName());
        }
    }
}
