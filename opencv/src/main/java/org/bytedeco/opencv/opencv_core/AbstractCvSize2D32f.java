package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvSize2D32f extends FloatPointer {
    static { Loader.load(); }
    public AbstractCvSize2D32f(Pointer p) { super(p); }

//    public CvSize2D32f(float width, float height) {
//        allocate(); width(width).height(height);
//    }

    public abstract float width();  public abstract CvSize2D32f width(float width);
    public abstract float height(); public abstract CvSize2D32f height(float height);

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            if (capacity() == 0) {
                return "(" + width() + ", " + height() + ")";
            }
            String s = "";
            long p = position();
            for (long i = 0; i < capacity(); i++) {
                position(i);
                s += (i == 0 ? "(" : " (") + width() + ", " + height() + ")";
            }
            position(p);
            return s;
        }
    }
}
