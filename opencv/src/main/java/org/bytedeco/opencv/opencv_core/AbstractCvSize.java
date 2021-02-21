package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvSize extends IntPointer {
    static { Loader.load(); }
    public AbstractCvSize(Pointer p) { super(p); }

//    public CvSize(int width, int height) {
//        allocate(); width(width).height(height);
//    }

    public abstract int width();  public abstract CvSize width(int width);
    public abstract int height(); public abstract CvSize height(int height);

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
    public static final CvSize ZERO = new CvSize().width(0).height(0).retainReference();
}
