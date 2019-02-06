package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvRect extends IntPointer {
    static { Loader.load(); }
    public AbstractCvRect(Pointer p) { super(p); }

//    public CvRect(int x, int y, int width, int height) {
//        allocate(); x(x).y(y).width(width).height(height);
//    }

    public abstract int x();
    public abstract int y();
    public abstract int width();
    public abstract int height();

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            if (capacity() == 0) {
                return "(" + x() + ", " + y() + "; " + width() + ", " + height() + ")";
            }
            String s = "";
            long p = position();
            for (long i = 0; i < capacity(); i++) {
                position(i);
                s += (i == 0 ? "(" : " (") + x() + ", " + y() + "; " + width() + ", " + height() + ")";
            }
            position(p);
            return s;
        }
    }
}
