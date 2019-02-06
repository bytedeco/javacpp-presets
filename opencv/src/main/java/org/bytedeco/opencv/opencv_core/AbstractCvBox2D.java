package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvBox2D extends FloatPointer {
    static { Loader.load(); }
    public AbstractCvBox2D(Pointer p) { super(p); }

//    public CvBox2D(CvPoint2D32f center, CvSize2D32f size, float angle) {
//        allocate(); center(center).size(size).angle(angle);
//    }

    public abstract CvPoint2D32f center(); public abstract CvBox2D center(CvPoint2D32f center);
    public abstract CvSize2D32f size();    public abstract CvBox2D size(CvSize2D32f size);
    public abstract float angle();         public abstract CvBox2D angle(float angle);

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            if (capacity() == 0) {
                return "(" + center() + ", " + size() + ", " + angle() + ")";
            }
            String s = "";
            long p = position();
            for (long i = 0; i < capacity(); i++) {
                position(i);
                s += (i == 0 ? "(" : " (") + center() + ", " + size() + ", " + angle() + ")";
            }
            position(p);
            return s;
        }
    }
}
