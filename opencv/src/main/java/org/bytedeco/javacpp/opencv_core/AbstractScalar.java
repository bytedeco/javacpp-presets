package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = opencv_core_presets.class)
public abstract class AbstractScalar extends DoublePointer {
    static { Loader.load(); }
    public AbstractScalar(Pointer p) { super(p); }

    public void scale(double s) {
        for (int i = 0; i < 4; i++) {
            put(i, get(i) * s);
        }
    }

    public double red()      { return get(2); }
    public double green()    { return get(1); }
    public double blue()     { return get(0); }
    public Scalar red  (double r) { put(2, r); return (Scalar)this; }
    public Scalar green(double g) { put(1, g); return (Scalar)this; }
    public Scalar blue (double b) { put(0, b); return (Scalar)this; }

    public double magnitude() {
        return Math.sqrt(get(0)*get(0) + get(1)*get(1) + get(2)*get(2) + get(3)*get(3));
    }

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            if (capacity() == 0) {
                return "(" + (float)get(0) + ", " + (float)get(1) + ", " +
                        (float)get(2) + ", " + (float)get(3) + ")";
            }
            String s = "";
            long p = position();
            for (long i = 0; i < capacity(); i++) {
                position(i);
                s += (i == 0 ? "(" : " (") + (float)get(0) + ", " + (float)get(1) + ", " +
                        (float)get(2) + ", " + (float)get(3) + ")";
            }
            position(p);
            return s;
        }
    }

    public static final Scalar
            ZERO    = new Scalar(0.0, 0.0, 0.0, 0.0),
            ONE     = new Scalar(1.0, 1.0, 1.0, 1.0),
            ONEHALF = new Scalar(0.5, 0.5, 0.5, 0.5),
            ALPHA1  = new Scalar(0.0, 0.0, 0.0, 1.0),
            ALPHA255= new Scalar(0.0, 0.0, 0.0, 255.0),

            WHITE   = opencv_core_helper.RGB(255, 255, 255),
            GRAY    = opencv_core_helper.RGB(128, 128, 128),
            BLACK   = opencv_core_helper.RGB(  0,   0,   0),
            RED     = opencv_core_helper.RGB(255,   0,   0),
            GREEN   = opencv_core_helper.RGB(  0, 255,   0),
            BLUE    = opencv_core_helper.RGB(  0,   0, 255),
            CYAN    = opencv_core_helper.RGB(  0, 255, 255),
            MAGENTA = opencv_core_helper.RGB(255,   0, 255),
            YELLOW  = opencv_core_helper.RGB(255, 255,   0);
}
