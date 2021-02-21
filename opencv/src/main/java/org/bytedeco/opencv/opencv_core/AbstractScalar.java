package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
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
            ZERO    = new Scalar(0.0, 0.0, 0.0, 0.0).retainReference(),
            ONE     = new Scalar(1.0, 1.0, 1.0, 1.0).retainReference(),
            ONEHALF = new Scalar(0.5, 0.5, 0.5, 0.5).retainReference(),
            ALPHA1  = new Scalar(0.0, 0.0, 0.0, 1.0).retainReference(),
            ALPHA255= new Scalar(0.0, 0.0, 0.0, 255.0).retainReference(),

            WHITE   = RGB(255, 255, 255).retainReference(),
            GRAY    = RGB(128, 128, 128).retainReference(),
            BLACK   = RGB(  0,   0,   0).retainReference(),
            RED     = RGB(255,   0,   0).retainReference(),
            GREEN   = RGB(  0, 255,   0).retainReference(),
            BLUE    = RGB(  0,   0, 255).retainReference(),
            CYAN    = RGB(  0, 255, 255).retainReference(),
            MAGENTA = RGB(255,   0, 255).retainReference(),
            YELLOW  = RGB(255, 255,   0).retainReference();
}
