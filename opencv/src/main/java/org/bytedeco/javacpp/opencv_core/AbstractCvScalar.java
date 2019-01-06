package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = opencv_core_presets.class)
public abstract class AbstractCvScalar extends DoublePointer {
    static { Loader.load(); }
    public AbstractCvScalar(Pointer p) { super(p); }

//    public CvScalar(double val0, double val1, double val2, double val3) {
//        allocate(); val(0, val0).val(1, val1).val(2, val2).val(3, val3);
//    }

    public abstract double/*[4]*/ val(int i); public abstract CvScalar val(int i, double val);
    public double getVal(int i)               { return val(i);      }
    public CvScalar setVal(int i, double val) { return val(i, val); }

    public abstract DoublePointer val();
    public DoublePointer getDoublePointerVal() { return val(); }
    public LongPointer getLongPointerVal() { return new LongPointer(val()); }

    public void scale(double s) {
        for (int i = 0; i < 4; i++) {
            val(i, val(i) * s);
        }
    }

    public double red()      { return val(2); }
    public double green()    { return val(1); }
    public double blue()     { return val(0); }
    public CvScalar red  (double r) { val(2, r); return (CvScalar)this; }
    public CvScalar green(double g) { val(1, g); return (CvScalar)this; }
    public CvScalar blue (double b) { val(0, b); return (CvScalar)this; }

    public double magnitude() {
        return Math.sqrt(val(0)*val(0) + val(1)*val(1) + val(2)*val(2) + val(3)*val(3));
    }

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            if (capacity() == 0) {
                return "(" + (float)val(0) + ", " + (float)val(1) + ", " +
                        (float)val(2) + ", " + (float)val(3) + ")";
            }
            String s = "";
            long p = position();
            for (long i = 0; i < capacity(); i++) {
                position(i);
                s += (i == 0 ? "(" : " (") + (float)val(0) + ", " + (float)val(1) + ", " +
                        (float)val(2) + ", " + (float)val(3) + ")";
            }
            position(p);
            return s;
        }
    }

    public static final CvScalar
            ZERO    = new CvScalar().val(0, 0.0).val(1, 0.0).val(2, 0.0).val(3, 0.0),
            ONE     = new CvScalar().val(0, 1.0).val(1, 1.0).val(2, 1.0).val(3, 1.0),
            ONEHALF = new CvScalar().val(0, 0.5).val(1, 0.5).val(2, 0.5).val(3, 0.5),
            ALPHA1  = new CvScalar().val(0, 0.0).val(1, 0.0).val(2, 0.0).val(3, 1.0),
            ALPHA255= new CvScalar().val(0, 0.0).val(1, 0.0).val(2, 0.0).val(3, 255.0),

            WHITE   = opencv_core_helper.CV_RGB(255, 255, 255),
            GRAY    = opencv_core_helper.CV_RGB(128, 128, 128),
            BLACK   = opencv_core_helper.CV_RGB(  0,   0,   0),
            RED     = opencv_core_helper.CV_RGB(255,   0,   0),
            GREEN   = opencv_core_helper.CV_RGB(  0, 255,   0),
            BLUE    = opencv_core_helper.CV_RGB(  0,   0, 255),
            CYAN    = opencv_core_helper.CV_RGB(  0, 255, 255),
            MAGENTA = opencv_core_helper.CV_RGB(255,   0, 255),
            YELLOW  = opencv_core_helper.CV_RGB(255, 255,   0);
}
