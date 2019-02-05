package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvPoint3D64f extends DoublePointer {
    static { Loader.load(); }
    public AbstractCvPoint3D64f(Pointer p) { super(p); }

//    public CvPoint3D64f(double[] pts, int offset, int length) {
//        this(length/3);
//        put(pts, offset, length);
//    }
//    public CvPoint3D64f(double ... pts) {
//        this(pts, 0, pts.length);
//    }

    public abstract double x(); public abstract CvPoint3D64f x(double x);
    public abstract double y(); public abstract CvPoint3D64f y(double y);
    public abstract double z(); public abstract CvPoint3D64f z(double z);

//    public double[] get() {
//        double[] pts = new double[capacity == 0 ? 3 : 3*capacity];
//        get(pts);
//        return pts;
//    }
    public CvPoint3D64f get(double[] pts) {
        return get(pts, 0, pts.length);
    }
    public CvPoint3D64f get(double[] pts, int offset, int length) {
        for (int i = 0; i < length/3; i++) {
            position(i);
            pts[offset + i*3  ] = x();
            pts[offset + i*3+1] = y();
            pts[offset + i*3+2] = z();
        }
        return (CvPoint3D64f)position(0);
    }

    public final CvPoint3D64f put(double[] pts, int offset, int length) {
        for (int i = 0; i < length/3; i++) {
            position(i); put(pts[offset + i*3], pts[offset + i*3+1], pts[offset + i*3+2]);
        }
        return (CvPoint3D64f)position(0);
    }
    public final CvPoint3D64f put(double ... pts) {
        return put(pts, 0, pts.length);
    }

    public CvPoint3D64f put(double x, double y, double z) {
        return x(x()).y(y()).z(z());
    }
    public CvPoint3D64f put(CvPoint o) {
        return x(o.x()).y(o.y()).z(0);
    }
    public CvPoint3D64f put(CvPoint2D32f o) {
        return x(o.x()).y(o.y()).z(0);
    }
    public CvPoint3D64f put(CvPoint2D64f o) {
        return x(o.x()).y(o.y()).z(0);
    }

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            if (capacity() == 0) {
                return "(" + (float)x() + ", " + (float)y() + ", " + (float)z() + ")";
            }
            String s = "";
            long p = position();
            for (long i = 0; i < capacity(); i++) {
                position(i);
                s += (i == 0 ? "(" : " (") + (float)x() + ", " + (float)y() + ", " + (float)z() + ")";
            }
            position(p);
            return s;
        }
    }
}
