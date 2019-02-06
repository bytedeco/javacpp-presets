package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvPoint3D32f extends FloatPointer {
    static { Loader.load(); }
    public AbstractCvPoint3D32f(Pointer p) { super(p); }

//    public CvPoint3D32f(double[] pts, int offset, int length) {
//        this(length/3);
//        put(pts, offset, length);
//    }
//    public CvPoint3D32f(double ... pts) {
//        this(pts, 0, pts.length);
//    }

    public abstract float x(); public abstract CvPoint3D32f x(float x);
    public abstract float y(); public abstract CvPoint3D32f y(float y);
    public abstract float z(); public abstract CvPoint3D32f z(float z);

//    public double[] get() {
//        double[] pts = new double[capacity == 0 ? 3 : 3*capacity];
//        get(pts);
//        return pts;
//    }
    public CvPoint3D32f get(double[] pts) {
        return get(pts, 0, pts.length);
    }
    public CvPoint3D32f get(double[] pts, int offset, int length) {
        for (int i = 0; i < length/3; i++) {
            position(i);
            pts[offset + i*3  ] = x();
            pts[offset + i*3+1] = y();
            pts[offset + i*3+2] = z();
        }
        return (CvPoint3D32f)position(0);
    }

    public final CvPoint3D32f put(double[] pts, int offset, int length) {
        for (int i = 0; i < length/3; i++) {
            position(i); put(pts[offset + i*3], pts[offset + i*3+1], pts[offset + i*3+2]);
        }
        return (CvPoint3D32f)position(0);
    }
    public final CvPoint3D32f put(double ... pts) {
        return put(pts, 0, pts.length);
    }

    public CvPoint3D32f put(double x, double y, double z) {
        return x((float)x).y((float)y).z((float)z);
    }
    public CvPoint3D32f put(CvPoint o) {
        return x(o.x()).y(o.y()).z(0);
    }
    public CvPoint3D32f put(CvPoint2D32f o) {
        return x(o.x()).y(o.y()).z(0);
    }
    public CvPoint3D32f put(CvPoint2D64f o) {
        return x((float)o.x()).y((float)o.y()).z(0);
    }

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            if (capacity() == 0) {
                return "(" + x() + ", " + y() + ", " + z() + ")";
            }
            String s = "";
            long p = position();
            for (long i = 0; i < capacity(); i++) {
                position(i);
                s += (i == 0 ? "(" : " (") + x() + ", " + y() + ", " + z() + ")";
            }
            position(p);
            return s;
        }
    }
}
