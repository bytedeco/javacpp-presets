package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvPoint extends IntPointer {
    static { Loader.load(); }
    public AbstractCvPoint(Pointer p) { super(p); }

//    public CvPoint(int[] pts, int offset, int length) {
//        this(length/2);
//        put(pts, offset, length);
//    }
//    public CvPoint(int ... pts) {
//        this(pts, 0, pts.length);
//    }
//    public CvPoint(byte shift, double[] pts, int offset, int length) {
//        this(length/2);
//        put(shift, pts, offset, length);
//    }
//    public CvPoint(byte shift, double ... pts) {
//        this(shift, pts, 0, pts.length);
//    }

    public abstract int x(); public abstract CvPoint x(int x);
    public abstract int y(); public abstract CvPoint y(int y);

//    public int[] get() {
//        int[] pts = new int[capacity == 0 ? 2 : 2*capacity];
//        get(pts);
//        return pts;
//    }
    public CvPoint get(int[] pts) {
        return get(pts, 0, pts.length);
    }
    public CvPoint get(int[] pts, int offset, int length) {
        for (int i = 0; i < length/2; i++) {
            position(i);
            pts[offset + i*2  ] = x();
            pts[offset + i*2+1] = y();
        }
        return (CvPoint)position(0);
    }

    public final CvPoint put(int[] pts, int offset, int length) {
        for (int i = 0; i < length/2; i++) {
            position(i); put(pts[offset + i*2], pts[offset + i*2+1]);
        }
        return (CvPoint)position(0);
    }
    public final CvPoint put(int ... pts) {
        return put(pts, 0, pts.length);
    }
    public final CvPoint put(byte shift, double[] pts, int offset, int length) {
        int[] a = new int[length];
        for (int i = 0; i < length; i++) {
            a[i] = (int)Math.round(pts[offset + i] * (1<<shift));
        }
        return put(a, 0, length);
    }
    public final CvPoint put(byte shift, double ... pts) {
        return put(shift, pts, 0, pts.length);
    }

    public CvPoint put(int x, int y) {
        return x(x).y(y);
    }
    public CvPoint put(CvPoint o) {
        return x(o.x()).y(o.y());
    }
    public CvPoint put(byte shift, CvPoint2D32f o) {
        x((int)Math.round(o.x() * (1<<shift)));
        y((int)Math.round(o.y() * (1<<shift)));
        return (CvPoint)this;
    }
    public CvPoint put(byte shift, CvPoint2D64f o) {
        x((int)Math.round(o.x() * (1<<shift)));
        y((int)Math.round(o.y() * (1<<shift)));
        return (CvPoint)this;
    }

    @Override public String toString() {
        if (isNull()) {
            return super.toString();
        } else {
            if (capacity() == 0) {
                return "(" + x() + ", " + y() + ")";
            }
            String s = "";
            long p = position();
            for (long i = 0; i < capacity(); i++) {
                position(i);
                s += (i == 0 ? "(" : " (") + x() + ", " + y() + ")";
            }
            position(p);
            return s;
        }
    }

    public static final CvPoint ZERO = new CvPoint().x(0).y(0).retainReference();
}
