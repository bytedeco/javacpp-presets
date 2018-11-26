package org.bytedeco.javacpp.opencv_imgproc;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core.IplConvKernel;

import static org.bytedeco.javacpp.opencv_imgproc.opencv_imgproc.cvCreateStructuringElementEx;
import static org.bytedeco.javacpp.opencv_imgproc.opencv_imgproc.cvReleaseStructuringElement;

public abstract class AbstractIplConvKernel extends Pointer {
    public AbstractIplConvKernel(Pointer p) { super(p); }

    /**
     * Calls cvCreateStructuringElementEx(), and registers a deallocator.
     * @return IplConvKernel created. Do not call cvReleaseStructuringElement() on it.
     */
    public static IplConvKernel create(int cols, int rows,
                                       int anchor_x, int anchor_y, int shape, int[] values/*=null*/) {
        IplConvKernel p = cvCreateStructuringElementEx(cols, rows,
                anchor_x, anchor_y, shape, values);
        if (p != null) {
            p.deallocator(new ReleaseDeallocator(p));
        }
        return p;
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void release() {
        deallocate();
    }
    static class ReleaseDeallocator extends IplConvKernel implements Deallocator {
        ReleaseDeallocator(IplConvKernel p) { super(p); }
        @Override public void deallocate() { cvReleaseStructuringElement(this); }
    }
}
