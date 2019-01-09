package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.javacpp.opencv_core.opencv_core.cvCloneMatND;
import static org.bytedeco.javacpp.opencv_core.opencv_core.cvCreateMatND;
import static org.bytedeco.javacpp.opencv_core.opencv_core.cvReleaseMatND;

@Properties(inherit = opencv_core_presets.class)
public abstract class AbstractCvMatND extends CvArr {
    public AbstractCvMatND(Pointer p) { super(p); }

    /**
     * Calls CvMatND(), and registers a deallocator.
     * @return CvMatND created. Do not call cvReleaseMatND() on it.
     */
    public static CvMatND create(int dims, int[] sizes, int type) {
        CvMatND m = cvCreateMatND(dims, sizes, type);
        if (m != null) {
            ((AbstractCvMatND)m).deallocator(new ReleaseDeallocator(m));
        }
        return m;
    }

    /**
     * Calls cvCloneMatND(), and registers a deallocator.
     * @return CvMatND cloned. Do not call cvReleaseMatND() on it.
     */
    @Override public CvMatND clone() {
        CvMatND m = cvCloneMatND((CvMatND)this);
        if (m != null) {
            m.deallocator(new ReleaseDeallocator(m));
        }
        return m;
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void release() {
        deallocate();
    }
    protected static class ReleaseDeallocator extends CvMatND implements Deallocator {
        ReleaseDeallocator(CvMatND p) { super(p); }
        @Override public void deallocate() { if (isNull()) return; cvReleaseMatND(this); setNull(); }
    }
}
