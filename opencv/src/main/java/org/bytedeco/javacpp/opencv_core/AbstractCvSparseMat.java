package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.javacpp.opencv_core.opencv_core.cvCloneSparseMat;
import static org.bytedeco.javacpp.opencv_core.opencv_core.cvCreateSparseMat;
import static org.bytedeco.javacpp.opencv_core.opencv_core.cvReleaseSparseMat;

@Properties(inherit = opencv_core_presets.class)
public abstract class AbstractCvSparseMat extends CvArr {
    public AbstractCvSparseMat(Pointer p) { super(p); }

    /**
     * Calls cvCreateSparseMat(), and registers a deallocator.
     * @return CvSparseMat created. Do not call cvReleaseSparseMat() on it.
     */
    public static CvSparseMat create(int dims, int[] sizes, int type) {
        CvSparseMat m = cvCreateSparseMat(dims, sizes, type);
        if (m != null) {
            m.deallocator(new ReleaseDeallocator(m));
        }
        return m;
    }

    /**
     * Calls cvCloneSparseMat(), and registers a deallocator.
     * @return CvSparseMat cloned. Do not call cvReleaseSparseMat() on it.
     */
    @Override public CvSparseMat clone() {
        CvSparseMat m = cvCloneSparseMat((CvSparseMat)this);
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
    protected static class ReleaseDeallocator extends CvSparseMat implements Deallocator {
        ReleaseDeallocator(CvSparseMat p) { super(p); }
        @Override public void deallocate() { if (isNull()) return; cvReleaseSparseMat(this); setNull(); }
    }
}
