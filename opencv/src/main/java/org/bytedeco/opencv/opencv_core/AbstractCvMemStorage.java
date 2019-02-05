package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvMemStorage extends Pointer {
    static { Loader.load(); }
    public AbstractCvMemStorage(Pointer p) { super(p); }

    /**
     * Calls cvCreateMemStorage(), and registers a deallocator.
     * @return CvMemStorage created. Do not call cvReleaseMemStorage() on it.
     */
    public static CvMemStorage create(int block_size) {
        CvMemStorage m = cvCreateMemStorage(block_size);
        if (m != null) {
            m.deallocator(new ReleaseDeallocator(m));
        }
        return m;
    }
    /**
     * Calls cvCreateMemStorage(0), and registers a deallocator.
     * @return CvMemStorage created. Do not call cvReleaseMemStorage() on it.
     */
    public static CvMemStorage create() {
        return create(0);
    }

    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void release() {
        deallocate();
    }
    protected static class ReleaseDeallocator extends CvMemStorage implements Deallocator {
        ReleaseDeallocator(CvMemStorage p) { super(p); }
        @Override public void deallocate() { cvReleaseMemStorage(this); }
    }
}
