package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvGraphScanner extends Pointer {
    public AbstractCvGraphScanner(Pointer p) { super(p); }

    /**
     * Calls cvCreateGraphScanner(), and registers a deallocator.
     * @return CvGraphScanner created. Do not call cvReleaseGraphScanner() on it.
     */
    public static CvGraphScanner create(CvGraph graph,
            CvGraphVtx vtx/*=null*/, int mask/*=CV_GRAPH_ALL_ITEMS*/) {
        CvGraphScanner g = cvCreateGraphScanner(graph, vtx, mask);
        if (g != null) {
            g.deallocator(new ReleaseDeallocator(g));
        }
        return g;
    }
    /**
     * Calls the deallocator, if registered, otherwise has no effect.
     */
    public void release() {
        deallocate();
    }
    protected static class ReleaseDeallocator extends CvGraphScanner implements Deallocator {
        ReleaseDeallocator(CvGraphScanner p) { super(p); }
        @Override public void deallocate() { cvReleaseGraphScanner(this); }
    }
}
