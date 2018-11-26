package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;

import static org.bytedeco.javacpp.opencv_core.opencv_core.cvCreateGraphScanner;
import static org.bytedeco.javacpp.opencv_core.opencv_core.cvReleaseGraphScanner;

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
