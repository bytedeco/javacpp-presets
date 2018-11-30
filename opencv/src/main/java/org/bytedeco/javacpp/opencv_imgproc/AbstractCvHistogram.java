package org.bytedeco.javacpp.opencv_imgproc;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.opencv_core.CvHistogram;

import static org.bytedeco.javacpp.opencv_imgproc.opencv_imgproc.cvReleaseHist;

@Properties(inherit = opencv_imgproc_presets.class)
public abstract class AbstractCvHistogram extends Pointer {
      public AbstractCvHistogram(Pointer p) { super(p); }

      /**
       * Calls cvCreateHist(), and registers a deallocator.
       * @return CvHistogram created. Do not call cvReleaseHist() on it.
       */
      public static CvHistogram create(int dims, int[] sizes, int type,
              float[][] ranges/*=null*/, int uniform/*=1*/) {
          CvHistogram h = opencv_imgproc_helper.cvCreateHist(dims, sizes, type, ranges, uniform);
          if (h != null) {
              h.deallocator(new ReleaseDeallocator(h));
          }
          return h;
      }

      /**
       * Calls the deallocator, if registered, otherwise has no effect.
       */
      public void release() {
          deallocate();
      }
      static class ReleaseDeallocator extends CvHistogram implements Deallocator {
          ReleaseDeallocator(CvHistogram p) { super(p); }
          @Override public void deallocate() { cvReleaseHist(this); }
      }
  }
