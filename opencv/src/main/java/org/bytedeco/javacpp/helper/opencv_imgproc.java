/*
 * Copyright (C) 2014-2015 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.javacpp.helper;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;

// required by javac to resolve circular dependencies
import org.bytedeco.javacpp.helper.opencv_core.*;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgproc.cvCreateStructuringElementEx;
import static org.bytedeco.javacpp.opencv_imgproc.cvReleaseHist;
import static org.bytedeco.javacpp.opencv_imgproc.cvReleaseStructuringElement;

public class opencv_imgproc extends org.bytedeco.javacpp.presets.opencv_imgproc {

    public static abstract class AbstractCvMoments extends Pointer {
        public AbstractCvMoments(Pointer p) { super(p); }

        public static ThreadLocal<CvMoments> createThreadLocal() {
            return new ThreadLocal<CvMoments>() {
                @Override protected CvMoments initialValue() {
                    return new CvMoments();
                }
            };
        }
    }

    public static int cvFindContours(CvArr image, CvMemStorage storage, CvSeq first_contour,
            int header_size/*=sizeof(CvContour)*/, int mode/*=CV_RETR_LIST*/, int method/*=CV_CHAIN_APPROX_SIMPLE*/) {
        return org.bytedeco.javacpp.opencv_imgproc.cvFindContours(image, storage, first_contour, header_size, mode, method, CvPoint.ZERO);
    }
    public static CvContourScanner cvStartFindContours(CvArr image, CvMemStorage storage,
            int header_size/*=sizeof(CvContour)*/, int mode/*=CV_RETR_LIST*/, int method/*=CV_CHAIN_APPROX_SIMPLE*/) {
        return org.bytedeco.javacpp.opencv_imgproc.cvStartFindContours(image, storage, header_size, mode, method, CvPoint.ZERO);
    }

    public static abstract class AbstractIplConvKernel extends Pointer {
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

    public static abstract class AbstractCvHistogram extends Pointer {
        public AbstractCvHistogram(Pointer p) { super(p); }

        /**
         * Calls cvCreateHist(), and registers a deallocator.
         * @return CvHistogram created. Do not call cvReleaseHist() on it.
         */
        public static CvHistogram create(int dims, int[] sizes, int type,
                float[][] ranges/*=null*/, int uniform/*=1*/) {
            CvHistogram h = cvCreateHist(dims, sizes, type, ranges, uniform);
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

    public static CvHistogram cvCreateHist(int dims, int[] sizes, int type,
            float[][] ranges/*=null*/, int uniform/*=1*/) {
        return org.bytedeco.javacpp.opencv_imgproc.cvCreateHist(dims, new IntPointer(sizes), type,
                ranges == null ? null : new PointerPointer(ranges), uniform);
    }
    public static void cvSetHistBinRanges(CvHistogram hist,
            float[][] ranges, int uniform/*=1*/) {
        org.bytedeco.javacpp.opencv_imgproc.cvSetHistBinRanges(hist,
                ranges == null ? null : new PointerPointer(ranges), uniform);
    }

    public static CvHistogram cvMakeHistHeaderForArray(int dims, int[] sizes, CvHistogram hist,
            float[] data, float[][] ranges/*=null*/, int uniform/*=1*/) {
        return org.bytedeco.javacpp.opencv_imgproc.cvMakeHistHeaderForArray(dims, new IntPointer(sizes), hist,
                new FloatPointer(data), ranges == null ? null : new PointerPointer(ranges), uniform);
    }
    public static CvHistogram cvMakeHistHeaderForArray(int dims, int[] sizes, CvHistogram hist,
            FloatPointer data, float[][] ranges/*=null*/, int uniform/*=1*/) {
        return org.bytedeco.javacpp.opencv_imgproc.cvMakeHistHeaderForArray(dims, new IntPointer(sizes), hist,
                new FloatPointer(data), ranges == null ? null : new PointerPointer(ranges), uniform);
    }

    public static void cvCalcArrHist(CvArr[] arr, CvHistogram hist, int accumulate/*=0*/, CvArr mask/*=null*/) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcArrHist(new CvArrArray(arr), hist, accumulate, mask);
    }
    public static void cvCalcHist(IplImage[] arr, CvHistogram hist, int accumulate/*=0*/, CvArr mask/*=null*/) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcHist(new IplImageArray(arr), hist, accumulate, mask);
    }
    public static void cvCalcHist(IplImageArray arr, CvHistogram hist,
            int accumulate/*=0*/, CvArr mask/*=null*/) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcArrHist(arr, hist, accumulate, mask);
    }

    public static void cvCalcArrBackProject(CvArr[] image, CvArr dst, CvHistogram hist) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcArrBackProject(new CvArrArray(image), dst, hist);
    }
    public static void cvCalcBackProject(IplImage[] image, CvArr dst, CvHistogram hist) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcBackProject(new IplImageArray(image), dst, hist);
    }
    public static void cvCalcBackProject(IplImageArray image,
            CvArr dst, CvHistogram hist) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcArrBackProject(image, dst, hist);
    }

    public static void cvCalcArrBackProjectPatch(CvArr[] image,
            CvArr dst, CvSize range, CvHistogram hist, int method, double factor) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcArrBackProjectPatch(new CvArrArray(image), dst, range, hist, method, factor);
    }
    public static void cvCalcBackProjectPatch(IplImage[] image,
            CvArr dst, CvSize range, CvHistogram hist, int method, double factor) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcBackProjectPatch(new IplImageArray(image), dst, range, hist, method, factor);
    }
    public static void cvCalcBackProjectPatch(IplImageArray image,
            CvArr dst, CvSize range, CvHistogram hist, int method, double factor) {
        org.bytedeco.javacpp.opencv_imgproc.cvCalcArrBackProjectPatch(image, dst, range, hist, method, factor);
    }

    public static void cvFillPoly(CvArr img, CvPoint[] pts, int[] npts,
            int contours, CvScalar color, int line_type/*=8*/, int shift/*=0*/) {
        org.bytedeco.javacpp.opencv_imgproc.cvFillPoly(img, new PointerPointer(pts),
                new IntPointer(npts), contours, color, line_type, shift);
    }

    public static void cvPolyLine(CvArr img, CvPoint[] pts,
            int[] npts, int contours, int is_closed, CvScalar color,
            int thickness/*=1*/, int line_type/*=8*/, int shift/*=0*/) {
        org.bytedeco.javacpp.opencv_imgproc.cvPolyLine(img, new PointerPointer(pts),
                new IntPointer(npts), contours, is_closed, color, thickness, line_type, shift);
    }

    public static void cvDrawPolyLine(CvArr img, CvPoint[] pts,
            int[] npts, int contours, int is_closed, CvScalar color,
            int thickness/*=1*/, int line_type/*=8*/, int shift/*=0*/) {
        cvPolyLine(img, pts, npts, contours, is_closed, color, thickness, line_type, shift);
    }

    public static void cvDrawContours(CvArr img, CvSeq contour, CvScalar external_color,
            CvScalar hole_color, int max_level, int thickness/*=1*/, int line_type/*=8*/) {
        org.bytedeco.javacpp.opencv_imgproc.cvDrawContours(img, contour, external_color,
                hole_color, max_level, thickness, line_type, CvPoint.ZERO);
    }
}
