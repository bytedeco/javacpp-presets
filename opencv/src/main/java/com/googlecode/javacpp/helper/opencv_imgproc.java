/*
 * Copyright (C) 2014 Samuel Audet
 *
 * This file is part of JavaCPP.
 *
 * JavaCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCPP.  If not, see <http://www.gnu.org/licenses/>.
 */

package com.googlecode.javacpp.helper;

import com.googlecode.javacpp.FloatPointer;
import com.googlecode.javacpp.IntPointer;
import com.googlecode.javacpp.Pointer;
import com.googlecode.javacpp.PointerPointer;

// required by javac to resolve circular dependencies
import com.googlecode.javacpp.helper.opencv_core.*;
import com.googlecode.javacpp.opencv_core.*;
import com.googlecode.javacpp.opencv_imgproc.*;
import static com.googlecode.javacpp.opencv_imgproc.cvCreateStructuringElementEx;
import static com.googlecode.javacpp.opencv_imgproc.cvReleaseHist;
import static com.googlecode.javacpp.opencv_imgproc.cvReleaseStructuringElement;

public class opencv_imgproc extends com.googlecode.javacpp.presets.opencv_imgproc {

    public static abstract class AbstractCvMoments extends Pointer {
        public AbstractCvMoments() { }
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
        return com.googlecode.javacpp.opencv_imgproc.cvFindContours(image, storage, first_contour, header_size, mode, method, CvPoint.ZERO);
    }
    public static CvContourScanner cvStartFindContours(CvArr image, CvMemStorage storage,
            int header_size/*=sizeof(CvContour)*/, int mode/*=CV_RETR_LIST*/, int method/*=CV_CHAIN_APPROX_SIMPLE*/) {
        return com.googlecode.javacpp.opencv_imgproc.cvStartFindContours(image, storage, header_size, mode, method, CvPoint.ZERO);
    }

    public static abstract class AbstractIplConvKernel extends Pointer {
        public AbstractIplConvKernel() { }
        public AbstractIplConvKernel(Pointer p) { super(p); }

        public static IplConvKernel create(int cols, int rows,
                int anchor_x, int anchor_y, int shape, int[] values/*=null*/) {
            IplConvKernel p = cvCreateStructuringElementEx(cols, rows,
                    anchor_x, anchor_y, shape, values);
            if (p != null) {
                p.deallocator(new ReleaseDeallocator(p));
            }
            return p;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends IplConvKernel implements Deallocator {
            ReleaseDeallocator(IplConvKernel p) { super(p); }
            @Override public void deallocate() { cvReleaseStructuringElement(this); }
        }
    }

    public static abstract class AbstractCvHistogram extends Pointer {
        public AbstractCvHistogram() { }
        public AbstractCvHistogram(Pointer p) { super(p); }

        public static CvHistogram create(int dims, int[] sizes, int type,
                float[][] ranges/*=null*/, int uniform/*=1*/) {
            CvHistogram h = cvCreateHist(dims, sizes, type, ranges, uniform);
            if (h != null) {
                h.deallocator(new ReleaseDeallocator(h));
            }
            return h;
        }

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
        return com.googlecode.javacpp.opencv_imgproc.cvCreateHist(dims, new IntPointer(sizes), type,
                ranges == null ? null : new PointerPointer(ranges), uniform);
    }
    public static void cvSetHistBinRanges(CvHistogram hist,
            float[][] ranges, int uniform/*=1*/) {
        com.googlecode.javacpp.opencv_imgproc.cvSetHistBinRanges(hist,
                ranges == null ? null : new PointerPointer(ranges), uniform);
    }

    public static CvHistogram cvMakeHistHeaderForArray(int dims, int[] sizes, CvHistogram hist,
            float[] data, float[][] ranges/*=null*/, int uniform/*=1*/) {
        return com.googlecode.javacpp.opencv_imgproc.cvMakeHistHeaderForArray(dims, new IntPointer(sizes), hist,
                new FloatPointer(data), ranges == null ? null : new PointerPointer(ranges), uniform);
    }
    public static CvHistogram cvMakeHistHeaderForArray(int dims, int[] sizes, CvHistogram hist,
            FloatPointer data, float[][] ranges/*=null*/, int uniform/*=1*/) {
        return com.googlecode.javacpp.opencv_imgproc.cvMakeHistHeaderForArray(dims, new IntPointer(sizes), hist,
                new FloatPointer(data), ranges == null ? null : new PointerPointer(ranges), uniform);
    }

    public static void cvCalcArrHist(CvArr[] arr, CvHistogram hist, int accumulate/*=0*/, CvArr mask/*=null*/) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcArrHist(new CvArrArray(arr), hist, accumulate, mask);
    }
    public static void cvCalcHist(IplImage[] arr, CvHistogram hist, int accumulate/*=0*/, CvArr mask/*=null*/) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcHist(new IplImageArray(arr), hist, accumulate, mask);
    }
    public static void cvCalcHist(IplImageArray arr, CvHistogram hist,
            int accumulate/*=0*/, CvArr mask/*=null*/) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcArrHist(arr, hist, accumulate, mask);
    }

    public static void cvCalcArrBackProject(CvArr[] image, CvArr dst, CvHistogram hist) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcArrBackProject(new CvArrArray(image), dst, hist);
    }
    public static void cvCalcBackProject(IplImage[] image, CvArr dst, CvHistogram hist) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcBackProject(new IplImageArray(image), dst, hist);
    }
    public static void cvCalcBackProject(IplImageArray image,
            CvArr dst, CvHistogram hist) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcArrBackProject(image, dst, hist);
    }

    public static void cvCalcArrBackProjectPatch(CvArr[] image,
            CvArr dst, CvSize range, CvHistogram hist, int method, double factor) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcArrBackProjectPatch(new CvArrArray(image), dst, range, hist, method, factor);
    }
    public static void cvCalcBackProjectPatch(IplImage[] image,
            CvArr dst, CvSize range, CvHistogram hist, int method, double factor) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcBackProjectPatch(new IplImageArray(image), dst, range, hist, method, factor);
    }
    public static void cvCalcBackProjectPatch(IplImageArray image,
            CvArr dst, CvSize range, CvHistogram hist, int method, double factor) {
        com.googlecode.javacpp.opencv_imgproc.cvCalcArrBackProjectPatch(image, dst, range, hist, method, factor);
    }

}
