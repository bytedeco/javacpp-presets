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

import com.googlecode.javacpp.Pointer;

// required by javac to resolve circular dependencies
import com.googlecode.javacpp.opencv_core.*;
import com.googlecode.javacpp.opencv_legacy.*;
import static com.googlecode.javacpp.opencv_legacy.CV_GLCM_ALL;
import static com.googlecode.javacpp.opencv_legacy.CV_GLCM_OPTIMIZATION_NONE;
import static com.googlecode.javacpp.opencv_legacy.cvCreate2DHMM;
import static com.googlecode.javacpp.opencv_legacy.cvCreateBGCodeBookModel;
import static com.googlecode.javacpp.opencv_legacy.cvCreateConDensation;
import static com.googlecode.javacpp.opencv_legacy.cvCreateFGDStatModel;
import static com.googlecode.javacpp.opencv_legacy.cvCreateGLCM;
import static com.googlecode.javacpp.opencv_legacy.cvCreateGaussianBGModel;
import static com.googlecode.javacpp.opencv_legacy.cvCreateObsInfo;
import static com.googlecode.javacpp.opencv_legacy.cvInitFaceTracker;
import static com.googlecode.javacpp.opencv_legacy.cvRelease2DHMM;
import static com.googlecode.javacpp.opencv_legacy.cvReleaseBGCodeBookModel;
import static com.googlecode.javacpp.opencv_legacy.cvReleaseBGStatModel;
import static com.googlecode.javacpp.opencv_legacy.cvReleaseConDensation;
import static com.googlecode.javacpp.opencv_legacy.cvReleaseFaceTracker;
import static com.googlecode.javacpp.opencv_legacy.cvReleaseGLCM;
import static com.googlecode.javacpp.opencv_legacy.cvReleaseObsInfo;

public class opencv_legacy {

    public static abstract class AbstractCvImgObsInfo extends Pointer {
        public AbstractCvImgObsInfo() { }
        public AbstractCvImgObsInfo(Pointer p) { super(p); }

        public static CvImgObsInfo create(CvSize numObs, int obsSize) {
            CvImgObsInfo p = cvCreateObsInfo(numObs, obsSize);
            if (p != null) {
                p.deallocator(new ReleaseDeallocator(p));
            }
            return p;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvImgObsInfo implements Deallocator {
            ReleaseDeallocator(CvImgObsInfo p) { super(p); }
            @Override public void deallocate() { cvReleaseObsInfo(this); }
        }
    }

    public static abstract class AbstractCvEHMM extends Pointer {
        public AbstractCvEHMM() { }
        public AbstractCvEHMM(Pointer p) { super(p); }

        public static CvEHMM create(int[] stateNumber, int[] numMix, int obsSize) {
            CvEHMM p = cvCreate2DHMM(stateNumber, numMix, obsSize);
            if (p != null) {
                p.deallocator(new ReleaseDeallocator(p));
            }
            return p;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvEHMM implements Deallocator {
            ReleaseDeallocator(CvEHMM p) { super(p); }
            @Override public void deallocate() { cvRelease2DHMM(this); }
        }
    }

    public static abstract class AbstractCvGLCM extends Pointer {
        public AbstractCvGLCM() { }
        public AbstractCvGLCM(Pointer p) { super(p); }

        public static CvGLCM create(IplImage srcImage, int stepMagnitude) {
            return create(srcImage, stepMagnitude, null, 0, CV_GLCM_OPTIMIZATION_NONE);
        }
        public static CvGLCM create(IplImage srcImage, int stepMagnitude,
                int[] stepDirections/*=null*/, int numStepDirections/*=0*/,
                int optimizationType/*=CV_GLCM_OPTIMIZATION_NONE*/) {
            CvGLCM p = cvCreateGLCM(srcImage, stepMagnitude, stepDirections,
                    numStepDirections, optimizationType);
            if (p != null) {
                p.deallocator(new ReleaseDeallocator(p));
            }
            return p;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvGLCM implements Deallocator {
            ReleaseDeallocator(CvGLCM p) { super(p); }
            @Override public void deallocate() { cvReleaseGLCM(this, CV_GLCM_ALL); }
        }
    }

    public static abstract class AbstractCvFaceTracker extends Pointer {
        public AbstractCvFaceTracker() { }
        public AbstractCvFaceTracker(Pointer p) { super(p); }

        public static CvFaceTracker create(CvFaceTracker pFaceTracking,
                IplImage imgGray, CvRect pRects, int nRects) {
            CvFaceTracker p = cvInitFaceTracker(new CvFaceTracker(), imgGray, pRects, nRects);
            if (p != null) {
                p.deallocator(new ReleaseDeallocator(p));
            }
            return p;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvFaceTracker implements Deallocator {
            ReleaseDeallocator(CvFaceTracker p) { super(p); }
            @Override public void deallocate() { cvReleaseFaceTracker(this); }
        }
    }

    public static abstract class AbstractCvConDensation extends Pointer {
        public AbstractCvConDensation() { }
        public AbstractCvConDensation(Pointer p) { super(p); }

        public static CvConDensation create(int dynam_params, int measure_params,
                int sample_count) {
            CvConDensation c = cvCreateConDensation(dynam_params, measure_params, sample_count);
            if (c != null) {
                c.deallocator(new ReleaseDeallocator(c));
            }
            return c;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvConDensation implements Deallocator {
            ReleaseDeallocator(CvConDensation p) { super(p); }
            @Override public void deallocate() { cvReleaseConDensation(this); }
        }
    }

    public static abstract class AbstractCvBGStatModel extends Pointer {
        public AbstractCvBGStatModel() { }
        public AbstractCvBGStatModel(Pointer p) { super(p); }

        public static CvBGStatModel create(IplImage first_frame, CvFGDStatModelParams parameters) {
            CvBGStatModel m = cvCreateFGDStatModel(first_frame, parameters);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }
        public static CvBGStatModel create(IplImage first_frame, CvGaussBGStatModelParams parameters) {
            CvBGStatModel m = cvCreateGaussianBGModel(first_frame, parameters);
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }

        public void release2() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvBGStatModel implements Deallocator {
            ReleaseDeallocator(CvBGStatModel p) { super(p); }
            @Override public void deallocate() { cvReleaseBGStatModel(this); }
        }
    }

    public static abstract class AbstractCvBGCodeBookModel extends Pointer {
        public AbstractCvBGCodeBookModel() { }
        public AbstractCvBGCodeBookModel(Pointer p) { super(p); }

        public static CvBGCodeBookModel create() {
            CvBGCodeBookModel m = cvCreateBGCodeBookModel();
            if (m != null) {
                m.deallocator(new ReleaseDeallocator(m));
            }
            return m;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvBGCodeBookModel implements Deallocator {
            ReleaseDeallocator(CvBGCodeBookModel p) { super(p); }
            @Override public void deallocate() { cvReleaseBGCodeBookModel(this); }
        }
    }

}
