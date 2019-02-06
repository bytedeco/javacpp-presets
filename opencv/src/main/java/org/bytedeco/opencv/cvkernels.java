/*
 * Copyright (C) 2009-2012 Samuel Audet
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

package org.bytedeco.opencv;

import java.nio.DoubleBuffer;
import java.util.Arrays;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.MemberSetter;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.opencv.opencv_core.CvMat;
import org.bytedeco.opencv.opencv_core.CvRect;
import org.bytedeco.opencv.opencv_core.CvScalar;
import org.bytedeco.opencv.opencv_core.IplImage;

import static org.bytedeco.javacpp.Loader.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit=org.bytedeco.opencv.presets.opencv_core.class, value={
    @Platform(define={"MAX_SIZE 16", "CV_INLINE static inline"}, include="cvkernels.h", compiler="fastfpu") })
public class cvkernels {
    static { load(); }

    public static class KernelData extends Pointer {
        static { load(); }
        public KernelData() { allocate(); }
        public KernelData(long size) { allocateArray(size); }
        public KernelData(Pointer p) { super(p); }
        private native void allocate();
        private native void allocateArray(long size);

        @Override public KernelData position(long position) {
            return (KernelData)super.position(position);
        }

        // input
        public native IplImage srcImg();         public native KernelData srcImg(IplImage srcImg);
        public native IplImage srcImg2();        public native KernelData srcImg2(IplImage srcImg2);
        public native IplImage subImg();         public native KernelData subImg(IplImage subImg);
        public native IplImage srcDotImg();      public native KernelData srcDotImg(IplImage srcDotImg);
        public native IplImage mask();           public native KernelData mask(IplImage mask);
        public native double zeroThreshold();    public native KernelData zeroThreshold(double zeroThreshold);
        public native double outlierThreshold(); public native KernelData outlierThreshold(double outlierThreshold);
        public native CvMat H1();                public native KernelData H1(CvMat H1);
        public native CvMat H2();                public native KernelData H2(CvMat H2);
        public native CvMat X();                 public native KernelData X (CvMat X);

        // output
        public native IplImage transImg();       public native KernelData transImg(IplImage transImg);
        public native IplImage dstImg();         public native KernelData dstImg(IplImage dstImg);
        public native int dstCount();            public native KernelData dstCount(int dstCount);
        public native int dstCountZero();        public native KernelData dstCountZero(int dstCountZero);
        public native int dstCountOutlier();     public native KernelData dstCountOutlier(int dstCountOutlier);
        public native double srcDstDot();        public native KernelData srcDstDot(double srcDstDot);
//        public native DoublePointer dstDstDot(); public native KernelData dstDstDot(DoublePointer dstDstDot);

        // Hack to let us use DoubleBuffer directly instead of DoublePointer, which also
        // provides us with Java references to boot, keeping the garbage collector happy
        private native @MemberSetter @Name("dstDstDot") KernelData setDstDstDot(DoubleBuffer dstDstDot);
        private DoubleBuffer[] dstDstDotBuffers = new DoubleBuffer[1];
        public DoubleBuffer dstDstDot() {
            return dstDstDotBuffers[(int)position];
        }
        public KernelData dstDstDot(DoubleBuffer dstDstDot) {
            if (dstDstDotBuffers.length < capacity) {
                dstDstDotBuffers = Arrays.copyOf(dstDstDotBuffers, (int)capacity);
            }
            dstDstDotBuffers[(int)position] = dstDstDot;
            return setDstDstDot(dstDstDot);
        }

        private native @Name("operator=") @ByRef KernelData put(@ByRef KernelData x);
    }

    public static native void multiWarpColorTransform32F(KernelData data, int size, CvRect roi, CvScalar fillColor);
    public static native void multiWarpColorTransform8U(KernelData data, int size, CvRect roi, CvScalar fillColor);
}
