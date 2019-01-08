/*
 * Copyright (C) 2014-2018 Samuel Audet
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

package org.bytedeco.javacpp.opencv_core;


import org.bytedeco.javacpp.IntPointer;

// required by javac to resolve circular dependencies
import static org.bytedeco.javacpp.opencv_core.opencv_core.CV_L2;
//import static org.bytedeco.javacpp.opencv_core.opencv_core.cvOpenFileStorage;
//import static org.bytedeco.javacpp.opencv_core.opencv_core.cvReleaseFileStorage;
import static org.bytedeco.javacpp.opencv_core.opencv_core.cvScalar;

public class opencv_core_helper extends opencv_core_presets {

    public static CvScalar CV_RGB(double r, double g, double b) {
        return cvScalar(b, g, r, 0);
    }

//    public static abstract class AbstractCvFileStorage extends Pointer {
//        public AbstractCvFileStorage(Pointer p) { super(p); }
//
//        /**
//         * Calls cvOpenFileStorage(), and registers a deallocator. Uses default encoding.
//         * @return CvFileStorage opened. Do not call cvReleaseFileStorage() on it.
//         */
//        public static CvFileStorage open(String filename, CvMemStorage memstorage, int flags) {
//            return open(filename, memstorage, flags, null);
//        }
//        /**
//         * Calls cvOpenFileStorage(), and registers a deallocator.
//         * @return CvFileStorage opened. Do not call cvReleaseFileStorage() on it.
//         */
//        public static CvFileStorage open(String filename, CvMemStorage memstorage, int flags, String encoding) {
//            CvFileStorage f = cvOpenFileStorage(filename, memstorage, flags, encoding);
//            if (f != null) {
//                f.deallocator(new ReleaseDeallocator(f));
//            }
//            return f;
//        }
//
//        /**
//         * Calls the deallocator, if registered, otherwise has no effect.
//         */
//        public void release() {
//            deallocate();
//        }
//        protected static class ReleaseDeallocator extends CvFileStorage implements Deallocator {
//            ReleaseDeallocator(CvFileStorage p) { super(p); }
//            @Override public void deallocate() { cvReleaseFileStorage(this); }
//        }
//    }

    public static int cvInitNArrayIterator(int count, CvArr[] arrs,
            CvArr mask, CvMatND stubs, CvNArrayIterator array_iterator, int flags/*=0*/) {
        return opencv_core.cvInitNArrayIterator(count, new CvArrArray(arrs),
                mask, stubs, array_iterator, flags);
    }

    public static void cvMixChannels(CvArr[] src, int src_count,
            CvArr[] dst, int dst_count, int[] from_to, int pair_count) {
        opencv_core.cvMixChannels(new CvArrArray(src), src_count,
                new CvArrArray(dst), dst_count, new IntPointer(from_to), pair_count);
    }

    public static void cvCalcCovarMatrix(CvArr[] vects, int count, CvArr cov_mat, CvArr avg, int flags) {
        opencv_core.cvCalcCovarMatrix(new CvArrArray(vects), count, cov_mat, avg, flags);
    }

    public static double cvNorm(CvArr arr1, CvArr arr2) {
        return opencv_core.cvNorm(arr1, arr2, CV_L2, null);
    }

    public static Scalar RGB(double r, double g, double b) {
        return new Scalar(b, g, r, 0);
    }
}
