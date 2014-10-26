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

package org.bytedeco.javacpp.helper;

import org.bytedeco.javacpp.Pointer;

// required by javac to resolve circular dependencies
import org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_video.cvCreateKalman;
import static org.bytedeco.javacpp.opencv_video.cvReleaseKalman;

public class opencv_video extends org.bytedeco.javacpp.presets.opencv_video {

    public static abstract class AbstractCvKalman extends Pointer {
        public AbstractCvKalman() { }
        public AbstractCvKalman(Pointer p) { super(p); }

        /**
         * Calls cvCreateKalman(), and registers a deallocator.
         * @return CvKalman created. Do not call cvReleaseKalman() on it.
         */
        public static CvKalman create(int dynam_params, int measure_params,
                int control_params/*=0*/) {
            CvKalman k = cvCreateKalman(dynam_params, measure_params, control_params);
            if (k != null) {
                k.deallocator(new ReleaseDeallocator(k));
            }
            return k;
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvKalman implements Deallocator {
            ReleaseDeallocator(CvKalman p) { super(p); }
            @Override public void deallocate() { cvReleaseKalman(this); }
        }
    }

}
