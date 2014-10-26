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
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_objdetect.cvLoadHaarClassifierCascade;
import static org.bytedeco.javacpp.opencv_objdetect.cvReleaseHaarClassifierCascade;

public class opencv_objdetect extends org.bytedeco.javacpp.presets.opencv_objdetect {

    public static abstract class AbstractCvHaarClassifierCascade extends Pointer {
        public AbstractCvHaarClassifierCascade() { }
        public AbstractCvHaarClassifierCascade(Pointer p) { super(p); }

        /**
         * Calls cvLoadHaarClassifierCascade(), and registers a deallocator.
         * @return CvHaarClassifierCascade loaded. Do not call cvReleaseHaarClassifierCascade() on it.
         */
        public static CvHaarClassifierCascade load(String directory,
                CvSize orig_window_size) {
            CvHaarClassifierCascade h = cvLoadHaarClassifierCascade(directory,
                    orig_window_size);
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
        static class ReleaseDeallocator extends CvHaarClassifierCascade implements Deallocator {
            ReleaseDeallocator(CvHaarClassifierCascade p) { super(p); }
            @Override public void deallocate() { cvReleaseHaarClassifierCascade(this); }
        }
    }

    public static CvSeq cvHaarDetectObjects(opencv_core.CvArr image, CvHaarClassifierCascade cascade,
            CvMemStorage storage, double scale_factor/*=1.1*/, int min_neighbors/*=3*/, int flags/*=0*/) {
        return org.bytedeco.javacpp.opencv_objdetect.cvHaarDetectObjects(image, cascade,
                storage, scale_factor, min_neighbors, flags, CvSize.ZERO, CvSize.ZERO);
    }

}
