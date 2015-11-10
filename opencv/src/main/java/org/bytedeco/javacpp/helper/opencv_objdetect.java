/*
 * Copyright (C) 2014 Samuel Audet
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

import org.bytedeco.javacpp.Pointer;

// required by javac to resolve circular dependencies
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_objdetect.cvLoadHaarClassifierCascade;
import static org.bytedeco.javacpp.opencv_objdetect.cvReleaseHaarClassifierCascade;

public class opencv_objdetect extends org.bytedeco.javacpp.presets.opencv_objdetect {

    public static abstract class AbstractCvHaarClassifierCascade extends Pointer {
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
