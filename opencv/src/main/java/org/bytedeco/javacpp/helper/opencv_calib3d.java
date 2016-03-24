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
import org.bytedeco.javacpp.opencv_calib3d.*;
import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_calib3d.cvCreatePOSITObject;
import static org.bytedeco.javacpp.opencv_calib3d.cvCreateStereoBMState;
import static org.bytedeco.javacpp.opencv_calib3d.cvReleasePOSITObject;
import static org.bytedeco.javacpp.opencv_calib3d.cvReleaseStereoBMState;

public class opencv_calib3d extends org.bytedeco.javacpp.presets.opencv_calib3d {

    public static abstract class AbstractCvPOSITObject extends Pointer {
        public AbstractCvPOSITObject(Pointer p) { super(p); }

        /**
         * Calls cvCreatePOSITObject(), and registers a deallocator.
         * @return CvPOSITObject created. Do not call cvReleasePOSITObject() on it.
         */
        public static CvPOSITObject create(CvPoint3D32f points, int point_count) {
            CvPOSITObject p = cvCreatePOSITObject(points, point_count);
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
        static class ReleaseDeallocator extends CvPOSITObject implements Deallocator {
            ReleaseDeallocator(CvPOSITObject p) { super(p); }
            @Override public void deallocate() { cvReleasePOSITObject(this); }
        }
    }

    public static abstract class AbstractCvStereoBMState extends Pointer {
        public AbstractCvStereoBMState(Pointer p) { super(p); }

        @Override public CvStereoBMState position(long position) {
            return (CvStereoBMState)super.position(position);
        }

        /**
         * Calls cvCreateStereoBMState(), and registers a deallocator.
         * @return CvStereoBMState created. Do not call cvReleaseStereoBMState() on it.
         */
        public static CvStereoBMState create(int preset, int numberOfDisparities) {
            CvStereoBMState p = cvCreateStereoBMState(preset, numberOfDisparities);
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
        static class ReleaseDeallocator extends CvStereoBMState implements Deallocator {
            ReleaseDeallocator(CvStereoBMState p) { super(p); }
            @Override public void deallocate() { cvReleaseStereoBMState(this); }
        }
    }

}
