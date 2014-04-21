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
import org.bytedeco.javacpp.opencv_calib3d.*;
import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_calib3d.cvCreatePOSITObject;
import static org.bytedeco.javacpp.opencv_calib3d.cvCreateStereoBMState;
import static org.bytedeco.javacpp.opencv_calib3d.cvReleasePOSITObject;
import static org.bytedeco.javacpp.opencv_calib3d.cvReleaseStereoBMState;

public class opencv_calib3d extends org.bytedeco.javacpp.presets.opencv_calib3d {

    public static abstract class AbstractCvPOSITObject extends Pointer {
        public AbstractCvPOSITObject() { }
        public AbstractCvPOSITObject(Pointer p) { super(p); }

        public static CvPOSITObject create(CvPoint3D32f points, int point_count) {
            CvPOSITObject p = cvCreatePOSITObject(points, point_count);
            if (p != null) {
                p.deallocator(new ReleaseDeallocator(p));
            }
            return p;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvPOSITObject implements Deallocator {
            ReleaseDeallocator(CvPOSITObject p) { super(p); }
            @Override public void deallocate() { cvReleasePOSITObject(this); }
        }
    }

    public static abstract class AbstractCvStereoBMState extends Pointer {
        public AbstractCvStereoBMState() { }
        public AbstractCvStereoBMState(Pointer p) { super(p); }

        @Override public CvStereoBMState position(int position) {
            return (CvStereoBMState)super.position(position);
        }

        public static CvStereoBMState create(int preset, int numberOfDisparities) {
            CvStereoBMState p = cvCreateStereoBMState(preset, numberOfDisparities);
            if (p != null) {
                p.deallocator(new ReleaseDeallocator(p));
            }
            return p;
        }
        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvStereoBMState implements Deallocator {
            ReleaseDeallocator(CvStereoBMState p) { super(p); }
            @Override public void deallocate() { cvReleaseStereoBMState(this); }
        }
    }

}
