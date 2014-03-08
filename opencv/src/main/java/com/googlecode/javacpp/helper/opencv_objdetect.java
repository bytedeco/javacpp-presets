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
import com.googlecode.javacpp.opencv_objdetect.*;
import static com.googlecode.javacpp.opencv_objdetect.cvLoadHaarClassifierCascade;
import static com.googlecode.javacpp.opencv_objdetect.cvReleaseHaarClassifierCascade;

public class opencv_objdetect {

    public static abstract class AbstractCvHaarClassifierCascade extends Pointer {
        public AbstractCvHaarClassifierCascade() { }
        public AbstractCvHaarClassifierCascade(Pointer p) { super(p); }

        public static CvHaarClassifierCascade load(String directory,
                CvSize orig_window_size) {
            CvHaarClassifierCascade h = cvLoadHaarClassifierCascade(directory,
                    orig_window_size);
            if (h != null) {
                h.deallocator(new ReleaseDeallocator(h));
            }
            return h;
        }

        public void release() {
            deallocate();
        }
        static class ReleaseDeallocator extends CvHaarClassifierCascade implements Deallocator {
            ReleaseDeallocator(CvHaarClassifierCascade p) { super(p); }
            @Override public void deallocate() { cvReleaseHaarClassifierCascade(this); }
        }
    }

}
