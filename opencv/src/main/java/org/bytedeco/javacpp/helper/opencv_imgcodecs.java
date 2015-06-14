/*
 * Copyright (C) 2015 Samuel Audet
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

// required by javac to resolve circular dependencies
import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvReleaseImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2BGRA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2RGBA;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;

public class opencv_imgcodecs extends org.bytedeco.javacpp.presets.opencv_imgcodecs {

    public static IplImage cvLoadImageBGRA(String filename) {
        IplImage imageBGR = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
        if (imageBGR == null) {
            return null;
        } else {
            IplImage imageBGRA = cvCreateImage(cvGetSize(imageBGR), imageBGR.depth(), 4);
            cvCvtColor(imageBGR, imageBGRA, CV_BGR2BGRA);
            cvReleaseImage(imageBGR);
            return imageBGRA;
        }
    }

    public static IplImage cvLoadImageRGBA(String filename) {
        IplImage imageBGR = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
        if (imageBGR == null) {
            return null;
        } else {
            IplImage imageRGBA = cvCreateImage(cvGetSize(imageBGR), imageBGR.depth(), 4);
            cvCvtColor(imageBGR, imageRGBA, CV_BGR2RGBA);
            cvReleaseImage(imageBGR);
            return imageRGBA;
        }
    }

}
