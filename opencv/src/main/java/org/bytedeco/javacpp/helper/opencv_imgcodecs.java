/*
 * Copyright (C) 2015 Samuel Audet
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
