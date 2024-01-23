/*
 * Copyright (C) 2013-2022 Samuel Audet
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

package org.bytedeco.opencv.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = opencv_core.class,
    value = {
        @Platform(include = {"<opencv2/imgproc/types_c.h>", "<opencv2/imgproc/imgproc_c.h>", "<opencv2/imgproc.hpp>",
            "<opencv2/imgproc/detail/gcgraph.hpp>"}, link = "opencv_imgproc@.409"),
        @Platform(value = "ios", preload = "libopencv_imgproc"),
        @Platform(value = "windows", link = "opencv_imgproc490")},
    target = "org.bytedeco.opencv.opencv_imgproc",
    global = "org.bytedeco.opencv.global.opencv_imgproc",
    helper = "org.bytedeco.opencv.helper.opencv_imgproc"
)
public class opencv_imgproc implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CvMoments").base("AbstractCvMoments"))
               .put(new Info("_CvContourScanner").pointerTypes("CvContourScanner"))
               .put(new Info("CvContourScanner").valueTypes("CvContourScanner").pointerTypes("@ByPtrPtr CvContourScanner"))
               .put(new Info("cvCalcBackProject").cppTypes("void", "IplImage**", "CvArr*", "CvHistogram*"))
               .put(new Info("cvCalcBackProjectPatch").cppTypes("void", "IplImage**", "CvArr*", "CvSize", "CvHistogram*", "int", "double"))
               .put(new Info("cv::Matx23d").cast().pointerTypes("DoublePointer"))
               .put(new Info("cv::Vec6f").cast().pointerTypes("FloatPointer"))
               .put(new Info("std::vector<cv::Vec2f>").pointerTypes("Vec2fVector").define())
               .put(new Info("std::vector<cv::Vec3f>").pointerTypes("Vec3fVector").define())
               .put(new Info("std::vector<cv::Vec4f>").pointerTypes("Vec4fVector").define())
               .put(new Info("std::vector<cv::Vec4i>").pointerTypes("Vec4iVector").define())
               .put(new Info("std::vector<cv::Vec6f>").pointerTypes("Vec6fVector").define())
               .put(new Info("cv::HoughLines").javaText(
                        "@Namespace(\"cv\") public static native void HoughLines( @ByVal Mat image, @ByVal Vec2fVector lines,\n"
                      + "                              double rho, double theta, int threshold,\n"
                      + "                              double srn/*=0*/, double stn/*=0*/,\n"
                      + "                              double min_theta/*=0*/, double max_theta/*=CV_PI*/ );\n"
                      + "@Namespace(\"cv\") public static native void HoughLines( @ByVal Mat image, @ByVal Vec3fVector lines,\n"
                      + "                              double rho, double theta, int threshold,\n"
                      + "                              double srn/*=0*/, double stn/*=0*/,\n"
                      + "                              double min_theta/*=0*/, double max_theta/*=CV_PI*/ );\n"))
               .put(new Info("cv::HoughLinesP").javaText(
                        "@Namespace(\"cv\") public static native void HoughLinesP( @ByVal Mat image, @ByVal Vec4iVector lines,\n"
                      + "                               double rho, double theta, int threshold,\n"
                      + "                               double minLineLength/*=0*/, double maxLineGap/*=0*/ );\n"))
               .put(new Info("cv::HoughCircles").javaText(
                        "@Namespace(\"cv\") public static native void HoughCircles( @ByVal Mat image, @ByVal Vec3fVector circles,\n"
                      + "                               int method, double dp, double minDist,\n"
                      + "                               double param1/*=100*/, double param2/*=100*/,\n"
                      + "                               int minRadius/*=0*/, int maxRadius/*=0*/ );\n"
                      + "@Namespace(\"cv\") public static native void HoughCircles( @ByVal Mat image, @ByVal Vec4fVector circles,\n"
                      + "                               int method, double dp, double minDist,\n"
                      + "                               double param1/*=100*/, double param2/*=100*/,\n"
                      + "                               int minRadius/*=0*/, int maxRadius/*=0*/ );\n"));

    }
}
