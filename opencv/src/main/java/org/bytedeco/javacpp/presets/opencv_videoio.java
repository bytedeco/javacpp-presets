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

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = opencv_imgcodecs.class, value = {
    @Platform(include = {"<opencv2/videoio/videoio_c.h>", "<opencv2/videoio.hpp>"}, link = "opencv_videoio@.3.0"),
    @Platform(value = "android", preload = {
        "native_camera_r2.2.0", "native_camera_r2.3.3", "native_camera_r3.0.1", "native_camera_r4.0.0", "native_camera_r4.0.3",
        "native_camera_r4.1.1", "native_camera_r4.2.0", "native_camera_r4.3.0", "native_camera_r4.4.0"}),
    @Platform(value = "windows", link = "opencv_videoio300", preload = {"opencv_ffmpeg300", "opencv_ffmpeg300_64"})},
        target = "org.bytedeco.javacpp.opencv_videoio")
public class opencv_videoio implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CV_FOURCC_DEFAULT").cppTypes("int"))
               .put(new Info("cvCaptureFromFile", "cvCaptureFromAVI").cppTypes("CvCapture*", "const char*"))
               .put(new Info("cvCaptureFromCAM").cppTypes("CvCapture*", "int"))
               .put(new Info("cvCreateAVIWriter").cppTypes("CvVideoWriter*", "const char*", "int", "double", "CvSize", "int"))
               .put(new Info("cvWriteToAVI").cppTypes("int", "CvVideoWriter*", "IplImage*"));
    }
}
