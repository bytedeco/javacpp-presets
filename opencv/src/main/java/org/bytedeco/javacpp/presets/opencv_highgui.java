/*
 * Copyright (C) 2013,2014 Samuel Audet
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
@Properties(inherit=opencv_imgproc.class, value={
    @Platform(include={"<opencv2/highgui/highgui_c.h>","<opencv2/highgui/highgui.hpp>"}, link="opencv_highgui@.2.4"),
    @Platform(value="windows", link="opencv_highgui2410", preload={"opencv_ffmpeg2410", "opencv_ffmpeg2410_64"}),
    @Platform(value="android", preload={"native_camera_r2.2.0", "native_camera_r2.3.3", "native_camera_r3.0.1", "native_camera_r4.0.0", "native_camera_r4.0.3",
                                        "native_camera_r4.1.1", "native_camera_r4.2.0", "native_camera_r4.3.0", "native_camera_r4.4.0"}) },
        target="org.bytedeco.javacpp.opencv_highgui", helper="org.bytedeco.javacpp.helper.opencv_highgui")
public class opencv_highgui implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cvFontQt").annotations("@Platform(\"linux\")").javaNames("cvFontQt"))
               .put(new Info("cvAddText").annotations("@Platform(\"linux\")").javaNames("cvAddText"))
               .put(new Info("cvDisplayOverlay").annotations("@Platform(\"linux\")").javaNames("cvDisplayOverlay"))
               .put(new Info("cvDisplayStatusBar").annotations("@Platform(\"linux\")").javaNames("cvDisplayStatusBar"))
               .put(new Info("cvSaveWindowParameters").annotations("@Platform(\"linux\")").javaNames("cvSaveWindowParameters"))
               .put(new Info("cvLoadWindowParameters").annotations("@Platform(\"linux\")").javaNames("cvLoadWindowParameters"))
               .put(new Info("cvStartLoop").annotations("@Platform(\"linux\")").javaNames("cvStartLoop"))
               .put(new Info("cvStopLoop").annotations("@Platform(\"linux\")").javaNames("cvStopLoop"))
               .put(new Info("cvCreateButton").annotations("@Platform(\"linux\")").javaNames("cvCreateButton"))
               .put(new Info("cvCaptureFromFile", "cvCaptureFromAVI").cppTypes("CvCapture*", "const char*"))
               .put(new Info("cvCaptureFromCAM").cppTypes("CvCapture*", "int"))
               .put(new Info("cvCreateAVIWriter").cppTypes("CvVideoWriter*", "const char*", "int", "double", "CvSize", "int"))
               .put(new Info("cvWriteToAVI").cppTypes("int", "CvVideoWriter*", "IplImage*"))
               .put(new Info("cvvInitSystem").cppTypes("int", "int", "char**"))
               .put(new Info("cvvNamedWindow").cppTypes("void", "const char*", "int"))
               .put(new Info("cvvShowImage").cppTypes("void", "const char*", "CvArr*"))
               .put(new Info("cvvResizeWindow").cppTypes("void", "const char*", "int", "int"))
               .put(new Info("cvvDestroyWindow").cppTypes("void", "const char*"))
               .put(new Info("cvvCreateTrackbar").cppTypes("int", "const char*", "const char*", "int*", "int", "CvTrackbarCallback"))
               .put(new Info("cvvLoadImage").cppTypes("IplImage*", "const char*"))
               .put(new Info("cvvSaveImage").cppTypes("int", "const char*", "CvArr*", "int*"))
               .put(new Info("cvvAddSearchPath", "cvAddSearchPath").cppTypes("void", "const char*"))
               .put(new Info("cvvWaitKey").cppTypes("int", "const char*"))
               .put(new Info("cvvWaitKeyEx").cppTypes("int", "const char*", "int"))
               .put(new Info("cvvConvertImage").cppTypes("void", "CvArr*", "CvArr*", "int"))
               .put(new Info("set_preprocess_func", "set_postprocess_func").cppTypes());
    }
}
