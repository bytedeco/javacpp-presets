/*
 * Copyright (C) 2013 Samuel Audet
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

package com.googlecode.javacpp.presets;

import com.googlecode.javacpp.Parser;
import com.googlecode.javacpp.annotation.Platform;
import com.googlecode.javacpp.annotation.Properties;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit=opencv_imgproc.class, target="com.googlecode.javacpp.opencv_highgui", value={
    @Platform(include="<opencv2/highgui/highgui_c.h>", link="opencv_highgui@.2.4"),
    @Platform(value="windows", link="opencv_highgui248", preload={"opencv_ffmpeg248", "opencv_ffmpeg248_64"}) })
public class opencv_highgui implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        new opencv_imgproc().map(infoMap);
        infoMap.put(new Parser.Info("cvCaptureFromFile", "cvCaptureFromAVI").cppTypes("CvCapture*", "const char*"))
               .put(new Parser.Info("cvCaptureFromCAM").cppTypes("CvCapture*", "int"))
               .put(new Parser.Info("cvCreateAVIWriter").cppTypes("CvVideoWriter*", "const char*", "int", "double", "CvSize", "int"))
               .put(new Parser.Info("cvWriteToAVI").cppTypes("int", "CvVideoWriter*", "IplImage*"))
               .put(new Parser.Info("cvvInitSystem").cppTypes("int", "int", "char**"))
               .put(new Parser.Info("cvvNamedWindow").cppTypes("void", "const char*", "int"))
               .put(new Parser.Info("cvvShowImage").cppTypes("void", "const char*", "CvArr*"))
               .put(new Parser.Info("cvvResizeWindow").cppTypes("void", "const char*", "int", "int"))
               .put(new Parser.Info("cvvDestroyWindow").cppTypes("void", "const char*"))
               .put(new Parser.Info("cvvCreateTrackbar").cppTypes("int", "const char*", "const char*", "int*", "int", "CvTrackbarCallback*"))
               .put(new Parser.Info("cvvLoadImage").cppTypes("IplImage*", "const char*"))
               .put(new Parser.Info("cvvSaveImage").cppTypes("int", "const char*", "CvArr*", "int*"))
               .put(new Parser.Info("cvvAddSearchPath", "cvAddSearchPath").cppTypes("void", "const char*"))
               .put(new Parser.Info("cvvWaitKey").cppTypes("int", "const char*"))
               .put(new Parser.Info("cvvWaitKeyEx").cppTypes("int", "const char*", "int"))
               .put(new Parser.Info("cvvConvertImage").cppTypes("void", "CvArr*", "CvArr*", "int"))
               .put(new Parser.Info("set_preprocess_func", "set_postprocess_func").cppTypes());
    }
}
