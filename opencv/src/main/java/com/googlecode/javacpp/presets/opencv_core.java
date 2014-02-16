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
@Properties(target="com.googlecode.javacpp.opencv_core", value={
    @Platform(include={"<opencv2/core/types_c.h>", "<opencv2/core/core_c.h>"}, link="opencv_core@.2.4", preload="tbb"),
    @Platform(value="windows", define="_WIN32_WINNT 0x0502", includepath="C:/opencv/build/include/",
        link="opencv_core248", preload={"msvcr100", "msvcp100"}),
    @Platform(value="windows-x86",    linkpath="C:/opencv/build/x86/vc10/lib/", preloadpath="C:/opencv/build/x86/vc10/bin/"),
    @Platform(value="windows-x86_64", linkpath="C:/opencv/build/x64/vc10/lib/", preloadpath="C:/opencv/build/x64/vc10/bin/") })
public class opencv_core implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        infoMap.put(new Parser.Info("__cplusplus").define(false))
               .put(new Parser.Info("defined __ICL", "defined __ICC", "defined __ECL", "defined __ECC",
                                    "defined __INTEL_COMPILER", "defined WIN32 || defined _WIN32").define(false))
               .put(new Parser.Info("CV_ENABLE_UNROLLED", "CV_CDECL", "CV_STDCALL", "CV_EXTERN_C", "CV_Func").cppTypes())
               .put(new Parser.Info("CV_DEFAULT", "CV_INLINE", "CV_EXPORTS").cppTypes().annotations())
               .put(new Parser.Info("CVAPI").text("#define CVAPI(rettype) rettype"))
               .put(new Parser.Info("CV_EXPORTS_W", "CV_EXPORTS_W_SIMPLE", "CV_EXPORTS_W_MAP",
                                    "CV_IN_OUT", "CV_OUT", "CV_PROP", "CV_PROP_RW", "CV_WRAP").cppTypes().annotations())
               .put(new Parser.Info("CV_MAT_DEPTH", "CV_8UC", "CV_8SC", "CV_16UC", "CV_16SC", "CV_32SC", "CV_32FC", "CV_64FC").cppTypes("int", "int"))
               .put(new Parser.Info("CV_MAKETYPE", "CV_MAKE_TYPE").cppTypes("int", "int", "int"))
               .put(new Parser.Info("CV_IS_MAT_CONT", "CV_IS_CONT_MAT").cppTypes("int", "int"))
               .put(new Parser.Info("CV_8UC1", "CV_8UC2", "CV_8UC3", "CV_8UC4",
                                    "CV_8SC1", "CV_8SC2", "CV_8SC3", "CV_8SC4",
                                    "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                                    "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                                    "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                                    "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                                    "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4").cppTypes("int").translate(true))
               .put(new Parser.Info("CV_WHOLE_ARR", "CV_WHOLE_SEQ").cppTypes("CvSlice").translate(true))
               .put(new Parser.Info("_IplROI").pointerTypes("IplROI"))
               .put(new Parser.Info("_IplImage").pointerTypes("IplImage"))
               .put(new Parser.Info("_IplTileInfo").pointerTypes("IplTileInfo"))
               .put(new Parser.Info("IplImage", "CvMat", "CvMatND", "CvSparseMat", "CvSeq").parent("CvArr"))
               .put(new Parser.Info("CvSet", "CvChain", "CvContour", "CvContourTree").parent("CvSeq"))
               .put(new Parser.Info("CvGraph").parent("CvSet"))
               .put(new Parser.Info("CvGraphVtx2D").parent("CvGraphVtx"))
               .put(new Parser.Info("CvChainPtReader").parent("CvSeqReader"))
               .put(new Parser.Info("cvGetSubArr").cppTypes("CvMat*", "CvArr*", "CvMat*", "CvRect"))
               .put(new Parser.Info("cvZero").cppTypes("void", "CvArr*"))
               .put(new Parser.Info("cvCvtScale", "cvScale", "cvCvtScaleAbs").cppTypes("void", "CvArr*", "CvArr*", "double", "double"))
               .put(new Parser.Info("cvConvert", "cvT").cppTypes("void", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvCheckArray").cppTypes("int", "CvArr*", "int", "double", "double"))
               .put(new Parser.Info("cvMatMulAddEx").cppTypes("void", "CvArr*", "CvArr*", "double", "CvArr*", "double", "CvArr*", "int"))
               .put(new Parser.Info("cvMatMulAddS").cppTypes("void", "CvArr*", "CvArr*", "CvMat*", "CvMat*"))
               .put(new Parser.Info("cvMirror", "cvInv").cppTypes("void", "CvArr*", "CvArr*", "int"))
               .put(new Parser.Info("cvMahalonobis").cppTypes("double", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvFFT").cppTypes("void", "CvArr*", "CvArr*", "int", "int"))
               .put(new Parser.Info("cvGraphFindEdge").cppTypes("CvGraphEdge*", "CvGraph*", "int", "int"))
               .put(new Parser.Info("cvGraphFindEdgeByPtr").cppTypes("CvGraphEdge*", "CvGraph*", "CvGraphVtx*", "CvGraphVtx*"))
               .put(new Parser.Info("cvDrawRect", "cvDrawLine").cppTypes("void", "CvArr*", "CvPoint", "CvPoint", "CvScalar", "int", "int", "int"))
               .put(new Parser.Info("cvDrawCircle").cppTypes("void", "CvArr*", "CvPoint", "int", "CvScalar", "int", "int", "int"))
               .put(new Parser.Info("cvDrawEllipse").cppTypes("void", "CvArr*", "CvPoint", "CvSize", "double", "double", "double", "CvScalar", "int", "int", "int"))
               .put(new Parser.Info("cvDrawPolyLine").cppTypes("void", "CvArr*", "CvPoint**", "int*", "int", "int", "CvScalar", "int", "int", "int"))
               .put(new Parser.Info("__CV_BEGIN__", "__CV_END__", "__CV_EXIT__").cppTypes());
    }
}
