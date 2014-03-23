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

package com.googlecode.javacpp.presets;

import com.googlecode.javacpp.Parser;
import com.googlecode.javacpp.annotation.Adapter;
import com.googlecode.javacpp.annotation.Cast;
import com.googlecode.javacpp.annotation.Platform;
import com.googlecode.javacpp.annotation.Properties;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 *
 * @author Samuel Audet
 */
@Properties(value={
    @Platform(include={"<opencv2/core/types_c.h>", "<opencv2/core/core_c.h>", "<opencv2/core/core.hpp>", "opencv_adapters.h"}, link="opencv_core@.2.4", preload="tbb"),
    @Platform(value="windows", define="_WIN32_WINNT 0x0502", includepath="C:/opencv/build/include/",
        link="opencv_core248", preload={"msvcr100", "msvcp100"}),
    @Platform(value="windows-x86",    linkpath="C:/opencv/build/x86/vc10/lib/", preloadpath={"C:/opencv/build/x86/vc10/bin/",
            "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/redist/x86/Microsoft.VC100.CRT/"}),
    @Platform(value="windows-x86_64", linkpath="C:/opencv/build/x64/vc10/lib/", preloadpath={"C:/opencv/build/x64/vc10/bin/",
            "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/redist/x64/Microsoft.VC100.CRT/"}) },
        target="com.googlecode.javacpp.opencv_core", helper="com.googlecode.javacpp.helper.opencv_core")
public class opencv_core implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        infoMap.put(new Parser.Info("opencv_adapters.h").skip(true))
               .put(new Parser.Info("__cplusplus").define(true))
               .put(new Parser.Info("defined __ICL", "defined __ICC", "defined __ECL", "defined __ECC",
                                    "defined __INTEL_COMPILER", "defined WIN32 || defined _WIN32").define(false))
               .put(new Parser.Info("CV_ENABLE_UNROLLED", "CV_CDECL", "CV_STDCALL", "CV_EXTERN_C", "CV_Func").cppTypes())
               .put(new Parser.Info("CV_DEFAULT", "CV_INLINE", "CV_EXPORTS").cppTypes().annotations())
               .put(new Parser.Info("CVAPI").cppText("#define CVAPI(rettype) rettype"))
               .put(new Parser.Info("CV_EXPORTS_W", "CV_EXPORTS_W_SIMPLE", "CV_EXPORTS_AS", "CV_EXPORTS_W_MAP",
                                    "CV_IN_OUT", "CV_OUT", "CV_PROP", "CV_PROP_RW", "CV_WRAP", "CV_WRAP_AS").cppTypes().annotations())
               .put(new Parser.Info("CvRNG").cast(true).valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Parser.Info("CV_MAT_DEPTH", "CV_8UC", "CV_8SC", "CV_16UC", "CV_16SC", "CV_32SC", "CV_32FC", "CV_64FC").cppTypes("int", "int"))
               .put(new Parser.Info("CV_MAKETYPE", "CV_MAKE_TYPE").cppTypes("int", "int", "int"))
               .put(new Parser.Info("CV_8UC1", "CV_8UC2", "CV_8UC3", "CV_8UC4",
                                    "CV_8SC1", "CV_8SC2", "CV_8SC3", "CV_8SC4",
                                    "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                                    "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                                    "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                                    "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                                    "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4").cppTypes("int").translate(true))
               .put(new Parser.Info("CV_MAT_CN", "CV_MAT_TYPE", "CV_IS_CONT_MAT", "CV_IS_MAT_CONT").cppTypes("int", "int"))
               .put(new Parser.Info("CV_WHOLE_ARR", "CV_WHOLE_SEQ").cppTypes("CvSlice").translate(true))
               .put(new Parser.Info("_IplROI").pointerTypes("IplROI"))
               .put(new Parser.Info("_IplImage").pointerTypes("IplImage"))
               .put(new Parser.Info("_IplTileInfo").pointerTypes("IplTileInfo"))
               .put(new Parser.Info("IplImage").base("AbstractIplImage"))
               .put(new Parser.Info("IplConvKernel").base("com.googlecode.javacpp.helper.opencv_imgproc.AbstractIplConvKernel"))
               .put(new Parser.Info("CvMat").base("AbstractCvMat"))
               .put(new Parser.Info("CvMatND").base("AbstractCvMatND"))
               .put(new Parser.Info("CvSparseMat").base("AbstractCvSparseMat"))
               .put(new Parser.Info("CvHistogram").base("com.googlecode.javacpp.helper.opencv_imgproc.AbstractCvHistogram"))
               .put(new Parser.Info("CvRect").base("AbstractCvRect"))
               .put(new Parser.Info("CvPoint").cast(true).pointerTypes("CvPoint", "IntBuffer", "int[]").base("AbstractCvPoint"))
               .put(new Parser.Info("CvPoint2D32f").cast(true).pointerTypes("CvPoint2D32f", "FloatBuffer", "float[]").base("AbstractCvPoint2D32f"))
               .put(new Parser.Info("CvPoint3D32f").cast(true).pointerTypes("CvPoint3D32f", "FloatBuffer", "float[]").base("AbstractCvPoint3D32f"))
               .put(new Parser.Info("CvPoint2D64f").cast(true).pointerTypes("CvPoint2D64f", "DoubleBuffer", "double[]").base("AbstractCvPoint2D64f"))
               .put(new Parser.Info("CvPoint3D64f").cast(true).pointerTypes("CvPoint3D64f", "DoubleBuffer", "double[]").base("AbstractCvPoint3D64f"))
               .put(new Parser.Info("CvSize").base("AbstractCvSize"))
               .put(new Parser.Info("CvSize2D32f").base("AbstractCvSize2D32f"))
               .put(new Parser.Info("CvBox2D").base("AbstractCvBox2D"))
               .put(new Parser.Info("CvScalar").base("AbstractCvScalar"))
               .put(new Parser.Info("CvMemStorage").base("AbstractCvMemStorage"))
               .put(new Parser.Info("CvSeq").base("AbstractCvSeq"))
               .put(new Parser.Info("CvSet").base("AbstractCvSet"))
               .put(new Parser.Info("CvChain", "CvContour", "CvContourTree").base("CvSeq"))
               .put(new Parser.Info("CvGraph").base("AbstractCvGraph"))
               .put(new Parser.Info("CvGraphVtx2D").base("CvGraphVtx"))
               .put(new Parser.Info("CvChainPtReader").base("CvSeqReader"))
               .put(new Parser.Info("CvFileStorage").base("AbstractCvFileStorage"))
               .put(new Parser.Info("CvGraphScanner").base("AbstractCvGraphScanner"))
               .put(new Parser.Info("CvFont").base("AbstractCvFont"))
               .put(new Parser.Info("cvGetSubArr").cppTypes("CvMat*", "CvArr*", "CvMat*", "CvRect"))
               .put(new Parser.Info("cvZero").cppTypes("void", "CvArr*"))
               .put(new Parser.Info("cvCvtScale", "cvScale", "cvCvtScaleAbs").cppTypes("void", "CvArr*", "CvArr*", "double", "double"))
               .put(new Parser.Info("cvConvert", "cvT").cppTypes("void", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvCheckArray").cppTypes("int", "CvArr*", "int", "double", "double"))
               .put(new Parser.Info("cvMatMulAdd").cppTypes("void", "CvArr*", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Parser.Info("cvMatMul").cppTypes("void", "CvArr*", "CvArr*", "CvArr*"))
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
               .put(new Parser.Info("__CV_BEGIN__", "__CV_END__", "__CV_EXIT__").cppTypes())

               .put(new Parser.Info("std::vector<std::vector<char> >", "std::vector<std::vector<unsigned char> >").cast(true).pointerTypes("ByteVectorVector").define(true))
               .put(new Parser.Info("std::vector<std::vector<int> >").pointerTypes("IntVectorVector").define(true))
               .put(new Parser.Info("std::vector<std::string>").pointerTypes("StringVector").define(true))
               .put(new Parser.Info("std::vector<std::vector<cv::Point_<int> > >").pointerTypes("PointVectorVector").define(true))
               .put(new Parser.Info("std::vector<std::vector<cv::Point_<float> > >").pointerTypes("Point2fVectorVector").define(true))
               .put(new Parser.Info("std::vector<std::vector<cv::Point_<double> > >").pointerTypes("Point2dVectorVector").define(true))
               .put(new Parser.Info("std::vector<std::vector<cv::Rect_<int> > >").pointerTypes("RectVectorVector").define(true))
               .put(new Parser.Info("std::vector<cv::Mat>").pointerTypes("MatVector").define(true))
               .put(new Parser.Info("cv::randu<int>").javaNames("randInt"))
               .put(new Parser.Info("cv::randu<float>").javaNames("randFloat"))
               .put(new Parser.Info("cv::randu<double>").javaNames("randDouble"))

               .put(new Parser.Info("CvModule::first", "CvModule::last", "CvType::first", "CvType::last",
                                    "cv::fromUtf16", "cv::toUtf16", "cv::Exception", "cv::Allocator", "cv::DataDepth", "cv::DataType", "cv::ParamType",
                                    "cv::_InputArray", "cv::_OutputArray", "cv::noArray", "cv::Mat_", "cv::SparseMat_",
                                    "cv::Matx_AddOp", "cv::Matx_SubOp", "cv::Matx_ScaleOp", "cv::Matx_MulOp", "cv::Matx_MatMulOp", "cv::Matx_TOp",
                                    "cv::Matx", "cv::Vec", "cv::MatIterator_", "cv::MatConstIterator_", "cv::Mat::MSize", "cv::Mat::MStep",
                                    "cv::MatCommaInitializer_", "cv::MatxCommaInitializer", "cv::VecCommaInitializer",
                                    "cv::MatConstIterator(cv::Mat*, int*)", "cv::SparseMatIterator(cv::SparseMat*, int*)",
                                    "cv::SparseMatIterator_", "cv::SparseMatConstIterator_", "cv::SparseMatConstIterator::operator--",
                                    "cv::AlgorithmInfoData", "cv::AlgorithmInfo::addParam", "cv::CommandLineParser").skip(true))

               .put(new Parser.Info("cv::Mat").base("AbstractMat"))
               .put(new Parser.Info("cv::Mat::zeros(int, int*, int)", "cv::Mat::ones(int, int*, int)").skip(true))
               .put(new Parser.Info("cv::Mat::size").javaText("public native @ByVal Size size();\n@MemberGetter public native int size(int i);"))
               .put(new Parser.Info("cv::Mat::step").javaText("@MemberGetter public native long step();\n@MemberGetter public native int step(int i);"))

               .put(new Parser.Info("cv::InputArray", "cv::OutputArray", "cv::InputOutputArray").skip(true)./*cast(true).*/pointerTypes("Mat"))
               .put(new Parser.Info("cv::InputArrayOfArrays", "cv::OutputArrayOfArrays", "cv::InputOutputArrayOfArrays").skip(true)./*cast(true).*/pointerTypes("MatVector"))

               .put(new Parser.Info("cv::Point_<int>").pointerTypes("Point"))
               .put(new Parser.Info("cv::Point_<float>").pointerTypes("Point2f"))
               .put(new Parser.Info("cv::Point_<double>").pointerTypes("Point2d"))
               .put(new Parser.Info("cv::Point3_<int>").pointerTypes("Point3i"))
               .put(new Parser.Info("cv::Point3_<float>").pointerTypes("Point3f"))
               .put(new Parser.Info("cv::Point3_<double>").pointerTypes("Point3d"))
               .put(new Parser.Info("cv::Size_<int>").pointerTypes("Size"))
               .put(new Parser.Info("cv::Size_<float>").pointerTypes("Size2f"))
               .put(new Parser.Info("cv::Size_<double>").pointerTypes("Size2d"))
               .put(new Parser.Info("cv::Rect_<int>").pointerTypes("Rect"))
               .put(new Parser.Info("cv::Rect_<float>").pointerTypes("Rectf"))
               .put(new Parser.Info("cv::Rect_<double>").pointerTypes("Rectd"))
               .put(new Parser.Info("cv::Scalar_<double>").pointerTypes("Scalar").base("Pointer"))

               .put(new Parser.Info("cv::Vec2i").pointerTypes("Point"))
               .put(new Parser.Info("cv::Vec2d").pointerTypes("Point2d"))
               .put(new Parser.Info("cv::Vec3d").pointerTypes("Point3d"))

               .put(new Parser.Info("cv::Ptr").annotations("@Ptr"));
    }

    @Retention(RetentionPolicy.RUNTIME) @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast({"cv::Ptr", "&"}) @Adapter("PtrAdapter") public @interface Ptr {
        /** @return template type */
        String value() default "";
    }
}
