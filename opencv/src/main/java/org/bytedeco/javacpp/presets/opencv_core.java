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

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import org.bytedeco.javacpp.annotation.Adapter;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(value={
    @Platform(include={"<opencv2/core/types_c.h>", "<opencv2/core/core_c.h>", "<opencv2/core/core.hpp>",
                       "<opencv2/core/operations.hpp>", "<opencv2/core/mat.hpp>", "opencv_adapters.h"}, link="opencv_core@.2.4", preload="tbb"),
    @Platform(value="windows", define="_WIN32_WINNT 0x0502", link="opencv_core2410", preload={"msvcr100", "msvcp100"}),
    @Platform(value="windows-x86", preloadpath={"C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/redist/x86/Microsoft.VC100.CRT/"}),
    @Platform(value="windows-x86_64", preloadpath={"C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/redist/x64/Microsoft.VC100.CRT/"}) },
        target="org.bytedeco.javacpp.opencv_core", helper="org.bytedeco.javacpp.helper.opencv_core")
public class opencv_core implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("opencv_adapters.h").skip())
               .put(new Info("__cplusplus").define())
               .put(new Info("defined __ICL", "defined __ICC", "defined __ECL", "defined __ECC",
                             "defined __INTEL_COMPILER", "defined WIN32 || defined _WIN32").define(false))
               .put(new Info("CV_ENABLE_UNROLLED", "CV_CDECL", "CV_STDCALL", "CV_EXTERN_C", "CV_Func").cppTypes())
               .put(new Info("CV_DEFAULT", "CV_INLINE", "CV_EXPORTS").cppTypes().annotations())
               .put(new Info("CVAPI").cppText("#define CVAPI(rettype) rettype"))
               .put(new Info("CV_EXPORTS_W", "CV_EXPORTS_W_SIMPLE", "CV_EXPORTS_AS", "CV_EXPORTS_W_MAP",
                             "CV_IN_OUT", "CV_OUT", "CV_PROP", "CV_PROP_RW", "CV_WRAP", "CV_WRAP_AS").cppTypes().annotations().cppText(""))
               .put(new Info("CvRNG").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("CV_MAT_DEPTH", "CV_8UC", "CV_8SC", "CV_16UC", "CV_16SC", "CV_32SC", "CV_32FC", "CV_64FC").cppTypes("int", "int"))
               .put(new Info("CV_MAKETYPE", "CV_MAKE_TYPE").cppTypes("int", "int", "int"))
               .put(new Info("CV_8UC1", "CV_8UC2", "CV_8UC3", "CV_8UC4",
                             "CV_8SC1", "CV_8SC2", "CV_8SC3", "CV_8SC4",
                             "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4").cppTypes("int").translate())
               .put(new Info("CV_MAT_CN", "CV_MAT_TYPE", "CV_IS_CONT_MAT", "CV_IS_MAT_CONT").cppTypes("int", "int"))
               .put(new Info("CV_WHOLE_ARR", "CV_WHOLE_SEQ").cppTypes("CvSlice").translate())
               .put(new Info("CvArr").skip().pointerTypes("CvArr"))
               .put(new Info("_IplROI").pointerTypes("IplROI"))
               .put(new Info("_IplImage").pointerTypes("IplImage"))
               .put(new Info("_IplTileInfo").pointerTypes("IplTileInfo"))
               .put(new Info("IplImage").base("AbstractIplImage"))
               .put(new Info("IplConvKernel").base("org.bytedeco.javacpp.helper.opencv_imgproc.AbstractIplConvKernel"))
               .put(new Info("CvMat").base("AbstractCvMat"))
               .put(new Info("CvMatND").base("AbstractCvMatND"))
               .put(new Info("CvSparseMat").base("AbstractCvSparseMat"))
               .put(new Info("CvHistogram").base("org.bytedeco.javacpp.helper.opencv_imgproc.AbstractCvHistogram"))
               .put(new Info("CvRect").base("AbstractCvRect"))
               .put(new Info("CvPoint").cast().pointerTypes("CvPoint", "IntBuffer", "int[]").base("AbstractCvPoint"))
               .put(new Info("CvPoint2D32f").cast().pointerTypes("CvPoint2D32f", "FloatBuffer", "float[]").base("AbstractCvPoint2D32f"))
               .put(new Info("CvPoint3D32f").cast().pointerTypes("CvPoint3D32f", "FloatBuffer", "float[]").base("AbstractCvPoint3D32f"))
               .put(new Info("CvPoint2D64f").cast().pointerTypes("CvPoint2D64f", "DoubleBuffer", "double[]").base("AbstractCvPoint2D64f"))
               .put(new Info("CvPoint3D64f").cast().pointerTypes("CvPoint3D64f", "DoubleBuffer", "double[]").base("AbstractCvPoint3D64f"))
               .put(new Info("CvSize").base("AbstractCvSize"))
               .put(new Info("CvSize2D32f").base("AbstractCvSize2D32f"))
               .put(new Info("CvBox2D").base("AbstractCvBox2D"))
               .put(new Info("CvScalar").base("AbstractCvScalar"))
               .put(new Info("CvMemStorage").base("AbstractCvMemStorage"))
               .put(new Info("CvSeq").base("AbstractCvSeq"))
               .put(new Info("CvSet").base("AbstractCvSet"))
               .put(new Info("CvChain", "CvContour", "CvContourTree").base("CvSeq"))
               .put(new Info("CvGraph").base("AbstractCvGraph"))
               .put(new Info("CvGraphVtx2D").base("CvGraphVtx"))
               .put(new Info("CvChainPtReader").base("CvSeqReader"))
               .put(new Info("CvFileStorage").base("AbstractCvFileStorage"))
               .put(new Info("CvGraphScanner").base("AbstractCvGraphScanner"))
               .put(new Info("CvFont").base("AbstractCvFont"))
               .put(new Info("cvGetSubArr").cppTypes("CvMat*", "CvArr*", "CvMat*", "CvRect"))
               .put(new Info("cvZero").cppTypes("void", "CvArr*"))
               .put(new Info("cvCvtScale", "cvScale", "cvCvtScaleAbs").cppTypes("void", "CvArr*", "CvArr*", "double", "double"))
               .put(new Info("cvConvert", "cvT").cppTypes("void", "CvArr*", "CvArr*"))
               .put(new Info("cvCheckArray").cppTypes("int", "CvArr*", "int", "double", "double"))
               .put(new Info("cvMatMulAdd").cppTypes("void", "CvArr*", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Info("cvMatMul").cppTypes("void", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Info("cvMatMulAddEx").cppTypes("void", "CvArr*", "CvArr*", "double", "CvArr*", "double", "CvArr*", "int"))
               .put(new Info("cvMatMulAddS").cppTypes("void", "CvArr*", "CvArr*", "CvMat*", "CvMat*"))
               .put(new Info("cvMirror", "cvInv").cppTypes("void", "CvArr*", "CvArr*", "int"))
               .put(new Info("cvMahalonobis").cppTypes("double", "CvArr*", "CvArr*", "CvArr*"))
               .put(new Info("cvFFT").cppTypes("void", "CvArr*", "CvArr*", "int", "int"))
               .put(new Info("cvGraphFindEdge").cppTypes("CvGraphEdge*", "CvGraph*", "int", "int"))
               .put(new Info("cvGraphFindEdgeByPtr").cppTypes("CvGraphEdge*", "CvGraph*", "CvGraphVtx*", "CvGraphVtx*"))
               .put(new Info("cvDrawRect", "cvDrawLine").cppTypes("void", "CvArr*", "CvPoint", "CvPoint", "CvScalar", "int", "int", "int"))
               .put(new Info("cvDrawCircle").cppTypes("void", "CvArr*", "CvPoint", "int", "CvScalar", "int", "int", "int"))
               .put(new Info("cvDrawEllipse").cppTypes("void", "CvArr*", "CvPoint", "CvSize", "double", "double", "double", "CvScalar", "int", "int", "int"))
               .put(new Info("cvDrawPolyLine").cppTypes("void", "CvArr*", "CvPoint**", "int*", "int", "int", "CvScalar", "int", "int", "int"))
               .put(new Info("__CV_BEGIN__", "__CV_END__", "__CV_EXIT__").cppTypes())

               .put(new Info("std::vector<std::vector<char> >", "std::vector<std::vector<unsigned char> >").cast().pointerTypes("ByteVectorVector").define())
               .put(new Info("std::vector<std::vector<int> >").pointerTypes("IntVectorVector").define())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<std::vector<cv::Point_<int> > >").pointerTypes("PointVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::Point_<float> > >").pointerTypes("Point2fVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::Point_<double> > >").pointerTypes("Point2dVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::Rect_<int> > >").pointerTypes("RectVectorVector").define())
               .put(new Info("std::vector<cv::Mat>").pointerTypes("MatVector").define())
               .put(new Info("cv::randu<int>").javaNames("randInt"))
               .put(new Info("cv::randu<float>").javaNames("randFloat"))
               .put(new Info("cv::randu<double>").javaNames("randDouble"))

               .put(new Info("CvModule::first", "CvModule::last", "CvType::first", "CvType::last",
                             "cv::fromUtf16", "cv::toUtf16", "cv::Exception", "cv::Allocator", "cv::DataDepth", "cv::DataType", "cv::ParamType",
                             "cv::_InputArray", "cv::_OutputArray", "cv::noArray", "cv::Mat_", "cv::SparseMat_",
                             "cv::Matx_AddOp", "cv::Matx_SubOp", "cv::Matx_ScaleOp", "cv::Matx_MulOp", "cv::Matx_MatMulOp", "cv::Matx_TOp",
                             "cv::Matx", "cv::Vec", "cv::MatIterator_", "cv::MatConstIterator_", "cv::Mat::MSize", "cv::Mat::MStep",
                             "cv::MatCommaInitializer_", "cv::MatxCommaInitializer", "cv::VecCommaInitializer",
                             "cv::MatConstIterator(cv::Mat*, int*)", "cv::SparseMatIterator(cv::SparseMat*, int*)",
                             "cv::SparseMatIterator_", "cv::SparseMatConstIterator_", "cv::SparseMatConstIterator::operator--",
                             "cv::AlgorithmInfoData", "cv::AlgorithmInfo::addParam", "cv::CommandLineParser").skip())

               .put(new Info("cv::Mat").base("AbstractMat"))
               .put(new Info("cv::Mat(int, int, int, void*, size_t)").javaText(
                       "public Mat(int rows, int cols, int type, Pointer data, @Cast(\"size_t\") long step/*=AUTO_STEP*/) { allocate(rows, cols, type, data, step); this.data = data; }\n"
                     + "private native void allocate(int rows, int cols, int type, Pointer data, @Cast(\"size_t\") long step/*=AUTO_STEP*/);\n"
                     + "private Pointer data; // a reference to prevent deallocation\n"
                     + "public Mat(int rows, int cols, int type, Pointer data) { this(rows, cols, type, data, AUTO_STEP); }\n"
                     + "public Mat(byte ... b) { this(b, false); }\n"
                     + "public Mat(byte[] b, boolean signed) { this(new BytePointer(b), signed); }\n"
                     + "public Mat(short ... s) { this(s, true); }\n"
                     + "public Mat(short[] s, boolean signed) { this(new ShortPointer(s), signed); }\n"
                     + "public Mat(int ... n) { this(new IntPointer(n)); }\n"
                     + "public Mat(double ... d) { this(new DoublePointer(d)); }\n"
                     + "public Mat(float ... f) { this(new FloatPointer(f)); }\n"
                     + "public Mat(BytePointer p, boolean signed) { this(p.limit - p.position, 1, signed ? CV_8SC1 : CV_8UC1, p); }\n"
                     + "public Mat(ShortPointer p, boolean signed) { this(p.limit - p.position, 1, signed ? CV_16SC1 : CV_16UC1, p); }\n"
                     + "public Mat(IntPointer p) { this(p.limit - p.position, 1, CV_32SC1, p); }\n"
                     + "public Mat(FloatPointer p) { this(p.limit - p.position, 1, CV_32FC1, p); }\n"
                     + "public Mat(DoublePointer p) { this(p.limit - p.position, 1, CV_64FC1, p); }\n"))
               .put(new Info("cv::Mat::zeros(int, int*, int)", "cv::Mat::ones(int, int*, int)").skip())
               .put(new Info("cv::Mat::size").javaText("public native @ByVal Size size();\n@MemberGetter public native int size(int i);"))
               .put(new Info("cv::Mat::step").javaText("@MemberGetter public native long step();\n@MemberGetter public native int step(int i);"))

               .put(new Info("cv::InputArray", "cv::OutputArray", "cv::InputOutputArray").skip()./*cast().*/pointerTypes("Mat"))
               .put(new Info("cv::InputArrayOfArrays", "cv::OutputArrayOfArrays", "cv::InputOutputArrayOfArrays").skip()./*cast().*/pointerTypes("MatVector"))

               .put(new Info("cv::Point_<int>").pointerTypes("Point").base("IntPointer"))
               .put(new Info("cv::Point_<float>").pointerTypes("Point2f").base("FloatPointer"))
               .put(new Info("cv::Point_<double>").pointerTypes("Point2d").base("DoublePointer"))
               .put(new Info("cv::Point3_<int>").pointerTypes("Point3i").base("IntPointer"))
               .put(new Info("cv::Point3_<float>").pointerTypes("Point3f").base("FloatPointer"))
               .put(new Info("cv::Point3_<double>").pointerTypes("Point3d").base("DoublePointer"))
               .put(new Info("cv::Size_<int>").pointerTypes("Size").base("IntPointer"))
               .put(new Info("cv::Size_<float>").pointerTypes("Size2f").base("FloatPointer"))
               .put(new Info("cv::Size_<double>").pointerTypes("Size2d").base("DoublePointer"))
               .put(new Info("cv::Rect_<int>").pointerTypes("Rect").base("IntPointer"))
               .put(new Info("cv::Rect_<float>").pointerTypes("Rectf").base("FloatPointer"))
               .put(new Info("cv::Rect_<double>").pointerTypes("Rectd").base("DoublePointer"))
               .put(new Info("cv::RotatedRect").pointerTypes("RotatedRect").base("FloatPointer"))
               .put(new Info("cv::Scalar_<double>").pointerTypes("Scalar").base("DoublePointer"))

               .put(new Info("cv::Vec2i").pointerTypes("Point"))
               .put(new Info("cv::Vec2d").pointerTypes("Point2d"))
               .put(new Info("cv::Vec3d").pointerTypes("Point3d"))

               .put(new Info("defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)", "defined __GNUC__",
                             "defined WIN32 || defined _WIN32 || defined WINCE").define(false))

               .put(new Info("cv::saturate_cast<uchar>").javaNames("saturateCastUchar"))
               .put(new Info("cv::saturate_cast<schar>").javaNames("saturateCastSchar"))
               .put(new Info("cv::saturate_cast<ushort>").javaNames("saturateCastUshort"))
               .put(new Info("cv::saturate_cast<short>").javaNames("saturateCastShort"))
               .put(new Info("cv::saturate_cast<int>").javaNames("saturateCastInt"))
               .put(new Info("cv::saturate_cast<unsigned>").javaNames("saturateCastUnsigned"))

               .put(new Info("cv::normL2Sqr", "cv::normL1").skip())

               .put(new Info("cv::Formatted(cv::Mat&, cv::Formatter*, int*)").javaText(
                       "public Formatted(@Const @ByRef Mat m, @Const Formatter fmt,\n"
                     + "              @StdVector IntPointer params) { allocate(m, fmt, params); }\n"
                     + "private native void allocate(@Const @ByRef Mat m, @Const Formatter fmt,\n"
                     + "              @Cast({\"\", \"std::vector<int>&\"}) @StdVector IntPointer params);"))

               .put(new Info("cv::Ptr").annotations("@Ptr"));
    }

    @Retention(RetentionPolicy.RUNTIME) @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast({"cv::Ptr", "&"}) @Adapter("PtrAdapter") public @interface Ptr {
        /** @return template type */
        String value() default "";
    }
}
