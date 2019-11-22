/*
 * Copyright (C) 2013-2019 Samuel Audet
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

import java.io.File;
import java.io.IOException;
import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.Arrays;
import java.util.List;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Adapter;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.openblas.presets.*;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = openblas.class,
    value = {
        @Platform(include = {"<opencv2/core/hal/interface.h>", "<opencv2/core/cvdef.h>", "<opencv2/core/hal/hal.hpp>", "<opencv2/core/fast_math.hpp>",
            "<algorithm>", "<map>", "<opencv2/core/saturate.hpp>", "<opencv2/core/version.hpp>", "<opencv2/core/base.hpp>", "<opencv2/core/cvstd.hpp>",
            "<opencv2/core/utility.hpp>", "<opencv2/core/types_c.h>", "<opencv2/core/core_c.h>", "<opencv2/core/types.hpp>", "<opencv2/core.hpp>",
            "<opencv2/core/cuda.hpp>", "<opencv2/core/ocl.hpp>", "<opencv2/core/operations.hpp>", "<opencv2/core/bufferpool.hpp>", "<opencv2/core/mat.hpp>",
            "<opencv2/core/persistence.hpp>", "<opencv2/core/optim.hpp>", "<opencv2/core/async.hpp>", "opencv_adapters.h"}, link = {"opencv_core@.4.1", "opencv_imgproc@.4.1"},
            resource = {"include", "lib", "sdk", "share", "x86", "x64", "OpenCVConfig.cmake", "OpenCVConfig-version.cmake", "python"}, linkresource = "lib",
            preload = {"gomp@.1", "opencv_cudev@.4.1"}, compiler = "cpp11"),
        @Platform(value = "android", preload = ""),
        @Platform(value = "ios", preload = {"liblibjpeg", "liblibpng", "liblibprotobuf", "liblibwebp", "libzlib", "libopencv_core"}),
        @Platform(value = "linux",        preloadpath = {"/usr/lib/", "/usr/lib32/", "/usr/lib64/"}),
        @Platform(value = "linux-armhf",  preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}),
        @Platform(value = "linux-arm64",  preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"}),
        @Platform(value = "linux-x86",    preloadpath = {"/usr/lib32/", "/usr/lib/"}),
        @Platform(value = "linux-x86_64", preloadpath = {"/usr/lib64/", "/usr/lib/"}),
        @Platform(value = "linux-ppc64",  preloadpath = {"/usr/lib/powerpc64-linux-gnu/", "/usr/lib/powerpc64le-linux-gnu/"}),
        @Platform(value = "windows", define = "_WIN32_WINNT 0x0502", link =  {"opencv_core412", "opencv_imgproc412"}, preload = {
            "api-ms-win-crt-locale-l1-1-0", "api-ms-win-crt-string-l1-1-0", "api-ms-win-crt-stdio-l1-1-0", "api-ms-win-crt-math-l1-1-0",
            "api-ms-win-crt-heap-l1-1-0", "api-ms-win-crt-runtime-l1-1-0", "api-ms-win-crt-convert-l1-1-0", "api-ms-win-crt-environment-l1-1-0",
            "api-ms-win-crt-time-l1-1-0", "api-ms-win-crt-filesystem-l1-1-0", "api-ms-win-crt-utility-l1-1-0", "api-ms-win-crt-multibyte-l1-1-0",
            "api-ms-win-core-string-l1-1-0", "api-ms-win-core-errorhandling-l1-1-0", "api-ms-win-core-timezone-l1-1-0", "api-ms-win-core-file-l1-1-0",
            "api-ms-win-core-namedpipe-l1-1-0", "api-ms-win-core-handle-l1-1-0", "api-ms-win-core-file-l2-1-0", "api-ms-win-core-heap-l1-1-0",
            "api-ms-win-core-libraryloader-l1-1-0", "api-ms-win-core-synch-l1-1-0", "api-ms-win-core-processthreads-l1-1-0",
            "api-ms-win-core-processenvironment-l1-1-0", "api-ms-win-core-datetime-l1-1-0", "api-ms-win-core-localization-l1-2-0",
            "api-ms-win-core-sysinfo-l1-1-0", "api-ms-win-core-synch-l1-2-0", "api-ms-win-core-console-l1-1-0", "api-ms-win-core-debug-l1-1-0",
            "api-ms-win-core-rtlsupport-l1-1-0", "api-ms-win-core-processthreads-l1-1-1", "api-ms-win-core-file-l1-2-0", "api-ms-win-core-profile-l1-1-0",
            "api-ms-win-core-memory-l1-1-0", "api-ms-win-core-util-l1-1-0", "api-ms-win-core-interlocked-l1-1-0", "ucrtbase",
            "vcruntime140", "msvcp140", "concrt140", "vcomp140", "opencv_cudev412"}),
        @Platform(value = "windows-x86", preloadpath = {"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x86/Microsoft.VC140.CRT/",
            "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x86/Microsoft.VC140.OpenMP/",
            "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x86/"}),
        @Platform(value = "windows-x86_64", preloadpath = {"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x64/Microsoft.VC140.CRT/",
            "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/redist/x64/Microsoft.VC140.OpenMP/",
            "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x64/"}),
        @Platform(value = {"linux-arm64", "linux-ppc64le", "linux-x86_64", "macosx-x86_64", "windows-x86_64"}, extension = "-gpu")},
    target = "org.bytedeco.opencv.opencv_core",
    global = "org.bytedeco.opencv.global.opencv_core",
    helper = "org.bytedeco.opencv.helper.opencv_core"
)
public class opencv_core implements LoadEnabled, InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "opencv"); }

    /** Returns {@code Loader.cacheResource("/org/bytedeco/opencv/" + Loader.getPlatform() + extension + "/python/")}. */
    public static File cachePackage() throws IOException {
        Loader.load(org.bytedeco.cpython.global.python.class);
        String path = Loader.load(opencv_core.class);
        if (path != null) {
            path = path.replace(File.separatorChar, '/');
            int i = path.indexOf("/org/bytedeco/opencv/" + Loader.getPlatform());
            int j = path.lastIndexOf("/");
            return Loader.cacheResource(path.substring(i, j) + "/python/");
        }
        return null;
    }

    /** Returns {@code {numpy.cachePackages(), opencv.cachePackage()}}. */
    public static File[] cachePackages() throws IOException {
        File[] path = org.bytedeco.numpy.global.numpy.cachePackages();
        path = Arrays.copyOf(path, path.length + 1);
        path[path.length - 1] = cachePackage();
        return path;
    }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        String extension = properties.getProperty("platform.extension");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");
        List<String> preloadpaths = properties.get("platform.preloadpath");

        // Only apply this at load time since we don't want to copy the CUDA libraries here
        if (!Loader.isLoadLibraries() || extension == null || !extension.equals("-gpu")) {
            return;
        }
        int i = 0;
        String[] libs = {"cudart", "cublasLt", "cublas", "cufft", "cudnn", "nppc", "nppial", "nppicc", "nppicom",
                         "nppidei", "nppif", "nppig", "nppim", "nppist", "nppisu", "nppitc", "npps"};
        for (String lib : libs) {
            switch (platform) {
                case "linux-arm64":
                case "linux-ppc64le":
                case "linux-x86_64":
                case "macosx-x86_64":
                    lib += lib.equals("cudnn") ? "@.7" : lib.equals("cudart") ? "@.10.1" : "@.10";
                    break;
                case "windows-x86_64":
                    lib += lib.equals("cudnn") ? "64_7" : lib.equals("cudart") ? "64_101" : "64_10";
                    break;
                default:
                    continue; // no CUDA
            }
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
        }

        String vcredistdir = System.getenv("VCToolsRedistDir");
        if (vcredistdir != null && vcredistdir.length() > 0) {
            switch (platform) {
                case "windows-x86":
                    preloadpaths.add(0, vcredistdir + "\\x86\\Microsoft.VC141.CRT");
                    preloadpaths.add(1, vcredistdir + "\\x86\\Microsoft.VC141.OpenMP");
                    break;
                case "windows-x86_64":
                    preloadpaths.add(0, vcredistdir + "\\x64\\Microsoft.VC141.CRT");
                    preloadpaths.add(1, vcredistdir + "\\x64\\Microsoft.VC141.OpenMP");
                    break;
                default:
                    // not Windows
            }
        }
    }

    public void map(InfoMap infoMap) {
        infoMap.putFirst(new Info("openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h", "lapacke.h", "lapacke_utils.h").skip())
               .put(new Info("algorithm", "map", "opencv_adapters.h").skip())
               .put(new Info("__cplusplus", "CV_StaticAssert", "CV__LEGACY_PERSISTENCE").define())
               .put(new Info("__OPENCV_BUILD", "defined __ICL", "defined __ICC", "defined __ECL", "defined __ECC", "defined __INTEL_COMPILER",
                             "defined WIN32 || defined _WIN32", "defined(__clang__)", "defined(__GNUC__)", "defined(_MSC_VER)",
                             "defined __GNUC__ || defined __clang__", "OPENCV_NOSTL_TRANSITIONAL", "CV_COLLECT_IMPL_DATA",
                             "__cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1800)", "CV_CXX11", "CV_FP16_TYPE", "__EMSCRIPTEN__").define(false))
               .put(new Info("CV_ENABLE_UNROLLED", "CV_CDECL", "CV_STDCALL", "CV_IMPL", "CV_EXTERN_C", "CV_Func",
                             "CV__ErrorNoReturn", "CV__ErrorNoReturn_", "CV_ErrorNoReturn", "CV_ErrorNoReturn_", "CV_USRTYPE1", "CV_Assert_1").cppTypes().cppText(""))
               .put(new Info("CV_DEFAULT", "CV_INLINE", "CV_ALWAYS_INLINE", "CV_EXPORTS", "CV_NEON", "CPU_HAS_NEON_FEATURE", "CV__DEBUG_NS_BEGIN", "CV__DEBUG_NS_END",
                             "CV_NORETURN", "CV_SUPPRESS_DEPRECATED_START", "CV_SUPPRESS_DEPRECATED_END", "CV_CATCH_ALL", "CV_NODISCARD").annotations().cppTypes())
               .put(new Info("CVAPI").cppText("#define CVAPI(rettype) rettype"))

               .put(new Info("CV_DEPRECATED").cppText("#define CV_DEPRECATED deprecated").cppTypes())
               .put(new Info("CV_DEPRECATED_EXTERNAL").cppText("#define CV_DEPRECATED_EXTERNAL deprecated").cppTypes())
               .put(new Info("deprecated").annotations("@Deprecated"))

               .put(new Info("CV_OVERRIDE").cppText("#define CV_OVERRIDE override").cppTypes())
               .put(new Info("override").annotations("@Override"))

               .put(new Info("CV_FINAL").cppText("#define CV_FINAL final").cppTypes())
               .put(new Info("final").annotations("final"))

               .put(new Info("CV_NOEXCEPT").cppText("#define CV_NOEXCEPT noexcept").cppTypes())
               .put(new Info("noexcept").annotations("@NoException"))

               .put(new Info("CV_CONSTEXPR").cppText("#define CV_CONSTEXPR constexpr").cppTypes())
               .put(new Info("constexpr").annotations("@Const"))

               .put(new Info("CV_EXPORTS_AS", "CV_WRAP_AS").annotations("@Name").cppTypes().cppText(""))
               .put(new Info("CV_EXPORTS_W", "CV_EXPORTS_W_SIMPLE", "CV_EXPORTS_W_MAP",
                             "CV_IN_OUT", "CV_OUT", "CV_PROP", "CV_PROP_RW", "CV_WRAP").annotations().cppTypes().cppText(""))
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
               .put(new Info("CV_VERSION").pointerTypes("String").translate(false))
               .put(new Info("CV_WHOLE_ARR", "CV_WHOLE_SEQ").cppTypes("CvSlice").translate())
               .put(new Info("std::uint32_t").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("cv::ocl::initializeContextFromHandle").skip())
               .put(new Info("cv::ocl::Platform").pointerTypes("OclPlatform"))
               .put(new Info("cv::ocl::Queue::Impl", "cv::ocl::Program::Impl", "cv::ocl::ProgramSource::Impl").cast().pointerTypes("Pointer"))
               .put(new Info("cv::ocl::Kernel(const char*, const cv::ocl::ProgramSource&, const cv::String&, cv::String*)").javaText(
                       "public Kernel(String kname, @Const @ByRef ProgramSource prog,\n"
                     + "            @Str String buildopts, @Str BytePointer errmsg) { allocate(kname, prog, buildopts, errmsg); }\n"
                     + "private native void allocate(String kname, @Const @ByRef ProgramSource prog,\n"
                     + "            @Str String buildopts, @Cast({\"\", \"cv::String*\"}) @Str BytePointer errmsg/*=NULL*/);"))
               .put(new Info("cv::ocl::Kernel::create(const char*, const cv::ocl::ProgramSource&, const cv::String&, cv::String*)").javaText(
                       "public native @Cast(\"bool\") boolean create(String kname, @Const @ByRef ProgramSource prog,\n"
                     + "            @Str String buildopts, @Cast({\"\", \"cv::String*\"}) @Str BytePointer errmsg/*=NULL*/);"))
               .put(new Info("CvArr").skip().pointerTypes("CvArr"))
               .put(new Info("_IplROI").pointerTypes("IplROI"))
               .put(new Info("_IplImage").pointerTypes("IplImage"))
               .put(new Info("_IplTileInfo").pointerTypes("IplTileInfo"))
               .put(new Info("IplImage").base("AbstractIplImage"))
               .put(new Info("IplConvKernel").base("org.bytedeco.opencv.opencv_imgproc.AbstractIplConvKernel"))
               .put(new Info("CvMat").base("AbstractCvMat"))
               .put(new Info("CvMatND").base("AbstractCvMatND"))
               .put(new Info("CvSparseMat").base("AbstractCvSparseMat"))
               .put(new Info("CvHistogram").base("org.bytedeco.opencv.opencv_imgproc.AbstractCvHistogram"))
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

               .put(new Info("uchar").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("std::vector<std::vector<char> >", "std::vector<std::vector<uchar> >").cast().pointerTypes("ByteVectorVector").define())
               .put(new Info("std::vector<std::vector<int> >").pointerTypes("IntVectorVector").define())
               .put(new Info("std::vector<cv::String>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<cv::Point>").pointerTypes("PointVector").define())
               .put(new Info("std::vector<cv::Point2f>").pointerTypes("Point2fVector").define())
               .put(new Info("std::vector<cv::Point2d>").pointerTypes("Point2dVector").define())
               .put(new Info("std::vector<cv::Point3i>", "std::vector<cv::Vec3i>").cast().pointerTypes("Point3iVector").define())
               .put(new Info("std::vector<cv::Point3f>").cast().pointerTypes("Point3fVector").define())
               .put(new Info("std::vector<cv::Size>").pointerTypes("SizeVector").define())
               .put(new Info("std::vector<cv::Rect>").pointerTypes("RectVector").define())
               .put(new Info("std::vector<cv::Rect2d>").pointerTypes("Rect2dVector").define())
               .put(new Info("std::vector<cv::Scalar>").pointerTypes("ScalarVector").define())
               .put(new Info("std::vector<cv::KeyPoint>").pointerTypes("KeyPointVector").define())
               .put(new Info("std::vector<cv::DMatch>").pointerTypes("DMatchVector").define())
               .put(new Info("std::vector<std::vector<cv::Point> >").pointerTypes("PointVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::Point2f> >").pointerTypes("Point2fVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::Point2d> >").pointerTypes("Point2dVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::Point3f> >").pointerTypes("Point3fVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::Rect> >").pointerTypes("RectVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::KeyPoint> >").pointerTypes("KeyPointVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::DMatch> >").pointerTypes("DMatchVectorVector").define())
               .put(new Info("std::vector<cv::Mat>").pointerTypes("MatVector").define())
               .put(new Info("std::vector<cv::UMat>").pointerTypes("UMatVector").define())
               .put(new Info("std::vector<cv::cuda::GpuMat>").pointerTypes("GpuMatVector").define())
               .put(new Info("std::vector<std::vector<cv::Mat> >").pointerTypes("MatVectorVector").define())
               .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
               .put(new Info("std::map<int,double>").pointerTypes("IntDoubleMap").define())
               .put(new Info("std::vector<std::pair<int,double> >").pointerTypes("IntDoublePairVector").define())
               .put(new Info("std::vector<std::pair<int,int> >").pointerTypes("IntIntPairVector").define())
               .put(new Info("std::vector<std::pair<cv::Mat,uchar> >").pointerTypes("MatBytePairVector").define())
               .put(new Info("std::vector<std::pair<cv::UMat,uchar> >").pointerTypes("UMatBytePairVector").define())
               .put(new Info("std::vector<cv::instr::NodeDataTls*>").pointerTypes("NodeDataTlsVector").define())
               .put(new Info("cv::TLSData<cv::instr::NodeDataTls>").pointerTypes("NodeDataTlsData").define())
               .put(new Info("cv::Node<cv::instr::NodeData>").pointerTypes("InstrNode").define())
               .put(new Info("cv::instr::NodeData::m_tls").javaText("@MemberGetter public native @ByRef NodeDataTlsData m_tls();"))
               .put(new Info("cv::SparseMat::Node").pointerTypes("SparseMat.Node"))
               .put(new Info("cv::ml::DTrees::Node").pointerTypes("DTrees.Node"))
               .put(new Info("cv::randu<int>").javaNames("intRand"))
               .put(new Info("cv::randu<float>").javaNames("floatRand"))
               .put(new Info("cv::randu<double>").javaNames("doubleRand"))

               .put(new Info("CvModule::first", "CvModule::last", "CvType::first", "CvType::last",
                             "cv::fromUtf16", "cv::toUtf16", "cv::Exception", "cv::Allocator", "cv::DataDepth", "cv::DataType", "cv::ParamType",
                             "cv::_InputArray", "cv::_OutputArray", "cv::Mat_", "cv::SparseMat_",
                             "cv::Matx_AddOp", "cv::Matx_SubOp", "cv::Matx_ScaleOp", "cv::Matx_MulOp", "cv::Matx_MatMulOp", "cv::Matx_TOp",
                             "cv::Matx", "cv::Vec", "cv::MatIterator_", "cv::MatConstIterator_", "cv::Mat::MSize", "cv::Mat::MStep",
                             "cv::MatCommaInitializer_", "cv::MatxCommaInitializer", "cv::VecCommaInitializer",
                             "cv::MatConstIterator(cv::Mat*, int*)", "cv::SparseMatIterator(cv::SparseMat*, int*)",
                             "cv::SparseMatIterator_", "cv::SparseMatConstIterator_", "cv::SparseMatConstIterator::operator --",
                             "cv::AlgorithmInfoData", "cv::AlgorithmInfo::addParam", "cv::CommandLineParser",
                             "cv::cvStartWriteRawData_Base64", "cv::cvWriteRawData_Base64", "cv::cvEndWriteRawData_Base64",
                             "cv::cvWriteMat_Base64", "cv::cvWriteMatND_Base64", "cv::FileStorage::Impl").skip())
               .put(new Info("cv::AutoBuffer<double>").cast().pointerTypes("Pointer"))

               .put(new Info("cv::Mat").base("AbstractMat"))
               .put(new Info("cv::noArray()").javaText("public static Mat noArray() { return null; }"))
               .put(new Info("cv::Mat(int, int, int, void*, size_t)").javaText(
                       "public Mat(int rows, int cols, int type, Pointer data, @Cast(\"size_t\") long step/*=AUTO_STEP*/) { super((Pointer)null); allocate(rows, cols, type, data, step); this.pointer = data; }\n"
                     + "private native void allocate(int rows, int cols, int type, Pointer data, @Cast(\"size_t\") long step/*=AUTO_STEP*/);\n"
                     + "private Pointer pointer; // a reference to prevent deallocation\n"
                     + "public Mat(int rows, int cols, int type, Pointer data) { this(rows, cols, type, data, AUTO_STEP); }\n"
                     + "public Mat(CvArr arr) { super(cvarrToMat(arr)); this.pointer = arr; }\n"
                     + "public Mat(Point points) { this(1, Math.max(1, points.limit() - points.position()), CV_32SC2, points); this.pointer = points; }\n"
                     + "public Mat(Point2f points) { this(1, Math.max(1, points.limit() - points.position()), CV_32FC2, points); this.pointer = points; }\n"
                     + "public Mat(Point2d points) { this(1, Math.max(1, points.limit() - points.position()), CV_64FC2, points); this.pointer = points; }\n"
                     + "public Mat(Point3i points) { this(1, Math.max(1, points.limit() - points.position()), CV_32SC3, points); this.pointer = points; }\n"
                     + "public Mat(Point3f points) { this(1, Math.max(1, points.limit() - points.position()), CV_32FC3, points); this.pointer = points; }\n"
                     + "public Mat(Point3d points) { this(1, Math.max(1, points.limit() - points.position()), CV_64FC3, points); this.pointer = points; }\n"
                     + "public Mat(Scalar scalar) { this(1, Math.max(1, scalar.limit() - scalar.position()), CV_64FC4, scalar); this.pointer = scalar; }\n"
                     + "public Mat(Scalar4i scalar) { this(1, Math.max(1, scalar.limit() - scalar.position()), CV_32SC4, scalar); this.pointer = scalar; }\n"
                     + "public Mat(byte ... b) { this(b, false); }\n"
                     + "public Mat(byte[] b, boolean signed) { this(new BytePointer(b), signed); }\n"
                     + "public Mat(short ... s) { this(s, true); }\n"
                     + "public Mat(short[] s, boolean signed) { this(new ShortPointer(s), signed); }\n"
                     + "public Mat(int ... n) { this(new IntPointer(n)); }\n"
                     + "public Mat(double ... d) { this(new DoublePointer(d)); }\n"
                     + "public Mat(float ... f) { this(new FloatPointer(f)); }\n"
                     + "private Mat(long rows, long cols, int type, Pointer data) { this((int)Math.min(rows, Integer.MAX_VALUE), (int)Math.min(cols, Integer.MAX_VALUE), type, data, AUTO_STEP); }\n"
                     + "public Mat(BytePointer p) { this(p, false); }\n"
                     + "public Mat(BytePointer p, boolean signed) { this(1, Math.max(1, p.limit() - p.position()), signed ? CV_8SC1 : CV_8UC1, p); }\n"
                     + "public Mat(ShortPointer p) { this(p, false); }\n"
                     + "public Mat(ShortPointer p, boolean signed) { this(1, Math.max(1, p.limit() - p.position()), signed ? CV_16SC1 : CV_16UC1, p); }\n"
                     + "public Mat(IntPointer p) { this(1, Math.max(1, p.limit() - p.position()), CV_32SC1, p); }\n"
                     + "public Mat(FloatPointer p) { this(1, Math.max(1, p.limit() - p.position()), CV_32FC1, p); }\n"
                     + "public Mat(DoublePointer p) { this(1, Math.max(1, p.limit() - p.position()), CV_64FC1, p); }\n"))
               .put(new Info("cv::Mat::zeros(int, int*, int)", "cv::Mat::ones(int, int*, int)").skip())
               .put(new Info("cv::Mat::size").javaText("public native @ByVal Size size();\n@MemberGetter public native int size(int i);"))
               .put(new Info("cv::Mat::step").javaText("@MemberGetter public native long step();\n@MemberGetter public native long step(int i);"))
               .put(new Info("cv::UMat::zeros(int, int*, int)", "cv::UMat::ones(int, int*, int)").skip())
               .put(new Info("cv::UMat::size").javaText("public native @ByVal Size size();\n@MemberGetter public native int size(int i);"))
               .put(new Info("cv::UMat::step").javaText("@MemberGetter public native long step();\n@MemberGetter public native long step(int i);"))
               .put(new Info("cv::DefaultDeleter<CvMat>").pointerTypes("CvMatDefaultDeleter"))

               .put(new Info("std::initializer_list", "_InputArray::KindFlag").skip())
               .put(new Info("cv::InputArray", "cv::OutputArray", "cv::InputOutputArray", "cv::_InputOutputArray")
                       .skip()./*cast().*/pointerTypes("Mat", "Mat", "Mat", "UMat", "UMat", "UMat", "GpuMat", "GpuMat", "GpuMat"))
               .put(new Info("cv::InputArrayOfArrays", "cv::OutputArrayOfArrays", "cv::InputOutputArrayOfArrays")
                       .skip()./*cast().*/pointerTypes("MatVector", "UMatVector", "GpuMatVector"))

               .put(new Info("cv::traits::Depth", "cv::traits::Type").skip())
               .put(new Info("cv::Complex<float>").pointerTypes("Complexf").base("FloatPointer"))
               .put(new Info("cv::Complex<double>").pointerTypes("Complexd").base("DoublePointer"))
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
               .put(new Info("cv::Rect_<float>").pointerTypes("Rect2f").base("FloatPointer"))
               .put(new Info("cv::Rect_<double>").pointerTypes("Rect2d").base("DoublePointer"))
               .put(new Info("cv::RotatedRect").pointerTypes("RotatedRect").base("FloatPointer"))
               .put(new Info("cv::Scalar_<double>").pointerTypes("Scalar").base("AbstractScalar"))
               .put(new Info("cv::Scalar_<int>").pointerTypes("Scalar4i").base("IntPointer"))
               .put(new Info("cv::Scalar_<float>").pointerTypes("Scalar4f").base("FloatPointer"))

               .put(new Info("cv::Vec2i").cast().pointerTypes("Point"))
               .put(new Info("cv::Vec2f").cast().pointerTypes("Point2f"))
               .put(new Info("cv::Vec2d").cast().pointerTypes("Point2d"))
               .put(new Info("cv::Vec3i").cast().pointerTypes("Point3i"))
               .put(new Info("cv::Vec3f").cast().pointerTypes("Point3f"))
               .put(new Info("cv::Vec3d").cast().pointerTypes("Point3d"))
               .put(new Info("cv::Vec4i").cast().pointerTypes("Scalar4i"))
               .put(new Info("cv::Vec4f").cast().pointerTypes("Scalar4f"))

               .put(new Info("defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)", "defined __GNUC__",
                             "defined WIN32 || defined _WIN32 || defined WINCE").define(false))

//               .put(new Info("cv::saturate_cast(uchar)", "cv::saturate_cast(ushort)", "cv::saturate_cast(unsigned)").skip())
               .put(new Info("cv::saturate_cast<uchar>").javaNames("ucharSaturateCast"))
               .put(new Info("cv::saturate_cast<schar>").javaNames("scharSaturateCast"))
               .put(new Info("cv::saturate_cast<ushort>").javaNames("ushortSaturateCast"))
               .put(new Info("cv::saturate_cast<short>").javaNames("shortSaturateCast"))
               .put(new Info("cv::saturate_cast<int>").javaNames("intSaturate"))
               .put(new Info("cv::saturate_cast<unsigned>").javaNames("unsignedSaturateCast"))
               .put(new Info("cv::saturate_cast<uint64>").javaNames("uint64SaturateCast"))
               .put(new Info("cv::saturate_cast<int64>").javaNames("int64SaturateCast"))
               .put(new Info("cv::saturate_cast<cv::float16_t>", "saturate_cast<float16_t>").javaNames("float16SaturateCast"))

               .put(new Info("cv::normL2Sqr", "cv::normL1", "cv::seqPopMulti").skip())

               .put(new Info("cv::Formatted(cv::Mat&, cv::Formatter*, int*)").javaText(
                       "public Formatted(@Const @ByRef Mat m, @Const Formatter fmt,\n"
                     + "              @StdVector IntPointer params) { allocate(m, fmt, params); }\n"
                     + "private native void allocate(@Const @ByRef Mat m, @Const Formatter fmt,\n"
                     + "              @Cast({\"\", \"std::vector<int>&\"}) @StdVector IntPointer params);"))

               .put(new Info("cv::MinProblemSolver", "cv::DownhillSolver", "cv::ConjGradSolver").purify())
               .put(new Info("cv::MinProblemSolver::Function").virtualize())

               .put(new Info("HAVE_OPENCV_CUDAOPTFLOW", "HAVE_OPENCV_CUDAWARPING",
                             "HAVE_OPENCV_CUDALEGACY", "HAVE_OPENCV_XFEATURES2D", "defined(HAVE_OPENCV_CUDAWARPING)",
                             "defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)",
                             "defined(HAVE_OPENCV_CUDA) && defined(HAVE_OPENCV_CUDAWARPING)").define(false))
               .put(new Info("operator cv::cuda::Stream::bool_type", // basically a check to see if the Stream is cv::cuda:Stream::Null
                             "cv::cuda::convertFp16").skip())
               .put(new Info("cv::min(cv::InputArray, cv::InputArray, cv::OutputArray)",
                             "cv::max(cv::InputArray, cv::InputArray, cv::OutputArray)").skip()) // don't work with GpuMat

               .put(new Info("std::function<void(const Range&)>").pointerTypes("Functor"))
               .put(new Info("cv::Ptr").skip().annotations("@Ptr"))
               .put(new Info("cv::String").skip().annotations("@Str").valueTypes("BytePointer", "String"));
    }

    public static class Functor extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Functor(Pointer p) { super(p); }
        protected Functor() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("const cv::Range*") Pointer range);
    }

    @Documented @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast({"cv::Ptr"}) @Adapter("PtrAdapter") public @interface Ptr {
        /** @return template type */
        String value() default "";
    }

    @Documented @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.METHOD, ElementType.PARAMETER})
    @Cast({"cv::String&"}) @Adapter("StrAdapter") public @interface Str { }
}
