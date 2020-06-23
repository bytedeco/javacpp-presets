package org.bytedeco.openpose.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.caffe.presets.*;
import org.bytedeco.opencv.presets.*;
import org.bytedeco.openblas.presets.*;
import org.bytedeco.hdf5.presets.*;


@Properties(
    inherit = {opencv_core.class, opencv_highgui.class, opencv_imgproc.class, opencv_imgcodecs.class, openblas.class, caffe.class, hdf5.class},
    value = {
        @Platform(
            value = {"linux", "macosx"},
            compiler = "cpp11",
            define = {
                "SHARED_PTR_NAMESPACE std",
                "UNIQUE_PTR_NAMESPACE std",
                "GPU_MODE CPU_ONLY"
            },
            include = {
                "openpose/utilities/enumClasses.hpp",
                "openpose/thread/enumClasses.hpp",
                "openpose/pose/enumClasses.hpp",
                "openpose/core/enumClasses.hpp",
                "openpose/gui/enumClasses.hpp",
                "openpose/producer/enumClasses.hpp",
                "openpose/filestream/enumClasses.hpp",
                //"openpose/gpu/enumClasses.hpp",
                "openpose/wrapper/enumClasses.hpp",

                "openpose/core/matrix.hpp",
                "openpose/core/array.hpp",
                "openpose/core/point.hpp",
                "openpose/core/rectangle.hpp",
                "openpose/core/string.hpp",
                "openpose/core/datum.hpp",

                "openpose/thread/worker.hpp",

                "openpose/utilities/flagsToOpenPose.hpp",

                "openpose/wrapper/wrapperStructExtra.hpp",
                "openpose/wrapper/wrapperStructFace.hpp",
                "openpose/wrapper/wrapperStructGui.hpp",
                "openpose/wrapper/wrapperStructHand.hpp",
                "openpose/wrapper/wrapperStructInput.hpp",
                "openpose/wrapper/wrapperStructOutput.hpp",
                "openpose/wrapper/wrapperStructPose.hpp",
                "openpose/producer/producer.hpp",
                "openpose/wrapper/wrapperAuxiliary.hpp",
                "openpose/wrapper/wrapper.hpp",
            },
            link = {
                "openpose@.1.6.0",
                "openpose_3d@.1.6.0",
                "openpose_calibration@.1.6.0",
                "openpose_core@.1.6.0",
                "openpose_face@.1.6.0",
                "openpose_filestream@.1.6.0",
                "openpose_gpu@.1.6.0",
                "openpose_gui@.1.6.0",
                "openpose_hand@.1.6.0",
                "openpose_net@.1.6.0",
                "openpose_pose@.1.6.0",
                "openpose_producer@.1.6.0",
                "openpose_thread@.1.6.0",
                "openpose_tracking@.1.6.0",
                "openpose_unity@.1.6.0",
                "openpose_utilities@.1.6.0",
                "openpose_wrapper@.1.6.0",
            },
            includepath = {"/usr/local/cuda/include/"},
            linkpath = "/usr/local/cuda/lib/"
        ),
        @Platform(
            value = {"linux-x86_64"},
            define = {"GPU_MODE CUDA"},
            extension = "-gpu"
        )
    },
    target = "org.bytedeco.openpose",
    global = "org.bytedeco.openpose.global.openpose",
    helper = "org.bytedeco.openpose.helper.openpose"
)
public class openpose implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap
        .put(new Info().javaText("import org.bytedeco.javacpp.annotation.Index;"))
        // get rid of this
        .put(new Info("OP_API").cppTypes().annotations())
        // undef things that have somehow been def'd
        .put(new Info("USE_3D_ADAM_MODEL").define(false))
        .put(new Info("USE_EIGEN").define(false))
        // name clash!
        .put(new Info("op::String").pointerTypes("OpString"))

        // conversion macros
        .put(new Info("OP_OP2CVMAT").cppTypes("cv::Mat", "op::Matrix"))
        .put(new Info("OP_OP2CVCONSTMAT").cppTypes("const cv::Mat", "op::Matrix"))
        .put(new Info("OP_CV2OPMAT").cppTypes("op::Matrix", "cv::Mat"))
        .put(new Info("OP_CV2OPCONSTMAT").cppTypes("const op::Matrix", "cv::Mat"))
        .put(new Info("OP_OP2CVVECTOR").cppTypes("std::vector<cv::Mat>", "std::vector<op::Matrix>"))
        .put(new Info("OP_CV2OPVECTOR").cppTypes("std::vector<op::Matrix>", "std::vector<cv::Mat>"))

        // template instanciations
        // Datum stuff
        .put(new Info("op::WrapperT<op::Datum,std::vector<std::shared_ptr<op::Datum> >,std::shared_ptr<std::vector<std::shared_ptr<op::Datum> > >,std::shared_ptr<op::Worker<std::shared_ptr<std::vector<std::shared_ptr<op::Datum> > > > > >").pointerTypes("OpWrapper").define())
        // stl containers
        .put(new Info("std::vector<op::HeatMapType>").pointerTypes("HeatMapTypeVector").define())
        .put(new Info("std::shared_ptr<op::Datum>").pointerTypes("Datum").annotations("@SharedPtr"))
        .put(new Info("std::vector<std::shared_ptr<op::Datum> >").pointerTypes("Datums").define())
        .put(new Info("std::shared_ptr<std::vector<std::shared_ptr<op::Datum> > >").pointerTypes("Datums").annotations("@SharedPtr"))
        // we dealt with these instanciations directly above
        .put(new Info("BASE_DATUM").skip())
        .put(new Info("BASE_DATUMS").skip())
        .put(new Info("BASE_DATUMS_SH").skip())
        
        // worker stuff
        .put(new Info("op::Worker<std::shared_ptr<std::vector<std::shared_ptr<op::Datum> > > >").pointerTypes("DatumsWorker").define())
        .put(new Info("std::shared_ptr<op::Worker<std::shared_ptr<std::vector<std::shared_ptr<op::Datum> > > > >").pointerTypes("DatumsWorker").annotations("@SharedPtr"))

        // numerical template instanciations
        .put(new Info("op::Point<int>").pointerTypes("IntPoint"))
        .put(new Info("op::Point<float>").pointerTypes("FloatPoint"))

        .put(new Info("op::Array<float>").pointerTypes("FloatArray"))
        .put(new Info("std::array<op::Array<float>,2>").pointerTypes("FloatArray2").define())
        .put(new Info("op::Array<long long>").pointerTypes("LongLongArray"))
        .put(new Info("std::array<float,3>").pointerTypes("Float3").define())
        .put(new Info("std::vector<std::array<float,3> >").pointerTypes("Float3Vector").define())
        .put(new Info("op::Rectangle<float>").pointerTypes("FloatRectangle"))
        .put(new Info("std::array<op::Rectangle<float>,2>").pointerTypes("FloatRectangle2").define())
        // some pairs
        .put(new Info("std::pair<int,std::string>").pointerTypes("IntStringPair").define())
        .put(new Info("std::pair<op::ProducerType,op::String>").pointerTypes("ProducerOpStringPair").define())

        // These don't match Java interface
        .put(new Info("op::Point<int>::toString").skip())
        .put(new Info("op::Point<float>::toString").skip())
        .put(new Info("op::Array<float>::toString").skip())
        .put(new Info("op::Array<long long>::toString").skip())
        .put(new Info("op::Rectangle<float>::toString").skip())
        .put(
            new Info(
                // utilities
                "op::ErrorMode", "op::LogMode", "op::Priority", "op::Extensions",
                // thread
                "op::ThreadManagerMode",
                // wrapper
                "op::PoseMode", "op::Detector", "op::WorkerType",
                // producer
                "op::ProducerFpsMode", "op::ProducerProperty", "op::ProducerType",
                // pose
                "op::PoseModel", "op::PoseProperty",
                // core
                "op::ScaleMode", "op::HeatMapType", "op::RenderMode", "op::ElementToRender",
                // gui
                "op::DisplayMode", "op::FullScreenMode",
                // filestream
                "op::DataFormat", "op::CocoJsonFormat"
            ).enumerate()
        )
        // hopefully this causes all strings to be cast to same type?
        // (taken from Caffe)
        .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
        // don't need this
        .put(new Info("OVERLOAD_C_OUT").cppText("#define OVERLOAD_C_OUT(x)"));
    }
}
