/*
 * Copyright (C) 2020 Frankie Robertson
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
    inherit = caffe.class,
    value = {
        @Platform(
            value = {"linux", "macosx"},
            compiler = "cpp11",
            define = {
                "SHARED_PTR_NAMESPACE std // overrides boost",
                "UNIQUE_PTR_NAMESPACE std // overrides boost",
                "GPU_MODE CPU_ONLY"
            },
            exclude = {
                // prevent linking unnecessarily with Boost
                "caffe/caffe.hpp",
                "caffe/common.hpp",
                "caffe/parallel.hpp",
                "caffe/solver.hpp",
                "caffe/util/blocking_queue.hpp",
                "caffe/util/benchmark.hpp",
                "caffe/util/io.hpp",
                "caffe/util/rng.hpp"
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
                "openpose@.1.7.0",
                "openpose_3d",
                "openpose_calibration",
                "openpose_core",
                "openpose_face",
                "openpose_filestream",
                "openpose_gpu",
                "openpose_gui",
                "openpose_hand",
                "openpose_net",
                "openpose_pose",
                "openpose_producer",
                "openpose_thread",
                "openpose_tracking",
                "openpose_unity",
                "openpose_utilities",
                "openpose_wrapper",
            },
            includepath = "/usr/local/cuda/include/",
            linkpath = "/usr/local/cuda/lib/"
        ),
        @Platform(
            value = "linux-x86_64",
            define = "GPU_MODE CUDA",
            extension = "-gpu"
        )
    },
    target = "org.bytedeco.openpose",
    global = "org.bytedeco.openpose.global.openpose"
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

    public static int POSE_MAX_PEOPLE = 127;
    public static float POSE_DEFAULT_ALPHA_KEYPOINT = 0.6f;
    public static float POSE_DEFAULT_ALPHA_HEAT_MAP = 0.7f;
    public static int FACE_MAX_FACES = 127;
    public static float FACE_DEFAULT_ALPHA_KEYPOINT = 0.6f;
    public static float FACE_DEFAULT_ALPHA_HEAT_MAP = 0.7f;
    public static int HAND_MAX_HANDS = 254;
    public static int HAND_NUMBER_PARTS = 21;
    public static float HAND_CCN_DECREASE_FACTOR = 8.0f;
    public static float HAND_DEFAULT_ALPHA_KEYPOINT = 0.6f;
    public static float HAND_DEFAULT_ALPHA_HEAT_MAP = 0.7f;
}
