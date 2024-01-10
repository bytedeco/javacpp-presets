/*
 * Copyright (C) 2021-2023 Samuel Audet
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
package org.bytedeco.depthai.presets;

import java.util.List;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import org.bytedeco.opencv.presets.opencv_imgproc;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = opencv_imgproc.class,
    value = {
        @Platform(
            compiler = "cpp14",
            define = {"SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std", "XLINK_USE_MX_ID_NAME ON"},
            include = {
                "XLink/XLinkPublicDefines.h",
                "XLink/XLinkTime.h",
                "depthai/depthai.hpp",
                "depthai/build/config.hpp",
//                "depthai/utility/span.hpp",
                "depthai/utility/Initialization.hpp",
                "depthai/utility/LockingQueue.hpp",
                "depthai/utility/Pimpl.hpp",
                "depthai/utility/Path.hpp",
                "depthai/utility/ProfilingData.hpp",
                "depthai-shared/utility/Serialization.hpp",
//                "depthai-shared/common/optional.hpp",
                "depthai-shared/common/CameraBoardSocket.hpp",
                "depthai-shared/common/CameraImageOrientation.hpp",
                "depthai-shared/common/CameraSensorType.hpp",
                "depthai-shared/common/CameraFeatures.hpp",
                "depthai-shared/common/ChipTemperature.hpp",
                "depthai-shared/common/ConnectionInterface.hpp",
                "depthai-shared/common/CpuUsage.hpp",
                "depthai-shared/common/DetectionNetworkType.hpp",
                "depthai-shared/common/DetectionParserOptions.hpp",
                "depthai-shared/common/MemoryInfo.hpp",
                "depthai-shared/common/Point2f.hpp",
                "depthai-shared/common/Point3f.hpp",
                "depthai-shared/common/Size2f.hpp",
                "depthai-shared/common/Rect.hpp",
                "depthai-shared/common/RotatedRect.hpp",
                "depthai-shared/common/Extrinsics.hpp",
                "depthai-shared/common/CameraModel.hpp",
                "depthai-shared/common/CameraInfo.hpp",
                "depthai-shared/common/Colormap.hpp",
                "depthai-shared/common/FrameEvent.hpp",
                "depthai-shared/common/Interpolation.hpp",
                "depthai-shared/common/MedianFilter.hpp",
                "depthai-shared/common/ProcessorType.hpp",
                "depthai-shared/common/StereoRectification.hpp",
                "depthai-shared/common/EepromData.hpp",
                "depthai-shared/common/TensorInfo.hpp",
                "depthai-shared/common/Timestamp.hpp",
                "depthai-shared/common/UsbSpeed.hpp",
                "depthai-shared/datatype/DatatypeEnum.hpp",
                "depthai-shared/datatype/RawBuffer.hpp",
                "depthai-shared/datatype/RawAprilTagConfig.hpp",
                "depthai-shared/datatype/RawAprilTags.hpp",
                "depthai-shared/datatype/RawIMUData.hpp",
                "depthai-shared/datatype/RawCameraControl.hpp",
                "depthai-shared/datatype/RawEdgeDetectorConfig.hpp",
                "depthai-shared/datatype/RawImgFrame.hpp",
                "depthai-shared/datatype/RawImgDetections.hpp",
                "depthai-shared/datatype/RawImageManipConfig.hpp",
                "depthai-shared/datatype/RawMessageGroup.hpp",
                "depthai-shared/datatype/RawNNData.hpp",
                "depthai-shared/datatype/RawSpatialImgDetections.hpp",
                "depthai-shared/datatype/RawSpatialLocationCalculatorConfig.hpp",
                "depthai-shared/datatype/RawSpatialLocations.hpp",
                "depthai-shared/datatype/RawStereoDepthConfig.hpp",
                "depthai-shared/datatype/RawSystemInformation.hpp",
                "depthai-shared/datatype/RawToFConfig.hpp",
                "depthai-shared/datatype/RawTracklets.hpp",
                "depthai-shared/device/BoardConfig.hpp",
                "depthai-shared/device/CrashDump.hpp",
                "depthai-shared/log/LogLevel.hpp",
                "depthai-shared/log/LogMessage.hpp",
                "depthai-shared/xlink/XLinkConstants.hpp",
                "depthai-shared/properties/Properties.hpp",
                "depthai-shared/properties/AprilTagProperties.hpp",
                "depthai-shared/properties/IMUProperties.hpp",
                "depthai-shared/properties/UVCProperties.hpp",
                "depthai-shared/properties/GlobalProperties.hpp",
                "depthai-shared/properties/CameraProperties.hpp",
                "depthai-shared/properties/ColorCameraProperties.hpp",
                "depthai-shared/properties/ImageManipProperties.hpp",
                "depthai-shared/properties/MessageDemuxProperties.hpp",
                "depthai-shared/properties/MonoCameraProperties.hpp",
                "depthai-shared/properties/EdgeDetectorProperties.hpp",
                "depthai-shared/properties/NeuralNetworkProperties.hpp",
                "depthai-shared/properties/DetectionNetworkProperties.hpp",
                "depthai-shared/properties/DetectionParserProperties.hpp",
                "depthai-shared/properties/ObjectTrackerProperties.hpp",
                "depthai-shared/properties/SPIOutProperties.hpp",
                "depthai-shared/properties/SpatialDetectionNetworkProperties.hpp",
                "depthai-shared/properties/SpatialLocationCalculatorProperties.hpp",
                "depthai-shared/properties/StereoDepthProperties.hpp",
                "depthai-shared/properties/SyncProperties.hpp",
                "depthai-shared/properties/SystemLoggerProperties.hpp",
                "depthai-shared/properties/ToFProperties.hpp",
                "depthai-shared/properties/VideoEncoderProperties.hpp",
                "depthai-shared/properties/WarpProperties.hpp",
                "depthai-shared/properties/XLinkInProperties.hpp",
                "depthai-shared/properties/XLinkOutProperties.hpp",
                "depthai-shared/pipeline/Assets.hpp",
                "depthai-shared/pipeline/NodeConnectionSchema.hpp",
                "depthai-shared/pipeline/NodeIoInfo.hpp",
                "depthai-shared/pipeline/NodeObjInfo.hpp",
                "depthai-shared/pipeline/PipelineSchema.hpp",
                "depthai/openvino/OpenVINO.hpp",
                "depthai/common/UsbSpeed.hpp",
                "depthai/common/CameraBoardSocket.hpp",
                "depthai/common/CameraExposureOffset.hpp",
                "depthai/common/CameraFeatures.hpp",
                "depthai/pipeline/datatype/ADatatype.hpp",
                "depthai/pipeline/datatype/AprilTagConfig.hpp",
                "depthai/pipeline/datatype/AprilTags.hpp",
                "depthai/pipeline/datatype/Buffer.hpp",
                "depthai/pipeline/datatype/IMUData.hpp",
                "depthai/pipeline/datatype/CameraControl.hpp",
                "depthai/pipeline/datatype/EdgeDetectorConfig.hpp",
                "depthai/pipeline/datatype/ImgFrame.hpp",
                "depthai/pipeline/datatype/ImgDetections.hpp",
                "depthai/pipeline/datatype/ImageManipConfig.hpp",
                "depthai/pipeline/datatype/MessageGroup.hpp",
                "depthai/pipeline/datatype/NNData.hpp",
                "depthai/pipeline/datatype/SpatialImgDetections.hpp",
                "depthai/pipeline/datatype/SpatialLocationCalculatorData.hpp",
                "depthai/pipeline/datatype/SpatialLocationCalculatorConfig.hpp",
                "depthai/pipeline/datatype/StereoDepthConfig.hpp",
                "depthai/pipeline/datatype/SystemInformation.hpp",
                "depthai/pipeline/datatype/ToFConfig.hpp",
                "depthai/pipeline/datatype/Tracklets.hpp",
                "depthai/pipeline/AssetManager.hpp",
                "depthai/pipeline/Node.hpp",
                "depthai/pipeline/Pipeline.hpp",
                "depthai/pipeline/nodes.hpp",
                "depthai/pipeline/datatypes.hpp",
                "depthai/pipeline/node/IMU.hpp",
                "depthai/pipeline/node/UVC.hpp",
                "depthai/pipeline/node/AprilTag.hpp",
                "depthai/pipeline/node/Camera.hpp",
                "depthai/pipeline/node/ColorCamera.hpp",
                "depthai/pipeline/node/ImageManip.hpp",
                "depthai/pipeline/node/MessageDemux.hpp",
                "depthai/pipeline/node/MonoCamera.hpp",
                "depthai/pipeline/node/EdgeDetector.hpp",
                "depthai/pipeline/node/NeuralNetwork.hpp",
                "depthai/pipeline/node/DetectionNetwork.hpp",
                "depthai/pipeline/node/DetectionParser.hpp",
                "depthai/pipeline/node/ObjectTracker.hpp",
                "depthai/pipeline/node/SPIOut.hpp",
                "depthai/pipeline/node/SpatialDetectionNetwork.hpp",
                "depthai/pipeline/node/SpatialLocationCalculator.hpp",
                "depthai/pipeline/node/StereoDepth.hpp",
                "depthai/pipeline/node/Sync.hpp",
                "depthai/pipeline/node/SystemLogger.hpp",
                "depthai/pipeline/node/ToF.hpp",
                "depthai/pipeline/node/VideoEncoder.hpp",
                "depthai/pipeline/node/Warp.hpp",
                "depthai/pipeline/node/XLinkIn.hpp",
                "depthai/pipeline/node/XLinkOut.hpp",
                "depthai/xlink/XLinkConnection.hpp",
                "depthai/xlink/XLinkStream.hpp",
                "depthai-bootloader-shared/Config.hpp",
                "depthai-bootloader-shared/Memory.hpp",
                "depthai-bootloader-shared/Section.hpp",
                "depthai-bootloader-shared/Type.hpp",
                "depthai/device/Version.hpp",
                "depthai/device/DataQueue.hpp",
                "depthai/device/DeviceBase.hpp",
                "depthai/device/CalibrationHandler.hpp",
                "depthai/device/CallbackHandler.hpp",
                "depthai/device/Device.hpp",
                "depthai/device/DeviceBootloader.hpp",
            },
            link = {"depthai-core", "depthai-opencv"}
        ),
        @Platform(value = "macosx", preload = "usb-1.0@.0", preloadpath = "/usr/local/lib/")
    },
    target = "org.bytedeco.depthai",
    global = "org.bytedeco.depthai.global.depthai"
)
public class depthai implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "depthai"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info().enumerate())
               .put(new Info("DEPTHAI_HAVE_OPENCV_SUPPORT").define(true))
               .put(new Info("XLINK_MAX_STREAM_RES", "defined(__cpp_lib_filesystem)", "defined(__cpp_lib_char8_t)").define(false))
               .put(new Info("NLOHMANN_DEFINE_TYPE_INTRUSIVE", "NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE", "NOP_STRUCTURE", "DEPTHAI_NODISCARD",
                             "DEPTHAI_NLOHMANN_DEFINE_TYPE_INTRUSIVE").cppTypes().annotations())

               .put(new Info("const char").valueTypes("@Cast(\"const char\") byte").pointerTypes("@Cast(\"const char*\") BytePointer", "@Cast(\"const char*\") ByteBuffer", "String"))
               .put(new Info("std::uint8_t").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("std::int16_t", "std::uint16_t").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
               .put(new Info("std::int32_t", "std::uint32_t", "dai::OpenVINO::Version").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
               .put(new Info("std::int64_t", "dai::Node::Id").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))
               .put(new Info("std::size_t").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "ByteBuffer", "String").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
               .put(new Info("std::basic_string<dai::Path::value_type>").cast().pointerTypes("Pointer"))

               .put(new Info("basic/containers").cppTypes("tl::optional"))
               .put(new Info("dai::XLinkStream::read").annotations("@Function"))
               .put(new Info("dai::bootloader::Config").pointerTypes("BootloaderConfig"))
               .put(new Info("dai::BoardConfig::USB").pointerTypes("BoardConfig.USB"))
               .put(new Info("dai::BoardConfig::GPIO").pointerTypes("BoardConfig.GPIO"))
               .put(new Info("dai::BoardConfig::UART").pointerTypes("BoardConfig.UART"))
               .put(new Info("dai::BoardConfig::IMU").pointerTypes("BoardConfig.IMU"))
               .put(new Info("dai::BoardConfig::UVC").pointerTypes("BoardConfig.UVC"))
               .put(new Info("dai::BoardConfig::Camera").pointerTypes("BoardConfig.Camera"))
               .put(new Info("dai::DeviceBase::Config", "dai::Device::Config").pointerTypes("DeviceBase.Config"))
               .put(new Info("auto", "span", "std::initializer_list", "std::weak_ptr", "dai::XLinkStream(dai::XLinkStream)",
                             "dai::RawStereoDepthConfig::CostAggregation::defaultHorizontalPenaltyCosts",
                             "dai::RawStereoDepthConfig::CostAggregation::defaultVerticalPenaltyCosts",
                             "dai::node::Camera::getScaledSize", "dai::MessageGroup::add", "getMonotonicTimestamp").skip())
               .put(new Info("std::chrono::microseconds", "std::chrono::milliseconds", "std::chrono::seconds", "std::runtime_error",
                             "std::chrono::nanoseconds", "std::chrono::duration<Rep,Period>", "std::chrono::duration<dai::ImgFrame,Period>",
                             "std::chrono::time_point<std::chrono::steady_clock,std::chrono::steady_clock::duration>",
                             "std::tuple<int,int>", "std::tuple<float,float>", "std::tuple<bool,dai::DeviceInfo>", "std::tuple<bool,std::string>",
                             "std::tuple<unsigned int,unsigned int>", "std::tuple<float,float,float,float>", "tl::optional<std::array<uint16_t,256> >",
                             "std::array<uint32_t,4>", "std::array<uint16_t,256>", "std::array<uint8_t,6>", "nlohmann::json",
                             "std::unordered_map<std::string,std::shared_ptr<dai::ADatatype> >::iterator",
                             "std::unordered_map<std::tuple<std::string,std::string>,dai::NodeIoInfo,dai::NodeObjInfo::IoInfoKey>",
                             "dai::copyable_unique_ptr<dai::Properties>", "nop::Status<void>").cast().pointerTypes("Pointer"))
               .put(new Info("std::shared_ptr<dai::Asset>").annotations("@SharedPtr").pointerTypes("Asset"))
               .put(new Info("std::shared_ptr<dai::ADatatype>").annotations("@SharedPtr").pointerTypes("ADatatype"))
               .put(new Info("std::shared_ptr<dai::Node>").annotations("@SharedPtr").pointerTypes("Node"))
               .put(new Info("std::vector<uint8_t>").pointerTypes("ByteVector").define())
               .put(new Info("std::vector<int>").pointerTypes("IntVector").define())
               .put(new Info("std::vector<std::string>").pointerTypes("StringVector").define())
               .put(new Info("std::vector<std::vector<float> >").pointerTypes("FloatVectorVector").define())
               .put(new Info("std::vector<std::pair<float,float> >").pointerTypes("FloatFloatPairVector").define())
               .put(new Info("std::vector<dai::Point2f>").pointerTypes("Point2fVector").define())
               .put(new Info("std::vector<std::shared_ptr<dai::Asset> >").pointerTypes("AssetVector").define())
               .put(new Info("std::vector<std::shared_ptr<dai::ADatatype> >").pointerTypes("ADatatypeVector").define())
               .put(new Info("std::vector<std::shared_ptr<dai::Node> >").pointerTypes("NodeVector").define())
               .put(new Info("const std::vector<std::pair<std::string,dai::AssetView> >",
                                   "std::vector<std::pair<std::string,dai::AssetView> >").pointerTypes("StringAssetViewPairVector").define())
               .put(new Info("std::unordered_set<dai::Node::Connection>").pointerTypes("ConnectionSet").define())
               .put(new Info("std::unordered_map<dai::CameraBoardSocket,std::string>").pointerTypes("CameraBoardSocketStringMap").define())
               .put(new Info("std::unordered_map<dai::CameraBoardSocket,dai::CameraInfo>").pointerTypes("CameraBoardSocketCameraInfoMap").define())
               .put(new Info("std::unordered_map<dai::CameraBoardSocket,dai::BoardConfig::Camera>").pointerTypes("CameraBoardSocketBoardConfigCameraMap").define())
               .put(new Info("std::unordered_map<dai::Node::Id,std::unordered_set<dai::Node::Connection> >").pointerTypes("NodeIdConnectionSetMap").define())
               .put(new Info("std::unordered_map<dai::Node::Id,std::shared_ptr<dai::Node> >").pointerTypes("NodeIdNodeMap").define())
               .put(new Info("std::unordered_map<int,int>").pointerTypes("IntIntMap").define())
               .put(new Info("std::unordered_map<int64_t,dai::NodeObjInfo>").pointerTypes("LongNodeObjInfoMap").define())
               .put(new Info("std::unordered_map<std::int8_t,dai::BoardConfig::GPIO>").pointerTypes("ByteGPIOMap").define())
               .put(new Info("std::unordered_map<std::int8_t,dai::BoardConfig::UART>").pointerTypes("ByteUARTMap").define())
               .put(new Info("std::unordered_map<std::string,dai::RawGroupMessage>").pointerTypes("StringRawGroupMessageMap").define())
               .put(new Info("std::unordered_map<std::string,dai::TensorInfo>").pointerTypes("StringTensorInfoMap").define())
               .put(new Info("std::unordered_map<std::string,dai::NodeIoInfo>").pointerTypes("StringNodeIoInfoMap").define())
               .put(new Info("std::unordered_map<std::string,dai::Node::Input*>").pointerTypes("StringNodeInputMap").define())
               .put(new Info("std::unordered_map<std::string,dai::Node::Output*>").pointerTypes("StringNodeOutputMap").define())
               .put(new Info("std::unordered_map<std::string,dai::Node::InputMap*>").pointerTypes("StringNodeInputMapMap").define())
               .put(new Info("std::unordered_map<std::string,dai::Node::OutputMap*>").pointerTypes("StringNodeOutputMapMap").define())
               .put(new Info("const std::unordered_map<std::string,dai::Node::Input>",
                                   "std::unordered_map<std::string,dai::Node::Input>").pointerTypes("StringNodeInputMap").define())
               .put(new Info("const std::unordered_map<std::string,dai::Node::Output>",
                                   "std::unordered_map<std::string,dai::Node::Output>").pointerTypes("StringNodeOutputMap").define())
               .put(new Info("std::map<std::string,std::vector<int> >").pointerTypes("StringIntVectorMap").define())
               .put(new Info("tl::optional<bool>").pointerTypes("BoolOptional").define())
               .put(new Info("tl::optional<float>").pointerTypes("FloatOptional").define())
               .put(new Info("tl::optional<int>", "tl::optional<uint32_t>", "tl::optional<std::int32_t>",
                             "tl::optional<std::uint32_t>", "tl::optional<dai::OpenVINO::Version>").cast().pointerTypes("IntOptional").define())
               .put(new Info("tl::optional<size_t>").pointerTypes("SizeTOptional").define())
               .put(new Info("tl::optional<std::string>").pointerTypes("StringOptional").define())
               .put(new Info("tl::optional<dai::BoardConfig::IMU>").pointerTypes("BoardConfigIMUOptional").define())
               .put(new Info("tl::optional<dai::BoardConfig::UVC>").pointerTypes("BoardConfigUVCOptional").define())
               .put(new Info("tl::optional<dai::CameraSensorType>").pointerTypes("CameraSensorTypeOptional").define())
               .put(new Info("tl::optional<dai::CameraImageOrientation>").pointerTypes("CameraImageOrientationOptional").define())
               .put(new Info("tl::optional<dai::EepromData>").pointerTypes("EepromDataOptional").define())
               .put(new Info("tl::optional<dai::LogLevel>", "tl::optional<LogLevel>").pointerTypes("LogLevelOptional").define())
               .put(new Info("tl::optional<dai::Version>", "tl::optional<Version>").pointerTypes("VersionOptional").define())
               .put(new Info("std::tuple<bool,float>").pointerTypes("BoolFloatTuple").define())
               .put(new Info("std::tuple<std::string,int,int>").pointerTypes("StringIntIntTuple").define())
               .put(new Info("std::tuple<std::string,std::string>").pointerTypes("StringStringTuple").define())
               .put(new Info("std::tuple<std::vector<std::vector<float> >,int,int>").pointerTypes("FloatVectorVectorIntIntTuple").define())
               .put(new Info("std::tuple<bool,std::string,std::vector<uint8_t> >").pointerTypes("BoolStringByteVectorTuple").define())

               .put(new Info("dai::Node").immutable().purify())
               .put(new Info("dai::Node::Connection").pointerTypes("Node.Connection"))
               .put(new Info("dai::Node::Input").pointerTypes("Node.Input"))
               .put(new Info("dai::Node::Output").pointerTypes("Node.Output"))
               .put(new Info("dai::Node::InputMap").pointerTypes("Node.InputMap"))
               .put(new Info("dai::Node::OutputMap").pointerTypes("Node.OutputMap"))
               .put(new Info("dai::Properties").pointerTypes("DaiProperties"))
               .put(new Info("dai::AprilTagConfig::Family").pointerTypes("RawAprilTagConfig.Family"))
               .put(new Info("dai::node::AprilTag").immutable().pointerTypes("AprilTagNode"))
               .put(new Info("dai::node::IMU", "dai::node::UVC", "dai::node::Camera", "dai::node::ColorCamera", "dai::node::ImageManip", "dai::node::MessageDemux", "dai::node::MonoCamera",
                             "dai::node::NeuralNetwork", "dai::node::EdgeDetector", "dai::node::DetectionNetwork", "dai::node::DetectionParser", "dai::node::ObjectTracker",
                             "dai::node::SPIOut", "dai::node::SpatialDetectionNetwork", "dai::node::SpatialLocationCalculator", "dai::node::StereoDepth", "dai::node::Sync",
                             "dai::node::SystemLogger", "dai::node::ToF", "dai::node::VideoEncoder", "dai::node::Warp", "dai::node::XLinkIn", "dai::node::XLinkOut").immutable())
               .put(new Info("dai::node::IMU::Properties").pointerTypes("IMUProperties"))
               .put(new Info("dai::node::UVC::Properties").pointerTypes("UVCProperties"))
               .put(new Info("dai::node::AprilTag::Properties").pointerTypes("AprilTagProperties"))
               .put(new Info("dai::node::Camera::Properties").pointerTypes("CameraProperties"))
               .put(new Info("dai::node::ColorCamera::Properties").pointerTypes("ColorCameraProperties"))
               .put(new Info("dai::node::MonoCamera::Properties").pointerTypes("MonoCameraProperties"))
               .put(new Info("dai::node::ImageManip::Properties").pointerTypes("ImageManipProperties"))
               .put(new Info("dai::node::MessageDemux::Properties").pointerTypes("MessageDemuxProperties"))
               .put(new Info("dai::node::EdgeDetector::Properties").pointerTypes("EdgeDetectorProperties"))
               .put(new Info("dai::node::NeuralNetwork::Properties").pointerTypes("NeuralNetworkProperties"))
               .put(new Info("dai::node::DetectionNetwork::Properties",
                             "dai::node::MobileNetDetectionNetwork::Properties",
                             "dai::node::YoloDetectionNetwork::Properties").pointerTypes("DetectionNetworkProperties"))
               .put(new Info("dai::node::DetectionParser::Properties").pointerTypes("DetectionParserProperties"))
               .put(new Info("dai::node::ObjectTracker::Properties").pointerTypes("ObjectTrackerProperties"))
               .put(new Info("dai::node::SPIOut::Properties").pointerTypes("SPIOutProperties"))
               .put(new Info("dai::node::SpatialDetectionNetwork::Properties").pointerTypes("SpatialDetectionNetworkProperties"))
               .put(new Info("dai::node::SpatialLocationCalculator::Properties").pointerTypes("SpatialLocationCalculatorProperties"))
               .put(new Info("dai::node::StereoDepth::Properties").pointerTypes("StereoDepthProperties"))
               .put(new Info("dai::node::Sync::Properties").pointerTypes("SyncProperties"))
               .put(new Info("dai::node::SystemLogger::Properties").pointerTypes("SystemLoggerProperties"))
               .put(new Info("dai::node::ToF::Properties").pointerTypes("ToFProperties"))
               .put(new Info("dai::node::VideoEncoder::Properties").pointerTypes("VideoEncoderProperties"))
               .put(new Info("dai::node::Warp::Properties").pointerTypes("WarpProperties"))
               .put(new Info("dai::node::XLinkIn::Properties").pointerTypes("XLinkInProperties"))
               .put(new Info("dai::node::XLinkOut::Properties").pointerTypes("XLinkOutProperties"))
               .put(new Info("dai::node::Camera::Properties::WarpMeshSource").pointerTypes("CameraProperties.WarpMeshSource"))
               .put(new Info("dai::node::ColorCamera::Properties::SensorResolution").pointerTypes("ColorCameraProperties.SensorResolution"))
               .put(new Info("dai::node::MonoCamera::Properties::SensorResolution").pointerTypes("MonoCameraProperties.SensorResolution"))
//               .put(new Info("dai::node::StereoDepth::Properties::DepthAlign").pointerTypes("StereoDepthProperties.DepthAlign"))
//               .put(new Info("dai::node::StereoDepth::Properties::MedianFilter").pointerTypes("StereoDepthProperties.MedianFilter"))
               .put(new Info("dai::node::VideoEncoder::Properties::Profile").pointerTypes("VideoEncoderProperties.Profile"))
               .put(new Info("dai::node::VideoEncoder::Properties::RateControlMode").pointerTypes("VideoEncoderProperties.RateControlMode"))
               .put(new Info("dai::node::Warp::Properties::Interpolation").pointerTypes("WarpProperties.Interpolation"))
               .put(new Info("dai::RawStereoDepthConfig::AlgorithmControl::DepthAlign",
                             "Properties::DepthAlign", "AlgorithmControl::DepthAlign").enumerate().pointerTypes("RawStereoDepthConfig.AlgorithmControl.DepthAlign"))
               .put(new Info("dai::RawStereoDepthConfig::AlgorithmControl::DepthUnit",
                             "Properties::DepthUnit", "AlgorithmControl::DepthUnit").enumerate().pointerTypes("RawStereoDepthConfig.AlgorithmControl.DepthUnit"))
               .put(new Info("dai::ToFConfig::DepthParams::TypeFMod").pointerTypes("RawToFConfig.DepthParams.TypeFMod"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::IMU,dai::IMUProperties>").pointerTypes("IMUPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::UVC,dai::UVCProperties>").pointerTypes("UVCPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::AprilTag,dai::AprilTagProperties>").pointerTypes("AprilTagPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::Camera,dai::CameraProperties>").pointerTypes("CameraPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::ColorCamera,dai::ColorCameraProperties>").pointerTypes("ColorCameraPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::MonoCamera,dai::MonoCameraProperties>").pointerTypes("MonoCameraPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::ImageManip,dai::ImageManipProperties>").pointerTypes("ImageManipPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::MessageDemux,dai::MessageDemuxProperties>").pointerTypes("MessageDemuxPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::EdgeDetector,dai::EdgeDetectorProperties>").pointerTypes("EdgeDetectorPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::NeuralNetwork,dai::NeuralNetworkProperties>").pointerTypes("NeuralNetworkPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::node::NeuralNetwork,dai::node::DetectionNetwork,dai::DetectionNetworkProperties>").pointerTypes("DetectionNetworkPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::node::DetectionNetwork,dai::node::MobileNetDetectionNetwork,dai::DetectionNetworkProperties>",
                             "dai::NodeCRTP<dai::node::DetectionNetwork,MobileNetDetectionNetwork,dai::DetectionNetworkProperties>").pointerTypes("MobileNetDetectionNetworkPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::node::DetectionNetwork,dai::node::YoloDetectionNetwork,dai::DetectionNetworkProperties>",
                             "dai::NodeCRTP<dai::node::DetectionNetwork,YoloDetectionNetwork,dai::DetectionNetworkProperties>").pointerTypes("YoloDetectionNetworkPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::ObjectTracker,dai::ObjectTrackerProperties>").pointerTypes("ObjectTrackerPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::SPIOut,dai::SPIOutProperties>").pointerTypes("SPIOutPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::node::DetectionNetwork,dai::node::SpatialDetectionNetwork,dai::SpatialDetectionNetworkProperties>").pointerTypes("SpatialDetectionNetworkPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::node::SpatialDetectionNetwork,dai::node::MobileNetSpatialDetectionNetwork,dai::SpatialDetectionNetworkProperties>",
                             "dai::NodeCRTP<dai::node::SpatialDetectionNetwork,MobileNetSpatialDetectionNetwork,dai::SpatialDetectionNetworkProperties>").pointerTypes("MobileNetSpatialDetectionNetworkPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::node::SpatialDetectionNetwork,dai::node::YoloSpatialDetectionNetwork,dai::SpatialDetectionNetworkProperties>",
                             "dai::NodeCRTP<dai::node::SpatialDetectionNetwork,YoloSpatialDetectionNetwork,dai::SpatialDetectionNetworkProperties>").pointerTypes("YoloSpatialDetectionNetworkPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::DetectionParser,dai::DetectionParserProperties>").pointerTypes("DetectionParserPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::SpatialLocationCalculator,dai::SpatialLocationCalculatorProperties>").pointerTypes("SpatialLocationCalculatorPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::StereoDepth,dai::StereoDepthProperties>").pointerTypes("StereoDepthPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::Sync,dai::SyncProperties>").pointerTypes("SyncPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::SystemLogger,dai::SystemLoggerProperties>").pointerTypes("SystemLoggerPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::ToF,dai::ToFProperties>").pointerTypes("ToFPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::VideoEncoder,dai::VideoEncoderProperties>").pointerTypes("VideoEncoderPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::Warp,dai::WarpProperties>").pointerTypes("WarpPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::XLinkIn,dai::XLinkInProperties>").pointerTypes("XLinkInPropertiesNode"))
               .put(new Info("dai::NodeCRTP<dai::Node,dai::node::XLinkOut,dai::XLinkOutProperties>").pointerTypes("XLinkOutPropertiesNode"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::IMUProperties>",
                             "dai::PropertiesSerializable<dai::Properties,IMUProperties>").pointerTypes("IMUPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::UVCProperties>",
                             "dai::PropertiesSerializable<dai::Properties,UVCProperties>").pointerTypes("UVCPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::AprilTagProperties>",
                             "dai::PropertiesSerializable<dai::Properties,AprilTagProperties>").pointerTypes("AprilTagPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::CameraProperties>",
                             "dai::PropertiesSerializable<dai::Properties,CameraProperties>").pointerTypes("CameraPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::ColorCameraProperties>",
                             "dai::PropertiesSerializable<dai::Properties,ColorCameraProperties>").pointerTypes("ColorCameraPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::MonoCameraProperties>",
                             "dai::PropertiesSerializable<dai::Properties,MonoCameraProperties>").pointerTypes("MonoCameraPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::ImageManipProperties>",
                             "dai::PropertiesSerializable<dai::Properties,ImageManipProperties>").pointerTypes("ImageManipPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::MessageDemuxProperties>",
                             "dai::PropertiesSerializable<dai::Properties,MessageDemuxProperties>").pointerTypes("MessageDemuxPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::EdgeDetectorProperties>",
                             "dai::PropertiesSerializable<dai::Properties,EdgeDetectorProperties>").pointerTypes("EdgeDetectorPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::GlobalProperties>",
                             "dai::PropertiesSerializable<dai::Properties,GlobalProperties>").pointerTypes("GlobalPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::NeuralNetworkProperties>",
                             "dai::PropertiesSerializable<dai::Properties,NeuralNetworkProperties>").pointerTypes("NeuralNetworkPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::NeuralNetworkProperties,dai::DetectionNetworkProperties>",
                             "dai::PropertiesSerializable<dai::NeuralNetworkProperties,DetectionNetworkProperties>").pointerTypes("DetectionNetworkPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::DetectionParserProperties>",
                             "dai::PropertiesSerializable<dai::Properties,DetectionParserProperties>").pointerTypes("DetectionParserPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::ObjectTrackerProperties>",
                             "dai::PropertiesSerializable<dai::Properties,ObjectTrackerProperties>").pointerTypes("ObjectTrackerPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::SPIOutProperties>",
                             "dai::PropertiesSerializable<dai::Properties,SPIOutProperties>").pointerTypes("SPIOutPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::DetectionNetworkProperties,dai::SpatialDetectionNetworkProperties>",
                             "dai::PropertiesSerializable<dai::DetectionNetworkProperties,SpatialDetectionNetworkProperties>").pointerTypes("SpatialDetectionNetworkPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::SpatialLocationCalculatorProperties>",
                             "dai::PropertiesSerializable<dai::Properties,SpatialLocationCalculatorProperties>").pointerTypes("SpatialLocationCalculatorPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::StereoDepthProperties>",
                             "dai::PropertiesSerializable<dai::Properties,StereoDepthProperties>").pointerTypes("StereoDepthPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::SyncProperties>",
                             "dai::PropertiesSerializable<dai::Properties,SyncProperties>").pointerTypes("SyncPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::SystemLoggerProperties>",
                             "dai::PropertiesSerializable<dai::Properties,SystemLoggerProperties>").pointerTypes("SystemLoggerPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::ToFProperties>",
                             "dai::PropertiesSerializable<dai::Properties,ToFProperties>").pointerTypes("ToFPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::VideoEncoderProperties>",
                             "dai::PropertiesSerializable<dai::Properties,VideoEncoderProperties>").pointerTypes("VideoEncoderPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::WarpProperties>",
                             "dai::PropertiesSerializable<dai::Properties,WarpProperties>").pointerTypes("WarpPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::XLinkInProperties>",
                             "dai::PropertiesSerializable<dai::Properties,XLinkInProperties>").pointerTypes("XLinkInPropertiesSerializable"))
               .put(new Info("dai::PropertiesSerializable<dai::Properties,dai::XLinkOutProperties>",
                             "dai::PropertiesSerializable<dai::Properties,XLinkOutProperties>").pointerTypes("XLinkOutPropertiesSerializable"))

               .put(new Info("dai::IMUReport::accuracy").javaNames("reportAccuracy"))
               .put(new Info("dai::DataInputQueue::send(const std::shared_ptr<dai::ADatatype>&)",
                             "dai::DataInputQueue::send(const std::shared_ptr<dai::ADatatype>&, std::chrono::milliseconds)").javaNames("sendSharedPtr"))

               .put(new Info("dai::Pipeline::create").javaText(
                       "public native @Name(\"create<dai::node::IMU>\") @SharedPtr IMU createIMU();\n"
                     + "public native @Name(\"create<dai::node::UVC>\") @SharedPtr UVC createUVC();\n"
                     + "public native @Name(\"create<dai::node::AprilTag>\") @SharedPtr AprilTagNode createAprilTag();\n"
                     + "public native @Name(\"create<dai::node::Camera>\") @SharedPtr Camera createCamera();\n"
                     + "public native @Name(\"create<dai::node::ColorCamera>\") @SharedPtr ColorCamera createColorCamera();\n"
                     + "public native @Name(\"create<dai::node::ImageManip>\") @SharedPtr ImageManip createImageManip();\n"
                     + "public native @Name(\"create<dai::node::MessageDemux>\") @SharedPtr MessageDemux createMessageDemux();\n"
                     + "public native @Name(\"create<dai::node::MonoCamera>\") @SharedPtr MonoCamera createMonoCamera();\n"
                     + "public native @Name(\"create<dai::node::NeuralNetwork>\") @SharedPtr NeuralNetwork createNeuralNetwork();\n"
//                     + "public native @Name(\"create<dai::node::DetectionNetwork>\") @SharedPtr DetectionNetwork createDetectionNetwork();\n"
                     + "public native @Name(\"create<dai::node::DetectionParser>\") @SharedPtr DetectionParser createDetectionParser();\n"
                     + "public native @Name(\"create<dai::node::MobileNetDetectionNetwork>\") @SharedPtr MobileNetDetectionNetwork createMobileNetDetectionNetwork();\n"
                     + "public native @Name(\"create<dai::node::YoloDetectionNetwork>\") @SharedPtr YoloDetectionNetwork createYoloDetectionNetwork();\n"
                     + "public native @Name(\"create<dai::node::ObjectTracker>\") @SharedPtr ObjectTracker createObjectTracker();\n"
                     + "public native @Name(\"create<dai::node::SPIOut>\") @SharedPtr SPIOut createSPIOut();\n"
//                     + "public native @Name(\"create<dai::node::SpatialDetectionNetwork>\") @SharedPtr SpatialDetectionNetwork createSpatialDetectionNetwork();\n"
                     + "public native @Name(\"create<dai::node::MobileNetSpatialDetectionNetwork>\") @SharedPtr MobileNetSpatialDetectionNetwork createMobileNetSpatialDetectionNetwork();\n"
                     + "public native @Name(\"create<dai::node::YoloSpatialDetectionNetwork>\") @SharedPtr YoloSpatialDetectionNetwork createYoloSpatialDetectionNetwork();\n"
                     + "public native @Name(\"create<dai::node::SpatialLocationCalculator>\") @SharedPtr SpatialLocationCalculator createSpatialLocationCalculator();\n"
                     + "public native @Name(\"create<dai::node::StereoDepth>\") @SharedPtr StereoDepth createStereoDepth();\n"
                     + "public native @Name(\"create<dai::node::Sync>\") @SharedPtr Sync createSync();\n"
                     + "public native @Name(\"create<dai::node::SystemLogger>\") @SharedPtr SystemLogger createSystemLogger();\n"
                     + "public native @Name(\"create<dai::node::ToF>\") @SharedPtr ToF createToF();\n"
                     + "public native @Name(\"create<dai::node::VideoEncoder>\") @SharedPtr VideoEncoder createVideoEncoder();\n"
                     + "public native @Name(\"create<dai::node::Warp>\") @SharedPtr Warp createWarp();\n"
                     + "public native @Name(\"create<dai::node::XLinkIn>\") @SharedPtr XLinkIn createXLinkIn();\n"
                     + "public native @Name(\"create<dai::node::XLinkOut>\") @SharedPtr XLinkOut createXLinkOut();\n"))
               .put(new Info("dai::DataOutputQueue::has").javaText(
                       "public native @Cast(\"bool\") boolean has();\n"
                     + "public native @Name(\"has<dai::AprilTagConfig>\") @Cast(\"bool\") boolean hasAprilTagConfig();\n"
                     + "public native @Name(\"has<dai::ImgFrame>\") @Cast(\"bool\") boolean hasImgFrame();\n"
                     + "public native @Name(\"has<dai::IMUData>\") @Cast(\"bool\") boolean hasIMUData();\n"
                     + "public native @Name(\"has<dai::SpatialLocationCalculatorConfig>\") @Cast(\"bool\") boolean hasSpatialLocationCalculatorConfig();\n"
                     + "public native @Name(\"has<dai::ImgDetections>\") @Cast(\"bool\") boolean hasImgDetections();\n"
                     + "public native @Name(\"has<dai::NNData>\") @Cast(\"bool\") boolean hasNNData();\n"
                     + "public native @Name(\"has<dai::Tracklets>\") @Cast(\"bool\") boolean hasTracklets();\n"
                     + "public native @Name(\"has<dai::SpatialImgDetections>\") @Cast(\"bool\") boolean hasSpatialImgDetections();\n"
                     + "public native @Name(\"has<dai::SpatialLocationCalculatorData>\") @Cast(\"bool\") boolean hasSpatialLocationCalculatorData();\n"
                     + "public native @Name(\"has<dai::StereoDepthConfig>\") @Cast(\"bool\") boolean hasStereoDepthConfig();\n"
                     + "public native @Name(\"has<dai::SystemInformation>\") @Cast(\"bool\") boolean hasSystemInformation();\n"))
               .put(new Info("dai::DataOutputQueue::tryGet").javaText(
                       "public native @SharedPtr @ByVal ADatatype tryGet();\n"
                     + "public native @Name(\"tryGet<dai::ADatatype>\") void tryGetVoid();\n"
                     + "public native @Name(\"tryGet<dai::AprilTagConfig>\") @SharedPtr AprilTagConfig tryGetAprilTagConfig();\n"
                     + "public native @Name(\"tryGet<dai::ImgFrame>\") @SharedPtr ImgFrame tryGetImgFrame();\n"
                     + "public native @Name(\"tryGet<dai::IMUData>\") @SharedPtr IMUData tryGetIMUData();\n"
                     + "public native @Name(\"tryGet<dai::SpatialLocationCalculatorConfig>\") @SharedPtr SpatialLocationCalculatorConfig tryGetSpatialLocationCalculatorConfig();\n"
                     + "public native @Name(\"tryGet<dai::ImgDetections>\") @SharedPtr ImgDetections tryGetImgDetections();\n"
                     + "public native @Name(\"tryGet<dai::NNData>\") @SharedPtr NNData tryGetNNData();\n"
                     + "public native @Name(\"tryGet<dai::Tracklets>\") @SharedPtr Tracklets tryGetTracklets();\n"
                     + "public native @Name(\"tryGet<dai::SpatialImgDetections>\") @SharedPtr SpatialImgDetections tryGetSpatialImgDetections();\n"
                     + "public native @Name(\"tryGet<dai::SpatialLocationCalculatorData>\") @SharedPtr SpatialLocationCalculatorData tryGetSpatialLocationCalculatorData();\n"
                     + "public native @Name(\"tryGet<dai::StereoDepthConfig>\") @SharedPtr StereoDepthConfig tryGetStereoDepthConfig();\n"
                     + "public native @Name(\"tryGet<dai::SystemInformation>\") @SharedPtr SystemInformation tryGetSystemInformation();\n"))
               .put(new Info("dai::DataOutputQueue::get").javaText(
                       "public native @SharedPtr @ByVal ADatatype get();\n"
                     + "public native @Name(\"get<dai::ADatatype>\") void getVoid();\n"
                     + "public native @Name(\"get<dai::AprilTagConfig>\") @SharedPtr AprilTagConfig getAprilTagConfig();\n"
                     + "public native @Name(\"get<dai::ImgFrame>\") @SharedPtr ImgFrame getImgFrame();\n"
                     + "public native @Name(\"get<dai::IMUData>\") @SharedPtr IMUData getIMUData();\n"
                     + "public native @Name(\"get<dai::SpatialLocationCalculatorConfig>\") @SharedPtr SpatialLocationCalculatorConfig getSpatialLocationCalculatorConfig();\n"
                     + "public native @Name(\"get<dai::ImgDetections>\") @SharedPtr ImgDetections getImgDetections();\n"
                     + "public native @Name(\"get<dai::NNData>\") @SharedPtr NNData getNNData();\n"
                     + "public native @Name(\"get<dai::Tracklets>\") @SharedPtr Tracklets getTracklets();\n"
                     + "public native @Name(\"get<dai::SpatialImgDetections>\") @SharedPtr SpatialImgDetections getSpatialImgDetections();\n"
                     + "public native @Name(\"get<dai::SpatialLocationCalculatorData>\") @SharedPtr SpatialLocationCalculatorData getSpatialLocationCalculatorData();\n"
                     + "public native @Name(\"get<dai::StereoDepthConfig>\") @SharedPtr StereoDepthConfig getStereoDepthConfig();\n"
                     + "public native @Name(\"get<dai::SystemInformation>\") @SharedPtr SystemInformation getSystemInformation();\n"))
               .put(new Info("dai::DataOutputQueue::front").javaText(
                       "public native @SharedPtr @ByVal ADatatype front();\n"
                     + "public native @Name(\"front<dai::ADatatype>\") void frontVoid();\n"
                     + "public native @Name(\"front<dai::AprilTagConfig>\") @SharedPtr AprilTagConfig frontAprilTagConfig();\n"
                     + "public native @Name(\"front<dai::ImgFrame>\") @SharedPtr ImgFrame frontImgFrame();\n"
                     + "public native @Name(\"front<dai::IMUData>\") @SharedPtr IMUData frontIMUData();\n"
                     + "public native @Name(\"front<dai::SpatialLocationCalculatorConfig>\") @SharedPtr SpatialLocationCalculatorConfig frontSpatialLocationCalculatorConfig();\n"
                     + "public native @Name(\"front<dai::ImgDetections>\") @SharedPtr ImgDetections frontImgDetections();\n"
                     + "public native @Name(\"front<dai::NNData>\") @SharedPtr NNData frontNNData();\n"
                     + "public native @Name(\"front<dai::Tracklets>\") @SharedPtr Tracklets frontTracklets();\n"
                     + "public native @Name(\"front<dai::SpatialImgDetections>\") @SharedPtr SpatialImgDetections frontSpatialImgDetections();\n"
                     + "public native @Name(\"front<dai::SpatialLocationCalculatorData>\") @SharedPtr SpatialLocationCalculatorData frontSpatialLocationCalculatorData();\n"
                     + "public native @Name(\"front<dai::StereoDepthConfig>\") @SharedPtr StereoDepthConfig frontStereoDepthConfig();\n"
                     + "public native @Name(\"front<dai::SystemInformation>\") @SharedPtr SystemInformation frontSystemInformation();\n"))
               .put(new Info("dai::DeviceBootloader::Version::toString", "dai::Version::toString",
                             "dai::DeviceInfo::toString", "dai::Node::toString").javaText("public native @StdString String toString();"))

               .put(new Info("std::function<std::shared_ptr<dai::RawBuffer>(dai::RawBuffer*)>").valueTypes("RawBufferCallback"))
               .put(new Info("std::function<void(dai::LogMessage)>").valueTypes("LogCallback"))
               .put(new Info("std::function<void(float)>").valueTypes("ProgressCallback"))
               .put(new Info("std::function<void(std::string,std::shared_ptr<dai::ADatatype>)>").valueTypes("NameMessageCallback"))
               .put(new Info("std::function<void(std::shared_ptr<dai::ADatatype>)>").valueTypes("MessageCallback"))
               .put(new Info("std::function<void()>").valueTypes("Callback"))
        ;
    }

    public static class RawBufferCallback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    RawBufferCallback(Pointer p) { super(p); }
        protected RawBufferCallback() { allocate(); }
        private native void allocate();
        public native @ByVal @Cast("std::shared_ptr<dai::RawBuffer>*") Pointer call(@ByVal @Cast("std::shared_ptr<dai::RawBuffer>*") Pointer p);
    }

    public static class LogCallback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    LogCallback(Pointer p) { super(p); }
        protected LogCallback() { allocate(); }
        private native void allocate();
        public native void call(@ByVal @Cast("dai::LogMessage*") Pointer p);
    }

    public static class NameMessageCallback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    NameMessageCallback(Pointer p) { super(p); }
        protected NameMessageCallback() { allocate(); }
        private native void allocate();
        public native void call(@ByVal @Cast("std::string*") Pointer name, @ByVal @Cast("std::shared_ptr<dai::ADatatype>*") Pointer message);
    }

    public static class MessageCallback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    MessageCallback(Pointer p) { super(p); }
        protected MessageCallback() { allocate(); }
        private native void allocate();
        public native void call(@ByVal @Cast("std::shared_ptr<dai::ADatatype>*") Pointer message);
    }

    public static class ProgressCallback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    ProgressCallback(Pointer p) { super(p); }
        protected ProgressCallback() { allocate(); }
        private native void allocate();
        public native void call(float f);
    }

    public static class Callback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Callback(Pointer p) { super(p); }
        protected Callback() { allocate(); }
        private native void allocate();
        public native void call();
    }
}
