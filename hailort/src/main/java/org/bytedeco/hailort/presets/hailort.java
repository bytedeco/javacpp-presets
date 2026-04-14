package org.bytedeco.hailort.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
        value = {
                @Platform(include = {
                        "hailo/hailort.h",
                        "hailo/platform.h"
                },
                        link = "hailort",
                        resource = {"lib"},
                        linkresource = "lib"),
                @Platform(value = "windows", define = "WIN32_LEAN_AND_MEAN", link = "libhailort", preload = "libhailort"),
                @Platform(value = "linux", preload = "hailort@.4.23.0"),
        },
        target = "org.bytedeco.hailort",
        global = "org.bytedeco.hailort.global.hailort",
        helper = "org.bytedeco.hailort.HailoHelper"
)
public class hailort implements InfoMapper {

    static {
        Loader.checkVersion("org.bytedeco", "hailort");
    }

    @Override
    public void map(InfoMap infoMap) {
        // types from platform.h
        infoMap.put(new Info("socket_t").cast().valueTypes("int").pointerTypes("IntPointer"));
        infoMap.put(new Info("port_t").cast().valueTypes("int").pointerTypes("IntPointer"));
        infoMap.put(new Info("sockaddr_in").cast().javaNames("org.bytedeco.systems.linux.sockaddr").pointerTypes("Pointer"));
        infoMap.put(new Info("INVALID_SOCKET").javaText("public static final int INVALID_SOCKET = (int)(-1);"));

        infoMap.put(new Info("hailort.h").linePatterns("#ifdef __cplusplus", "#endif").skip());

        // these variables are defined in HailoHelper
        infoMap.put(new Info("hailo_status").cast().valueTypes("int").pointerTypes("IntPointer"));
        infoMap.put(new Info("HAILO_STATUS_VARIABLES").skip());

        // macros and constants
        infoMap.put(new Info("HAILO_INFINITE").javaText("public static final long HAILO_INFINITE = 0xFFFFFFFFL;"));
        infoMap.put(new Info("HAILO_PCIE_ANY_DOMAIN").javaText("public static final long HAILO_PCIE_ANY_DOMAIN = 0xFFFFFFFFL;"));
        infoMap.put(new Info("INVALID_QUANT_INFO").skip());
        infoMap.put(new Info("HAILO_DEFAULT_SOCKADDR").skip());
        infoMap.put(new Info("HAILO_ETH_INPUT_STREAM_PARAMS_DEFAULT").skip());
        infoMap.put(new Info("HAILO_ETH_OUTPUT_STREAM_PARAMS_DEFAULT").skip());
        infoMap.put(new Info("HAILO_MIPI_INPUT_STREAM_PARAMS_DEFAULT").skip());
        infoMap.put(new Info("HAILO_DEFAULT_TRANSFORM_PARAMS").skip());
        infoMap.put(new Info("HAILO_PCIE_STREAM_PARAMS_DEFAULT").skip());
        infoMap.put(new Info("HAILO_ACTIVATE_NETWORK_GROUP_PARAMS_DEFAULT").skip());
        infoMap.put(new Info("HAILO_RANDOM_SEED").javaText("public static final long HAILO_RANDOM_SEED = 0xFFFFFFFFL;"));
        infoMap.put(new Info("HAILO_ETH_ADDRESS_ANY").javaText("public static final String HAILO_ETH_ADDRESS_ANY = \"0.0.0.0\";"));
        infoMap.put(new Info("HAILORTAPI").cppTypes().annotations());
        infoMap.put(new Info("EMPTY_STRUCT_PLACEHOLDER").skip());
        infoMap.put(new Info("HAILO_UNIQUE_VDEVICE_GROUP_ID").javaText("public static final String HAILO_UNIQUE_VDEVICE_GROUP_ID = \"UNIQUE\";"));
        infoMap.put(new Info("HAILO_DEFAULT_VDEVICE_GROUP_ID").javaText("public static final String HAILO_DEFAULT_VDEVICE_GROUP_ID = HAILO_UNIQUE_VDEVICE_GROUP_ID;"));
        infoMap.put(new Info("HAILO_DEFAULT_INIT_SAMPLING_PERIOD_US").javaText("public static final int HAILO_DEFAULT_INIT_SAMPLING_PERIOD_US = 1100;"));
        infoMap.put(new Info("HAILO_DEFAULT_INIT_AVERAGING_FACTOR").javaText("public static final int HAILO_DEFAULT_INIT_AVERAGING_FACTOR = 256;"));
        infoMap.put(new Info("hailo_release_input_vstreams").javaText(
                "public static native @Cast(\"hailo_status\") int hailo_release_input_vstreams(" +
                        "@Cast(\"const hailo_input_vstream*\") _hailo_input_vstream input_vstreams, " +
                        "@Cast(\"size_t\") long inputs_count);"));
        infoMap.put(new Info("hailo_release_output_vstreams").javaText(
                "public static native @Cast(\"hailo_status\") int hailo_release_output_vstreams(" +
                        "@Cast(\"const hailo_output_vstream*\") _hailo_output_vstream output_vstreams, " +
                        "@Cast(\"size_t\") long outputs_count);"));
        infoMap.put(new Info("hailo_clear_input_vstreams").javaText(
                "public static native @Cast(\"hailo_status\") int hailo_clear_input_vstreams(" +
                        "@Cast(\"const hailo_input_vstream*\") _hailo_input_vstream input_vstreams, " +
                        "@Cast(\"size_t\") long inputs_count);"));
        infoMap.put(new Info("hailo_clear_output_vstreams").javaText(
                "public static native @Cast(\"hailo_status\") int hailo_clear_output_vstreams(" +
                        "@Cast(\"const hailo_output_vstream*\") _hailo_output_vstream output_vstreams, " +
                        "@Cast(\"size_t\") long outputs_count);"));
        infoMap.put(new Info("hailo_detections_t").skip());

        // mapping e.g. typedef struct _hailo_configured_network_group *hailo_configured_network_group;
        infoMap.put(new Info("hailo_hef").valueTypes("_hailo_hef").pointerTypes("@ByPtrPtr _hailo_hef"));
        infoMap.put(new Info("hailo_configured_network_group").valueTypes("_hailo_configured_network_group").pointerTypes("@ByPtrPtr _hailo_configured_network_group"));
        infoMap.put(new Info("hailo_activated_network_group").valueTypes("_hailo_activated_network_group").pointerTypes("@ByPtrPtr _hailo_activated_network_group"));
        infoMap.put(new Info("hailo_device").valueTypes("_hailo_device").pointerTypes("@ByPtrPtr _hailo_device"));
        infoMap.put(new Info("hailo_vdevice").valueTypes("_hailo_vdevice").pointerTypes("@ByPtrPtr _hailo_vdevice"));
        infoMap.put(new Info("hailo_input_stream").valueTypes("_hailo_input_stream").pointerTypes("@ByPtrPtr _hailo_input_stream"));
        infoMap.put(new Info("hailo_input_vstream").valueTypes("_hailo_input_vstream").pointerTypes("@ByPtrPtr _hailo_input_vstream"));
        infoMap.put(new Info("hailo_input_transform_context").valueTypes("_hailo_input_transform_context").pointerTypes("@ByPtrPtr _hailo_input_transform_context"));
        infoMap.put(new Info("hailo_output_stream").valueTypes("_hailo_output_stream").pointerTypes("@ByPtrPtr _hailo_output_stream"));
        infoMap.put(new Info("hailo_output_demuxer").valueTypes("_hailo_output_demuxer").pointerTypes("@ByPtrPtr _hailo_output_demuxer"));
        infoMap.put(new Info("hailo_output_transform_context").valueTypes("_hailo_output_transform_context").pointerTypes("@ByPtrPtr _hailo_output_transform_context"));
        infoMap.put(new Info("hailo_output_vstream").valueTypes("_hailo_output_vstream").pointerTypes("@ByPtrPtr _hailo_output_vstream"));

    }
}
