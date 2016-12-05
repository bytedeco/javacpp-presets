/*
 * Copyright notice to add here.
 */
package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Namespace;
//import org.bytedeco.javacpp.RealSense.context;
//import org.bytedeco.javacpp.RealSense.device;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Jérémy Laviole
 */
@Properties(target = "org.bytedeco.javacpp.RealSense",
        value = {
            @Platform(value = "linux-x86",
                    include = {"<librealsense/rs.h>",
        "<librealsense/rs.hpp>",
        "<librealsense/rscore.hpp>",
        "<librealsense/rsutil.h>"},
            link = {"realsense@"})
        })

@Platform(include = {"<stdexcept>", "<mutex>"})

public class RealSense implements InfoMapper {

    public void map(InfoMap infoMap) {

        infoMap.put(new Info("std::runtime_error").cast().pointerTypes("Pointer"))
                .put(new Info("log_severity").cast().valueTypes("int").pointerTypes("IntPointer"))
                .put(new Info("std::timed_mutex").cast().pointerTypes("Pointer"))
                .put(new Info("rs_device::start_fw_logger").virtualize())
                .put(new Info("rs_device::stop_wf_logger").virtualize())
                .put(new Info("std::function<void()>").cast().pointerTypes("Fn"))
                .put(new Info("std::function<void(rs::motion_data)>").cast().pointerTypes("Pointer"))
                .put(new Info("std::function<void(rs::frame)>").cast().pointerTypes("Pointer"))
                .put(new Info("std::function<void(rs::timestamp_data)>").cast().pointerTypes("Pointer"));
    }
    
}
