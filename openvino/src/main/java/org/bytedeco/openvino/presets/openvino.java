package org.bytedeco.openvino.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
    value = {
        @Platform(include = "<inference_engine.hpp>", link = "inference_engine")
    },
    target = "org.bytedeco.openvino",
    global = "org.bytedeco.openvino.global.openvino"
)
public class openvino implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "openvino"); }

    public void map(InfoMap infoMap) {
        // infoMap.put(new Info("gzFile").valueTypes("gzFile"));
    }
}