package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    inherit = {
        opencv_core.class,
        opencv_cuda.class
    },
    value = {
        @Platform(
            include = "<opencv2/cudaarithm.hpp>",
            link = "opencv_cudaarithm@.3.2"
        )
    },
    target = "org.bytedeco.javacpp.opencv_cudaarithm"
)
public class opencv_cudaarithm implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {

    }

}
