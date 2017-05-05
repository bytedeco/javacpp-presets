package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    inherit = {
        opencv_core.class,
        opencv_imgproc.class,
        opencv_cuda.class
    },
    value = {
        @Platform(
            include = {
                "<opencv2/cudafilters.hpp>"
            },
            link = {
                "opencv_cudafilters@.3.2"
            }
        )
    },
    target = "org.bytedeco.javacpp.opencv_cudafilters"
)
public class opencv_cudafilters implements InfoMapper {
    @Override
    public void map(InfoMap infoMap) {
    }
}
