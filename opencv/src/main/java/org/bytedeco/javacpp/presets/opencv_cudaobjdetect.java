package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    inherit = {
        opencv_core.class,
        opencv_cuda.class
    },
    value = {
        @Platform(
            include = "<opencv2/cudaobjdetect.hpp>",
            link = "opencv_cudaobjdetect@.3.2"
        )
    },
    target = "org.bytedeco.javacpp.opencv_cudaobjdetect"
)
public class opencv_cudaobjdetect implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv::cuda::CascadeClassifier")
                        .pointerTypes("CudaCascadeClassifier"));
    }

}
