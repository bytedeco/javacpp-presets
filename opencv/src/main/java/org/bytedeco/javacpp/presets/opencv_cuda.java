package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Defines bindings for the OpenCV CUDA header {@code opencv2/core/cuda.hpp}
 */
@Properties(
    inherit = opencv_core.class,
    value = {
        @Platform(
            include = "<opencv2/core/cuda.hpp>",
            link = "opencv_core@.3.2"
        )
    },
    target = "org.bytedeco.javacpp.opencv_cuda"
)
public class opencv_cuda implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
        infoMap.putFirst(new Info("cv::cuda::GpuMat").skip(false))
               .putFirst(new Info("cv::InputArray", "cv::OutputArray", "cv::InputOutputArray", "cv::_InputOutputArray").skip().pointerTypes("Mat", "UMat", "GpuMat"))
               .put(new Info("cv::cuda::Stream").pointerTypes("Stream"))
               .put(new Info("operator cv::cuda::Stream::bool_type").skip()); // basically a check to see if the Stream is cv::cuda:Stream::Null
    }

}
