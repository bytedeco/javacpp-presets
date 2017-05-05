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
                "<opencv2/cudaimgproc.hpp>"
            },
            link = {
                "opencv_cudaimgproc@.3.2"
            }
        )
    },
    target = "org.bytedeco.javacpp.opencv_cudaimgproc"
)
public class opencv_cudaimgproc implements InfoMapper {
    @Override
    public void map(InfoMap infoMap) {
        // The parser generates a cyclic reference (class CLAHE extends CLAHE).
        // This forces the parser to generate a proper declaration
        infoMap.put(new Info("cv::cuda::CLAHE").skip());
    }
}
