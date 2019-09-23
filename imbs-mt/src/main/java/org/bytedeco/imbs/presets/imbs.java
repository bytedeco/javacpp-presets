package org.bytedeco.imbs.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.opencv.presets.*;

@Properties(
        inherit = {
                opencv_core.class,
                opencv_highgui.class,
                opencv_imgproc.class,
                opencv_features2d.class,
                opencv_stitching.class,
                opencv_aruco.class,
                opencv_bgsegm.class,
                opencv_bioinspired.class,
                opencv_face.class,
                opencv_img_hash.class,
                opencv_quality.class,
                opencv_saliency.class,
                opencv_structured_light.class,
                opencv_superres.class,
                opencv_text.class,
                opencv_tracking.class,
                opencv_videostab.class,
                opencv_xphoto.class,
        },
        value = {

                @Platform(
                        link = {"imbs-mt"},
                        include = {"imbs.hpp"}
                ),
        },
        target = "org.bytedeco.imbs",
        global = "org.bytedeco.imbs.global.imbs"
)
public class imbs implements InfoMapper {
    @Override
    public void map(InfoMap map) {
        map.put(new Info("cv::Vec3b").cast().pointerTypes("Point3i"))
       .put(new Info( "string").annotations("@StdString").valueTypes("BytePointer", "String").pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"));
    }
}
