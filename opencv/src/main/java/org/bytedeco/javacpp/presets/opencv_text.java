package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * Wrapper for OpenCV module text, part of OpenCV_Contrib.
 *
 * @author Bram Biesbrouck
 */
@Properties(inherit = {opencv_highgui.class, opencv_ml.class}, value = {
    @Platform(include = {"<opencv2/text.hpp>", "<opencv2/text/erfilter.hpp>", "<opencv2/text/ocr.hpp>", "opencv_adapters.h"},
              link = "opencv_text@.3.2"),
    @Platform(value = "windows", link = "opencv_text320")},
              target = "org.bytedeco.javacpp.opencv_text")
public class opencv_text implements InfoMapper {
    public void map(InfoMap infoMap) {
	
	infoMap.put(new Info("std::deque<int>").pointerTypes("IntDeque").define());
	infoMap.put(new Info("std::vector<cv::text::ERStat>").pointerTypes("ERStatVector").define());
	infoMap.put(new Info("std::vector<std::vector<cv::text::ERStat> >").pointerTypes("ERStatVectorVector").define());
	infoMap.put(new Info("std::vector<double>").pointerTypes("DoubleVector").define());
	infoMap.put(new Info("std::vector<std::string>").pointerTypes("StdStringVector").define());
	
	infoMap.put(new Info("std::vector<cv::Vec2i>").pointerTypes("PointVector").cast());
	infoMap.put(new Info("std::vector<std::vector<cv::Vec2i> >").pointerTypes("PointVectorVector").cast());
	
    }
}
