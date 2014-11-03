/*
 * Copyright (C) 2014 Samuel Audet
 *
 * This file is part of JavaCPP.
 *
 * JavaCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version (subject to the "Classpath" exception
 * as provided in the LICENSE.txt file that accompanied this code).
 *
 * JavaCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JavaCPP.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit={opencv_calib3d.class, opencv_objdetect.class, opencv_video.class, opencv_ml.class, opencv_nonfree.class}, value={
    @Platform(include={"<opencv2/contrib/contrib.hpp>", "<opencv2/contrib/detection_based_tracker.hpp>",
        "<opencv2/contrib/hybridtracker.hpp>", "<opencv2/contrib/retina.hpp>", "<opencv2/contrib/openfabmap.hpp>"},
        link="opencv_contrib@.2.4", preload={"opencv_gpu@.2.4", "opencv_ocl@.2.4"}),
    @Platform(value="windows", include={"<opencv2/contrib/contrib.hpp>",
        "<opencv2/contrib/hybridtracker.hpp>", "<opencv2/contrib/retina.hpp>", "<opencv2/contrib/openfabmap.hpp>"},
        link="opencv_contrib2410", preload={"opencv_gpu2410", "opencv_ocl2410"}) },
        target="org.bytedeco.javacpp.opencv_contrib")
public class opencv_contrib implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info().javaText("import org.bytedeco.javacpp.annotation.Index;"))
               .put(new Info("std::vector<std::vector<cv::Vec2i> >").cast().pointerTypes("PointVectorVector"))
               .put(new Info("std::map<int,std::string>").pointerTypes("IntStringMap").define())
               .put(new Info("std::vector<std::pair<cv::Rect_<int>,int> >").pointerTypes("RectIntPairVector").define())
               .put(new Info("std::valarray<float>").pointerTypes("FloatValArray").define())
               .put(new Info("CvFuzzyMeanShiftTracker::kernel", "cv::Mesh3D::allzero",
                             "cv::SpinImageModel::calcSpinMapCoo", "cv::SpinImageModel::geometricConsistency", "cv::SpinImageModel::groupingCreteria",
                             "cv::CvMeanShiftTracker()", "cv::CvFeatureTracker()").skip());
    }
}
