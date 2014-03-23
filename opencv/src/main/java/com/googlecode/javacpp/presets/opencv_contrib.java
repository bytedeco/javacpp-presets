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

package com.googlecode.javacpp.presets;

import com.googlecode.javacpp.Parser;
import com.googlecode.javacpp.annotation.Platform;
import com.googlecode.javacpp.annotation.Properties;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit={opencv_calib3d.class, opencv_objdetect.class, opencv_video.class, opencv_ml.class}, value={
    @Platform(include={"<opencv2/contrib/contrib.hpp>", "<opencv2/contrib/detection_based_tracker.hpp>",
        "<opencv2/contrib/hybridtracker.hpp>", "<opencv2/contrib/retina.hpp>", "<opencv2/contrib/openfabmap.hpp>"}, link="opencv_contrib@.2.4"),
    @Platform(value="windows", include={"<opencv2/contrib/contrib.hpp>",
        "<opencv2/contrib/hybridtracker.hpp>", "<opencv2/contrib/retina.hpp>", "<opencv2/contrib/openfabmap.hpp>"}, link="opencv_contrib248") },
        target="com.googlecode.javacpp.opencv_contrib")
public class opencv_contrib implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        infoMap.put(new Parser.Info().javaText("import com.googlecode.javacpp.annotation.Index;"))
               .put(new Parser.Info("std::vector<std::vector<cv::Vec2i> >").cast(true).pointerTypes("PointVectorVector").define(false))
               .put(new Parser.Info("std::vector<std::pair<cv::Rect_<int>,int> >").pointerTypes("RectIntPairVector").define(true))
               .put(new Parser.Info("std::valarray<float>").pointerTypes("FloatValArray").define(true))
               .put(new Parser.Info("CvFuzzyMeanShiftTracker::kernel", "cv::Mesh3D::allzero",
                                    "cv::SpinImageModel::calcSpinMapCoo", "cv::SpinImageModel::geometricConsistency", "cv::SpinImageModel::groupingCreteria",
                                    "cv::CvMeanShiftTracker()", "cv::CvFeatureTracker()").skip(true));
    }
}
