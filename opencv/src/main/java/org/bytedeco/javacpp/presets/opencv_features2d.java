/*
 * Copyright (C) 2014,2015 Samuel Audet
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
@Properties(inherit = {opencv_highgui.class, opencv_flann.class, opencv_ml.class}, value = {
    @Platform(include = "<opencv2/features2d.hpp>", link = "opencv_features2d@.3.0"),
    @Platform(value = "windows", link = "opencv_features2d300")},
        target = "org.bytedeco.javacpp.opencv_features2d")
public class opencv_features2d implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info().javaText("import org.bytedeco.javacpp.annotation.Index;"))
               .put(new Info("std::vector<std::vector<cv::KeyPoint> >").pointerTypes("KeyPointVectorVector").define())
               .put(new Info("std::vector<std::vector<cv::DMatch> >").pointerTypes("DMatchVectorVector").define())
               .put(new Info("cv::FREAK(cv::FREAK&)", "cv::FREAK::operator=").skip());
    }
}
