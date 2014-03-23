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
@Properties(inherit={opencv_highgui.class, opencv_flann.class}, value={
    @Platform(include="<opencv2/features2d/features2d.hpp>", link="opencv_features2d@.2.4"),
    @Platform(value="windows", link="opencv_features2d248") },
        target="com.googlecode.javacpp.opencv_features2d")
public class opencv_features2d implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        infoMap.put(new Parser.Info().javaText("import com.googlecode.javacpp.annotation.Index;"))
               .put(new Parser.Info("std::vector<std::vector<cv::KeyPoint> >").pointerTypes("KeyPointVectorVector").define(true))
               .put(new Parser.Info("std::vector<std::vector<cv::DMatch> >").pointerTypes("DMatchVectorVector").define(true))
               .put(new Parser.Info("cv::FREAK(cv::FREAK&)", "cv::FREAK::operator=").skip(true));
    }
}
