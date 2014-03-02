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
@Properties(inherit={opencv_calib3d.class, opencv_features2d.class, opencv_objdetect.class,
        opencv_photo.class, opencv_ml.class, opencv_legacy.class, opencv_video.class}, target="com.googlecode.javacpp.opencv_nonfree", value={
    @Platform(include={"<opencv2/nonfree/nonfree.hpp>", "<opencv2/nonfree/features2d.hpp>"}, link="opencv_nonfree@.2.4", preload={"opencv_gpu@.2.4", "opencv_ocl@.2.4"}),
    @Platform(value="windows", link="opencv_nonfree248", preload={"opencv_gpu248", "opencv_ocl248"}) })
public class opencv_nonfree implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
    }
}
