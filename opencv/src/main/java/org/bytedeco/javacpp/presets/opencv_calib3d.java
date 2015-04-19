/*
 * Copyright (C) 2013,2014,2015 Samuel Audet
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
@Properties(inherit = {opencv_highgui.class, opencv_features2d.class}, value = {
    @Platform(include = {"<opencv2/calib3d/calib3d_c.h>", "<opencv2/calib3d.hpp>"}, link = "opencv_calib3d@.3.0"),
    @Platform(value = "windows", link = "opencv_calib3d300")},
        target = "org.bytedeco.javacpp.opencv_calib3d", helper = "org.bytedeco.javacpp.helper.opencv_calib3d")
public class opencv_calib3d implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("CvPOSITObject").base("AbstractCvPOSITObject"))
               .put(new Info("CvStereoBMState").base("AbstractCvStereoBMState"))
               .put(new Info("cv::fisheye::CALIB_USE_INTRINSIC_GUESS").javaNames("FISHEYE_CALIB_USE_INTRINSIC_GUESS"))
               .put(new Info("cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC").javaNames("FISHEYE_CALIB_RECOMPUTE_EXTRINSIC"))
               .put(new Info("cv::fisheye::CALIB_CHECK_COND").javaNames("FISHEYE_CALIB_CHECK_COND"))
               .put(new Info("cv::fisheye::CALIB_FIX_SKEW").javaNames("FISHEYE_CALIB_FIX_SKEW"))
               .put(new Info("cv::fisheye::CALIB_FIX_K1").javaNames("FISHEYE_CALIB_FIX_K1"))
               .put(new Info("cv::fisheye::CALIB_FIX_K2").javaNames("FISHEYE_CALIB_FIX_K2"))
               .put(new Info("cv::fisheye::CALIB_FIX_K3").javaNames("FISHEYE_CALIB_FIX_K3"))
               .put(new Info("cv::fisheye::CALIB_FIX_K4").javaNames("FISHEYE_CALIB_FIX_K4"))
               .put(new Info("cv::fisheye::CALIB_FIX_INTRINSIC").javaNames("FISHEYE_CALIB_FIX_INTRINSIC"))
               .put(new Info("Affine3d").pointerTypes("Mat"));
    }
}
