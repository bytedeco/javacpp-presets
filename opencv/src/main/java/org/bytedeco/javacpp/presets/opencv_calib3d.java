/*
 * Copyright (C) 2013-2015 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    @Platform(include = {"<opencv2/calib3d/calib3d_c.h>", "<opencv2/calib3d.hpp>"}, link = "opencv_calib3d@.3.2"),
    @Platform(value = "windows", link = "opencv_calib3d320")},
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
               .put(new Info("cv::fisheye::CALIB_FIX_PRINCIPAL_POINT").javaNames("FISHEYE_CALIB_FIX_PRINCIPAL_POINT"))
               .put(new Info("Affine3d").pointerTypes("Mat"));
    }
}
