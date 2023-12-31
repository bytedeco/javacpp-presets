/*
 * Copyright (C) 2015-2022 Samuel Audet
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

package org.bytedeco.opencv.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    inherit = opencv_video.class,
    value = {
        @Platform(include = {"<opencv2/ximgproc.hpp>", "opencv2/ximgproc/edge_filter.hpp", "opencv2/ximgproc/disparity_filter.hpp",
            "opencv2/ximgproc/sparse_match_interpolator.hpp", "opencv2/ximgproc/structured_edge_detection.hpp", "opencv2/ximgproc/edgeboxes.hpp", "opencv2/ximgproc/seeds.hpp",
            "opencv2/ximgproc/segmentation.hpp", "opencv2/ximgproc/fast_hough_transform.hpp", "opencv2/ximgproc/estimated_covariance.hpp", "opencv2/ximgproc/weighted_median_filter.hpp",
            "opencv2/ximgproc/slic.hpp", "opencv2/ximgproc/lsc.hpp", "opencv2/ximgproc/paillou_filter.hpp", "opencv2/ximgproc/fast_line_detector.hpp",
            "opencv2/ximgproc/deriche_filter.hpp", "opencv2/ximgproc/peilin.hpp", "opencv2/ximgproc/fourier_descriptors.hpp", "opencv2/ximgproc/ridgefilter.hpp",
            "opencv2/ximgproc/brightedges.hpp", "opencv2/ximgproc/run_length_morphology.hpp", "opencv2/ximgproc/edgepreserving_filter.hpp",
            "opencv2/ximgproc/color_match.hpp", "opencv2/ximgproc/radon_transform.hpp", "opencv2/ximgproc/find_ellipses.hpp"}, link = "opencv_ximgproc@.409"),
        @Platform(value = "ios", preload = "libopencv_ximgproc"),
        @Platform(value = "windows", link = "opencv_ximgproc490")},
    target = "org.bytedeco.opencv.opencv_ximgproc",
    global = "org.bytedeco.opencv.global.opencv_ximgproc"
)
public class opencv_ximgproc implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv::ximgproc::segmentation::PointSet").skip())
               .put(new Info("cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategyColor",
                             "cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategySize",
                             "cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategyTexture",
                             "cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategyFill").purify());
    }
}
