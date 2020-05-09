/*
 * Copyright (C) 2018 Samuel Audet
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

@Properties(
    inherit = {opencv_plot.class, opencv_video.class, opencv_dnn.class},
    value = {
        @Platform(
            include = {
                "<opencv2/tracking.hpp>",
                "<opencv2/tracking/feature.hpp>",
                "<opencv2/tracking/kalman_filters.hpp>",
                "<opencv2/tracking/onlineMIL.hpp>",
                "<opencv2/tracking/onlineBoosting.hpp>",
                "<opencv2/tracking/tldDataset.hpp>",
                "<opencv2/tracking/tracker.hpp>",
            },
            link = "opencv_tracking@.4.3"
        ),
        @Platform(value = "ios", preload = "libopencv_tracking"),
        @Platform(value = "windows", link = "opencv_tracking430")
    },
    target = "org.bytedeco.opencv.opencv_tracking",
    global = "org.bytedeco.opencv.global.opencv_tracking"
)
public class opencv_tracking implements InfoMapper {
    @Override public void map(InfoMap infoMap) {
        infoMap.put(new Info().javaText("import org.bytedeco.javacpp.annotation.Index;"))
               .put(new Info("override").annotations()) // pure virtual functions are not mapped unless virtualized, so disable override annotation
               .put(new Info("cv::Ptr<cv::Tracker>").annotations("@Ptr").pointerTypes("Tracker"))
               .put(new Info("cv::Ptr<cv::TrackerFeature>").annotations("@Ptr").pointerTypes("TrackerFeature"))
               .put(new Info("cv::Ptr<cv::TrackerTargetState>").annotations("@Ptr").pointerTypes("TrackerTargetState"))
               .put(new Info("cv::Ptr<cv::TrackerSamplerAlgorithm>").annotations("@Ptr").pointerTypes("TrackerSamplerAlgorithm"))
               .put(new Info("std::vector<cv::Ptr<cv::Tracker> >").pointerTypes("TrackerVector").define())
               .put(new Info("std::vector<cv::ConfidenceMap>").pointerTypes("ConfidenceMapVector").define())
               .put(new Info("std::vector<std::pair<cv::Ptr<cv::TrackerTargetState>,float> >").pointerTypes("ConfidenceMap").define())
               .put(new Info("std::vector<std::pair<cv::String,cv::Ptr<cv::TrackerFeature> > >").pointerTypes("StringTrackerFeaturePairVector").define())
               .put(new Info("std::vector<std::pair<cv::String,cv::Ptr<cv::TrackerSamplerAlgorithm> > >").pointerTypes("StringTrackerSamplerAlgorithmPairVector").define())
               .put(new Info("std::vector<cv::Ptr<cv::TrackerTargetState> >").pointerTypes("Trajectory").define())
               .put(new Info("cv::CvHaarEvaluator::setWinSize").annotations("@Function"))
               .put(new Info("cv::TrackerMIL", "cv::TrackerBoosting", "cv::TrackerMedianFlow", "cv::TrackerTLD", "cv::TrackerGOTURN", "cv::TrackerMOSSE").purify());
    }
}
