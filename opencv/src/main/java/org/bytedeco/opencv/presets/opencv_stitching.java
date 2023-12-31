/*
 * Copyright (C) 2014-2022 Samuel Audet
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
    inherit = {opencv_objdetect.class, opencv_video.class, opencv_xfeatures2d.class},
    value = {
        @Platform(include = {
            "<opencv2/stitching/detail/warpers.hpp>", "<opencv2/stitching/detail/matchers.hpp>", "<opencv2/stitching/detail/util.hpp>",
            "<opencv2/stitching/detail/camera.hpp>", "<opencv2/stitching/detail/motion_estimators.hpp>", "<opencv2/stitching/detail/exposure_compensate.hpp>",
            "<opencv2/stitching/detail/seam_finders.hpp>", "<opencv2/stitching/detail/blenders.hpp>", "<opencv2/stitching/detail/autocalib.hpp>",
            "<opencv2/stitching/detail/timelapsers.hpp>", "<opencv2/stitching/warpers.hpp>", "<opencv2/stitching.hpp>"},
            link = "opencv_stitching@.409", preload = {"opencv_cuda@.409", "opencv_cudaarithm@.409", "opencv_cudafilters@.409",
            "opencv_cudaimgproc@.409", "opencv_cudawarping@.409", "opencv_cudafeatures2d@.409", "opencv_cudalegacy@.409"}),
        @Platform(value = "ios", preload = "libopencv_stitching"),
        @Platform(value = "windows", link = "opencv_stitching490", preload = {"opencv_cuda490", "opencv_cudaarithm490", "opencv_cudafilters490",
            "opencv_cudaimgproc490", "opencv_cudawarping490", "opencv_cudafeatures2d490", "opencv_cudalegacy490"})},
    target = "org.bytedeco.opencv.opencv_stitching",
    global = "org.bytedeco.opencv.global.opencv_stitching"
)
public class opencv_stitching implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("override").annotations()) // we are not exposing all subclasses, so disable override annotation
               .put(new Info().javaText("import org.bytedeco.javacpp.annotation.Index;"))
               .put(new Info("cv::detail::BlocksCompensator").purify())
               .put(new Info("cv::detail::PlaneWarper").pointerTypes("DetailPlaneWarper").base("RotationWarper"))
               .put(new Info("cv::detail::SphericalWarper").pointerTypes("DetailSphericalWarper").base("RotationWarper"))
               .put(new Info("cv::detail::CylindricalWarper").pointerTypes("DetailCylindricalWarper").base("RotationWarper"))
               .put(new Info("cv::detail::FisheyeWarper").pointerTypes("DetailFisheyeWarper").base("RotationWarper"))
               .put(new Info("cv::detail::StereographicWarper").pointerTypes("DetailStereographicWarper").base("RotationWarper"))
               .put(new Info("cv::detail::CompressedRectilinearWarper").pointerTypes("DetailCompressedRectilinearWarper").base("RotationWarper"))
               .put(new Info("cv::detail::CompressedRectilinearPortraitWarper").pointerTypes("DetailCompressedRectilinearPortraitWarper").base("RotationWarper"))
               .put(new Info("cv::detail::PaniniWarper").pointerTypes("DetailPaniniWarper").base("RotationWarper"))
               .put(new Info("cv::detail::PaniniPortraitWarper").pointerTypes("DetailPaniniPortraitWarper").base("RotationWarper"))
               .put(new Info("cv::detail::MercatorWarper").pointerTypes("DetailMercatorWarper").base("RotationWarper"))
               .put(new Info("cv::detail::TransverseMercatorWarper").pointerTypes("DetailTransverseMercatorWarper").base("RotationWarper"))
               .put(new Info("cv::detail::PlaneWarperGpu").pointerTypes("DetailPlaneWarperGpu").base("RotationWarper"))
               .put(new Info("cv::detail::SphericalWarperGpu").pointerTypes("DetailSphericalWarperGpu").base("RotationWarper"))
               .put(new Info("cv::detail::CylindricalWarperGpu").pointerTypes("DetailCylindricalWarperGpu").base("RotationWarper"))
               .put(new Info("cv::detail::SphericalPortraitWarper", "cv::detail::CylindricalPortraitWarper", "cv::detail::PlanePortraitWarper").base("RotationWarper"))
               .put(new Info("std::vector<cv::detail::CameraParams>").pointerTypes("CameraParamsVector").define())
               .put(new Info("std::vector<cv::detail::ImageFeatures>").pointerTypes("ImageFeaturesVector").define())
               .put(new Info("std::vector<cv::detail::MatchesInfo>").pointerTypes("MatchesInfoVector").define())
               .put(new Info("cv::PlaneWarperGpu").pointerTypes("PlaneWarperGpu"))
               .put(new Info("cv::CylindricalWarperGpu").pointerTypes("CylindricalWarperGpu"))
               .put(new Info("cv::SphericalWarperGpu").pointerTypes("SphericalWarperGpu"))
               .put(new Info("cv::detail::SurfFeaturesFinderGpu").pointerTypes("SurfFeaturesFinderGpu"))
               .put(new Info("cv::detail::GraphCutSeamFinderGpu").pointerTypes("GraphCutSeamFinderGpu"));
    }
}

