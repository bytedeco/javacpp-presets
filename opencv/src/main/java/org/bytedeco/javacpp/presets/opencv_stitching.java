/*
 * Copyright (C) 2014-2018 Samuel Audet
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
@Properties(inherit = {opencv_objdetect.class, opencv_video.class, opencv_xfeatures2d.class}, value = {
    @Platform(include = {
        "<opencv2/stitching/detail/warpers.hpp>", "<opencv2/stitching/detail/matchers.hpp>", "<opencv2/stitching/detail/util.hpp>",
        "<opencv2/stitching/detail/camera.hpp>", "<opencv2/stitching/detail/motion_estimators.hpp>", "<opencv2/stitching/detail/exposure_compensate.hpp>",
        "<opencv2/stitching/detail/seam_finders.hpp>", "<opencv2/stitching/detail/blenders.hpp>", "<opencv2/stitching/detail/autocalib.hpp>",
        "<opencv2/stitching/detail/timelapsers.hpp>", "<opencv2/stitching/warpers.hpp>", "<opencv2/stitching.hpp>"},
              link = "opencv_stitching@.4.0", preload = {"opencv_cuda@.4.0", "opencv_cudaarithm@.4.0", "opencv_cudafilters@.4.0",
              "opencv_cudaimgproc@.4.0", "opencv_cudafeatures2d@.4.0", "opencv_cudalegacy@.4.0", "opencv_cudawarping@.4.0"}),
    @Platform(value = "ios", preload = "libopencv_stitching"),
    @Platform(value = "windows", link = "opencv_stitching401", preload = {"opencv_cuda401", "opencv_cudaarithm401", "opencv_cudafilters401",
              "opencv_cudaimgproc401", "opencv_cudafeatures2d401", "opencv_cudalegacy401", "opencv_cudawarping401"})},
        target = "org.bytedeco.javacpp.opencv_stitching")
public class opencv_stitching implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("override").annotations()) // we are not exposing all subclasses, so disable override annotation
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
               .put(new Info("cv::PlaneWarperGpu").pointerTypes("PlaneWarperGpu"))
               .put(new Info("cv::CylindricalWarperGpu").pointerTypes("CylindricalWarperGpu"))
               .put(new Info("cv::SphericalWarperGpu").pointerTypes("SphericalWarperGpu"))
               .put(new Info("cv::detail::SurfFeaturesFinderGpu").pointerTypes("SurfFeaturesFinderGpu"))
               .put(new Info("cv::detail::GraphCutSeamFinderGpu").pointerTypes("GraphCutSeamFinderGpu"));
    }
}

