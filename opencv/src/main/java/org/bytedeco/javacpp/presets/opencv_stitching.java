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
@Properties(inherit = {opencv_calib3d.class, opencv_features2d.class, opencv_objdetect.class, opencv_ml.class, opencv_video.class}, value = {
    @Platform(include = {
        "<opencv2/stitching/detail/warpers.hpp>", "<opencv2/stitching/detail/matchers.hpp>", "<opencv2/stitching/detail/util.hpp>",
        "<opencv2/stitching/detail/camera.hpp>", "<opencv2/stitching/detail/motion_estimators.hpp>", "<opencv2/stitching/detail/exposure_compensate.hpp>",
        "<opencv2/stitching/detail/seam_finders.hpp>", "<opencv2/stitching/detail/blenders.hpp>", "<opencv2/stitching/detail/autocalib.hpp>",
        "<opencv2/stitching/detail/timelapsers.hpp>", "<opencv2/stitching/warpers.hpp>", "<opencv2/stitching.hpp>"},
              link = "opencv_stitching@.3.0", preload = "opencv_cuda@.3.0"),
    @Platform(value = "windows", link = "opencv_stitching300", preload = "opencv_cuda300")},
        target = "org.bytedeco.javacpp.opencv_stitching")
public class opencv_stitching implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info().javaText("import org.bytedeco.javacpp.annotation.Index;"))
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
               .put(new Info("cv::detail::GraphCutSeamFinderGpu").pointerTypes("GraphCutSeamFinderGpu"))
               .putFirst(new Info("cv::InputArray").skip()./*cast().*/pointerTypes("MatVector"));
    }
}
