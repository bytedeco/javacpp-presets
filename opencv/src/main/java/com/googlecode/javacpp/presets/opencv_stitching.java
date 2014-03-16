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
@Properties(inherit={opencv_calib3d.class, opencv_features2d.class, opencv_objdetect.class, opencv_nonfree.class,
        opencv_photo.class, opencv_ml.class, opencv_legacy.class, opencv_video.class}, value={
    @Platform(include={"<opencv2/stitching/detail/warpers.hpp>", "<opencv2/stitching/detail/matchers.hpp>", "<opencv2/stitching/detail/util.hpp>",
        "<opencv2/stitching/detail/camera.hpp>", "<opencv2/stitching/detail/motion_estimators.hpp>", "<opencv2/stitching/detail/exposure_compensate.hpp>",
        "<opencv2/stitching/detail/seam_finders.hpp>", "<opencv2/stitching/detail/blenders.hpp>", "<opencv2/stitching/detail/autocalib.hpp>",
        "<opencv2/stitching/warpers.hpp>", "<opencv2/stitching/stitcher.hpp>"}, link="opencv_stitching@.2.4"),
    @Platform(value="windows", link="opencv_stitching248") },
        target="com.googlecode.javacpp.opencv_stitching")
public class opencv_stitching implements Parser.InfoMapper {
    public void map(Parser.InfoMap infoMap) {
        infoMap.put(new Parser.Info().javaText("import com.googlecode.javacpp.annotation.Index;"))
               .put(new Parser.Info("std::vector<std::pair<cv::Mat,unsigned char> >").pointerTypes("MatBytePairVector").define(true))
               .put(new Parser.Info("cv::detail::PlaneWarper").pointerTypes("DetailPlaneWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::SphericalWarper").pointerTypes("DetailSphericalWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::CylindricalWarper").pointerTypes("DetailCylindricalWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::FisheyeWarper").pointerTypes("DetailFisheyeWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::StereographicWarper").pointerTypes("DetailStereographicWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::CompressedRectilinearWarper").pointerTypes("DetailCompressedRectilinearWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::CompressedRectilinearPortraitWarper").pointerTypes("DetailCompressedRectilinearPortraitWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::PaniniWarper").pointerTypes("DetailPaniniWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::PaniniPortraitWarper").pointerTypes("DetailPaniniPortraitWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::MercatorWarper").pointerTypes("DetailMercatorWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::TransverseMercatorWarper").pointerTypes("DetailTransverseMercatorWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::PlaneWarperGpu").pointerTypes("DetailPlaneWarperGpu").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::SphericalWarperGpu").pointerTypes("DetailSphericalWarperGpu").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::CylindricalWarperGpu").pointerTypes("DetailCylindricalWarperGpu").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::SphericalPortraitWarper", "cv::detail::CylindricalPortraitWarper", "cv::detail::PlanePortraitWarper").base("RotationWarper"))
               .put(new Parser.Info("cv::detail::WaveCorrectKind").cast(true).valueTypes("int"));
    }
}
