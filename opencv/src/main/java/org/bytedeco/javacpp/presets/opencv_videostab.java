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
@Properties(inherit = {opencv_calib3d.class, opencv_features2d.class, opencv_photo.class, opencv_video.class,
                       opencv_videoio.class, opencv_ml.class}, value = {
    @Platform(include = {
        "<opencv2/videostab/frame_source.hpp>", "<opencv2/videostab/log.hpp>", "<opencv2/videostab/fast_marching.hpp>",
        "<opencv2/videostab/optical_flow.hpp>", "<opencv2/videostab/motion_core.hpp>", "<opencv2/videostab/outlier_rejection.hpp>",
        "<opencv2/videostab/global_motion.hpp>", "<opencv2/videostab/motion_stabilizing.hpp>", "<opencv2/videostab/inpainting.hpp>",
        "<opencv2/videostab/deblurring.hpp>", "<opencv2/videostab/wobble_suppression.hpp>", "<opencv2/videostab/stabilizer.hpp>",
        "<opencv2/videostab/ring_buffer.hpp>", "<opencv2/videostab.hpp>"}, link = "opencv_videostab@.3.0", preload = "opencv_cuda@.3.0"),
    @Platform(value = "windows", link = "opencv_videostab300", preload = "opencv_cuda300")},
        target = "org.bytedeco.javacpp.opencv_videostab")
public class opencv_videostab implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv::videostab::DensePyrLkOptFlowEstimatorGpu").pointerTypes("DensePyrLkOptFlowEstimatorGpu"));
    }
}
