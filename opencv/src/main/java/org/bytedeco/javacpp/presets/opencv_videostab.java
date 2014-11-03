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
@Properties(inherit={opencv_calib3d.class, opencv_features2d.class, opencv_objdetect.class, opencv_photo.class,
        opencv_nonfree.class, opencv_video.class, opencv_ml.class, opencv_legacy.class}, value={
    @Platform(include={"<opencv2/videostab/frame_source.hpp>", "<opencv2/videostab/log.hpp>", "<opencv2/videostab/fast_marching.hpp>",
        "<opencv2/videostab/optical_flow.hpp>", "<opencv2/videostab/global_motion.hpp>", "<opencv2/videostab/motion_stabilizing.hpp>",
        "<opencv2/videostab/inpainting.hpp>", "<opencv2/videostab/deblurring.hpp>", "<opencv2/videostab/stabilizer.hpp>",
        "<opencv2/videostab/videostab.hpp>"}, link="opencv_videostab@.2.4", preload={"opencv_gpu@.2.4", "opencv_ocl@.2.4"}),
    @Platform(value="windows", link="opencv_videostab2410", preload={"opencv_gpu2410", "opencv_ocl2410"}) },
        target="org.bytedeco.javacpp.opencv_videostab")
public class opencv_videostab implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("cv::videostab::DensePyrLkOptFlowEstimatorGpu").annotations("@Platform(not=\"android\")").pointerTypes("DensePyrLkOptFlowEstimatorGpu"));
    }
}
