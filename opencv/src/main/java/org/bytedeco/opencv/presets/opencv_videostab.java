/*
 * Copyright (C) 2014-2019 Samuel Audet
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

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.ByRef;
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
    inherit = {opencv_objdetect.class, opencv_optflow.class, opencv_photo.class},
    value = {
        @Platform(include = {
            "<opencv2/videostab/frame_source.hpp>", "<opencv2/videostab/log.hpp>", "<opencv2/videostab/fast_marching.hpp>",
            "<opencv2/videostab/optical_flow.hpp>", "<opencv2/videostab/motion_core.hpp>", "<opencv2/videostab/outlier_rejection.hpp>",
            "<opencv2/videostab/global_motion.hpp>", "<opencv2/videostab/motion_stabilizing.hpp>", "<opencv2/videostab/inpainting.hpp>",
            "<opencv2/videostab/deblurring.hpp>", "<opencv2/videostab/wobble_suppression.hpp>", "<opencv2/videostab/stabilizer.hpp>",
            "<opencv2/videostab/ring_buffer.hpp>", "<opencv2/videostab.hpp>"}, link = "opencv_videostab@.405",
            preload = {"opencv_cuda@.405", "opencv_cudaarithm@.405", "opencv_cudafilters@.405",
                "opencv_cudaimgproc@.405", "opencv_cudafeatures2d@.405", "opencv_cudalegacy@.405",
                "opencv_cudaoptflow@.405", "opencv_cudawarping@.405"}),
        @Platform(value = "ios", preload = "libopencv_videostab"),
        @Platform(value = "windows", link = "opencv_videostab455",
            preload = {"opencv_cuda455", "opencv_cudaarithm455", "opencv_cudafilters455",
                "opencv_cudaimgproc455", "opencv_cudafeatures2d455", "opencv_cudalegacy455",
                "opencv_cudaoptflow455", "opencv_cudawarping455"})},
    target = "org.bytedeco.opencv.opencv_videostab",
    global = "org.bytedeco.opencv.global.opencv_videostab"
)
public class opencv_videostab implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("override").annotations()) // pure virtual functions are not mapped unless virtualized, so disable override annotation
               .put(new Info("std::function<void(Mat&)>").pointerTypes("MaskCallback"))
               .put(new Info("cv::videostab::IFrameSource").virtualize());
    }

    public static class MaskCallback extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    MaskCallback(Pointer p) { super(p); }
        protected MaskCallback() { allocate(); }
        private native void allocate();
        public native void call(@ByRef @Cast("cv::Mat*") Pointer image);
    }
}

