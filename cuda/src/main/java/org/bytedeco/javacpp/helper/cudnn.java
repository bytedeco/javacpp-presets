/*
 * Copyright (C) 2017 Samuel Audet
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

package org.bytedeco.javacpp.helper;

import org.bytedeco.javacpp.cudnn.cudnnConvolutionStruct;

import static org.bytedeco.javacpp.cudnn.cudnnSetConvolution2dDescriptor_v4;

/**
 *
 * @author Samuel Audet
 */
public class cudnn extends org.bytedeco.javacpp.presets.cudnn {

    public static int cudnnSetConvolution2dDescriptor(cudnnConvolutionStruct convDesc,
            int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w, int mode) {
        return cudnnSetConvolution2dDescriptor_v4(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode);
    }

}
