/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
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
package org.bytedeco.pytorch.vision.models;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.AdaptiveAvgPool2dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm2dImpl;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.MaxPool2dImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.nn.options.Conv2dOptions;
import org.bytedeco.pytorch.LongOptional;

import static org.bytedeco.pytorch.global.torch.relu;
import static org.bytedeco.pytorch.global.torch.flatten;

/**
 * Helpers to build torchvision-like conv modules with JavaCPP ExpandingArray options.
 */
public final class ConvHelpers {
    private ConvHelpers() {}

    /** ExpandingArray&lt;2&gt; from a single value (k → {k,k}). */
    public static LongPointer k2(long k) {
        return new LongPointer(new long[]{k, k});
    }

    public static LongPointer k2(long h, long w) {
        return new LongPointer(new long[]{h, w});
    }

    public static Conv2dImpl conv3x3(long in, long out, long stride, long padding) {
        Conv2dOptions opt = new Conv2dOptions(in, out, k2(3));
        opt.stride(k2(stride));
        opt.padding().put(k2(padding));
        return new Conv2dImpl(opt);
    }

    public static Conv2dImpl conv2d(long in, long out, long kernel, long stride, long padding, boolean bias) {
        Conv2dOptions opt = new Conv2dOptions(in, out, k2(kernel));
        opt.stride(k2(stride));
        opt.padding().put(k2(padding));
        opt.bias(bias);
        return new Conv2dImpl(opt);
    }

    public static Conv2dImpl conv2d(long in, long out, long kernel) {
        return new Conv2dImpl(new Conv2dOptions(in, out, k2(kernel)));
    }

    public static BatchNorm2dImpl bn2d(long features) {
        return new BatchNorm2dImpl(new BatchNormOptions(features));
    }

    public static MaxPool2dImpl maxPool2d(long kernel) {
        return new MaxPool2dImpl(k2(kernel));
    }

    public static AdaptiveAvgPool2dImpl adaptiveAvgPool2d(long size) {
        // ExpandingArrayWithOptionalElem — single LongOptional often works as output size
        LongOptional lo = new LongOptional(size);
        return new AdaptiveAvgPool2dImpl(lo);
    }

    public static ReLUImpl reluMod() {
        return new ReLUImpl();
    }

    public static LinearImpl linear(long in, long out) {
        return new LinearImpl(in, out);
    }
}
