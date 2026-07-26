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
package org.bytedeco.pytorch.utils.vision.models;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.BatchNorm2dImpl;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.MaxPool2dImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.nn.options.Conv2dOptions;

import static org.bytedeco.pytorch.global.torch.adaptive_avg_pool2d;
import static org.bytedeco.pytorch.global.torch.flatten;
import static org.bytedeco.pytorch.global.torch.max_pool2d;
import static org.bytedeco.pytorch.global.torch.relu;

/**
 * torchvision.models-style factories. Architectures are real {@link Module}s;
 * pretrained weights are optional (random init by default).
 */
public final class Models {
    private Models() {}

    /** ExpandingArray&lt;2&gt; single-value ctor used by module options. */
    private static LongPointer exp2ptr(long v) {
        return new LongPointer(new long[]{v, v});
    }

    /** Functional ops take long[] / long... not LongPointer. */
    private static long[] k2(long v) {
        return new long[]{v, v};
    }

    private static Conv2dImpl conv(long in, long out, long k, long stride, boolean bias) {
        Conv2dOptions opt = new Conv2dOptions(in, out, exp2ptr(k));
        opt.stride(exp2ptr(stride));
        opt.bias(bias);
        return new Conv2dImpl(opt);
    }

    private static BatchNorm2dImpl bn(long f) {
        return new BatchNorm2dImpl(new BatchNormOptions(f));
    }

    // -------------------------------------------------------------------------
    // Simple / training-friendly
    // -------------------------------------------------------------------------

    /** Tiny MLP classifier on flattened CHW images — always safe for mini-train benches. */
    public static final class SimpleClassifier extends Module {
        final LinearImpl fc1, fc2, fc3;
        final long inFeatures;

        public SimpleClassifier(long inFeatures, long hidden, long numClasses) {
            super("SimpleClassifier");
            this.inFeatures = inFeatures;
            fc1 = register_module("fc1", new LinearImpl(inFeatures, hidden));
            fc2 = register_module("fc2", new LinearImpl(hidden, hidden / 2));
            fc3 = register_module("fc3", new LinearImpl(hidden / 2, numClasses));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor flat = x.reshape(x.size(0), -1);
            Tensor h = relu(fc1.forward(flat));
            h = relu(fc2.forward(h));
            return fc3.forward(h);
        }
    }

    /** Small conv net (LeNet-ish) for CHW inputs. */
    public static final class SimpleCNN extends Module {
        final Conv2dImpl conv1, conv2;
        final LinearImpl fc1, fc2;
        final long numClasses;

        public SimpleCNN(long inChannels, long numClasses) {
            super("SimpleCNN");
            this.numClasses = numClasses;
            conv1 = register_module("conv1", conv(inChannels, 16, 3, 1, true));
            conv2 = register_module("conv2", conv(16, 32, 3, 1, true));
            // After two 2x2 pools on 32x32 → 8x8; use adaptive path via flatten of pooled
            fc1 = register_module("fc1", new LinearImpl(32 * 8 * 8, 128));
            fc2 = register_module("fc2", new LinearImpl(128, numClasses));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = relu(conv1.forward(x));
            h = max_pool2d(h, k2(2));
            h = relu(conv2.forward(h));
            h = max_pool2d(h, k2(2));
            // adaptive to 8x8 if needed
            if (h.size(2) != 8 || h.size(3) != 8) {
                h = adaptive_avg_pool2d(h, k2(8));
            }
            h = flatten(h, 1L, -1L);
            h = relu(fc1.forward(h));
            return fc2.forward(h);
        }
    }

    // -------------------------------------------------------------------------
    // ResNet
    // -------------------------------------------------------------------------

    public static final class BasicBlock extends Module {
        final Conv2dImpl conv1, conv2;
        final BatchNorm2dImpl bn1, bn2;
        final Conv2dImpl downsampleConv;
        final BatchNorm2dImpl downsampleBn;
        final long stride;

        public BasicBlock(long inPlanes, long planes, long stride) {
            super("BasicBlock");
            this.stride = stride;
            conv1 = register_module("conv1", conv(inPlanes, planes, 3, stride, false));
            bn1 = register_module("bn1", bn(planes));
            conv2 = register_module("conv2", conv(planes, planes, 3, 1, false));
            bn2 = register_module("bn2", bn(planes));
            if (stride != 1 || inPlanes != planes) {
                downsampleConv = register_module("downsample_conv", conv(inPlanes, planes, 1, stride, false));
                downsampleBn = register_module("downsample_bn", bn(planes));
            } else {
                downsampleConv = null;
                downsampleBn = null;
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor identity = x;
            Tensor out = relu(bn1.forward(conv1.forward(x)));
            out = bn2.forward(conv2.forward(out));
            if (downsampleConv != null) {
                identity = downsampleBn.forward(downsampleConv.forward(x));
            }
            return relu(out.add(identity));
        }
    }

    public static final class ResNet extends Module {
        final Conv2dImpl conv1;
        final BatchNorm2dImpl bn1;
        final BasicBlock[] layer1, layer2, layer3, layer4;
        final LinearImpl fc;
        long inPlanes = 64;

        public ResNet(int[] layers, long numClasses) {
            super("ResNet");
            conv1 = register_module("conv1", conv(3, 64, 7, 2, false));
            bn1 = register_module("bn1", bn(64));
            layer1 = makeLayer("layer1", 64, layers[0], 1);
            layer2 = makeLayer("layer2", 128, layers[1], 2);
            layer3 = makeLayer("layer3", 256, layers[2], 2);
            layer4 = makeLayer("layer4", 512, layers[3], 2);
            fc = register_module("fc", new LinearImpl(512, numClasses));
        }

        private BasicBlock[] makeLayer(String name, long planes, int blocks, long stride) {
            BasicBlock[] out = new BasicBlock[blocks];
            out[0] = register_module(name + "_0", new BasicBlock(inPlanes, planes, stride));
            inPlanes = planes;
            for (int i = 1; i < blocks; i++) {
                out[i] = register_module(name + "_" + i, new BasicBlock(inPlanes, planes, 1));
            }
            return out;
        }

        private static Tensor runBlocks(BasicBlock[] blocks, Tensor x) {
            Tensor h = x;
            for (BasicBlock b : blocks) {
                h = b.forward(h);
            }
            return h;
        }

        /** Backbone features before the classification head — shape {@code [N, 512]}. */
        public Tensor features(Tensor x) {
            Tensor h = relu(bn1.forward(conv1.forward(x)));
            h = max_pool2d(h, k2(3)); // approx; kernel 3
            h = runBlocks(layer1, h);
            h = runBlocks(layer2, h);
            h = runBlocks(layer3, h);
            h = runBlocks(layer4, h);
            h = adaptive_avg_pool2d(h, k2(1));
            return flatten(h, 1L, -1L);
        }

        /** Feature dimension produced by {@link #features(Tensor)}. */
        public long featureDim() { return 512L; }

        @Override
        public Tensor forward(Tensor x) {
            return fc.forward(features(x));
        }
    }

    public static ResNet resnet18(long numClasses) {
        return new ResNet(new int[]{2, 2, 2, 2}, numClasses);
    }

    public static ResNet resnet34(long numClasses) {
        return new ResNet(new int[]{3, 4, 6, 3}, numClasses);
    }

    // -------------------------------------------------------------------------
    // AlexNet (simplified)
    // -------------------------------------------------------------------------

    public static final class AlexNet extends Module {
        final Conv2dImpl c1, c2, c3, c4, c5;
        final LinearImpl fc1, fc2, fc3;

        public AlexNet(long numClasses) {
            super("AlexNet");
            c1 = register_module("features_0", conv(3, 64, 11, 4, true));
            c2 = register_module("features_3", conv(64, 192, 5, 1, true));
            c3 = register_module("features_6", conv(192, 384, 3, 1, true));
            c4 = register_module("features_8", conv(384, 256, 3, 1, true));
            c5 = register_module("features_10", conv(256, 256, 3, 1, true));
            fc1 = register_module("classifier_1", new LinearImpl(256 * 6 * 6, 4096));
            fc2 = register_module("classifier_4", new LinearImpl(4096, 4096));
            fc3 = register_module("classifier_6", new LinearImpl(4096, numClasses));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = max_pool2d(relu(c1.forward(x)), k2(3));
            h = max_pool2d(relu(c2.forward(h)), k2(3));
            h = relu(c3.forward(h));
            h = relu(c4.forward(h));
            h = max_pool2d(relu(c5.forward(h)), k2(3));
            h = adaptive_avg_pool2d(h, k2(6));
            h = flatten(h, 1L, -1L);
            h = relu(fc1.forward(h));
            h = relu(fc2.forward(h));
            return fc3.forward(h);
        }
    }

    public static AlexNet alexnet(long numClasses) {
        return new AlexNet(numClasses);
    }

    // -------------------------------------------------------------------------
    // MobileNetV2-lite / VGG-lite (API surface)
    // -------------------------------------------------------------------------

    public static final class MobileNetV2 extends Module {
        final Conv2dImpl stem;
        final BatchNorm2dImpl stemBn;
        final Conv2dImpl mid;
        final BatchNorm2dImpl midBn;
        final LinearImpl classifier;

        public MobileNetV2(long numClasses) {
            super("MobileNetV2");
            stem = register_module("stem", conv(3, 32, 3, 2, false));
            stemBn = register_module("stem_bn", bn(32));
            mid = register_module("mid", conv(32, 128, 3, 2, false));
            midBn = register_module("mid_bn", bn(128));
            classifier = register_module("classifier", new LinearImpl(128, numClasses));
        }

        /** Backbone features before classifier — shape {@code [N, 128]}. */
        public Tensor features(Tensor x) {
            Tensor h = relu(stemBn.forward(stem.forward(x)));
            h = relu(midBn.forward(mid.forward(h)));
            h = adaptive_avg_pool2d(h, k2(1));
            return flatten(h, 1L, -1L);
        }

        public long featureDim() { return 128L; }

        @Override
        public Tensor forward(Tensor x) {
            return classifier.forward(features(x));
        }
    }

    public static MobileNetV2 mobilenet_v2(long numClasses) {
        return new MobileNetV2(numClasses);
    }

    public static final class VGG extends Module {
        final Conv2dImpl[] convs;
        final LinearImpl fc1, fc2, fc3;
        final int[] cfg;

        public VGG(int[] cfg, long numClasses) {
            super("VGG");
            this.cfg = cfg;
            convs = new Conv2dImpl[cfg.length];
            long in = 3;
            int idx = 0;
            for (int v : cfg) {
                if (v > 0) {
                    Conv2dImpl c = register_module("conv" + idx, conv(in, v, 3, 1, true));
                    convs[idx] = c;
                    in = v;
                }
                idx++;
            }
            fc1 = register_module("fc1", new LinearImpl(in * 7 * 7, 4096));
            fc2 = register_module("fc2", new LinearImpl(4096, 4096));
            fc3 = register_module("fc3", new LinearImpl(4096, numClasses));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = x;
            for (int i = 0; i < cfg.length; i++) {
                if (cfg[i] < 0) {
                    h = max_pool2d(h, k2(2));
                } else if (convs[i] != null) {
                    h = relu(convs[i].forward(h));
                }
            }
            h = adaptive_avg_pool2d(h, k2(7));
            h = flatten(h, 1L, -1L);
            h = relu(fc1.forward(h));
            h = relu(fc2.forward(h));
            return fc3.forward(h);
        }
    }

    /** VGG11 cfg: channels, -1 means MaxPool. */
    public static VGG vgg11(long numClasses) {
        return new VGG(new int[]{64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}, numClasses);
    }

    public static VGG vgg16(long numClasses) {
        return new VGG(new int[]{
                64, 64, -1,
                128, 128, -1,
                256, 256, 256, -1,
                512, 512, 512, -1,
                512, 512, 512, -1
        }, numClasses);
    }

    /** Factory by name (torchvision.get_model style, random weights). */
    public static Module get_model(String name, long numClasses) {
        String n = name == null ? "" : name.toLowerCase().replace('-', '_');
        return switch (n) {
            case "resnet18" -> resnet18(numClasses);
            case "resnet34" -> resnet34(numClasses);
            case "alexnet" -> alexnet(numClasses);
            case "mobilenet_v2", "mobilenetv2" -> mobilenet_v2(numClasses);
            case "vgg11" -> vgg11(numClasses);
            case "vgg16" -> vgg16(numClasses);
            case "simple_cnn", "simplecnn" -> new SimpleCNN(3, numClasses);
            default -> new SimpleClassifier(3 * 224 * 224, 256, numClasses);
        };
    }

    public static Module getModel(String name, long numClasses) {
        return get_model(name, numClasses);
    }
}
