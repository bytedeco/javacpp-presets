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
package org.bytedeco.pytorch.vision.transforms;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.vision.transforms.functional.F;
import org.bytedeco.pytorch.vision.utils.ImageTensors;

import java.awt.image.BufferedImage;
import java.util.Objects;
import java.util.Random;

/**
 * Common torchvision.transforms classes.
 * Image ops accept {@link BufferedImage} / {@link ImageData} / {@link Tensor}.
 */
public final class Transforms {
    private Transforms() {}

    public static final class Resize implements Transform<Object, BufferedImage> {
        private final int height;
        private final int width;

        public Resize(int size) {
            this(size, size);
        }

        public Resize(int height, int width) {
            this.height = height;
            this.width = width;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.resize(input, height, width);
        }
    }

    public static final class CenterCrop implements Transform<Object, BufferedImage> {
        private final int height;
        private final int width;

        public CenterCrop(int size) {
            this(size, size);
        }

        public CenterCrop(int height, int width) {
            this.height = height;
            this.width = width;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.centerCrop(input, height, width);
        }
    }

    public static final class RandomCrop implements Transform<Object, BufferedImage> {
        private final int height;
        private final int width;
        private final int padding;
        private final boolean padIfNeeded;
        private final int fill;
        private final Random random;

        public RandomCrop(int size) {
            this(size, size, 0, false, 0, new Random());
        }

        public RandomCrop(int height, int width) {
            this(height, width, 0, false, 0, new Random());
        }

        public RandomCrop(int height, int width, Random random) {
            this(height, width, 0, false, 0, random);
        }

        public RandomCrop(int height, int width, int padding, boolean padIfNeeded, int fill, Random random) {
            this.height = height;
            this.width = width;
            this.padding = Math.max(0, padding);
            this.padIfNeeded = padIfNeeded;
            this.fill = fill;
            this.random = random != null ? random : new Random();
        }

        @Override
        public BufferedImage forward(Object input) {
            BufferedImage src = F.asBufferedImage(input);
            if (padding > 0) {
                src = F.pad(src, padding, padding, padding, padding, fill);
            }
            int w = src.getWidth();
            int h = src.getHeight();
            if (padIfNeeded) {
                int padH = Math.max(0, height - h);
                int padW = Math.max(0, width - w);
                if (padH > 0 || padW > 0) {
                    src = F.pad(src, padH / 2, padW / 2, padH - padH / 2, padW - padW / 2, fill);
                    w = src.getWidth();
                    h = src.getHeight();
                }
            }
            int top = h > height ? random.nextInt(h - height + 1) : 0;
            int left = w > width ? random.nextInt(w - width + 1) : 0;
            return F.crop(src, top, left, Math.min(height, h), Math.min(width, w));
        }
    }

    public static final class RandomResizedCrop implements Transform<Object, BufferedImage> {
        private final int height;
        private final int width;
        private final double minScale;
        private final double maxScale;
        private final double minRatio;
        private final double maxRatio;
        private final Random random;

        public RandomResizedCrop(int size) {
            this(size, size, 0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, new Random());
        }

        public RandomResizedCrop(int size, double minScale, double maxScale, Random random) {
            this(size, size, minScale, maxScale, 3.0 / 4.0, 4.0 / 3.0, random);
        }

        /** Full torchvision-style: size, scale=[min,max], ratio=[min,max]. */
        public RandomResizedCrop(int size, double[] scale, double[] ratio) {
            this(size, size,
                    scale != null && scale.length > 0 ? scale[0] : 0.08,
                    scale != null && scale.length > 1 ? scale[1] : 1.0,
                    ratio != null && ratio.length > 0 ? ratio[0] : 3.0 / 4.0,
                    ratio != null && ratio.length > 1 ? ratio[1] : 4.0 / 3.0,
                    new Random());
        }

        public RandomResizedCrop(int height, int width, double minScale, double maxScale,
                                 double minRatio, double maxRatio, Random random) {
            this.height = height;
            this.width = width;
            this.minScale = minScale;
            this.maxScale = maxScale;
            this.minRatio = minRatio;
            this.maxRatio = maxRatio;
            this.random = random != null ? random : new Random();
        }

        @Override
        public BufferedImage forward(Object input) {
            BufferedImage src = F.asBufferedImage(input);
            int w = src.getWidth();
            int h = src.getHeight();
            int area = h * w;
            for (int attempt = 0; attempt < 10; attempt++) {
                double target = area * (minScale + (maxScale - minScale) * random.nextDouble());
                double logMin = Math.log(minRatio);
                double logMax = Math.log(maxRatio);
                double aspect = Math.exp(logMin + (logMax - logMin) * random.nextDouble());
                int cropW = Math.max(1, (int) Math.round(Math.sqrt(target * aspect)));
                int cropH = Math.max(1, (int) Math.round(Math.sqrt(target / aspect)));
                if (cropW <= w && cropH <= h) {
                    int top = h > cropH ? random.nextInt(h - cropH + 1) : 0;
                    int left = w > cropW ? random.nextInt(w - cropW + 1) : 0;
                    BufferedImage cropped = F.crop(src, top, left, cropH, cropW);
                    return F.resize(cropped, height, width);
                }
            }
            // fallback: center crop to match aspect then resize
            return F.resize(F.centerCrop(src, Math.min(h, w), Math.min(h, w)), height, width);
        }
    }

    public static final class RandomHorizontalFlip implements Transform<Object, Object> {
        private final double p;
        private final Random random;

        public RandomHorizontalFlip() {
            this(0.5);
        }

        public RandomHorizontalFlip(double p) {
            this(p, new Random());
        }

        public RandomHorizontalFlip(double p, Random random) {
            this.p = p;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() < p) {
                return F.hflip(input);
            }
            return input instanceof Tensor ? input : F.asBufferedImage(input);
        }
    }

    public static final class RandomVerticalFlip implements Transform<Object, Object> {
        private final double p;
        private final Random random;

        public RandomVerticalFlip() {
            this(0.5);
        }

        public RandomVerticalFlip(double p) {
            this(p, new Random());
        }

        public RandomVerticalFlip(double p, Random random) {
            this.p = p;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() < p) {
                return F.vflip(input);
            }
            return input instanceof Tensor ? input : F.asBufferedImage(input);
        }
    }

    public static final class RandomRotation implements Transform<Object, BufferedImage> {
        private final double degrees;
        private final Random random;

        public RandomRotation(double degrees) {
            this(degrees, new Random());
        }

        public RandomRotation(double degrees, Random random) {
            this.degrees = degrees;
            this.random = random != null ? random : new Random();
        }

        @Override
        public BufferedImage forward(Object input) {
            double angle = (random.nextDouble() * 2 - 1) * degrees;
            return F.rotate(input, angle);
        }
    }

    public static final class Pad implements Transform<Object, BufferedImage> {
        private final int top, left, bottom, right, fill;

        public Pad(int padding) {
            this(padding, padding, padding, padding, 0);
        }

        public Pad(int top, int left, int bottom, int right, int fill) {
            this.top = top;
            this.left = left;
            this.bottom = bottom;
            this.right = right;
            this.fill = fill;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.pad(input, top, left, bottom, right, fill);
        }
    }

    public static final class Grayscale implements Transform<Object, BufferedImage> {
        private final int numOutputChannels;

        public Grayscale() {
            this(1);
        }

        public Grayscale(int numOutputChannels) {
            this.numOutputChannels = numOutputChannels;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.toGrayscale(input, numOutputChannels);
        }
    }

    public static final class RandomGrayscale implements Transform<Object, Object> {
        private final double p;
        private final Random random;

        public RandomGrayscale() {
            this(0.1);
        }

        public RandomGrayscale(double p) {
            this(p, new Random());
        }

        public RandomGrayscale(double p, Random random) {
            this.p = p;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() < p) {
                return F.toGrayscale(input, 3);
            }
            return input instanceof Tensor ? input : F.asBufferedImage(input);
        }
    }

    public static final class ColorJitter implements Transform<Object, BufferedImage> {
        private final float brightness, contrast, saturation, hue;
        private final Random random;

        public ColorJitter(float brightness, float contrast, float saturation, float hue) {
            this(brightness, contrast, saturation, hue, new Random());
        }

        public ColorJitter(float brightness, float contrast, float saturation, float hue, Random random) {
            this.brightness = brightness;
            this.contrast = contrast;
            this.saturation = saturation;
            this.hue = hue;
            this.random = random != null ? random : new Random();
        }

        @Override
        public BufferedImage forward(Object input) {
            BufferedImage img = F.asBufferedImage(input);
            if (brightness > 0) {
                float f = 1f + (random.nextFloat() * 2 - 1f) * brightness;
                img = F.adjustBrightness(img, Math.max(0f, f));
            }
            if (contrast > 0) {
                float f = 1f + (random.nextFloat() * 2 - 1f) * contrast;
                img = F.adjustContrast(img, Math.max(0f, f));
            }
            if (saturation > 0) {
                float f = 1f + (random.nextFloat() * 2 - 1f) * saturation;
                img = F.adjustSaturation(img, Math.max(0f, f));
            }
            if (hue > 0) {
                float f = (random.nextFloat() * 2 - 1f) * hue;
                img = F.adjustHue(img, f);
            }
            return img;
        }
    }

    public static final class GaussianBlur implements Transform<Object, BufferedImage> {
        private final int kernelSize;
        private final double sigma;

        public GaussianBlur(int kernelSize) {
            this(kernelSize, 1.0);
        }

        public GaussianBlur(int kernelSize, double sigma) {
            this.kernelSize = kernelSize;
            this.sigma = sigma;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.gaussianBlur(input, kernelSize, sigma);
        }
    }

    public static final class ToTensor implements Transform<Object, Tensor> {
        @Override
        public Tensor forward(Object input) {
            return F.toTensor(input);
        }
    }

    public static final class ToPILImage implements Transform<Object, BufferedImage> {
        @Override
        public BufferedImage forward(Object input) {
            if (input instanceof Tensor t) {
                return ImageTensors.toBufferedImage(t);
            }
            return F.asBufferedImage(input);
        }
    }

    public static final class Normalize implements Transform<Object, Tensor> {
        private final float[] mean;
        private final float[] std;

        public Normalize(float[] mean, float[] std) {
            this.mean = Objects.requireNonNull(mean).clone();
            this.std = Objects.requireNonNull(std).clone();
        }

        public Normalize(double[] mean, double[] std) {
            this.mean = toFloat(mean);
            this.std = toFloat(std);
        }

        private static float[] toFloat(double[] d) {
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return f;
        }

        @Override
        public Tensor forward(Object input) {
            Tensor t = input instanceof Tensor tt ? tt : F.toTensor(input);
            return F.normalize(t, mean, std);
        }
    }

    public static final class ConvertImageDtype implements Transform<Object, Tensor> {
        private final org.bytedeco.pytorch.global.torch.ScalarType dtype;

        public ConvertImageDtype() {
            this(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        }

        public ConvertImageDtype(org.bytedeco.pytorch.global.torch.ScalarType dtype) {
            this.dtype = dtype != null ? dtype : org.bytedeco.pytorch.global.torch.ScalarType.Float;
        }

        @Override
        public Tensor forward(Object input) {
            Tensor t = input instanceof Tensor tt ? tt : F.toTensor(input);
            return t.to(dtype);
        }
    }

    public static final class Lambda<T, R> implements Transform<T, R> {
        private final java.util.function.Function<T, R> fn;

        public Lambda(java.util.function.Function<T, R> fn) {
            this.fn = Objects.requireNonNull(fn);
        }

        @Override
        public R forward(T input) {
            return fn.apply(input);
        }
    }

    public static final class FiveCrop implements Transform<Object, BufferedImage[]> {
        private final int size;

        public FiveCrop(int size) {
            this.size = size;
        }

        @Override
        public BufferedImage[] forward(Object input) {
            BufferedImage src = F.asBufferedImage(input);
            int w = src.getWidth();
            int h = src.getHeight();
            int s = size;
            BufferedImage tl = F.crop(src, 0, 0, s, s);
            BufferedImage tr = F.crop(src, 0, w - s, s, s);
            BufferedImage bl = F.crop(src, h - s, 0, s, s);
            BufferedImage br = F.crop(src, h - s, w - s, s, s);
            BufferedImage center = F.centerCrop(src, s, s);
            return new BufferedImage[]{tl, tr, bl, br, center};
        }
    }

    public static final class TenCrop implements Transform<Object, BufferedImage[]> {
        private final int size;

        public TenCrop(int size) {
            this.size = size;
        }

        @Override
        public BufferedImage[] forward(Object input) {
            FiveCrop five = new FiveCrop(size);
            BufferedImage[] a = five.forward(input);
            BufferedImage flipped = F.hflip(input);
            BufferedImage[] b = five.forward(flipped);
            BufferedImage[] out = new BufferedImage[10];
            System.arraycopy(a, 0, out, 0, 5);
            System.arraycopy(b, 0, out, 5, 5);
            return out;
        }
    }

    /** Identity transform. */
    public static final class Identity implements Transform<Object, Object> {
        @Override
        public Object forward(Object input) {
            return input;
        }
    }

    // -------------------------------------------------------------------------
    // Deterministic geometric (torchvision.transforms)
    // -------------------------------------------------------------------------

    /** Always horizontal flip (HorizontalFlip / RandomHorizontalFlip(p=1)). */
    public static final class HorizontalFlip implements Transform<Object, BufferedImage> {
        @Override
        public BufferedImage forward(Object input) {
            return F.hflip(input);
        }
    }

    /** Always vertical flip. */
    public static final class VerticalFlip implements Transform<Object, BufferedImage> {
        @Override
        public BufferedImage forward(Object input) {
            return F.vflip(input);
        }
    }

    /** Deterministic rotation (counter-clockwise degrees). */
    public static final class Rotate implements Transform<Object, BufferedImage> {
        private final double degrees;

        public Rotate(double degrees) {
            this.degrees = degrees;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.rotate(input, degrees);
        }
    }

    /**
     * Affine transform: rotate + translate + scale + shear.
     * {@code translate} is fraction of image size if values in (0,1], else pixels when |v|≥1.
     */
    public static final class Affine implements Transform<Object, BufferedImage> {
        private final double degrees;
        private final double[] translate;
        private final double scale;
        private final double[] shear;
        private final int fill;

        public Affine(double degrees) {
            this(degrees, null, 1.0, null, 0);
        }

        public Affine(double degrees, double[] translate, double scale, double[] shear, int fill) {
            this.degrees = degrees;
            this.translate = translate == null ? null : translate.clone();
            this.scale = scale <= 0 ? 1.0 : scale;
            this.shear = shear == null ? null : shear.clone();
            this.fill = fill;
        }

        @Override
        public BufferedImage forward(Object input) {
            BufferedImage src = F.asBufferedImage(input);
            double[] txy = resolveTranslate(src.getWidth(), src.getHeight(), translate);
            return F.affine(src, degrees, txy, scale, shear, fill);
        }
    }

    /** Deterministic perspective given explicit start/end corner points. */
    public static final class Perspective implements Transform<Object, BufferedImage> {
        private final double[][] startpoints;
        private final double[][] endpoints;
        private final int fill;

        public Perspective(double[][] startpoints, double[][] endpoints) {
            this(startpoints, endpoints, 0);
        }

        public Perspective(double[][] startpoints, double[][] endpoints, int fill) {
            this.startpoints = copyPoints(startpoints);
            this.endpoints = copyPoints(endpoints);
            this.fill = fill;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.perspective(input, startpoints, endpoints, fill);
        }
    }

    // -------------------------------------------------------------------------
    // Random geometric
    // -------------------------------------------------------------------------

    /**
     * Random affine. {@code degrees} is half-range (±degrees) or {min,max}.
     * {@code translate} max fractions of W/H; {@code scale} {min,max}; {@code shear} half-range or pair.
     */
    public static final class RandomAffine implements Transform<Object, BufferedImage> {
        private final double degMin, degMax;
        private final double[] translateMax; // fractions
        private final double scaleMin, scaleMax;
        private final double shearXMin, shearXMax, shearYMin, shearYMax;
        private final int fill;
        private final Random random;

        public RandomAffine(double degrees) {
            this(degrees, null, null, null, 0, new Random());
        }

        public RandomAffine(double degrees, double[] translate, double[] scale, double[] shear, int fill) {
            this(degrees, translate, scale, shear, fill, new Random());
        }

        public RandomAffine(double degrees, double[] translate, double[] scale, double[] shear,
                            int fill, Random random) {
            this.degMin = -Math.abs(degrees);
            this.degMax = Math.abs(degrees);
            this.translateMax = translate == null ? null : translate.clone();
            if (scale != null && scale.length >= 2) {
                this.scaleMin = scale[0];
                this.scaleMax = scale[1];
            } else {
                this.scaleMin = 1.0;
                this.scaleMax = 1.0;
            }
            if (shear == null || shear.length == 0) {
                this.shearXMin = this.shearXMax = this.shearYMin = this.shearYMax = 0;
            } else if (shear.length == 1) {
                this.shearXMin = -Math.abs(shear[0]);
                this.shearXMax = Math.abs(shear[0]);
                this.shearYMin = this.shearYMax = 0;
            } else if (shear.length == 2) {
                this.shearXMin = shear[0];
                this.shearXMax = shear[1];
                this.shearYMin = this.shearYMax = 0;
            } else {
                this.shearXMin = shear[0];
                this.shearXMax = shear[1];
                this.shearYMin = shear[2];
                this.shearYMax = shear.length > 3 ? shear[3] : shear[2];
            }
            this.fill = fill;
            this.random = random != null ? random : new Random();
        }

        @Override
        public BufferedImage forward(Object input) {
            BufferedImage src = F.asBufferedImage(input);
            double angle = degMin + (degMax - degMin) * random.nextDouble();
            double[] txy = new double[]{0, 0};
            if (translateMax != null) {
                double maxTx = translateMax.length > 0 ? translateMax[0] * src.getWidth() : 0;
                double maxTy = translateMax.length > 1 ? translateMax[1] * src.getHeight() : maxTx;
                txy[0] = (random.nextDouble() * 2 - 1) * maxTx;
                txy[1] = (random.nextDouble() * 2 - 1) * maxTy;
            }
            double sc = scaleMin + (scaleMax - scaleMin) * random.nextDouble();
            double[] sh = new double[]{
                    shearXMin + (shearXMax - shearXMin) * random.nextDouble(),
                    shearYMin + (shearYMax - shearYMin) * random.nextDouble()
            };
            return F.affine(src, angle, txy, sc, sh, fill);
        }
    }

    /**
     * Random perspective distortion.
     *
     * @param distortionScale in [0,1], fraction of width/height used to jitter corners
     */
    public static final class RandomPerspective implements Transform<Object, Object> {
        private final double distortionScale;
        private final double p;
        private final int fill;
        private final Random random;

        public RandomPerspective() {
            this(0.5, 0.5, 0, new Random());
        }

        public RandomPerspective(double distortionScale, double p) {
            this(distortionScale, p, 0, new Random());
        }

        public RandomPerspective(double distortionScale, double p, int fill, Random random) {
            this.distortionScale = Math.max(0, Math.min(1, distortionScale));
            this.p = p;
            this.fill = fill;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() >= p) {
                return input instanceof Tensor ? input : F.asBufferedImage(input);
            }
            BufferedImage src = F.asBufferedImage(input);
            int w = src.getWidth();
            int h = src.getHeight();
            double halfW = distortionScale * w / 2.0;
            double halfH = distortionScale * h / 2.0;
            // startpoints: image corners tl,tr,br,bl
            double[][] start = {
                    {0, 0}, {w - 1, 0}, {w - 1, h - 1}, {0, h - 1}
            };
            double[][] end = {
                    {jitter(0, halfW), jitter(0, halfH)},
                    {jitter(w - 1, halfW), jitter(0, halfH)},
                    {jitter(w - 1, halfW), jitter(h - 1, halfH)},
                    {jitter(0, halfW), jitter(h - 1, halfH)}
            };
            return F.perspective(src, start, end, fill);
        }

        private double jitter(double base, double halfRange) {
            return base + (random.nextDouble() * 2 - 1) * halfRange;
        }
    }

    /**
     * Random erasing / Cutout (typically after ToTensor). Operates on CHW float tensor or image.
     */
    public static final class RandomErasing implements Transform<Object, Object> {
        private final double p;
        private final double scaleMin, scaleMax;
        private final double ratioMin, ratioMax;
        private final float[] value;
        private final boolean inplace;
        private final Random random;

        public RandomErasing() {
            this(0.5, 0.02, 0.33, 0.3, 3.3, new float[]{0f}, false, new Random());
        }

        public RandomErasing(double p) {
            this(p, 0.02, 0.33, 0.3, 3.3, new float[]{0f}, false, new Random());
        }

        public RandomErasing(double p, double scaleMin, double scaleMax,
                             double ratioMin, double ratioMax, float[] value,
                             boolean inplace, Random random) {
            this.p = p;
            this.scaleMin = scaleMin;
            this.scaleMax = scaleMax;
            this.ratioMin = ratioMin;
            this.ratioMax = ratioMax;
            this.value = value == null ? new float[]{0f} : value.clone();
            this.inplace = inplace;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() >= p) {
                return input;
            }
            Tensor t = input instanceof Tensor tt ? tt : F.toTensor(input);
            long[] sizes = ImageTensors.sizes(t);
            if (sizes.length < 3) {
                return t;
            }
            int H = (int) sizes[sizes.length == 3 ? 1 : 2];
            int W = (int) sizes[sizes.length == 3 ? 2 : 3];
            int area = H * W;
            for (int attempt = 0; attempt < 10; attempt++) {
                double target = area * (scaleMin + (scaleMax - scaleMin) * random.nextDouble());
                double aspect = Math.exp(Math.log(ratioMin) + (Math.log(ratioMax) - Math.log(ratioMin)) * random.nextDouble());
                int h = Math.max(1, (int) Math.round(Math.sqrt(target * aspect)));
                int w = Math.max(1, (int) Math.round(Math.sqrt(target / aspect)));
                if (h < H && w < W) {
                    int i = random.nextInt(H - h + 1);
                    int j = random.nextInt(W - w + 1);
                    return F.erase(t, i, j, h, w, value, inplace);
                }
            }
            return t;
        }
    }

    // -------------------------------------------------------------------------
    // Random photometric
    // -------------------------------------------------------------------------

    public static final class RandomSolarize implements Transform<Object, Object> {
        private final double threshold;
        private final double p;
        private final Random random;

        public RandomSolarize(double threshold) {
            this(threshold, 0.5, new Random());
        }

        public RandomSolarize(double threshold, double p) {
            this(threshold, p, new Random());
        }

        public RandomSolarize(double threshold, double p, Random random) {
            this.threshold = threshold;
            this.p = p;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() < p) {
                return F.solarize(input, threshold);
            }
            return input instanceof Tensor ? input : F.asBufferedImage(input);
        }
    }

    public static final class RandomAutocontrast implements Transform<Object, Object> {
        private final double p;
        private final Random random;

        public RandomAutocontrast() {
            this(0.5);
        }

        public RandomAutocontrast(double p) {
            this(p, new Random());
        }

        public RandomAutocontrast(double p, Random random) {
            this.p = p;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() < p) {
                return F.autocontrast(input);
            }
            return input instanceof Tensor ? input : F.asBufferedImage(input);
        }
    }

    public static final class RandomEqualize implements Transform<Object, Object> {
        private final double p;
        private final Random random;

        public RandomEqualize() {
            this(0.5);
        }

        public RandomEqualize(double p) {
            this(p, new Random());
        }

        public RandomEqualize(double p, Random random) {
            this.p = p;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() < p) {
                return F.equalize(input);
            }
            return input instanceof Tensor ? input : F.asBufferedImage(input);
        }
    }

    public static final class RandomAdjustSharpness implements Transform<Object, Object> {
        private final float sharpnessFactor;
        private final double p;
        private final Random random;

        public RandomAdjustSharpness(float sharpnessFactor) {
            this(sharpnessFactor, 0.5, new Random());
        }

        public RandomAdjustSharpness(float sharpnessFactor, double p) {
            this(sharpnessFactor, p, new Random());
        }

        public RandomAdjustSharpness(float sharpnessFactor, double p, Random random) {
            this.sharpnessFactor = sharpnessFactor;
            this.p = p;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() < p) {
                return F.adjustSharpness(input, sharpnessFactor);
            }
            return input instanceof Tensor ? input : F.asBufferedImage(input);
        }
    }

    /** Deterministic sharpness adjust. */
    public static final class AdjustSharpness implements Transform<Object, BufferedImage> {
        private final float factor;

        public AdjustSharpness(float factor) {
            this.factor = factor;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.adjustSharpness(input, factor);
        }
    }

    /** Deterministic solarize. */
    public static final class Solarize implements Transform<Object, BufferedImage> {
        private final double threshold;

        public Solarize(double threshold) {
            this.threshold = threshold;
        }

        @Override
        public BufferedImage forward(Object input) {
            return F.solarize(input, threshold);
        }
    }

    /** Deterministic autocontrast. */
    public static final class Autocontrast implements Transform<Object, BufferedImage> {
        @Override
        public BufferedImage forward(Object input) {
            return F.autocontrast(input);
        }
    }

    /** Deterministic equalize. */
    public static final class Equalize implements Transform<Object, BufferedImage> {
        @Override
        public BufferedImage forward(Object input) {
            return F.equalize(input);
        }
    }

    // ---- helpers ----

    private static double[] resolveTranslate(int w, int h, double[] translate) {
        if (translate == null) {
            return new double[]{0, 0};
        }
        double tx = translate.length > 0 ? translate[0] : 0;
        double ty = translate.length > 1 ? translate[1] : 0;
        // treat values in (-1,1) exclusive of ±1 as fractions
        if (Math.abs(tx) > 0 && Math.abs(tx) < 1.0) {
            tx *= w;
        }
        if (Math.abs(ty) > 0 && Math.abs(ty) < 1.0) {
            ty *= h;
        }
        return new double[]{tx, ty};
    }

    private static double[][] copyPoints(double[][] pts) {
        Objects.requireNonNull(pts, "points");
        double[][] out = new double[pts.length][];
        for (int i = 0; i < pts.length; i++) {
            out[i] = pts[i] == null ? null : pts[i].clone();
        }
        return out;
    }
}
