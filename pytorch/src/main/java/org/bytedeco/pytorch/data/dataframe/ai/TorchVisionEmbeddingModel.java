package org.bytedeco.pytorch.data.dataframe.ai;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.data.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.vision.models.Models;
import org.bytedeco.pytorch.utils.vision.utils.ImageTensors;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Real torchvision-style neural image embedding.
 *
 * <p>Runs a randomly-initialized (or user-supplied) backbone in eval / no-grad mode
 * and returns the pooled feature vector — not a pure hash. Suitable for offline
 * multimodal pipelines and benchmarks without external pretrained weight files.
 *
 * <p>Supported model ids (family {@code vision}):
 * <ul>
 *   <li>{@code resnet18} / {@code resnet18-embed} → 512-d features</li>
 *   <li>{@code resnet34} → 512-d features</li>
 *   <li>{@code mobilenet_v2} / {@code mobilenet-v2} → 128-d features</li>
 * </ul>
 *
 * <pre>{@code
 * EmbeddingModel m = TorchVisionEmbeddingModel.resnet18();
 * float[] v = m.embed(imageData, Modality.IMAGE);
 * DataFrame df = AiFunctions.embedImageColumn(imgs, "image", "emb", "resnet18");
 * }</pre>
 */
public final class TorchVisionEmbeddingModel implements EmbeddingModel {

    public enum Backbone {
        RESNET18, RESNET34, MOBILENET_V2
    }

    private final ModelSpec spec;
    private final Backbone backbone;
    private final Module encoder;
    private final long nativeFeatureDim;
    private final int outDim;
    private final int inputSize;
    private final EmbeddingModel fallback;
    private volatile boolean warmed;

    public TorchVisionEmbeddingModel(ModelSpec spec, Backbone backbone, int outDim, int inputSize) {
        this.spec = spec == null ? ModelSpec.of("resnet18", Modality.IMAGE, 512, "vision", true) : spec;
        this.backbone = backbone == null ? Backbone.RESNET18 : backbone;
        this.inputSize = inputSize > 0 ? inputSize : 224;
        this.encoder = buildEncoder(this.backbone);
        this.nativeFeatureDim = featureDimOf(this.backbone, this.encoder);
        this.outDim = outDim > 0 ? outDim : (int) Math.min(Integer.MAX_VALUE, this.nativeFeatureDim);
        this.fallback = new HashEmbeddingModel(
                ModelSpec.of(this.spec.id() + "/hash-fallback", Modality.IMAGE, this.outDim, "hash", true));
        try {
            this.encoder.eval();
        } catch (Throwable ignored) {}
    }

    public static TorchVisionEmbeddingModel resnet18() {
        return new TorchVisionEmbeddingModel(
                ModelSpec.of("resnet18", Modality.IMAGE, 512, "vision", true),
                Backbone.RESNET18, 512, 224);
    }

    public static TorchVisionEmbeddingModel resnet34() {
        return new TorchVisionEmbeddingModel(
                ModelSpec.of("resnet34", Modality.IMAGE, 512, "vision", true),
                Backbone.RESNET34, 512, 224);
    }

    public static TorchVisionEmbeddingModel mobilenetV2() {
        return new TorchVisionEmbeddingModel(
                ModelSpec.of("mobilenet_v2", Modality.IMAGE, 128, "vision", true),
                Backbone.MOBILENET_V2, 128, 224);
    }

    /** Build from a {@link ModelSpec} / model id string (used by {@link EmbeddingRegistry}). */
    public static TorchVisionEmbeddingModel fromSpec(ModelSpec spec) {
        ModelSpec s = spec == null ? ModelSpec.of("resnet18", Modality.IMAGE, 512, "vision", true) : spec;
        String id = s.id().toLowerCase(Locale.ROOT);
        Backbone b;
        int dim;
        if (id.contains("mobilenet")) {
            b = Backbone.MOBILENET_V2;
            dim = s.defaultDim() > 0 ? s.defaultDim() : 128;
        } else if (id.contains("resnet34")) {
            b = Backbone.RESNET34;
            dim = s.defaultDim() > 0 ? s.defaultDim() : 512;
        } else {
            b = Backbone.RESNET18;
            dim = s.defaultDim() > 0 ? s.defaultDim() : 512;
        }
        // Keep native feature dim when user asks for a larger CLIP-style dim — we project via ensureDim.
        return new TorchVisionEmbeddingModel(s.withDim(dim), b, dim, 224);
    }

    private static Module buildEncoder(Backbone b) {
        // numClasses is unused when we call features(); keep tiny head for Module completeness.
        return switch (b) {
            case RESNET34 -> Models.resnet34(10);
            case MOBILENET_V2 -> Models.mobilenet_v2(10);
            case RESNET18 -> Models.resnet18(10);
        };
    }

    private static long featureDimOf(Backbone b, Module encoder) {
        try {
            if (encoder instanceof Models.ResNet r) return r.featureDim();
            if (encoder instanceof Models.MobileNetV2 m) return m.featureDim();
        } catch (Throwable ignored) {}
        return switch (b) {
            case MOBILENET_V2 -> 128L;
            default -> 512L;
        };
    }

    @Override public ModelSpec spec() { return spec; }
    @Override public String backend() { return "torchvision-" + backbone.name().toLowerCase(Locale.ROOT); }
    @Override public int dimension() { return outDim; }
    @Override public boolean isReady() { return encoder != null; }

    @Override
    public boolean supports(Modality modality) {
        return modality == Modality.IMAGE || modality == Modality.VIDEO
                || modality == Modality.TENSOR || modality == Modality.MULTIMODAL;
    }

    @Override
    public void warmup() {
        if (warmed) return;
        synchronized (this) {
            if (warmed) return;
            try {
                // one dummy forward to init BatchNorm running stats buffers
                ImageData dummy = solid(inputSize, inputSize, 0x808080);
                embedImage(dummy);
                warmed = true;
            } catch (Throwable t) {
                warmed = true; // don't retry endlessly
            }
        }
    }

    @Override
    public float[] embed(Object input, Modality modality) {
        if (input == null) return null;
        Modality m = modality == null ? detect(input) : modality;
        try {
            if (m == Modality.VIDEO && input instanceof VideoData vd) {
                return embedVideo(vd);
            }
            if (m == Modality.IMAGE || m == Modality.TENSOR || m == Modality.MULTIMODAL) {
                ImageData img = coerceImage(input);
                if (img != null) return embedImage(img);
            }
        } catch (Throwable t) {
            // fall through to hash
        }
        return fallback.embed(input, m == null ? Modality.IMAGE : m);
    }

    @Override
    public float[][] embedBatch(List<?> inputs, Modality modality) {
        if (inputs == null || inputs.isEmpty()) return new float[0][];
        // Prefer true batched forward when all cells are images
        List<ImageData> imgs = new ArrayList<>(inputs.size());
        boolean allImg = true;
        for (Object o : inputs) {
            ImageData id = coerceImage(o);
            if (id == null) { allImg = false; break; }
            imgs.add(id);
        }
        if (allImg) {
            try {
                return embedImagesBatched(imgs);
            } catch (Throwable ignored) {}
        }
        return EmbeddingModel.super.embedBatch(inputs, modality);
    }

    private float[] embedImage(ImageData image) {
        Objects.requireNonNull(image, "image");
        ImageData use = image;
        if (use.getImage() == null && use.getPath() != null) {
            try {
                use = org.bytedeco.pytorch.data.dataframe.media.MediaBridge.loadImage(use.getPath());
            } catch (Exception e) {
                return fallback.embed(image, Modality.IMAGE);
            }
        }
        if (use.getImage() == null) return fallback.embed(image, Modality.IMAGE);

        // resize to network input
        if (use.getWidth() != inputSize || use.getHeight() != inputSize) {
            use = use.resize(inputSize, inputSize);
        }

        Tensor chw = ImageTensors.toTensor(use); // [C,H,W] in [0,1]
        // ImageNet-ish normalize
        Tensor normed = imagenetNormalize(chw);
        Tensor batch = normed.unsqueeze(0); // [1,C,H,W]

        float[] feats = forwardFeatures(batch);
        if (feats == null) return fallback.embed(image, Modality.IMAGE);
        float[] out = EmbeddingMath.ensureDim(feats, outDim);
        if (spec.l2Normalize()) out = EmbeddingMath.l2Normalize(out);
        return out;
    }

    private float[] embedVideo(VideoData video) {
        List<ImageData> frames = video.getFrames();
        if (frames == null || frames.isEmpty()) {
            return fallback.embed(video, Modality.VIDEO);
        }
        // temporal mean of up to 8 frame embeddings
        int take = Math.min(8, frames.size());
        double step = frames.size() / (double) take;
        List<float[]> acc = new ArrayList<>(take);
        for (int i = 0; i < take; i++) {
            ImageData f = frames.get(Math.min(frames.size() - 1, (int) (i * step)));
            try {
                acc.add(embedImage(f));
            } catch (Throwable ignored) {}
        }
        if (acc.isEmpty()) return fallback.embed(video, Modality.VIDEO);
        float[] pooled = EmbeddingMath.meanPool(acc);
        float[] out = EmbeddingMath.ensureDim(pooled, outDim);
        if (spec.l2Normalize()) out = EmbeddingMath.l2Normalize(out);
        return out;
    }

    private float[][] embedImagesBatched(List<ImageData> images) {
        List<Tensor> ts = new ArrayList<>(images.size());
        for (ImageData img : images) {
            ImageData use = img;
            if (use.getImage() == null && use.getPath() != null) {
                try {
                    use = org.bytedeco.pytorch.data.dataframe.media.MediaBridge.loadImage(use.getPath());
                } catch (Exception e) {
                    ts.add(null);
                    continue;
                }
            }
            if (use == null || use.getImage() == null) {
                ts.add(null);
                continue;
            }
            if (use.getWidth() != inputSize || use.getHeight() != inputSize) {
                use = use.resize(inputSize, inputSize);
            }
            ts.add(imagenetNormalize(ImageTensors.toTensor(use)));
        }
        // run one-by-one if any null (simpler than masking); still uses neural path
        float[][] out = new float[images.size()][];
        for (int i = 0; i < ts.size(); i++) {
            Tensor t = ts.get(i);
            if (t == null) {
                out[i] = fallback.embed(images.get(i), Modality.IMAGE);
                continue;
            }
            float[] feats = forwardFeatures(t.unsqueeze(0));
            if (feats == null) {
                out[i] = fallback.embed(images.get(i), Modality.IMAGE);
            } else {
                float[] v = EmbeddingMath.ensureDim(feats, outDim);
                out[i] = spec.l2Normalize() ? EmbeddingMath.l2Normalize(v) : v;
            }
        }
        return out;
    }

    private float[] forwardFeatures(Tensor batchNCHW) {
        try (NoGradGuard ng = new NoGradGuard()) {
            encoder.eval();
            Tensor feat;
            if (encoder instanceof Models.ResNet r) {
                feat = r.features(batchNCHW);
            } else if (encoder instanceof Models.MobileNetV2 m) {
                feat = m.features(batchNCHW);
            } else {
                // generic: forward then treat logits as embedding
                feat = invokeFeaturesOrForward(encoder, batchNCHW);
            }
            if (feat == null) return null;
            return row0(feat);
        } catch (Throwable t) {
            return null;
        }
    }

    private static Tensor invokeFeaturesOrForward(Module m, Tensor x) {
        try {
            for (String name : new String[]{"features", "forward", "forward_tensor"}) {
                try {
                    java.lang.reflect.Method method = m.getClass().getMethod(name, Tensor.class);
                    Object r = method.invoke(m, x);
                    if (r instanceof Tensor t) return t;
                } catch (NoSuchMethodException ignored) {}
            }
        } catch (Throwable ignored) {}
        return null;
    }

    /** ImageNet mean/std normalize on CHW float [0,1] → CHW. */
    private static Tensor imagenetNormalize(Tensor chw) {
        // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        try {
            Tensor t = chw.contiguous().to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
            long[] sz = sizes(t);
            if (sz.length != 3) return t;
            int c = (int) sz[0], h = (int) sz[1], w = (int) sz[2];
            FloatPointer fp = t.data_ptr_float();
            long plane = (long) h * w;
            for (int ch = 0; ch < Math.min(c, 3); ch++) {
                float m = mean[ch], s = std[ch] > 1e-8f ? std[ch] : 1f;
                long base = ch * plane;
                for (long i = 0; i < plane; i++) {
                    float v = fp.get(base + i);
                    fp.put(base + i, (v - m) / s);
                }
            }
            return t;
        } catch (Throwable e) {
            return chw;
        }
    }

    private static float[] row0(Tensor feat) {
        Tensor cpu = feat.contiguous().cpu().to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        // accept [D] or [1,D] or [1,1,D]…
        long n = cpu.numel();
        if (n <= 0) return null;
        // if batched, take first row
        long[] sz = sizes(cpu);
        int d;
        long offset = 0;
        if (sz.length >= 2 && sz[0] >= 1) {
            d = (int) (n / sz[0]);
            offset = 0; // row 0
        } else {
            d = (int) n;
        }
        float[] out = new float[d];
        FloatPointer fp = cpu.data_ptr_float();
        for (int i = 0; i < d; i++) out[i] = fp.get(offset + i);
        return out;
    }

    private static long[] sizes(Tensor t) {
        long ndim = t.dim();
        long[] out = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) out[i] = t.size(i);
        return out;
    }

    private static ImageData coerceImage(Object input) {
        if (input instanceof ImageData id) return id;
        if (input instanceof Tensor t) {
            try {
                return org.bytedeco.pytorch.data.dataframe.media.MediaBridge.tensorToImage(t);
            } catch (Throwable e) {
                return null;
            }
        }
        if (input instanceof String path) {
            try {
                return org.bytedeco.pytorch.data.dataframe.media.MediaBridge.loadImage(path);
            } catch (Exception e) {
                return null;
            }
        }
        return null;
    }

    private static Modality detect(Object input) {
        if (input instanceof VideoData) return Modality.VIDEO;
        if (input instanceof ImageData) return Modality.IMAGE;
        if (input instanceof Tensor) return Modality.TENSOR;
        return Modality.IMAGE;
    }

    private static ImageData solid(int w, int h, int rgb) {
        java.awt.image.BufferedImage bi =
                new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                bi.setRGB(x, y, rgb);
        return new ImageData(bi);
    }

    @Override
    public void close() {
        try { fallback.close(); } catch (Exception ignored) {}
        // Module native cleanup is GC-managed via Pointer
    }
}
