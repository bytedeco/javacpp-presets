package org.bytedeco.pytorch.data.dataframe.ai;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.bytedeco.pytorch.data.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.data.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.data.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.data.dataframe.dtype.TensorData;
import org.bytedeco.pytorch.data.dataframe.dtype.VideoData;

/**
 * Always-available pure-Java embedding backend using feature hashing /
 * signal statistics. Used as default and as fallback when Torch weights
 * are not present. Produces deterministic, L2-normalized vectors.
 *
 * <p>Supports TEXT / IMAGE / AUDIO / VIDEO / TENSOR modalities.
 */
public final class HashEmbeddingModel implements EmbeddingModel {
    private final ModelSpec spec;
    private final int dim;

    public HashEmbeddingModel(ModelSpec spec) {
        this.spec = spec == null ? ModelSpec.HASH_TEXT : spec;
        this.dim = this.spec.defaultDim();
    }

    public HashEmbeddingModel(String modelId, Modality modality, int dim) {
        this(ModelSpec.of(modelId, modality, dim, "hash", true));
    }

    public static HashEmbeddingModel forText(int dim) {
        return new HashEmbeddingModel(ModelSpec.of("hash-text", Modality.TEXT, dim, "hash", true));
    }

    public static HashEmbeddingModel forImage(int dim) {
        return new HashEmbeddingModel(ModelSpec.of("hash-image", Modality.IMAGE, dim, "hash", true));
    }

    public static HashEmbeddingModel forAudio(int dim) {
        return new HashEmbeddingModel(ModelSpec.of("hash-audio", Modality.AUDIO, dim, "hash", true));
    }

    public static HashEmbeddingModel forVideo(int dim) {
        return new HashEmbeddingModel(ModelSpec.of("hash-video", Modality.VIDEO, dim, "hash", true));
    }

    public static HashEmbeddingModel fromSpec(ModelSpec spec) {
        // force family=hash but keep requested dim / modality
        ModelSpec s = ModelSpec.of(
            spec.id().startsWith("hash") ? spec.id() : "hash-" + spec.id(),
            spec.modality() == Modality.MULTIMODAL ? Modality.TEXT : spec.modality(),
            spec.defaultDim(), "hash", true);
        return new HashEmbeddingModel(s);
    }

    @Override public ModelSpec spec() { return spec; }
    @Override public String backend() { return "hash"; }
    @Override public int dimension() { return dim; }

    @Override
    public float[] embed(Object input, Modality modality) {
        if (input == null) return null;
        Modality m = modality == null ? spec.modality() : modality;
        float[] raw = switch (m) {
            case TEXT -> embedText(input);
            case IMAGE -> embedImage(input);
            case AUDIO -> embedAudio(input);
            case VIDEO -> embedVideo(input);
            case TENSOR, MULTIMODAL -> embedGeneric(input);
        };
        if (raw == null) return null;
        float[] sized = EmbeddingMath.ensureDim(raw, dim);
        return spec.l2Normalize() ? EmbeddingMath.l2Normalize(sized) : sized;
    }

    private float[] embedText(Object input) {
        if (input instanceof EmbeddingData ed) return ed.getVector();
        return EmbeddingMath.hashEmbedText(String.valueOf(input), dim);
    }

    private float[] embedImage(Object input) {
        if (input instanceof EmbeddingData ed) return ed.getVector();
        if (input instanceof float[] f) return EmbeddingMath.hashEmbedSignal(f, dim);
        ImageData img = asImage(input);
        if (img == null) {
            // path or string fallback
            return EmbeddingMath.hashEmbedText(String.valueOf(input), dim);
        }
        float[] rgb = toRgbFloat(img);
        int h = Math.max(1, img.getHeight());
        int w = Math.max(1, img.getWidth());
        // also blend ImageData.extractEmbedding when available for richer features
        try {
            float[] deep = img.extractEmbedding(Math.min(256, dim));
            float[] hash = EmbeddingMath.hashEmbedImageRgb(rgb, h, w, dim);
            return mix(hash, EmbeddingMath.ensureDim(deep, dim), 0.55f);
        } catch (Exception e) {
            return EmbeddingMath.hashEmbedImageRgb(rgb, h, w, dim);
        }
    }

    private float[] embedAudio(Object input) {
        if (input instanceof EmbeddingData ed) return ed.getVector();
        AudioData aud = asAudio(input);
        if (aud == null) return EmbeddingMath.hashEmbedText(String.valueOf(input), dim);
        try {
            // prefer MFCC mean-pool when samples present
            if (aud.getSamples() != null && aud.getSamples().length > 0) {
                float[][] mfcc = aud.mfcc(Math.min(13, dim));
                float[] pooled = EmbeddingMath.meanPool(mfcc);
                float[] sig = EmbeddingMath.hashEmbedSignal(aud.getSamples(), dim);
                return mix(EmbeddingMath.ensureDim(pooled, dim), sig, 0.6f);
            }
        } catch (Exception ignored) {}
        if (aud.getSamples() != null) return EmbeddingMath.hashEmbedSignal(aud.getSamples(), dim);
        return EmbeddingMath.hashEmbedText(aud.getPath() == null ? "audio" : aud.getPath(), dim);
    }

    private float[] embedVideo(Object input) {
        if (input instanceof EmbeddingData ed) return ed.getVector();
        VideoData vid = asVideo(input);
        if (vid == null) return EmbeddingMath.hashEmbedText(String.valueOf(input), dim);
        List<ImageData> frames = vid.getFrames();
        if (frames == null || frames.isEmpty()) {
            return EmbeddingMath.hashEmbedText(
                vid.getPath() == null ? "video" : vid.getPath(), dim);
        }
        // sample up to 8 frames uniformly
        int take = Math.min(8, frames.size());
        List<float[]> emb = new ArrayList<>(take);
        double step = frames.size() / (double) take;
        for (int i = 0; i < take; i++) {
            int idx = Math.min(frames.size() - 1, (int) Math.floor(i * step));
            emb.add(embedImage(frames.get(idx)));
        }
        return EmbeddingMath.ensureDim(EmbeddingMath.meanPool(emb), dim);
    }

    private float[] embedGeneric(Object input) {
        if (input instanceof EmbeddingData ed) return EmbeddingMath.ensureDim(ed.getVector(), dim);
        if (input instanceof float[] f) return EmbeddingMath.ensureDim(f, dim);
        if (input instanceof double[] d) {
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return EmbeddingMath.ensureDim(f, dim);
        }
        if (input instanceof TensorData td) return EmbeddingMath.ensureDim(td.getData(), dim);
        if (input instanceof ImageData) return embedImage(input);
        if (input instanceof AudioData) return embedAudio(input);
        if (input instanceof VideoData) return embedVideo(input);
        return embedText(input);
    }

    private static float[] mix(float[] a, float[] b, float alphaA) {
        if (a == null) return b;
        if (b == null) return a;
        int n = Math.min(a.length, b.length);
        float[] out = new float[Math.max(a.length, b.length)];
        float alphaB = 1f - alphaA;
        for (int i = 0; i < n; i++) out[i] = alphaA * a[i] + alphaB * b[i];
        if (a.length > n) System.arraycopy(a, n, out, n, a.length - n);
        return EmbeddingMath.l2Normalize(out);
    }

    static ImageData asImage(Object v) {
        if (v instanceof ImageData id) return id;
        if (v instanceof BufferedImage bi) return new ImageData(bi);
        if (v instanceof String path) {
            try { return ImageData.load(path); } catch (Exception e) { return null; }
        }
        return null;
    }

    static AudioData asAudio(Object v) {
        if (v instanceof AudioData ad) return ad;
        if (v instanceof String path) {
            try { return AudioData.loadFromFile(path); } catch (Exception e) { return null; }
        }
        if (v instanceof float[] s) return new AudioData(s, 16000, 1);
        return null;
    }

    static VideoData asVideo(Object v) {
        if (v instanceof VideoData vd) return vd;
        if (v instanceof String path) return new VideoData(path);
        return null;
    }

    static float[] toRgbFloat(ImageData img) {
        BufferedImage bi = img.getImage();
        if (bi == null) return new float[0];
        int w = bi.getWidth(), h = bi.getHeight();
        float[] out = new float[w * h * 3];
        int k = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = bi.getRGB(x, y);
                out[k++] = ((rgb >> 16) & 0xFF) / 255f;
                out[k++] = ((rgb >> 8) & 0xFF) / 255f;
                out[k++] = (rgb & 0xFF) / 255f;
            }
        }
        return out;
    }
}
