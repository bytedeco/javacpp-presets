package org.bytedeco.pytorch.dataframe.ai;

import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Descriptor of an embedding model (Daft-style {@code model="clip-vit-base-patch32"}).
 *
 * <pre>
 *   ModelSpec clip = ModelSpec.parse("clip-vit-base-patch32");
 *   ModelSpec bge  = ModelSpec.of("bge-small-zh", Modality.TEXT, 512);
 * </pre>
 */
public final class ModelSpec {
    private final String id;
    private final Modality modality;
    private final int defaultDim;
    private final String family;   // clip | bge | wav2vec | hash | custom
    private final boolean l2Normalize;

    private ModelSpec(String id, Modality modality, int defaultDim, String family, boolean l2Normalize) {
        this.id = Objects.requireNonNull(id);
        this.modality = modality == null ? Modality.TEXT : modality;
        this.defaultDim = defaultDim > 0 ? defaultDim : 384;
        this.family = family == null ? "custom" : family;
        this.l2Normalize = l2Normalize;
    }

    public static ModelSpec of(String id, Modality modality, int dim) {
        return new ModelSpec(id, modality, dim, guessFamily(id), true);
    }

    public static ModelSpec of(String id, Modality modality, int dim, String family, boolean l2Normalize) {
        return new ModelSpec(id, modality, dim, family, l2Normalize);
    }

    /**
     * Parse common model id strings used by Daft / HuggingFace-style names.
     */
    public static ModelSpec parse(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return of("hash-text", Modality.TEXT, 384);
        }
        String id = modelId.trim();
        String lower = id.toLowerCase(Locale.ROOT);

        if (lower.contains("clip")) {
            int dim = lower.contains("large") ? 768 : 512;
            return of(id, Modality.MULTIMODAL, dim, "clip", true);
        }
        if (lower.contains("bge") || lower.contains("e5") || lower.contains("minilm")
                || lower.contains("sentence") || lower.startsWith("text-")) {
            int dim = lower.contains("large") ? 1024 : (lower.contains("base") ? 768 : 384);
            if (lower.contains("small")) dim = 512;
            return of(id, Modality.TEXT, dim, "bge", true);
        }
        if (lower.contains("wav2vec") || lower.contains("whisper") || lower.contains("hubert")
                || lower.startsWith("audio-")) {
            return of(id, Modality.AUDIO, 768, "wav2vec", true);
        }
        if (lower.contains("video") || lower.contains("xclip") || lower.contains("videomae")) {
            return of(id, Modality.VIDEO, 512, "video", true);
        }
        if (lower.contains("resnet") || lower.contains("vit") || lower.contains("image")) {
            int dim = lower.contains("large") ? 1024 : 768;
            return of(id, Modality.IMAGE, dim, "vision", true);
        }
        if (lower.startsWith("hash-") || lower.contains("hash")) {
            Modality m = lower.contains("image") ? Modality.IMAGE
                : lower.contains("audio") ? Modality.AUDIO
                : lower.contains("video") ? Modality.VIDEO
                : Modality.TEXT;
            return of(id, m, 384, "hash", true);
        }
        // default: text encoder
        return of(id, Modality.TEXT, 384, "custom", true);
    }

    private static String guessFamily(String id) {
        String l = id.toLowerCase(Locale.ROOT);
        if (l.contains("clip")) return "clip";
        if (l.contains("bge") || l.contains("e5")) return "bge";
        if (l.contains("wav2vec") || l.contains("whisper")) return "wav2vec";
        if (l.contains("hash")) return "hash";
        if (l.contains("video")) return "video";
        if (l.contains("vit") || l.contains("resnet")) return "vision";
        return "custom";
    }

    public String id() { return id; }
    public Modality modality() { return modality; }
    public int defaultDim() { return defaultDim; }
    public String family() { return family; }
    public boolean l2Normalize() { return l2Normalize; }

    public ModelSpec withDim(int dim) {
        return new ModelSpec(id, modality, dim, family, l2Normalize);
    }

    public ModelSpec withNormalize(boolean norm) {
        return new ModelSpec(id, modality, defaultDim, family, norm);
    }

    public boolean supports(Modality m) {
        if (modality == Modality.MULTIMODAL) {
            return m == Modality.TEXT || m == Modality.IMAGE || m == Modality.MULTIMODAL;
        }
        return modality == m || m == Modality.TENSOR;
    }

    @Override public String toString() {
        return "ModelSpec{id=" + id + ", modality=" + modality + ", dim=" + defaultDim + ", family=" + family + "}";
    }

    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ModelSpec m)) return false;
        return id.equals(m.id);
    }

    @Override public int hashCode() { return id.hashCode(); }

    // well-known presets
    public static final ModelSpec HASH_TEXT = parse("hash-text");
    public static final ModelSpec HASH_IMAGE = parse("hash-image");
    public static final ModelSpec HASH_AUDIO = parse("hash-audio");
    public static final ModelSpec HASH_VIDEO = parse("hash-video");
    public static final ModelSpec CLIP_VIT_B32 = parse("clip-vit-base-patch32");
    public static final ModelSpec CLIP_VIT_L14 = parse("clip-vit-large-patch14");
    public static final ModelSpec BGE_SMALL_ZH = parse("bge-small-zh");
    public static final ModelSpec BGE_BASE_EN = parse("bge-base-en");
    public static final ModelSpec WAV2VEC2_BASE = parse("wav2vec2-base");
    public static final ModelSpec VIDEO_MAE = parse("videomae-base");

    public static Set<ModelSpec> builtins() {
        return Set.of(HASH_TEXT, HASH_IMAGE, HASH_AUDIO, HASH_VIDEO,
            CLIP_VIT_B32, CLIP_VIT_L14, BGE_SMALL_ZH, BGE_BASE_EN, WAV2VEC2_BASE, VIDEO_MAE);
    }
}
