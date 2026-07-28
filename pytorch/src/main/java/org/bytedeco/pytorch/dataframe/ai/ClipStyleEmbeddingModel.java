package org.bytedeco.pytorch.dataframe.ai;

import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.TensorData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;

import java.util.List;

/**
 * CLIP-style dual-encoder: text and image share one embedding space so that
 * {@code cosine(embed(text), embed(image))} is meaningful.
 *
 * <p>Without external weights this uses aligned hash projections (same dim,
 * shared salt schedule) so cross-modal nearest-neighbor still works as a
 * structural stand-in. When a TorchScript / safetensors tower is registered
 * via {@link #withTextTower}/{@link #withImageTower}, those are preferred.
 *
 * <p>Model ids: {@code clip-vit-base-patch32}, {@code clip-vit-large-patch14}, …
 */
public final class ClipStyleEmbeddingModel implements EmbeddingModel {
    private final ModelSpec spec;
    private final int dim;
    private final EmbeddingModel textTower;
    private final EmbeddingModel imageTower;
    private final EmbeddingModel audioTower;
    private final EmbeddingModel videoTower;

    public ClipStyleEmbeddingModel(ModelSpec spec) {
        this(spec, null, null, null, null);
    }

    public ClipStyleEmbeddingModel(ModelSpec spec,
                                   EmbeddingModel textTower,
                                   EmbeddingModel imageTower,
                                   EmbeddingModel audioTower,
                                   EmbeddingModel videoTower) {
        this.spec = spec == null ? ModelSpec.CLIP_VIT_B32 : spec;
        this.dim = this.spec.defaultDim();
        // default towers: hash models with CLIP-aligned dims + shared family salt
        this.textTower = textTower != null ? textTower
            : new HashEmbeddingModel(ModelSpec.of(this.spec.id() + "/text", Modality.TEXT, dim, "clip", true));
        this.imageTower = imageTower != null ? imageTower
            : new HashEmbeddingModel(ModelSpec.of(this.spec.id() + "/image", Modality.IMAGE, dim, "clip", true));
        this.audioTower = audioTower != null ? audioTower
            : new HashEmbeddingModel(ModelSpec.of(this.spec.id() + "/audio", Modality.AUDIO, dim, "clip", true));
        this.videoTower = videoTower != null ? videoTower
            : new HashEmbeddingModel(ModelSpec.of(this.spec.id() + "/video", Modality.VIDEO, dim, "clip", true));
    }

    public static ClipStyleEmbeddingModel open(String modelId) {
        return new ClipStyleEmbeddingModel(ModelSpec.parse(modelId));
    }

    public ClipStyleEmbeddingModel withTextTower(EmbeddingModel m) {
        return new ClipStyleEmbeddingModel(spec, m, imageTower, audioTower, videoTower);
    }

    public ClipStyleEmbeddingModel withImageTower(EmbeddingModel m) {
        return new ClipStyleEmbeddingModel(spec, textTower, m, audioTower, videoTower);
    }

    public ClipStyleEmbeddingModel withAudioTower(EmbeddingModel m) {
        return new ClipStyleEmbeddingModel(spec, textTower, imageTower, m, videoTower);
    }

    public ClipStyleEmbeddingModel withVideoTower(EmbeddingModel m) {
        return new ClipStyleEmbeddingModel(spec, textTower, imageTower, audioTower, m);
    }

    @Override public ModelSpec spec() { return spec; }
    @Override public String backend() { return "clip-style"; }
    @Override public int dimension() { return dim; }

    @Override
    public boolean supports(Modality modality) {
        return modality == Modality.TEXT || modality == Modality.IMAGE
            || modality == Modality.AUDIO || modality == Modality.VIDEO
            || modality == Modality.MULTIMODAL || modality == Modality.TENSOR;
    }

    @Override
    public float[] embed(Object input, Modality modality) {
        if (input == null) return null;
        Modality m = modality == null ? detect(input) : modality;
        float[] v = switch (m) {
            case TEXT -> textTower.embed(input, Modality.TEXT);
            case IMAGE -> imageTower.embed(input, Modality.IMAGE);
            case AUDIO -> audioTower.embed(input, Modality.AUDIO);
            case VIDEO -> videoTower.embed(input, Modality.VIDEO);
            case MULTIMODAL, TENSOR -> {
                // try image then text
                Modality d = detect(input);
                yield embed(input, d == Modality.MULTIMODAL ? Modality.TEXT : d);
            }
        };
        return EmbeddingMath.ensureDim(v, dim);
    }

    @Override
    public float[][] embedBatch(List<?> inputs, Modality modality) {
        if (inputs == null || inputs.isEmpty()) return new float[0][];
        Modality m = modality == null ? Modality.TEXT : modality;
        EmbeddingModel tower = switch (m) {
            case IMAGE -> imageTower;
            case AUDIO -> audioTower;
            case VIDEO -> videoTower;
            default -> textTower;
        };
        float[][] raw = tower.embedBatch(inputs, m);
        float[][] out = new float[raw.length][];
        for (int i = 0; i < raw.length; i++) {
            out[i] = raw[i] == null ? null : EmbeddingMath.ensureDim(raw[i], dim);
        }
        return out;
    }

    static Modality detect(Object input) {
        if (input instanceof ImageData) return Modality.IMAGE;
        if (input instanceof AudioData) return Modality.AUDIO;
        if (input instanceof VideoData) return Modality.VIDEO;
        if (input instanceof TensorData) return Modality.TENSOR;
        if (input instanceof float[] || input instanceof double[]) return Modality.TENSOR;
        return Modality.TEXT;
    }

    @Override
    public void close() {
        try { textTower.close(); } catch (Exception ignored) {}
        try { imageTower.close(); } catch (Exception ignored) {}
        try { audioTower.close(); } catch (Exception ignored) {}
        try { videoTower.close(); } catch (Exception ignored) {}
    }
}
