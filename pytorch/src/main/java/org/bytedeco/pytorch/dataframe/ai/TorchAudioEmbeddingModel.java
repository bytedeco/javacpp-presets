package org.bytedeco.pytorch.dataframe.ai;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.audio.models.AudioModels;
import org.bytedeco.pytorch.utils.audio.utils.AudioTensors;

import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Real torchaudio-style neural audio embedding.
 *
 * <p>Runs an M5 / Wav2Letter-lite raw-waveform CNN in eval / no-grad mode and
 * returns the pooled feature vector. Random-init weights are fine for structural
 * benchmarks; callers may later load safetensors into the same Module.
 *
 * <p>Supported model ids (family {@code wav2vec} / {@code audio}):
 * <ul>
 *   <li>{@code m5} / {@code m5-audio} → 256-d features</li>
 *   <li>{@code wav2letter} / {@code wav2letter-lite} → 128-d features</li>
 *   <li>{@code wav2vec2-base} (structural stand-in via M5 backbone) → 256-d</li>
 * </ul>
 *
 * <pre>{@code
 * EmbeddingModel m = TorchAudioEmbeddingModel.m5();
 * float[] v = m.embed(audioData, Modality.AUDIO);
 * DataFrame df = AiFunctions.embedAudioColumn(waves, "audio", "emb", "m5");
 * }</pre>
 */
public final class TorchAudioEmbeddingModel implements EmbeddingModel {

    public enum Backbone {
        M5, WAV2LETTER_LITE
    }

    private final ModelSpec spec;
    private final Backbone backbone;
    private final Module encoder;
    private final long nativeFeatureDim;
    private final int outDim;
    private final int targetSampleRate;
    private final int maxSamples;
    private final EmbeddingModel fallback;
    private volatile boolean warmed;

    public TorchAudioEmbeddingModel(ModelSpec spec, Backbone backbone, int outDim,
                                    int targetSampleRate, int maxSamples) {
        this.spec = spec == null
                ? ModelSpec.of("m5", Modality.AUDIO, 256, "wav2vec", true) : spec;
        this.backbone = backbone == null ? Backbone.M5 : backbone;
        this.targetSampleRate = targetSampleRate > 0 ? targetSampleRate : 16000;
        this.maxSamples = maxSamples > 0 ? maxSamples : this.targetSampleRate * 2; // 2s
        this.encoder = buildEncoder(this.backbone);
        this.nativeFeatureDim = featureDimOf(this.backbone, this.encoder);
        this.outDim = outDim > 0 ? outDim : (int) this.nativeFeatureDim;
        this.fallback = new HashEmbeddingModel(
                ModelSpec.of(this.spec.id() + "/hash-fallback", Modality.AUDIO, this.outDim, "hash", true));
        try {
            this.encoder.eval();
        } catch (Throwable ignored) {}
    }

    public static TorchAudioEmbeddingModel m5() {
        return new TorchAudioEmbeddingModel(
                ModelSpec.of("m5", Modality.AUDIO, 256, "wav2vec", true),
                Backbone.M5, 256, 16000, 32000);
    }

    public static TorchAudioEmbeddingModel wav2letter() {
        return new TorchAudioEmbeddingModel(
                ModelSpec.of("wav2letter-lite", Modality.AUDIO, 128, "wav2vec", true),
                Backbone.WAV2LETTER_LITE, 128, 16000, 32000);
    }

    /** Build from {@link ModelSpec} / model id (used by {@link EmbeddingRegistry}). */
    public static TorchAudioEmbeddingModel fromSpec(ModelSpec spec) {
        ModelSpec s = spec == null
                ? ModelSpec.of("m5", Modality.AUDIO, 256, "wav2vec", true) : spec;
        String id = s.id().toLowerCase(Locale.ROOT);
        Backbone b;
        int dim;
        if (id.contains("wav2letter") || id.contains("w2l")) {
            b = Backbone.WAV2LETTER_LITE;
            dim = s.defaultDim() > 0 ? s.defaultDim() : 128;
        } else {
            // m5, wav2vec*, whisper* structural stand-in
            b = Backbone.M5;
            dim = s.defaultDim() > 0 ? Math.min(s.defaultDim(), 256) : 256;
            // If user asked for 768 (wav2vec2-base), keep outDim=768 via ensureDim pad/project
            if (s.defaultDim() > 256) dim = s.defaultDim();
        }
        return new TorchAudioEmbeddingModel(s.withDim(dim), b, dim, 16000, 32000);
    }

    private static Module buildEncoder(Backbone b) {
        return switch (b) {
            case WAV2LETTER_LITE -> AudioModels.wav2letter_lite(1, 10);
            case M5 -> AudioModels.m5(1, 10);
        };
    }

    private static long featureDimOf(Backbone b, Module encoder) {
        try {
            if (encoder instanceof AudioModels.M5 m) return m.featureDim();
            if (encoder instanceof AudioModels.Wav2LetterLite w) return w.featureDim();
        } catch (Throwable ignored) {}
        return switch (b) {
            case WAV2LETTER_LITE -> 128L;
            case M5 -> 256L;
        };
    }

    @Override public ModelSpec spec() { return spec; }
    @Override public String backend() { return "torchaudio-" + backbone.name().toLowerCase(Locale.ROOT); }
    @Override public int dimension() { return outDim; }
    @Override public boolean isReady() { return encoder != null; }

    @Override
    public boolean supports(Modality modality) {
        return modality == Modality.AUDIO || modality == Modality.TENSOR
                || modality == Modality.MULTIMODAL;
    }

    @Override
    public void warmup() {
        if (warmed) return;
        synchronized (this) {
            if (warmed) return;
            try {
                float[] tone = new float[targetSampleRate / 4];
                for (int i = 0; i < tone.length; i++) {
                    tone[i] = (float) Math.sin(2 * Math.PI * 440 * i / (double) targetSampleRate) * 0.2f;
                }
                embedAudio(new AudioData(tone, targetSampleRate, 1));
                warmed = true;
            } catch (Throwable t) {
                warmed = true;
            }
        }
    }

    @Override
    public float[] embed(Object input, Modality modality) {
        if (input == null) return null;
        Modality m = modality == null ? detect(input) : modality;
        try {
            AudioData audio = coerceAudio(input);
            if (audio != null) return embedAudio(audio);
        } catch (Throwable ignored) {}
        return fallback.embed(input, m == null ? Modality.AUDIO : m);
    }

    @Override
    public float[][] embedBatch(List<?> inputs, Modality modality) {
        // Variable-length audio → per-row forward (still neural)
        return EmbeddingModel.super.embedBatch(inputs, modality);
    }

    private float[] embedAudio(AudioData audio) {
        Objects.requireNonNull(audio, "audio");
        AudioData use = audio;
        // mono
        if (use.getChannels() > 1) {
            try {
                use = MediaBridge.toMono(use);
            } catch (Throwable ignored) {}
        }
        // resample
        if (use.getSampleRate() > 0 && use.getSampleRate() != targetSampleRate) {
            try {
                use = MediaBridge.resample(use, targetSampleRate);
            } catch (Throwable ignored) {}
        }
        float[] samples = use.getSamples();
        if (samples == null || samples.length == 0) {
            return fallback.embed(audio, Modality.AUDIO);
        }

        // trim / pad to maxSamples
        float[] window = fitLength(samples, maxSamples);
        Tensor wave = AudioTensors.toTensor(window, 1); // [1, T] or [C,T]
        // ensure [N, C, T] = [1, 1, T]
        if (wave.dim() == 1) {
            wave = wave.unsqueeze(0).unsqueeze(0);
        } else if (wave.dim() == 2) {
            wave = wave.unsqueeze(0);
        }

        float[] feats = forwardFeatures(wave);
        if (feats == null) return fallback.embed(audio, Modality.AUDIO);
        float[] out = EmbeddingMath.ensureDim(feats, outDim);
        if (spec.l2Normalize()) out = EmbeddingMath.l2Normalize(out);
        return out;
    }

    private float[] forwardFeatures(Tensor batchNCT) {
        try (NoGradGuard ng = new NoGradGuard()) {
            encoder.eval();
            Tensor feat;
            if (encoder instanceof AudioModels.M5 m) {
                feat = m.features(batchNCT);
            } else if (encoder instanceof AudioModels.Wav2LetterLite w) {
                feat = w.features(batchNCT);
            } else {
                feat = invokeFeaturesOrForward(encoder, batchNCT);
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

    private static float[] fitLength(float[] samples, int target) {
        if (samples.length == target) return samples;
        float[] out = new float[target];
        if (samples.length > target) {
            System.arraycopy(samples, 0, out, 0, target);
        } else {
            System.arraycopy(samples, 0, out, 0, samples.length);
            // zero-pad remainder
        }
        return out;
    }

    private static float[] row0(Tensor feat) {
        Tensor cpu = feat.contiguous().cpu().to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        long n = cpu.numel();
        if (n <= 0) return null;
        long[] sz = sizes(cpu);
        int d;
        if (sz.length >= 2 && sz[0] >= 1) {
            d = (int) (n / sz[0]);
        } else {
            d = (int) n;
        }
        float[] out = new float[d];
        FloatPointer fp = cpu.data_ptr_float();
        for (int i = 0; i < d; i++) out[i] = fp.get(i);
        return out;
    }

    private static long[] sizes(Tensor t) {
        long ndim = t.dim();
        long[] out = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) out[i] = t.size(i);
        return out;
    }

    private static AudioData coerceAudio(Object input) {
        if (input instanceof AudioData ad) return ad;
        if (input instanceof Tensor t) {
            try {
                return AudioTensors.toAudioData(t, 16000);
            } catch (Throwable e) {
                return null;
            }
        }
        if (input instanceof String path) {
            try {
                return MediaBridge.loadAudio(path, 16000, true);
            } catch (Exception e) {
                return null;
            }
        }
        if (input instanceof float[] f) {
            return new AudioData(f, 16000, 1);
        }
        return null;
    }

    private static Modality detect(Object input) {
        if (input instanceof AudioData) return Modality.AUDIO;
        if (input instanceof Tensor) return Modality.TENSOR;
        return Modality.AUDIO;
    }

    @Override
    public void close() {
        try { fallback.close(); } catch (Exception ignored) {}
    }
}
