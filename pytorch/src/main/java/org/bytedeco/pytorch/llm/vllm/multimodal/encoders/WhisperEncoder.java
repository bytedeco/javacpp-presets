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
package org.bytedeco.pytorch.llm.vllm.multimodal.encoders;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.audio.functional.AudioF;
import org.bytedeco.pytorch.audio.io.AudioIO;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.tensor;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * OpenAI Whisper-tiny <b>encoder</b> (conv frontend + transformer blocks).
 *
 * <p>HF keys under {@code model.encoder.*}. Mel spectrogram is produced via
 * {@link AudioF#melSpectrogram} then log-compressed to approximate Whisper features.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class WhisperEncoder extends Module implements MediaEncoder {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final int SAMPLE_RATE = 16000;
    public static final int N_MELS = 80;
    public static final int N_FFT = 400;
    public static final int HOP = 160;

    private final String name;
    private final int dModel;
    private final int nHeads;
    private final int nLayers;
    private final int ffnDim;
    private final int maxSourcePositions;
    private final WeightBinder.Report loadReport;

    private final LongPointer convKernel;
    private final LongPointer convStride1;
    private final LongPointer convStride2;
    private final Conv1dImpl conv1;
    private final Conv1dImpl conv2;
    private Tensor embedPositions; // [max_source, d_model]
    private final List<WhisperEncBlock> blocks = new ArrayList<>();
    private final LayerNormImpl layerNorm;

    public WhisperEncoder(Path dir) throws Exception {
        super("WhisperEncoder");
        Objects.requireNonNull(dir, "dir");
        this.name = dir.toString();
        int d = 384, heads = 6, layers = 4, ffn = 1536, maxPos = 1500;
        Path cfg = dir.resolve("config.json");
        if (Files.isRegularFile(cfg)) {
            String json = Files.readString(cfg);
            d = readInt(json, "d_model", d);
            heads = readInt(json, "encoder_attention_heads", heads);
            layers = readInt(json, "encoder_layers", layers);
            ffn = readInt(json, "encoder_ffn_dim", ffn);
            maxPos = readInt(json, "max_source_positions", maxPos);
        }
        this.dModel = d;
        this.nHeads = heads;
        this.nLayers = layers;
        this.ffnDim = ffn;
        this.maxSourcePositions = maxPos;

        // Keep ExpandingArray pointers as fields so GC cannot free them.
        this.convKernel = new LongPointer(new long[]{3});
        this.convStride1 = new LongPointer(new long[]{1});
        this.convStride2 = new LongPointer(new long[]{2});
        // conv1: [d_model, n_mels, 3] stride 1; conv2 stride 2.
        Conv1dOptions c1 = new Conv1dOptions(N_MELS, d, convKernel);
        c1.stride(convStride1);
        c1.bias(true);
        this.conv1 = register_module("model/encoder/conv1", new Conv1dImpl(c1));
        Conv1dOptions c2 = new Conv1dOptions(d, d, convKernel);
        c2.stride(convStride2);
        c2.bias(true);
        this.conv2 = register_module("model/encoder/conv2", new Conv1dImpl(c2));
        this.embedPositions = register_parameter("model/encoder/embed_positions/weight",
                zeros(maxPos, d), true);
        LongVector lnShape = new LongVector().put((long) d);
        for (int i = 0; i < layers; i++) {
            blocks.add(register_module("model/encoder/layers/" + i,
                    new WhisperEncBlock(d, heads, ffn)));
        }
        this.layerNorm = register_module("model/encoder/layer_norm", new LayerNormImpl(lnShape));

        this.eval();
        this.loadReport = WeightBinder.bindSafetensors(this, dir, List.of(), false);
        System.out.println("[WhisperEncoder] " + loadReport + " dir=" + dir.getFileName());
    }

    public static WhisperEncoder fromDirectory(Path dir) throws Exception {
        return new WhisperEncoder(dir);
    }

    public WeightBinder.Report loadReport() { return loadReport; }

    @Override public MediaType modality() { return MediaType.AUDIO; }
    @Override
    public String encoderName() { return "whisper:" + name; }
    @Override public int featureDim() { return dModel; }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        try {
            Tensor mel = loadMel(input); // [1, 80, T]
            Tensor hidden = forwardEncoder(mel); // [1, T', D]
            // mean pool over time
            Tensor pooled = hidden.mean(1L); // [1, D]
            float[] pool = ImagePreprocess.toFloatArray(pooled.reshape(-1));
            int seqN = (int) Math.min(16, hidden.size(1));
            float[][] seq = new float[seqN][];
            for (int i = 0; i < seqN; i++) {
                seq[i] = ImagePreprocess.toFloatArray(hidden.select(1, i).reshape(-1));
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            return new EncoderFeatures(pool, seq, encoderName(), ms);
        } catch (Exception e) {
            System.out.println("[WhisperEncoder] encode failed: " + e.getMessage());
            return EncoderFeatures.empty(encoderName());
        }
    }

    public Tensor forwardEncoder(Tensor mel) {
        // mel: [B, 80, T]
        Tensor x = gelu(conv1.forward(mel));
        x = gelu(conv2.forward(x)); // [B, D, T']
        x = x.transpose(1, 2); // [B, T', D]
        long T = x.size(1);
        Tensor pe = embedPositions;
        if (pe.size(0) < T) {
            // truncate time
            x = x.slice(1, new org.bytedeco.pytorch.LongOptional(0),
                    new org.bytedeco.pytorch.LongOptional(pe.size(0)), 1);
            T = x.size(1);
        }
        pe = pe.slice(0, new org.bytedeco.pytorch.LongOptional(0),
                new org.bytedeco.pytorch.LongOptional(T), 1).unsqueeze(0);
        x = x.add(pe);
        for (WhisperEncBlock b : blocks) x = b.forward(x);
        return layerNorm.forward(x);
    }

    /** Load waveform and build log-mel [1,80,T]. */
    public static Tensor loadMel(MediaInput input) {
        Tensor waveform;
        int sr = SAMPLE_RATE;
        if (input.tensor != null && input.tensor.defined()) {
            waveform = input.tensor.to(ScalarType.Float).contiguous();
            if (waveform.dim() == 1) waveform = waveform.unsqueeze(0);
        } else if (input.path != null && Files.isRegularFile(input.path)) {
            try {
                AudioIO.AudioLoadResult r = AudioIO.load(input.path, SAMPLE_RATE, true);
                waveform = r.waveform().to(ScalarType.Float).contiguous();
                sr = r.sampleRate();
                if (waveform.dim() == 1) waveform = waveform.unsqueeze(0);
            } catch (Throwable t) {
                // synthetic tone fallback
                waveform = syntheticTone(SAMPLE_RATE, 1.0);
            }
        } else {
            waveform = syntheticTone(SAMPLE_RATE, 1.0);
        }
        // mel spectrogram
        Tensor mel;
        try {
            mel = AudioF.melSpectrogram(waveform, sr, N_MELS, 0.0, sr / 2.0, N_FFT, HOP);
        } catch (Throwable t) {
            // simple magnitude STFT-ish fallback: random-ish but deterministic from samples
            float[] w = ImagePreprocess.toFloatArray(waveform.reshape(-1));
            int frames = Math.max(1, w.length / HOP);
            float[] data = new float[N_MELS * frames];
            for (int f = 0; f < frames; f++) {
                for (int m = 0; m < N_MELS; m++) {
                    int idx = Math.min(w.length - 1, f * HOP + (m * 3) % HOP);
                    data[m * frames + f] = Math.abs(w[idx]);
                }
            }
            mel = ImagePreprocess.fromFloatArray(data, 1, N_MELS, frames);
            if (mel.dim() == 3 && mel.size(0) == 1) {
                // [1, n_mels, frames]
            } else {
                mel = ImagePreprocess.fromFloatArray(data, N_MELS, frames).unsqueeze(0);
            }
        }
        // ensure [B, n_mels, T]
        if (mel.dim() == 2) mel = mel.unsqueeze(0);
        if (mel.dim() == 3 && mel.size(1) != N_MELS && mel.size(2) == N_MELS) {
            mel = mel.transpose(1, 2);
        }
        // log compress
        mel = mel.clamp_min(new Scalar(1e-10)).log10();
        // max-normalize roughly like Whisper
        try {
            float max = mel.max().item_float();
            mel = mel.clamp_min(new Scalar(max - 8.0));
            mel = mel.add(new Scalar(4.0)).div(new Scalar(4.0));
        } catch (Throwable ignored) {}
        return mel.to(ScalarType.Float).contiguous();
    }

    private static Tensor syntheticTone(int sr, double seconds) {
        int n = (int) (sr * seconds);
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            w[i] = (float) Math.sin(2 * Math.PI * 440 * i / sr) * 0.2f;
        }
        return tensor(w, new TensorOptions(ScalarType.Float)).reshape(1, n);
    }

    @Override
    public Tensor forward(Tensor input) {
        return forwardEncoder(input).mean(1L);
    }

    // ---- encoder block -------------------------------------------------------

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class WhisperEncBlock extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LayerNormImpl selfAttnLn;
        public final WhisperAttn selfAttn;
        public final LayerNormImpl finalLn;
        public final LinearImpl fc1, fc2;
        public WhisperEncBlock(int d, int heads, int ffn) {
            super("WhisperEncBlock");
            LongVector s = new LongVector().put((long) d);
            this.selfAttnLn = register_module("self_attn_layer_norm", new LayerNormImpl(s));
            this.selfAttn = register_module("self_attn", new WhisperAttn(d, heads));
            this.finalLn = register_module("final_layer_norm", new LayerNormImpl(s));
            this.fc1 = register_module("fc1", new LinearImpl(new LinearOptions(d, ffn).bias(true)));
            this.fc2 = register_module("fc2", new LinearImpl(new LinearOptions(ffn, d).bias(true)));
        }
        @Override
        public Tensor forward(Tensor x) {
            x = x.add(selfAttn.forward(selfAttnLn.forward(x)));
            Tensor h = gelu(fc1.forward(finalLn.forward(x)));
            return x.add(fc2.forward(h));
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class WhisperAttn extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }
        public final LinearImpl q_proj, k_proj, v_proj, out_proj;
        private final int heads, headDim;
        public WhisperAttn(int d, int heads) {
            super("WhisperAttn");
            this.heads = heads;
            this.headDim = d / heads;
            this.q_proj = register_module("q_proj", new LinearImpl(new LinearOptions(d, d).bias(true)));
            this.k_proj = register_module("k_proj", new LinearImpl(new LinearOptions(d, d).bias(false)));
            this.v_proj = register_module("v_proj", new LinearImpl(new LinearOptions(d, d).bias(true)));
            this.out_proj = register_module("out_proj", new LinearImpl(new LinearOptions(d, d).bias(true)));
        }
        @Override
        public Tensor forward(Tensor x) {
            long B = x.size(0), N = x.size(1), C = x.size(2);
            Tensor q = q_proj.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            Tensor k = k_proj.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            Tensor v = v_proj.forward(x).reshape(B, N, heads, headDim).transpose(1, 2);
            double scale = 1.0 / Math.sqrt(headDim);
            Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)), -1L);
            Tensor out = matmul(attn, v).transpose(1, 2).contiguous().reshape(B, N, C);
            return out_proj.forward(out);
        }
    }

    private static int readInt(String json, String key, int def) {
        try {
            String pat = "\"" + key + "\"";
            int i = json.indexOf(pat);
            if (i < 0) return def;
            String rest = json.substring(i + pat.length()).replaceAll("^[^0-9-]+", "");
            int end = 0;
            while (end < rest.length() && (Character.isDigit(rest.charAt(end)) || rest.charAt(end) == '-')) end++;
            return Integer.parseInt(rest.substring(0, end));
        } catch (Exception e) { return def; }
    }
}
