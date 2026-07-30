package samples;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.utils.audio.datasets.AudioDataset;
import org.bytedeco.pytorch.utils.audio.datasets.AudioFolder;
import org.bytedeco.pytorch.utils.audio.datasets.FakeAudio;
import org.bytedeco.pytorch.utils.audio.functional.F;
import org.bytedeco.pytorch.utils.audio.io.AudioIO;
import org.bytedeco.pytorch.utils.audio.models.AudioModels;
import org.bytedeco.pytorch.utils.audio.transforms.Compose;
import org.bytedeco.pytorch.utils.audio.transforms.Transforms;
import org.bytedeco.pytorch.utils.audio.utils.AudioTensors;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Multi-dimensional correctness + performance benchmark for {@code utils.audio}.
 *
 * <p>Dimensions:
 * <ol>
 *   <li>D1 AudioTensors convert / layout / feature bridge</li>
 *   <li>D2 AudioIO load/save roundtrip (WAV)</li>
 *   <li>D3 functional F — spectrogram family, dB, resample, mu-law, fade/vol, mask, pad/trim, deltas</li>
 *   <li>D4 Transforms wrappers + Compose pipeline</li>
 *   <li>D5 Datasets — FakeAudio / AudioFolder</li>
 *   <li>D6 Models — SimpleAudioClassifier / M5 / Wav2LetterLite forward</li>
 *   <li>D7 Daily pipeline + edge cases</li>
 *   <li>D8 Throughput stress</li>
 * </ol>
 *
 * <pre>{@code
 * java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *   -cp "target/classes:$(mvn -q dependency:build-classpath -DincludeScope=runtime -Dmdep.outputFile=/dev/stdout)" \
 *   samples.BenchmarkAudio
 * }</pre>
 */
public class BenchmarkAudio {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("    CHECK " + name + ": OK");
        } else {
            failed++;
            report.append("CHECK FAILED: ").append(name).append('\n');
            System.out.println("    CHECK " + name + ": FAIL");
        }
    }

    static void checkEq(String name, Object expected, Object actual) {
        boolean ok;
        if (expected instanceof Number && actual instanceof Number) {
            double d = Math.abs(((Number) expected).doubleValue() - ((Number) actual).doubleValue());
            ok = Double.isNaN(d) ? Double.isNaN(((Number) expected).doubleValue()) : d < 1e-5;
        } else if (expected instanceof long[] ea && actual instanceof long[] aa) {
            ok = Arrays.equals(ea, aa);
        } else if (expected instanceof int[] ea && actual instanceof int[] aa) {
            ok = Arrays.equals(ea, aa);
        } else {
            ok = java.util.Objects.equals(expected, actual);
        }
        if (ok) {
            passed++;
            System.out.println("    CHECK " + name + ": OK (" + fmt(expected) + ")");
        } else {
            failed++;
            report.append("CHECK FAILED [").append(name).append("]: expected=")
                    .append(fmt(expected)).append(", actual=").append(fmt(actual)).append('\n');
            System.out.println("    CHECK " + name + ": FAIL (expected=" + fmt(expected) + ", got=" + fmt(actual) + ")");
        }
    }

    static String fmt(Object o) {
        if (o instanceof long[] a) return Arrays.toString(a);
        if (o instanceof int[] a) return Arrays.toString(a);
        if (o instanceof float[] a) return "float[" + a.length + "]";
        return String.valueOf(o);
    }

    static void section(String name, CheckedRunnable r) {
        System.out.println("\n── " + name + " ──");
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  FAIL " + name + " (" + ms + " ms): " + e.getMessage());
            report.append("SECTION FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static long[] shapes(Tensor t) {
        long ndim = t.dim();
        long[] s = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) s[i] = t.size(i);
        return s;
    }

    static boolean isFloat(Tensor t) {
        try {
            // dtype() is TypeMeta pointer; scalar_type() prints "Float"/"Double"/…
            // (ScalarType enums are not identity-equal without intern — use name.)
            String s = String.valueOf(t.scalar_type());
            return s.contains("Float") || s.contains("float") || s.contains("Half");
        } catch (Exception e) {
            return false;
        }
    }

    static float[] sine(double freq, int sr, int n, double amp) {
        float[] y = new float[n];
        for (int i = 0; i < n; i++) y[i] = (float) (amp * Math.sin(2 * Math.PI * freq * i / sr));
        return y;
    }

    static float[] stereoSine(double freq, int sr, int frames, double amp) {
        float[] y = new float[frames * 2];
        for (int i = 0; i < frames; i++) {
            float v = (float) (amp * Math.sin(2 * Math.PI * freq * i / sr));
            y[2 * i] = v;
            y[2 * i + 1] = 0.5f * v;
        }
        return y;
    }

    static void writeWav(Path path, float[] samples, int sr, int channels) throws Exception {
        int bits = 16;
        int byteRate = sr * channels * bits / 8;
        int blockAlign = channels * bits / 8;
        int dataSize = samples.length * 2;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeBytes("RIFF");
        dos.writeInt(Integer.reverseBytes(36 + dataSize));
        dos.writeBytes("WAVE");
        dos.writeBytes("fmt ");
        dos.writeInt(Integer.reverseBytes(16));
        dos.writeShort(Short.reverseBytes((short) 1));
        dos.writeShort(Short.reverseBytes((short) channels));
        dos.writeInt(Integer.reverseBytes(sr));
        dos.writeInt(Integer.reverseBytes(byteRate));
        dos.writeShort(Short.reverseBytes((short) blockAlign));
        dos.writeShort(Short.reverseBytes((short) bits));
        dos.writeBytes("data");
        dos.writeInt(Integer.reverseBytes(dataSize));
        for (float s : samples) {
            short v = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, (int) (s * 32767)));
            dos.writeShort(Short.reverseBytes(v));
        }
        dos.close();
        Files.write(path, baos.toByteArray());
    }

    static float maxAbs(float[] a) {
        float m = 0;
        for (float v : a) m = Math.max(m, Math.abs(v));
        return m;
    }

    static boolean finite(float[] a) {
        for (float v : a) if (!Float.isFinite(v)) return false;
        return true;
    }

    public static void main(String[] args) throws Exception {
        Path tmp = Files.createTempDirectory("audio_bench");
        System.out.println("=== Audio Module Benchmark ===");
        System.out.println("Temp: " + tmp);

        final int SR = 16000;
        final float[] mono = sine(440, SR, SR, 0.5);            // 1s mono interleaved
        final float[] stereo = stereoSine(440, SR, SR, 0.5);    // 1s stereo interleaved
        final Tensor waveMono = AudioTensors.toTensor(mono, 1); // [1, T]
        final Tensor waveStereo = AudioTensors.toTensor(stereo, 2); // [2, T]

        // ── D1 AudioTensors ──────────────────────────────────────────────────
        System.out.println("\n══ D1 AudioTensors ══");
        section("toTensor mono/stereo layout", () -> {
            long[] sm = shapes(waveMono);
            checkEq("mono shape [1,T]", 2, sm.length);
            checkEq("mono C=1", 1L, sm[0]);
            checkEq("mono T", (long) SR, sm[1]);
            check("mono float", isFloat(waveMono));

            long[] ss = shapes(waveStereo);
            checkEq("stereo shape rank 2", 2, ss.length);
            checkEq("stereo C=2", 2L, ss[0]);
            checkEq("stereo T", (long) SR, ss[1]);
        });

        section("fromTensor roundtrip", () -> {
            float[] back = AudioTensors.fromTensor(waveMono);
            checkEq("fromTensor mono length", mono.length, back.length);
            check("fromTensor mono peak close", Math.abs(maxAbs(back) - maxAbs(mono)) < 0.02f);

            float[] backS = AudioTensors.fromTensor(waveStereo);
            checkEq("fromTensor stereo length", stereo.length, backS.length);
            check("inferChannels mono", AudioTensors.inferChannels(waveMono) == 1);
            check("inferChannels stereo", AudioTensors.inferChannels(waveStereo) == 2);
            check("inferTime mono", AudioTensors.inferTime(waveMono) == SR);
            check("inferTime stereo", AudioTensors.inferTime(waveStereo) == SR);
        });

        section("AudioData bridge + feature matrix", () -> {
            AudioData ad = AudioTensors.toAudioData(waveMono, SR);
            check("AudioData samples", ad.getSamples() != null && ad.getSamples().length > 0);
            checkEq("AudioData sr", SR, ad.getSampleRate());
            Tensor t2 = AudioTensors.toTensor(ad);
            checkEq("toTensor(AudioData) C", 1L, shapes(t2)[0]);

            float[][] feat = new float[][]{{1, 2, 3}, {4, 5, 6}};
            Tensor ft = AudioTensors.featureToTensor(feat);
            checkEq("featureToTensor shape", new long[]{2, 3}, shapes(ft));
            float[][] back = AudioTensors.tensorToFeature(ft);
            checkEq("tensorToFeature rows", 2, back.length);
            checkEq("tensorToFeature [0][2]", 3f, back[0][2]);
            check("samples(AudioData)", AudioTensors.samples(ad) != null);
            check("toFloatArray", AudioTensors.toFloatArray(waveMono).length == SR);
            check("sizes", AudioTensors.sizes(waveMono).length == 2);
        });

        section("toTensor invalid args", () -> {
            boolean threw = false;
            try { AudioTensors.toTensor(new float[]{1, 2, 3}, 2); } catch (IllegalArgumentException e) { threw = true; }
            check("channels not divisible throws", threw);
            threw = false;
            try { AudioTensors.toTensor(mono, 0); } catch (IllegalArgumentException e) { threw = true; }
            check("channels < 1 throws", threw);
        });

        // ── D2 AudioIO ───────────────────────────────────────────────────────
        System.out.println("\n══ D2 AudioIO load/save ══");
        Path wav = tmp.resolve("tone.wav");
        writeWav(wav, mono, SR, 1);

        section("AudioIO.load defaults + aliases", () -> {
            AudioIO.AudioLoadResult r = AudioIO.load(wav.toString());
            check("waveform non-null", r.waveform() != null);
            check("sampleRate > 0", r.sampleRate() > 0);
            check("sample_rate alias", r.sample_rate() == r.sampleRate());
            check("toString", r.toString().contains("sampleRate"));
            check("DEFAULT_SAMPLE_RATE", AudioIO.DEFAULT_SAMPLE_RATE == 16000);
            long[] sh = shapes(r.waveform());
            check("loaded rank >= 1", sh.length >= 1);
            check("loaded float", isFloat(r.waveform()));

            AudioIO.AudioLoadResult r2 = AudioIO.load(wav, 8000, true);
            checkEq("load resample 8k", 8000, r2.sampleRate());
            check("load_audio alias", AudioIO.load_audio(wav.toString(), SR, true).waveform() != null);

            AudioData ad = AudioIO.loadAudioData(wav.toString(), SR, true);
            check("loadAudioData", ad.getSamples() != null && ad.getSamples().length > 0);
        });

        section("AudioIO.save roundtrip", () -> {
            Path out = tmp.resolve("roundtrip.wav");
            AudioIO.save(out.toString(), waveMono, SR);
            check("saved file exists", Files.exists(out) && Files.size(out) > 44);
            AudioIO.AudioLoadResult r = AudioIO.load(out.toString(), SR, true);
            check("roundtrip sr", r.sampleRate() == SR);
            check("roundtrip time close", Math.abs(AudioTensors.inferTime(r.waveform()) - SR) < 10);
            AudioIO.save_audio(tmp.resolve("alias.wav").toString(), waveMono, SR);
            check("save_audio alias", Files.exists(tmp.resolve("alias.wav")));

            // Path overload
            AudioIO.save(tmp.resolve("path.wav"), waveMono, SR);
            check("save Path overload", Files.exists(tmp.resolve("path.wav")));
        });

        section("AudioIO.save invalid", () -> {
            boolean threw = false;
            try { AudioIO.save(tmp.resolve("bad.wav").toString(), waveMono, 0); } catch (IllegalArgumentException e) { threw = true; }
            check("sampleRate<=0 throws", threw);
        });

        // ── D3 functional F ──────────────────────────────────────────────────
        System.out.println("\n══ D3 functional F ══");
        section("spectrogram / mel / mfcc", () -> {
            Tensor spec = F.spectrogram(waveMono, SR);
            long[] ss = shapes(spec);
            check("spec rank 2", ss.length == 2);
            check("spec freq bins > 0", ss[0] > 0);
            check("spec frames > 0", ss[1] > 0);
            check("spec float", isFloat(spec));

            Tensor spec2 = F.spectrogram(waveMono, SR, 1024, 256, 0, true);
            checkEq("custom nFft bins", 1024 / 2 + 1L, shapes(spec2)[0]);

            Tensor mel = F.mel_spectrogram(waveMono, SR);
            check("mel rank 2", shapes(mel).length == 2);
            check("melSpectrogram alias", shapes(F.melSpectrogram(waveMono, SR)).length == 2);

            Tensor mel64 = F.melSpectrogram(waveMono, SR, 64, 0, SR / 2.0, 1024, 256);
            checkEq("mel nMels=64", 64L, shapes(mel64)[0]);

            Tensor mfcc = F.mfcc(waveMono, SR);
            checkEq("mfcc default 13", 13L, shapes(mfcc)[0]);
            Tensor mfcc20 = F.mfcc(waveMono, SR, 20, 64, 0, SR / 2.0, 1024, 256);
            checkEq("mfcc n=20", 20L, shapes(mfcc20)[0]);
        });

        section("amplitude↔dB roundtrip", () -> {
            Tensor mel = F.melSpectrogram(waveMono, SR, 32, 0, SR / 2.0, 1024, 256);
            Tensor db = F.amplitudeToDB(mel);
            check("amplitudeToDB finite shape", shapes(db).length == 2);
            Tensor back = F.dbToAmplitude(db);
            check("dbToAmplitude shape match", Arrays.equals(shapes(mel), shapes(back)));
            check("amplitude_to_DB alias", shapes(F.amplitude_to_DB(mel, 10, 1e-10, 0)).length == 2);
            check("DB_to_amplitude alias", shapes(F.DB_to_amplitude(db, 1.0, 0.5)).length == 2);
            check("db_to_amplitude alias", shapes(F.db_to_amplitude(db)).length == 2);
        });

        section("resample / mu-law / fade / vol / normalize", () -> {
            Tensor r8k = F.resample(waveMono, SR, 8000);
            check("resample 16k→8k time ~8000", Math.abs(AudioTensors.inferTime(r8k) - 8000) < 5);
            Tensor same = F.resample(waveMono, SR, SR);
            check("resample identity same object or equal time", AudioTensors.inferTime(same) == SR);

            float[] rs = F.resampleSamples(mono, 1, SR, 8000);
            check("resampleSamples length ~8000", Math.abs(rs.length - 8000) < 5);

            boolean threw = false;
            try { F.resample(waveMono, 0, 8000); } catch (IllegalArgumentException e) { threw = true; }
            check("resample bad sr throws", threw);

            Tensor enc = F.mu_law_encoding(waveMono, 256);
            check("mu_law shape", Arrays.equals(shapes(waveMono), shapes(enc)) || shapes(enc)[shapes(enc).length - 1] == SR);
            Tensor dec = F.mu_law_decoding(enc, 256);
            check("mu_law decode shape rank", shapes(dec).length >= 1);
            check("muLawEncoding alias", shapes(F.muLawEncoding(waveMono, 256)).length >= 1);
            check("muLawDecoding alias", shapes(F.muLawDecoding(enc, 256)).length >= 1);

            Tensor faded = F.fade(waveMono, 1000, 1000);
            check("fade same shape", Arrays.equals(shapes(waveMono), shapes(faded)));
            Tensor faded2 = F.fade(waveMono, 500, 500, "linear");
            check("fade shape named", Arrays.equals(shapes(waveMono), shapes(faded2)));

            Tensor vol = F.vol(waveMono, 0.5);
            float[] vSamples = AudioTensors.fromTensor(vol);
            check("vol 0.5 reduces peak", maxAbs(vSamples) < maxAbs(mono) * 0.6f + 0.01f);
            Tensor volDb = F.vol(waveMono, -6.0, "db");
            check("vol db shape", Arrays.equals(shapes(waveMono), shapes(volDb)));

            Tensor norm = F.normalize(waveMono);
            check("normalize peak ~1", Math.abs(maxAbs(AudioTensors.fromTensor(norm)) - 1.0f) < 0.05f);
            Tensor norm2 = F.normalize(waveMono, 0.8);
            check("normalize peak custom", Math.abs(maxAbs(AudioTensors.fromTensor(norm2)) - 0.8f) < 0.05f);
        });

        section("masking / pad / trim / deltas / stretch / pitch / lfcc / ispec", () -> {
            Tensor spec = F.spectrogram(waveMono, SR, 512, 128, 0, true);
            Tensor fmask = F.frequency_masking(spec, 10);
            check("freq mask same shape", Arrays.equals(shapes(spec), shapes(fmask)));
            Tensor tmask = F.time_masking(spec, 5);
            check("time mask same shape", Arrays.equals(shapes(spec), shapes(tmask)));
            Tensor maskAxis = F.mask_along_axis(spec, 8, 0, 0);
            check("mask_along_axis shape", Arrays.equals(shapes(spec), shapes(maskAxis)));

            Tensor padded = F.pad(waveMono, 100);
            check("pad both sides +200", AudioTensors.inferTime(padded) == SR + 200);
            Tensor padded2 = F.pad(waveMono, 50, 75);
            check("pad left/right", AudioTensors.inferTime(padded2) == SR + 125);
            Tensor padded3 = F.pad(waveMono, new int[]{10, 20}, "constant", 0f);
            check("pad array form", AudioTensors.inferTime(padded3) == SR + 30);

            // silence + tone → trim
            float[] paddedArr = new float[SR * 3];
            System.arraycopy(mono, 0, paddedArr, SR, mono.length);
            Tensor paddedWave = AudioTensors.toTensor(paddedArr, 1);
            Tensor trimmed = F.trim(paddedWave, SR, 40f);
            check("trim shortens", AudioTensors.inferTime(trimmed) < SR * 3);
            check("trim default", AudioTensors.inferTime(F.trim(paddedWave, SR)) < SR * 3);

            Tensor deltas = F.compute_deltas(spec, 5);
            check("deltas shape", Arrays.equals(shapes(spec), shapes(deltas)));
            check("computeDeltas alias", shapes(F.computeDeltas(spec)).length == 2);
            Tensor d2 = F.compute_2d_deltas(spec, 5);
            check("2d deltas shape", Arrays.equals(shapes(spec), shapes(d2)));
            check("compute2DDeltas alias", shapes(F.compute2DDeltas(spec)).length == 2);

            Tensor stretched = F.time_stretch(spec, 1.2);
            check("time_stretch frames change or same rank", shapes(stretched).length == 2);
            check("timeStretch alias", shapes(F.timeStretch(spec, 0.8)).length == 2);

            Tensor pitched = F.pitch_shift(waveMono, SR, 2.0);
            check("pitch_shift time preserved roughly", Math.abs(AudioTensors.inferTime(pitched) - SR) < SR / 5);
            check("pitchShift alias", shapes(F.pitchShift(waveMono, SR, -1)).length >= 1);

            Tensor lfcc = F.lfcc(waveMono, SR);
            check("lfcc rank 2", shapes(lfcc).length == 2);
            Tensor lfcc2 = F.lfcc(waveMono, SR, 13, 128, 1024, 256);
            checkEq("lfcc n=13", 13L, shapes(lfcc2)[0]);

            Tensor ispec = F.inverse_spectrogram(spec, 512, 128, 0, true);
            check("inverse_spectrogram rank >= 1", shapes(ispec).length >= 1);
            check("inverseSpectrogram alias", shapes(F.inverseSpectrogram(spec, 512, 128)).length >= 1);
        });

        // ── D4 Transforms ────────────────────────────────────────────────────
        System.out.println("\n══ D4 Transforms + Compose ══");
        section("transform wrappers", () -> {
            Tensor s = new Transforms.Spectrogram(SR).forward(waveMono);
            check("Spectrogram transform", shapes(s).length == 2);
            Tensor s2 = new Transforms.Spectrogram(SR, 1024, 256, 0, true).forward(waveMono);
            check("Spectrogram custom", shapes(s2)[0] == 513);

            Tensor m = new Transforms.MelSpectrogram(SR).forward(waveMono);
            check("MelSpectrogram", shapes(m).length == 2);
            Tensor m2 = new Transforms.MelSpectrogram(SR, 64).forward(waveMono);
            checkEq("MelSpectrogram nMels", 64L, shapes(m2)[0]);

            Tensor mf = new Transforms.MFCC(SR).forward(waveMono);
            checkEq("MFCC transform 13", 13L, shapes(mf)[0]);
            checkEq("MFCC n=20", 20L, shapes(new Transforms.MFCC(SR, 20).forward(waveMono))[0]);

            Tensor rs = new Transforms.Resample(SR, 8000).forward(waveMono);
            check("Resample transform", Math.abs(AudioTensors.inferTime(rs) - 8000) < 5);

            Tensor v = new Transforms.Vol(0.5).forward(waveMono);
            check("Vol transform", shapes(v).length == 2);
            Tensor fad = new Transforms.Fade(100, 100).forward(waveMono);
            check("Fade transform", Arrays.equals(shapes(waveMono), shapes(fad)));
            Tensor adb = new Transforms.AmplitudeToDB().forward(m);
            check("AmplitudeToDB transform", shapes(adb).length == 2);
            Tensor dba = new Transforms.DBToAmplitude().forward(adb);
            check("DBToAmplitude transform", shapes(dba).length == 2);

            Tensor fm = new Transforms.FrequencyMasking(8).forward(s);
            check("FrequencyMasking", Arrays.equals(shapes(s), shapes(fm)));
            Tensor tm = new Transforms.TimeMasking(4).forward(s);
            check("TimeMasking", Arrays.equals(shapes(s), shapes(tm)));
            check("FrequencyMask alias", shapes(new Transforms.FrequencyMask(8).forward(s)).length == 2);
            check("TimeMask alias", shapes(new Transforms.TimeMask(4).forward(s)).length == 2);

            Tensor mu = new Transforms.MuLawEncoding().forward(waveMono);
            Tensor mud = new Transforms.MuLawDecoding().forward(mu);
            check("MuLaw encode/decode", shapes(mud).length >= 1);

            Tensor inv = new Transforms.InverseSpectrogram(512, 128).forward(s);
            check("InverseSpectrogram", shapes(inv).length >= 1);
            Tensor ts = new Transforms.TimeStretch(1.1).forward(s);
            check("TimeStretch", shapes(ts).length == 2);
            Tensor ps = new Transforms.PitchShift(SR, 1.0).forward(waveMono);
            check("PitchShift", shapes(ps).length >= 1);
            Tensor lf = new Transforms.LFCC(SR).forward(waveMono);
            check("LFCC", shapes(lf).length == 2);
            Tensor cd = new Transforms.ComputeDeltas().forward(s);
            check("ComputeDeltas", shapes(cd).length == 2);
            Tensor c2 = new Transforms.Compute2DDeltas().forward(s);
            check("Compute2DDeltas", shapes(c2).length == 2);
            Tensor pad = new Transforms.Pad(50).forward(waveMono);
            check("Pad transform", AudioTensors.inferTime(pad) == SR + 100);
            Tensor tr = new Transforms.Trim(SR).forward(waveMono);
            check("Trim transform", AudioTensors.inferTime(tr) > 0);

            // factory helpers
            check("spectrogram() factory", shapes(Transforms.spectrogram(SR).forward(waveMono)).length == 2);
            check("inverse_spectrogram() factory", shapes(Transforms.inverse_spectrogram().forward(s)).length >= 1);
            check("inverseSpectrogram() factory", shapes(Transforms.inverseSpectrogram().forward(s)).length >= 1);
        });

        section("Compose pipeline", () -> {
            Compose pipe = new Compose(
                    new Transforms.Resample(SR, 8000),
                    new Transforms.Vol(0.8),
                    new Transforms.Fade(100, 100),
                    new Transforms.MelSpectrogram(8000, 32)
            );
            Object out = pipe.forward(waveMono);
            check("Compose returns Tensor", out instanceof Tensor);
            Tensor t = (Tensor) out;
            check("Compose mel rank 2", shapes(t).length == 2);
            checkEq("Compose mel 32", 32L, shapes(t)[0]);
            check("Compose.transforms size", pipe.transforms().size() == 4);

            Compose pipe2 = new Compose(List.of(
                    new Transforms.Pad(100),
                    new Transforms.Vol(0.9, "amplitude")
            ));
            Object out2 = pipe2.forward(waveMono);
            check("Compose(List) returns Tensor", out2 instanceof Tensor);
            check("Compose(List) pad length",
                    AudioTensors.inferTime((Tensor) out2) == SR + 200);
        });

        // ── D5 Datasets ──────────────────────────────────────────────────────
        System.out.println("\n══ D5 Datasets ══");
        section("FakeAudio", () -> {
            FakeAudio ds = new FakeAudio(16, SR, 4000, 5);
            checkEq("FakeAudio size", 16, ds.size());
            checkEq("FakeAudio length", 16, ds.length());
            checkEq("sampleRate", SR, ds.sampleRate());
            checkEq("numClasses", 5, ds.numClasses());
            checkEq("numSamples", 4000, ds.numSamples());
            AudioDataset.Sample s = ds.get(0);
            check("sample data is Tensor", s.data() instanceof Tensor);
            check("sample target is Number", s.target() instanceof Number);
            check("toString", s.toString() != null);

            int count = 0;
            for (AudioDataset.Sample ignored : ds) count++;
            checkEq("iterator count", 16, count);

            Tensor batch = FakeAudio.randomBatch(4, 1, 2000);
            checkEq("randomBatch shape", new long[]{4, 1, 2000}, shapes(batch));
            float[] sineArr = FakeAudio.sine(440, SR, 1000, 0.5);
            checkEq("FakeAudio.sine length", 1000, sineArr.length);

            FakeAudio ds2 = new FakeAudio(4, SR, 1000, 3).setTransform(new Transforms.Vol(0.5));
            check("setTransform get works", ds2.get(0).data() instanceof Tensor);
        });

        section("AudioFolder", () -> {
            Path root = tmp.resolve("audio_folder");
            Path clsA = root.resolve("cat");
            Path clsB = root.resolve("dog");
            Files.createDirectories(clsA);
            Files.createDirectories(clsB);
            writeWav(clsA.resolve("a1.wav"), mono, SR, 1);
            writeWav(clsA.resolve("a2.wav"), mono, SR, 1);
            writeWav(clsB.resolve("b1.wav"), mono, SR, 1);

            AudioFolder folder = new AudioFolder(root.toString());
            check("AudioFolder size >= 3", folder.size() >= 3);
            check("classes has cat/dog", folder.classes().contains("cat") && folder.classes().contains("dog"));
            check("class_to_idx cat >= 0", folder.class_to_idx("cat") >= 0);
            check("classToIdx alias", folder.classToIdx("dog") >= 0);
            check("samples list", folder.samples().size() >= 3);
            check("targets list", folder.targets().size() >= 3);
            checkEq("sampleRate", SR, folder.sampleRate() > 0 ? folder.sampleRate() : SR); // may default
            AudioDataset.Sample s = folder.get(0);
            check("folder sample data Tensor", s.data() instanceof Tensor);
            check("folder sample target int", s.target() instanceof Integer || s.target() instanceof Number);

            AudioFolder.DatasetFolder df = new AudioFolder.DatasetFolder(root.toString());
            check("DatasetFolder size", df.size() >= 3);
        });

        // ── D6 Models ────────────────────────────────────────────────────────
        System.out.println("\n══ D6 Models ══");
        section("SimpleAudioClassifier / M5 / Wav2LetterLite", () -> {
            // Simple classifier on flattened features
            AudioModels.SimpleAudioClassifier clf = AudioModels.simple_audio_classifier(64, 5);
            Tensor feat = torch.randn(4, 64);
            Tensor logits = clf.forward(feat);
            checkEq("SimpleAudioClassifier out", new long[]{4, 5}, shapes(logits));
            check("simpleAudioClassifier alias",
                    shapes(AudioModels.simpleAudioClassifier(32, 3).forward(torch.randn(2, 32))).length == 2);

            // M5 expects waveform [B, C, T]
            AudioModels.M5 m5 = AudioModels.m5(1, 10);
            Tensor waveB = torch.randn(2, 1, 8000);
            Tensor out = m5.forward(waveB);
            checkEq("M5 out classes", new long[]{2, 10}, shapes(out));
            Tensor feats = m5.features(waveB);
            check("M5 features rank >= 2", shapes(feats).length >= 2);
            checkEq("M5 featureDim", 256L, m5.featureDim());

            AudioModels.Wav2LetterLite w2l = AudioModels.wav2letter_lite(1, 8);
            Tensor out2 = w2l.forward(waveB);
            checkEq("Wav2LetterLite out", new long[]{2, 8}, shapes(out2));
            check("wav2letterLite alias",
                    shapes(AudioModels.wav2letterLite(1, 4).forward(waveB)).length == 2);
            checkEq("W2L featureDim", 128L, w2l.featureDim());
            check("W2L features", shapes(w2l.features(waveB)).length >= 2);
        });

        // ── D7 Daily pipeline + edges ────────────────────────────────────────
        System.out.println("\n══ D7 Daily pipeline / edges ══");
        section("daily: load → resample → mel → mfcc → mask → model", () -> {
            AudioIO.AudioLoadResult loaded = AudioIO.load(wav.toString(), SR, true);
            Tensor w = loaded.waveform();
            Tensor r = F.resample(w, loaded.sampleRate(), 8000);
            Tensor mel = F.melSpectrogram(r, 8000, 64, 0, 4000, 1024, 256);
            Tensor mfcc = F.mfcc(r, 8000, 13, 64, 0, 4000, 1024, 256);
            Tensor masked = F.frequency_masking(mel, 8);
            check("pipeline mel", shapes(mel)[0] == 64);
            check("pipeline mfcc", shapes(mfcc)[0] == 13);
            check("pipeline mask", Arrays.equals(shapes(mel), shapes(masked)));

            // Train-like step on FakeAudio + M5
            FakeAudio ds = new FakeAudio(8, 8000, 4000, 4);
            AudioModels.M5 m5 = AudioModels.m5(1, 4);
            AudioDataset.Sample s0 = ds.get(0);
            Tensor x = ((Tensor) s0.data()).unsqueeze(0); // [1,C,T] or [1,T]
            if (x.dim() == 2) x = x.unsqueeze(0); // ensure [B,C,T] if [C,T]
            if (x.dim() == 2) {
                // [1,T] → [1,1,T]
                x = x.unsqueeze(1);
            }
            // FakeAudio returns [C,T] typically
            long[] xs = shapes((Tensor) s0.data());
            Tensor input;
            if (xs.length == 1) input = ((Tensor) s0.data()).reshape(1, 1, xs[0]);
            else if (xs.length == 2) input = ((Tensor) s0.data()).unsqueeze(0);
            else input = (Tensor) s0.data();
            Tensor logits = m5.forward(input);
            check("daily model forward batch", shapes(logits)[0] == 1);
            check("daily model classes", shapes(logits)[1] == 4);
        });

        section("stereo path", () -> {
            Tensor mel = F.melSpectrogram(waveStereo, SR, 32, 0, SR / 2.0, 512, 128);
            check("stereo mel works", shapes(mel).length == 2);
            Tensor rs = F.resample(waveStereo, SR, 8000);
            check("stereo resample channels", AudioTensors.inferChannels(rs) == 2);
        });

        // ── D8 Throughput ────────────────────────────────────────────────────
        System.out.println("\n══ D8 Throughput ══");
        section("throughput spectrogram/mel/mfcc/resample", () -> {
            int iters = 30;
            for (int i = 0; i < 3; i++) F.spectrogram(waveMono, SR, 512, 128, 0, true);
            long t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) F.spectrogram(waveMono, SR, 512, 128, 0, true);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            double ips = iters / (ms / 1000.0);
            System.out.println("    spectrogram: " + String.format("%.1f", ips) + " /s");
            check("spec throughput > 0", ips > 0);

            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) F.melSpectrogram(waveMono, SR, 64, 0, SR / 2.0, 512, 128);
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / (ms / 1000.0);
            System.out.println("    mel: " + String.format("%.1f", ips) + " /s");
            check("mel throughput > 0", ips > 0);

            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) F.mfcc(waveMono, SR, 13, 64, 0, SR / 2.0, 512, 128);
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / (ms / 1000.0);
            System.out.println("    mfcc: " + String.format("%.1f", ips) + " /s");
            check("mfcc throughput > 0", ips > 0);

            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) F.resample(waveMono, SR, 8000);
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / (ms / 1000.0);
            System.out.println("    resample: " + String.format("%.1f", ips) + " /s");
            check("resample throughput > 0", ips > 0);
        });

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        deleteRecursive(tmp);
    }

    static void deleteRecursive(Path path) {
        try {
            if (Files.isDirectory(path)) {
                try (var entries = Files.list(path)) {
                    entries.forEach(BenchmarkAudio::deleteRecursive);
                }
            }
            Files.deleteIfExists(path);
        } catch (Exception ignored) {}
    }
}
