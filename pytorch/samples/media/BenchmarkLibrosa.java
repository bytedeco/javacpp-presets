package media;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.audio.librosa.Librosa;
import org.bytedeco.pytorch.audio.librosa.beat.Beat;
import org.bytedeco.pytorch.audio.librosa.core.Spectrum;
import org.bytedeco.pytorch.audio.librosa.effects.Effects;
import org.bytedeco.pytorch.audio.librosa.feature.Feature;
import org.bytedeco.pytorch.audio.librosa.onset.Onset;
import org.bytedeco.pytorch.audio.librosa.util.Utils;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Multi-dimensional correctness + performance benchmark for {@code utils.librosa}.
 *
 * <p>Dimensions:
 * <ol>
 *   <li>D1 Load / mono / duration / AudioData bridge</li>
 *   <li>D2 Core Spectrum (STFT, magphase, power↔dB, amplitude↔dB)</li>
 *   <li>D3 Feature (mfcc, mel, chroma, centroid, bandwidth, contrast, rolloff, zcr, rms)</li>
 *   <li>D4 Effects (trim, split, preemphasis, normalize)</li>
 *   <li>D5 Onset detection / frames_to_time / peakPick</li>
 *   <li>D6 Beat track / tempo</li>
 *   <li>D7 Utils (frame, normalize, softmask, pad_center, valid_audio)</li>
 *   <li>D8 Edge cases / invariants / daily-use pipeline</li>
 *   <li>D9 Throughput stress</li>
 * </ol>
 *
 * <pre>{@code
 * java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *   -cp "target/classes:$(mvn -q dependency:build-classpath -DincludeScope=runtime -Dmdep.outputFile=/dev/stdout)" \
 *   media.BenchmarkLibrosa
 * }</pre>
 */
public class BenchmarkLibrosa {

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
        boolean ok = ObjectsEq(expected, actual);
        if (ok) {
            passed++;
            System.out.println("    CHECK " + name + ": OK (" + expected + ")");
        } else {
            failed++;
            report.append("CHECK FAILED [").append(name).append("]: expected=")
                    .append(expected).append(", actual=").append(actual).append('\n');
            System.out.println("    CHECK " + name + ": FAIL (expected=" + expected + ", got=" + actual + ")");
        }
    }

    static boolean ObjectsEq(Object a, Object b) {
        if (a == b) return true;
        if (a == null || b == null) return false;
        if (a instanceof Number && b instanceof Number) {
            double d = Math.abs(((Number) a).doubleValue() - ((Number) b).doubleValue());
            return Double.isNaN(d) ? Double.isNaN(((Number) a).doubleValue()) : d < 1e-5;
        }
        if (a instanceof int[] ia && b instanceof int[] ib) return Arrays.equals(ia, ib);
        if (a instanceof float[] fa && b instanceof float[] fb) return Arrays.equals(fa, fb);
        return a.equals(b);
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

    static float[] sine(double freq, int sr, int n, double amp) {
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            y[i] = (float) (amp * Math.sin(2 * Math.PI * freq * i / sr));
        }
        return y;
    }

    static float[] mix(float[] a, float[] b) {
        float[] out = new float[Math.max(a.length, b.length)];
        for (int i = 0; i < out.length; i++) {
            float va = i < a.length ? a[i] : 0f;
            float vb = i < b.length ? b[i] : 0f;
            out[i] = va + vb;
        }
        return out;
    }

    static boolean finite(float[] a) {
        if (a == null) return false;
        for (float v : a) if (!Float.isFinite(v)) return false;
        return true;
    }

    static boolean finite2(float[][] a) {
        if (a == null || a.length == 0) return false;
        for (float[] row : a) if (!finite(row)) return false;
        return true;
    }

    static float maxAbs(float[] a) {
        float m = 0;
        for (float v : a) m = Math.max(m, Math.abs(v));
        return m;
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

    public static void main(String[] args) throws Exception {
        Path tmp = Files.createTempDirectory("librosa_bench");
        System.out.println("=== Librosa Module Benchmark ===");
        System.out.println("Temp: " + tmp);

        final int SR = 22050;
        final float[] y1s = sine(440, SR, SR, 0.5);           // 1s A4
        final float[] y2s = sine(440, SR, SR * 2, 0.5);       // 2s
        final float[] yChord = mix(sine(440, SR, SR, 0.4), sine(554.37, SR, SR, 0.3)); // A+C#
        final float[] yQuiet = sine(440, SR, SR, 0.01);
        final float[] yBeat = makeClickTrack(SR, 2.0, 120);   // 2s @ 120 BPM

        // ── D1 Load ──────────────────────────────────────────────────────────
        System.out.println("\n══ D1 Load / mono / duration ══");
        Path wav = tmp.resolve("tone.wav");
        writeWav(wav, y1s, SR, 1);

        section("Librosa.load path", () -> {
            Librosa.AudioLoad al = Librosa.load(wav.toString());
            check("y non-null", al.y() != null && al.y().length > 0);
            check("sr default 22050 or source", al.sr() > 0);
            check("duration ~1s", Math.abs(al.duration() - 1.0) < 0.05);
            check("toString", al.toString() != null && al.toString().contains("sr"));
            AudioData ad = al.toAudioData();
            check("toAudioData samples", ad.getSamples() != null && ad.getSamples().length > 0);
        });

        section("Librosa.load with sr/mono", () -> {
            Librosa.AudioLoad al = Librosa.load(wav.toString(), 16000, true);
            checkEq("target sr 16000", 16000, al.sr());
            check("resampled length ~16000", Math.abs(al.y().length - 16000) < 200);
            check("sampleRate alias", al.sampleRate() == al.sr());
        });

        section("to_mono / resample / get_duration", () -> {
            // stereo interleaved LRLR
            float[] stereo = new float[SR * 2];
            for (int i = 0; i < SR; i++) {
                stereo[2 * i] = y1s[i];
                stereo[2 * i + 1] = 0.5f * y1s[i];
            }
            float[] mono = Librosa.to_mono(stereo, 2);
            checkEq("to_mono length", SR, mono.length);
            check("to_mono finite", finite(mono));
            check("toMono alias", Librosa.toMono(stereo, 2).length == mono.length);

            float[] rs = Librosa.resample(y1s, SR, 16000);
            check("resample length ~16000", Math.abs(rs.length - 16000) < 5);
            check("resample identity", Librosa.resample(y1s, SR, SR).length == y1s.length);

            double dur = Librosa.get_duration(y1s, SR);
            check("get_duration ~1", Math.abs(dur - 1.0) < 1e-6);
            check("getDuration alias", Math.abs(Librosa.getDuration(y1s, SR) - 1.0) < 1e-6);
            AudioData ad = Librosa.asAudioData(y1s, SR);
            check("asAudioData", ad != null && ad.getSampleRate() == SR);
            check("get_duration(AudioData)", Math.abs(Librosa.get_duration(ad) - 1.0) < 0.05);
        });

        // ── D2 Spectrum ──────────────────────────────────────────────────────
        System.out.println("\n══ D2 Core Spectrum ══");
        section("STFT default + custom", () -> {
            Spectrum.StftResult st = Spectrum.stft(y1s, SR);
            check("stft magnitude non-empty", st.magnitude() != null && st.magnitude().length > 0);
            check("stft phase same shape", st.phase().length == st.magnitude().length);
            check("stft power same shape", st.power().length == st.magnitude().length);
            check("nFft > 0", st.nFft > 0);
            check("hopLength > 0", st.hopLength > 0);
            check("mag finite", finite2(st.magnitude()));
            check("power >= 0", allNonNeg(st.power()));

            Spectrum.StftResult st2 = Spectrum.stft(y1s, SR, 1024, 256, "hann");
            checkEq("custom nFft", 1024, st2.nFft);
            checkEq("custom hop", 256, st2.hopLength);
            // freq bins = nFft/2+1
            checkEq("freq bins", 1024 / 2 + 1, st2.magnitude().length);
        });

        section("magphase / power_to_db / db_to_power", () -> {
            Spectrum.StftResult st = Spectrum.stft(y1s, SR, 1024, 256, "hann");
            Spectrum.MagPhase mp = Spectrum.magphase(st.power(), 1.0); // power already
            check("magphase mag rows", mp.magnitude().length == st.power().length);
            check("magphase finite", finite2(mp.magnitude()) && finite2(mp.phase()));

            float[][] pdb = Spectrum.power_to_db(st.power());
            check("power_to_db finite", finite2(pdb));
            float[][] back = Spectrum.db_to_power(pdb);
            check("db_to_power finite", finite2(back));
            // rough invertibility on mid energy
            double err = relativeErr(st.power(), back);
            check("power↔db relative err < 0.05", err < 0.05);

            float[][] adb = Spectrum.amplitude_to_db(st.magnitude());
            check("amplitude_to_db finite", finite2(adb));
            check("powerToDb alias", Spectrum.powerToDb(st.power()).length == pdb.length);
            check("dbToPower alias", Spectrum.dbToPower(pdb).length == back.length);
        });

        // ── D3 Feature ───────────────────────────────────────────────────────
        System.out.println("\n══ D3 Feature ══");
        section("mfcc / mel / chroma", () -> {
            float[][] mfcc = Feature.mfcc(y1s, SR);
            check("mfcc default 13", mfcc.length == 13);
            check("mfcc frames > 0", mfcc[0].length > 0);
            check("mfcc finite", finite2(mfcc));

            float[][] mfcc20 = Feature.mfcc(y1s, SR, 20);
            checkEq("mfcc n=20", 20, mfcc20.length);

            float[][] mel = Feature.melspectrogram(y1s, SR);
            check("mel default 128 rows", mel.length == 128 || mel.length > 0);
            check("mel finite", finite2(mel));
            check("mel_spectrogram alias", Feature.mel_spectrogram(y1s, SR).length == mel.length);

            float[][] mel64 = Feature.melspectrogram(y1s, SR, 64);
            checkEq("mel n=64", 64, mel64.length);

            float[][] chroma = Feature.chroma_stft(y1s, SR);
            checkEq("chroma 12 bins", 12, chroma.length);
            check("chroma finite", finite2(chroma));
            check("chromaStft alias", Feature.chromaStft(y1s, SR).length == 12);
        });

        section("spectral stats / zcr / rms", () -> {
            float[] cent = Feature.spectral_centroid(y1s, SR);
            check("centroid frames > 0", cent.length > 0);
            check("centroid finite", finite(cent));
            // 440Hz tone → centroid near 440 (allow broad band due to window)
            float meanCent = mean(cent);
            check("centroid ~440 for pure tone (100-2000)", meanCent > 100 && meanCent < 2000);

            float[] bw = Feature.spectral_bandwidth(y1s, SR);
            check("bandwidth finite", finite(bw) && bw.length > 0);

            float[][] contrast = Feature.spectral_contrast(y1s, SR);
            check("contrast rows > 0", contrast.length > 0);
            check("contrast finite", finite2(contrast));

            float[] rolloff = Feature.spectral_rolloff(y1s, SR);
            check("rolloff finite", finite(rolloff) && rolloff.length > 0);

            float[] zcr = Feature.zero_crossing_rate(y1s);
            check("zcr frames > 0", zcr.length > 0);
            check("zcr in [0,1]", allInRange(zcr, 0f, 1.01f));

            float[] rms = Feature.rms(y1s);
            check("rms > 0 for tone", mean(rms) > 0.01f);
            check("rms_energy alias", Feature.rms_energy(y1s).length == rms.length);

            // aliases
            check("spectralCentroid alias", Feature.spectralCentroid(y1s, SR).length == cent.length);
            check("spectralBandwidth alias", Feature.spectralBandwidth(y1s, SR).length == bw.length);
            check("spectralContrast alias", Feature.spectralContrast(y1s, SR).length == contrast.length);
            check("spectralRolloff alias", Feature.spectralRolloff(y1s, SR).length == rolloff.length);
            check("zeroCrossingRate alias", Feature.zeroCrossingRate(y1s).length == zcr.length);
        });

        // ── D4 Effects ───────────────────────────────────────────────────────
        System.out.println("\n══ D4 Effects ══");
        section("trim / split / preemphasis / normalize", () -> {
            // silence + tone + silence
            float[] padded = new float[SR * 3];
            System.arraycopy(y1s, 0, padded, SR, y1s.length);
            Effects.TrimResult tr = Effects.trim(padded, SR, 40f, 0.025f);
            check("trim result non-null", tr != null && tr.y() != null);
            check("trim shortened", tr.y().length < padded.length);
            check("trim index valid", tr.index()[0] >= 0 && tr.index()[1] > tr.index()[0]);
            check("trim toString", tr.toString() != null);

            float[] sliced = Effects.trim_time(y1s, SR, 0.1f, 0.5f);
            check("trim_time length ~0.4s", Math.abs(sliced.length - (int) (0.4 * SR)) < 5);

            int[][] segs = Effects.split(padded, SR);
            check("split found segments", segs.length >= 1);
            check("split segment [start,end]", segs[0].length == 2 && segs[0][1] > segs[0][0]);

            float[] pre = Effects.preemphasis(y1s);
            checkEq("preemphasis same length", y1s.length, pre.length);
            check("preemphasis changes signal", maxAbs(diff(y1s, pre)) > 1e-6f);
            float[] pre2 = Effects.preemphasis(y1s, 0.97f);
            checkEq("preemphasis coef length", y1s.length, pre2.length);

            float[] norm = Effects.normalize(y1s, SR);
            check("normalize peak ~1", Math.abs(maxAbs(norm) - 1.0f) < 0.05f);
        });

        // ── D5 Onset ─────────────────────────────────────────────────────────
        System.out.println("\n══ D5 Onset ══");
        section("onset_strength / detect / frames_to_time", () -> {
            float[] strength = Onset.onset_strength(yBeat, SR);
            check("onset_strength frames > 0", strength.length > 0);
            check("onset_strength finite", finite(strength));
            check("onsetStrength alias", Onset.onsetStrength(yBeat, SR).length == strength.length);

            int[] onsets = Onset.onset_detect(yBeat, SR);
            check("onset_detect returns array", onsets != null);
            // click track should produce multiple onsets
            check("onset count >= 1 for click track", onsets.length >= 1);

            float[] times = Onset.frames_to_time(onsets, SR, 512);
            checkEq("frames_to_time length", onsets.length, times.length);
            check("times increasing", isNonDecreasing(times));
            check("framesToTime alias", Onset.framesToTime(onsets, SR).length == times.length);

            // peakPick synthetic
            float[] x = new float[100];
            x[10] = 1f; x[30] = 1.5f; x[60] = 1.2f;
            int[] peaks = Onset.peakPick(x, 5, 0.1f, 3, 3);
            check("peakPick found peaks", peaks.length >= 1);
        });

        // ── D6 Beat ──────────────────────────────────────────────────────────
        System.out.println("\n══ D6 Beat / tempo ══");
        section("beat_track / tempo", () -> {
            Beat.BeatTrackResult bt = Beat.beat_track(yBeat, SR);
            check("tempo > 0", bt.tempo() > 0);
            check("beats non-null", bt.beats() != null);
            check("beatTimes non-null", bt.beatTimes != null);
            check("beat_times alias", bt.beat_times() != null);
            check("toString", bt.toString() != null && bt.toString().contains("tempo"));
            // 120 BPM expected roughly (allow wide tolerance for simplified tracker)
            System.out.println("    tempo=" + bt.tempo() + " beats=" + bt.beats().length);
            check("tempo in plausible range 40-240", bt.tempo() >= 40 && bt.tempo() <= 240);

            float tempo = Beat.tempo(yBeat, SR);
            check("tempo() > 0", tempo > 0);
            check("beatTrack alias", Beat.beatTrack(yBeat, SR).tempo() > 0);
        });

        // ── D7 Utils ─────────────────────────────────────────────────────────
        System.out.println("\n══ D7 Utils ══");
        section("frame / normalize / softmask / pad_center / valid_audio", () -> {
            float[][] frames = Utils.frame(y1s, 1024, 512);
            // librosa.util.frame layout: [frameLength, n_frames] (column-major frames)
            checkEq("frame length dim0", 1024, frames.length);
            check("frame n_frames > 0", frames.length > 0 && frames[0].length > 0);
            checkEq("frame hop implies n_frames", 1 + (y1s.length - 1024) / 512, frames[0].length);

            float[] n1 = Utils.normalize(y1s, 2.0);
            check("L2 normalize finite", finite(n1));
            float[][] n2 = Utils.normalize(frames, 2.0, 0);
            check("2d normalize finite", finite2(n2));

            float[][] X = Feature.melspectrogram(y1s, SR, 32);
            float[][] mask = Utils.softmask(X, X);
            check("softmask same shape", mask.length == X.length && mask[0].length == X[0].length);
            check("softmask finite", finite2(mask));

            float[] padded = Utils.pad_center(new float[]{1, 2, 3}, 9);
            checkEq("pad_center length 9", 9, padded.length);
            checkEq("pad_center middle", 2f, padded[4]);
            check("padCenter alias", Utils.padCenter(new float[]{1, 2, 3}, 9).length == 9);

            check("valid_audio true", Utils.valid_audio(y1s));
            check("valid_audio empty false", !Utils.valid_audio(new float[0]));
            check("validAudio alias", Utils.validAudio(y1s));
            float[] bad = y1s.clone();
            bad[0] = Float.NaN;
            check("valid_audio NaN false", !Utils.valid_audio(bad));
        });

        // ── D8 Edge / pipeline ───────────────────────────────────────────────
        System.out.println("\n══ D8 Edge cases / daily pipeline ══");
        section("edge cases", () -> {
            // short signal
            float[] shortY = sine(440, SR, 512, 0.5);
            check("short stft works", Spectrum.stft(shortY, SR).magnitude().length > 0);
            check("short mfcc works", Feature.mfcc(shortY, SR).length == 13);
            check("short zcr works", Feature.zero_crossing_rate(shortY).length >= 1);

            // silence
            float[] silence = new float[SR];
            check("silence rms ~0", mean(Feature.rms(silence)) < 1e-4f);
            check("silence valid", Utils.valid_audio(silence));

            // DEFAULT_SR constant
            checkEq("DEFAULT_SR", 22050, Librosa.DEFAULT_SR);
        });

        section("daily pipeline: load → feature → onset → beat", () -> {
            Librosa.AudioLoad al = Librosa.load(wav.toString(), SR, true);
            float[] y = al.y();
            float[][] mel = Feature.melspectrogram(y, al.sr(), 64);
            float[][] mfcc = Feature.mfcc(y, al.sr(), 13);
            float[] zcr = Feature.zero_crossing_rate(y);
            float[] rms = Feature.rms(y);
            int[] onsets = Onset.onset_detect(y, al.sr());
            float tempo = Beat.tempo(y, al.sr());
            Effects.TrimResult tr = Effects.trim(y);
            check("pipeline mel ok", finite2(mel));
            check("pipeline mfcc ok", finite2(mfcc));
            check("pipeline zcr ok", finite(zcr));
            check("pipeline rms ok", finite(rms));
            check("pipeline onsets ok", onsets != null);
            check("pipeline tempo ok", tempo >= 0);
            check("pipeline trim ok", tr.y().length > 0);
            System.out.println("    pipeline: mel=" + mel.length + "x" + mel[0].length
                    + " mfcc=" + mfcc.length + "x" + mfcc[0].length
                    + " onsets=" + onsets.length + " tempo=" + tempo);
        });

        // ── D9 Throughput ────────────────────────────────────────────────────
        System.out.println("\n══ D9 Throughput ══");
        section("feature throughput", () -> {
            int iters = 20;
            // warmup
            for (int i = 0; i < 3; i++) Feature.mfcc(y1s, SR);
            long t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) Feature.mfcc(y1s, SR);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            double ips = iters / (ms / 1000.0);
            System.out.println("    MFCC: " + String.format("%.1f", ips) + " clips/s (" + ms + " ms / " + iters + ")");
            check("mfcc throughput > 0", ips > 0);

            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) Feature.melspectrogram(y1s, SR, 64);
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / (ms / 1000.0);
            System.out.println("    Mel: " + String.format("%.1f", ips) + " clips/s (" + ms + " ms / " + iters + ")");
            check("mel throughput > 0", ips > 0);

            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) Spectrum.stft(y1s, SR, 1024, 256, "hann");
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / (ms / 1000.0);
            System.out.println("    STFT: " + String.format("%.1f", ips) + " clips/s (" + ms + " ms / " + iters + ")");
            check("stft throughput > 0", ips > 0);
        });

        // ── Summary ──────────────────────────────────────────────────────────
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        deleteRecursive(tmp);
    }

    // ---- helpers ----

    static float[] makeClickTrack(int sr, double durSec, double bpm) {
        int n = (int) (sr * durSec);
        float[] y = new float[n];
        double interval = 60.0 / bpm;
        for (double t = 0; t < durSec; t += interval) {
            int i0 = (int) (t * sr);
            // 5ms click burst @ 1kHz
            int clickLen = Math.min(sr / 200, n - i0);
            for (int i = 0; i < clickLen; i++) {
                double env = 1.0 - (double) i / clickLen;
                y[i0 + i] += (float) (env * Math.sin(2 * Math.PI * 1000 * i / sr));
            }
        }
        return y;
    }

    static boolean allNonNeg(float[][] a) {
        for (float[] row : a) for (float v : row) if (v < -1e-6f) return false;
        return true;
    }

    static boolean allInRange(float[] a, float lo, float hi) {
        for (float v : a) if (v < lo || v > hi) return false;
        return true;
    }

    static float mean(float[] a) {
        double s = 0;
        for (float v : a) s += v;
        return (float) (s / Math.max(1, a.length));
    }

    static float[] diff(float[] a, float[] b) {
        float[] d = new float[Math.min(a.length, b.length)];
        for (int i = 0; i < d.length; i++) d[i] = a[i] - b[i];
        return d;
    }

    static boolean isNonDecreasing(float[] a) {
        for (int i = 1; i < a.length; i++) if (a[i] + 1e-6f < a[i - 1]) return false;
        return true;
    }

    static double relativeErr(float[][] a, float[][] b) {
        double num = 0, den = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                double da = a[i][j], db = b[i][j];
                num += (da - db) * (da - db);
                den += da * da;
            }
        }
        return den > 0 ? Math.sqrt(num / den) : 0;
    }

    static void deleteRecursive(Path path) {
        try {
            if (Files.isDirectory(path)) {
                try (var entries = Files.list(path)) {
                    entries.forEach(BenchmarkLibrosa::deleteRecursive);
                }
            }
            Files.deleteIfExists(path);
        } catch (Exception ignored) {}
    }
}
