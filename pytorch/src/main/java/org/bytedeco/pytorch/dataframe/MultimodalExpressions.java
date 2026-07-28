package org.bytedeco.pytorch.dataframe;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.*;
import javax.imageio.ImageIO;

import org.bytedeco.pytorch.dataframe.ai.AiFunctions;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.BinaryData;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.TensorData;
import org.bytedeco.pytorch.dataframe.dtype.VectorData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;

/**
 * Daft-style multimodal expression namespaces and nodes:
 * {@code col("img").image().resize(w,h)}, {@code col("a").audio().mfcc()},
 * {@code col("t").tensor().l2Norm()}, {@code col("s").text().clean()}, …
 *
 * <p>Delegates to existing dtype implementations ({@link ImageData}, {@link AudioData},
 * {@link VideoData}, {@link TensorData}, {@link EmbeddingData}, {@link VectorData}).
 */
public final class MultimodalExpressions {
    private MultimodalExpressions() {}

    // ================================================================
    // Shared unwrap helpers
    // ================================================================

    static ImageData asImage(Object v) {
        if (v == null) return null;
        if (v instanceof ImageData id) return id;
        if (v instanceof BufferedImage bi) return new ImageData(bi);
        if (v instanceof byte[] bytes) return new ImageData(bytes);
        if (v instanceof String path) {
            try { return ImageData.load(path); } catch (Exception e) { return new ImageData(path); }
        }
        return null;
    }

    static AudioData asAudio(Object v) {
        if (v == null) return null;
        if (v instanceof AudioData ad) return ad;
        if (v instanceof String path) {
            try { return AudioData.loadFromFile(path); } catch (Exception e) { return new AudioData(path); }
        }
        if (v instanceof float[] samples) return new AudioData(samples, 16000, 1);
        return null;
    }

    static VideoData asVideo(Object v) {
        if (v == null) return null;
        if (v instanceof VideoData vd) return vd;
        if (v instanceof String path) return new VideoData(path);
        if (v instanceof List<?> list) {
            List<ImageData> frames = new ArrayList<>();
            for (Object o : list) {
                ImageData id = asImage(o);
                if (id != null) frames.add(id);
            }
            return new VideoData(frames, 30.0);
        }
        return null;
    }

    static float[] asFloatVector(Object v) {
        if (v == null) return null;
        if (v instanceof float[] f) return f;
        if (v instanceof double[] d) {
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return f;
        }
        if (v instanceof EmbeddingData ed) return ed.getVector();
        if (v instanceof VectorData vd) return vectorDataToFloat(vd);
        if (v instanceof TensorData td) return td.getData();
        if (v instanceof Number n) return new float[]{n.floatValue()};
        if (v instanceof List<?> list) {
            float[] out = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object o = list.get(i);
                out[i] = o instanceof Number ? ((Number) o).floatValue() : 0f;
            }
            return out;
        }
        return null;
    }

    static float[] vectorDataToFloat(VectorData vd) {
        if (vd == null) return null;
        try {
            double[] data = vd.getAsDoubleArray();
            if (data == null) return null;
            float[] f = new float[data.length];
            for (int i = 0; i < data.length; i++) f[i] = (float) data[i];
            return f;
        } catch (Exception e) {
            return null;
        }
    }

    static int product(int[] shape) {
        int p = 1;
        if (shape != null) for (int s : shape) p *= s;
        return p;
    }

    static TensorData asTensor(Object v) {
        if (v == null) return null;
        if (v instanceof TensorData td) return td;
        if (v instanceof float[] f) return new TensorData(f, new int[]{f.length});
        if (v instanceof double[] d) return new TensorData(d, new int[]{d.length});
        if (v instanceof EmbeddingData ed) return new TensorData(ed.getVector(), new int[]{ed.getDimension()});
        if (v instanceof VectorData vd) {
            float[] f = vectorDataToFloat(vd);
            if (f != null) return new TensorData(f, new int[]{f.length});
        }
        if (v instanceof ImageData id) {
            float[] arr = imageToFloatArray(id);
            if (arr != null) {
                int h = id.getHeight(), w = id.getWidth(), c = Math.max(1, id.getChannels());
                return new TensorData(arr, new int[]{h, w, c});
            }
        }
        return null;
    }

    static float[] imageToFloatArray(ImageData id) {
        if (id == null) return null;
        BufferedImage img = id.getImage();
        if (img == null) return null;
        int w = img.getWidth(), h = img.getHeight();
        float[] out = new float[h * w * 3];
        int k = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                out[k++] = ((rgb >> 16) & 0xFF) / 255f;
                out[k++] = ((rgb >> 8) & 0xFF) / 255f;
                out[k++] = (rgb & 0xFF) / 255f;
            }
        }
        return out;
    }

    static String asText(Object v) {
        if (v == null) return null;
        return v.toString();
    }

    // ================================================================
    // Image namespace
    // ================================================================

    public static final class ImageNameSpace {
        private final Expression parent;
        ImageNameSpace(Expression parent) { this.parent = parent; }

        public Expression decode() { return new ImageUnaryExpr(parent, ImageOp.DECODE); }
        public Expression resize(int w, int h) { return new ImageResizeExpr(parent, w, h); }
        public Expression crop(int x, int y, int w, int h) { return new ImageCropExpr(parent, x, y, w, h); }
        public Expression pad(int top, int right, int bottom, int left) {
            return new ImagePadExpr(parent, top, right, bottom, left, 0);
        }
        public Expression pad(int top, int right, int bottom, int left, int color) {
            return new ImagePadExpr(parent, top, right, bottom, left, color);
        }
        public Expression rotate(double angleDegrees) { return new ImageRotateExpr(parent, angleDegrees); }
        public Expression flip(String axis) {
            boolean horizontal = axis == null || !axis.toLowerCase(Locale.ROOT).startsWith("v");
            return new ImageFlipExpr(parent, horizontal);
        }
        public Expression flipHorizontal() { return flip("horizontal"); }
        public Expression flipVertical() { return flip("vertical"); }
        public Expression toGrayscale() { return new ImageUnaryExpr(parent, ImageOp.GRAYSCALE); }
        public Expression normalize(float[] mean, float[] std) {
            return new ImageNormalizeExpr(parent, mean, std);
        }
        public Expression denoise() { return new ImageUnaryExpr(parent, ImageOp.DENOISE); }
        public Expression sharpen() { return new ImageUnaryExpr(parent, ImageOp.SHARPEN); }
        public Expression blur() { return new ImageUnaryExpr(parent, ImageOp.BLUR); }
        public Expression blur(int kernelSize) { return new ImageUnaryExpr(parent, ImageOp.BLUR); }
        public Expression equalizeHist() { return new ImageUnaryExpr(parent, ImageOp.EQUALIZE); }
        public Expression toArray() { return new ImageUnaryExpr(parent, ImageOp.TO_ARRAY); }
        public Expression toArray(String dtype) { return toArray(); }
        public Expression info() { return new ImageUnaryExpr(parent, ImageOp.INFO); }
        public Expression encode() { return encode("JPEG", 85); }
        public Expression encode(String format, int quality) {
            return new ImageEncodeExpr(parent, format == null ? "JPEG" : format, quality);
        }
        public Expression toEmbedding() { return toEmbedding(256); }
        public Expression toEmbedding(int dim) { return new ImageEmbedExpr(parent, dim); }
        /** Embed via registered model (e.g. {@code "clip-vit-base-patch32"}). */
        public Expression toEmbedding(String modelId) {
            return AiFunctions.embedImage(parent, modelId);
        }
        public Expression toEmbedding(String modelId, int dim) {
            // dim hint — resolve model then ensure; for hash models dim is in ModelSpec
            return AiFunctions.embedImage(parent, modelId);
        }
        public Expression phash() { return new ImageUnaryExpr(parent, ImageOp.PHASH); }
    }

    enum ImageOp { DECODE, GRAYSCALE, DENOISE, SHARPEN, BLUR, EQUALIZE, TO_ARRAY, INFO, PHASH }

    static final class ImageUnaryExpr extends Expression {
        final Expression child;
        final ImageOp op;
        ImageUnaryExpr(Expression child, ImageOp op) { this.child = child; this.op = op; }
        @Override public Object eval(int row, DataFrame df) {
            Object raw = child.eval(row, df);
            ImageData img = asImage(raw);
            if (img == null && op != ImageOp.DECODE) return null;
            try {
                return switch (op) {
                    case DECODE -> img != null ? img : asImage(raw);
                    case GRAYSCALE -> img.toGrayscale();
                    case DENOISE -> img.medianFilter(3);
                    case SHARPEN -> img.sharpen();
                    case BLUR -> img.gaussianBlur();
                    case EQUALIZE -> img.equalizeHistogram();
                    case TO_ARRAY -> imageToFloatArray(img);
                    case INFO -> {
                        Map<String, Object> m = new LinkedHashMap<>();
                        m.put("width", img.getWidth());
                        m.put("height", img.getHeight());
                        m.put("channels", img.getChannels());
                        m.put("format", img.getFormat());
                        yield m;
                    }
                    case PHASH -> simplePHash(img);
                };
            } catch (Exception e) {
                return null;
            }
        }
        @Override public String suggestedName() { return "image_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static String simplePHash(ImageData img) {
        if (img == null || img.getImage() == null) return null;
        ImageData g = img.resize(8, 8).toGrayscale();
        BufferedImage bi = g.getImage();
        long hash = 0;
        long sum = 0;
        int[] pix = new int[64];
        for (int i = 0, y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++, i++) {
                int rgb = bi.getRGB(x, y) & 0xFF;
                pix[i] = rgb;
                sum += rgb;
            }
        long avg = sum / 64;
        for (int i = 0; i < 64; i++) if (pix[i] >= avg) hash |= (1L << i);
        return Long.toHexString(hash);
    }

    static final class ImageResizeExpr extends Expression {
        final Expression child; final int w, h;
        ImageResizeExpr(Expression child, int w, int h) { this.child = child; this.w = w; this.h = h; }
        @Override public Object eval(int row, DataFrame df) {
            ImageData img = asImage(child.eval(row, df));
            return img == null ? null : img.resize(w, h);
        }
        @Override public String suggestedName() { return "image_resize(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ImageCropExpr extends Expression {
        final Expression child; final int x, y, w, h;
        ImageCropExpr(Expression child, int x, int y, int w, int h) {
            this.child = child; this.x = x; this.y = y; this.w = w; this.h = h;
        }
        @Override public Object eval(int row, DataFrame df) {
            ImageData img = asImage(child.eval(row, df));
            return img == null ? null : img.crop(x, y, w, h);
        }
        @Override public String suggestedName() { return "image_crop(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ImagePadExpr extends Expression {
        final Expression child; final int top, right, bottom, left, color;
        ImagePadExpr(Expression child, int top, int right, int bottom, int left, int color) {
            this.child = child; this.top = top; this.right = right;
            this.bottom = bottom; this.left = left; this.color = color;
        }
        @Override public Object eval(int row, DataFrame df) {
            ImageData img = asImage(child.eval(row, df));
            if (img == null || img.getImage() == null) return null;
            BufferedImage src = img.getImage();
            int nw = src.getWidth() + left + right;
            int nh = src.getHeight() + top + bottom;
            BufferedImage out = new BufferedImage(nw, nh, src.getType() == 0 ? BufferedImage.TYPE_INT_RGB : src.getType());
            Graphics2D g = out.createGraphics();
            g.setColor(new Color(color));
            g.fillRect(0, 0, nw, nh);
            g.drawImage(src, left, top, null);
            g.dispose();
            return new ImageData(out);
        }
        @Override public String suggestedName() { return "image_pad(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ImageRotateExpr extends Expression {
        final Expression child; final double angleDeg;
        ImageRotateExpr(Expression child, double angleDeg) { this.child = child; this.angleDeg = angleDeg; }
        @Override public Object eval(int row, DataFrame df) {
            ImageData img = asImage(child.eval(row, df));
            return img == null ? null : img.rotate(Math.toRadians(angleDeg));
        }
        @Override public String suggestedName() { return "image_rotate(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ImageFlipExpr extends Expression {
        final Expression child; final boolean horizontal;
        ImageFlipExpr(Expression child, boolean horizontal) { this.child = child; this.horizontal = horizontal; }
        @Override public Object eval(int row, DataFrame df) {
            ImageData img = asImage(child.eval(row, df));
            return img == null ? null : img.flip(horizontal);
        }
        @Override public String suggestedName() { return "image_flip(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ImageNormalizeExpr extends Expression {
        final Expression child; final float[] mean, std;
        ImageNormalizeExpr(Expression child, float[] mean, float[] std) {
            this.child = child; this.mean = mean; this.std = std;
        }
        @Override public Object eval(int row, DataFrame df) {
            ImageData img = asImage(child.eval(row, df));
            float[] arr = imageToFloatArray(img);
            if (arr == null) return null;
            for (int i = 0; i < arr.length; i++) {
                int c = i % 3;
                float m = mean != null && c < mean.length ? mean[c] : 0.5f;
                float s = std != null && c < std.length ? std[c] : 0.5f;
                if (s == 0) s = 1f;
                arr[i] = (arr[i] - m) / s;
            }
            return arr;
        }
        @Override public String suggestedName() { return "image_normalize(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ImageEncodeExpr extends Expression {
        final Expression child; final String format; final int quality;
        ImageEncodeExpr(Expression child, String format, int quality) {
            this.child = child; this.format = format; this.quality = quality;
        }
        @Override public Object eval(int row, DataFrame df) {
            ImageData img = asImage(child.eval(row, df));
            if (img == null || img.getImage() == null) return null;
            try {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                String fmt = format.toLowerCase(Locale.ROOT).replace("jpeg", "jpg");
                if (fmt.equals("jpg")) fmt = "jpg";
                ImageIO.write(img.getImage(), fmt.equals("jpg") ? "jpg" : fmt, baos);
                return new BinaryData("image", baos.toByteArray());
            } catch (Exception e) {
                return null;
            }
        }
        @Override public String suggestedName() { return "image_encode(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class ImageEmbedExpr extends Expression {
        final Expression child; final int dim;
        ImageEmbedExpr(Expression child, int dim) { this.child = child; this.dim = dim; }
        @Override public Object eval(int row, DataFrame df) {
            ImageData img = asImage(child.eval(row, df));
            if (img == null) return null;
            try {
                float[] emb = img.extractEmbedding(dim);
                return new EmbeddingData(emb, "image-hash");
            } catch (Exception e) {
                return null;
            }
        }
        @Override public String suggestedName() { return "image_embed(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    // ================================================================
    // Audio namespace
    // ================================================================

    public static final class AudioNameSpace {
        private final Expression parent;
        AudioNameSpace(Expression parent) { this.parent = parent; }

        public Expression decode() { return new AudioUnaryExpr(parent, AudioOp.DECODE); }
        public Expression resample(int sampleRate) { return new AudioResampleExpr(parent, sampleRate); }
        public Expression trim(float startSec, float endSec) { return new AudioTrimExpr(parent, startSec, endSec); }
        public Expression normalize() { return new AudioUnaryExpr(parent, AudioOp.NORMALIZE); }
        public Expression normalizeDb(float db) { return normalize(); }
        public Expression toMono() { return new AudioUnaryExpr(parent, AudioOp.TO_MONO); }
        public Expression toStereo() { return new AudioUnaryExpr(parent, AudioOp.TO_STEREO); }
        public Expression spectrogram() { return new AudioUnaryExpr(parent, AudioOp.SPECTROGRAM); }
        public Expression mfcc() { return mfcc(13); }
        public Expression mfcc(int nMfcc) { return new AudioMfccExpr(parent, nMfcc); }
        public Expression metadata() { return new AudioUnaryExpr(parent, AudioOp.METADATA); }
        public Expression denoise() { return new AudioUnaryExpr(parent, AudioOp.DENOISE); }
        public Expression encode(String format) { return new AudioUnaryExpr(parent, AudioOp.ENCODE); }
        public Expression toEmbedding() { return new AudioUnaryExpr(parent, AudioOp.TO_EMBEDDING); }
        /** Embed via registered model (e.g. {@code "wav2vec2-base"}). */
        public Expression toEmbedding(String modelId) {
            return AiFunctions.embedAudio(parent, modelId);
        }
    }

    enum AudioOp { DECODE, NORMALIZE, TO_MONO, TO_STEREO, SPECTROGRAM, METADATA, DENOISE, ENCODE, TO_EMBEDDING }

    static final class AudioUnaryExpr extends Expression {
        final Expression child; final AudioOp op;
        AudioUnaryExpr(Expression child, AudioOp op) { this.child = child; this.op = op; }
        @Override public Object eval(int row, DataFrame df) {
            Object raw = child.eval(row, df);
            AudioData aud = asAudio(raw);
            if (aud == null && op != AudioOp.DECODE) return null;
            try {
                return switch (op) {
                    case DECODE -> aud != null ? aud : asAudio(raw);
                    case NORMALIZE -> aud.normalize();
                    case TO_MONO -> toMonoAudio(aud);
                    case TO_STEREO -> toStereoAudio(aud);
                    case SPECTROGRAM -> aud.melSpectrogram();
                    case METADATA -> {
                        Map<String, Object> m = new LinkedHashMap<>();
                        m.put("sample_rate", aud.getSampleRate());
                        m.put("channels", aud.getChannels());
                        m.put("duration", aud.getDuration());
                        m.put("format", aud.getFormat());
                        yield m;
                    }
                    case DENOISE -> aud.denoise(1.0f);
                    case ENCODE -> new BinaryData("audio", aud.getRawBytes() != null ? aud.getRawBytes() : new byte[0]);
                    case TO_EMBEDDING -> {
                        float[][] mf = aud.mfcc(13);
                        float[] emb = averageFrames(mf);
                        yield new EmbeddingData(emb, "mfcc");
                    }
                };
            } catch (Exception e) {
                return null;
            }
        }
        @Override public String suggestedName() { return "audio_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static AudioData toMonoAudio(AudioData aud) {
        if (aud == null || aud.getSamples() == null) return aud;
        if (aud.getChannels() <= 1) return aud;
        float[] s = aud.getSamples();
        int ch = aud.getChannels();
        int frames = s.length / ch;
        float[] mono = new float[frames];
        for (int i = 0; i < frames; i++) {
            float sum = 0;
            for (int c = 0; c < ch; c++) sum += s[i * ch + c];
            mono[i] = sum / ch;
        }
        return new AudioData(mono, aud.getSampleRate(), 1);
    }

    static AudioData toStereoAudio(AudioData aud) {
        if (aud == null || aud.getSamples() == null) return aud;
        if (aud.getChannels() >= 2) return aud;
        float[] s = aud.getSamples();
        float[] stereo = new float[s.length * 2];
        for (int i = 0; i < s.length; i++) {
            stereo[i * 2] = s[i];
            stereo[i * 2 + 1] = s[i];
        }
        return new AudioData(stereo, aud.getSampleRate(), 2);
    }

    static float[] averageFrames(float[][] frames) {
        if (frames == null || frames.length == 0) return new float[0];
        int n = frames[0].length;
        float[] out = new float[n];
        for (float[] f : frames) {
            for (int i = 0; i < n && i < f.length; i++) out[i] += f[i];
        }
        for (int i = 0; i < n; i++) out[i] /= frames.length;
        return out;
    }

    static final class AudioResampleExpr extends Expression {
        final Expression child; final int sr;
        AudioResampleExpr(Expression child, int sr) { this.child = child; this.sr = sr; }
        @Override public Object eval(int row, DataFrame df) {
            AudioData aud = asAudio(child.eval(row, df));
            if (aud == null || aud.getSamples() == null) return null;
            if (aud.getSampleRate() == sr) return aud;
            // linear interpolation resample
            float[] src = aud.getSamples();
            int ch = Math.max(1, aud.getChannels());
            int srcFrames = src.length / ch;
            double ratio = (double) sr / aud.getSampleRate();
            int dstFrames = Math.max(1, (int) Math.round(srcFrames * ratio));
            float[] dst = new float[dstFrames * ch];
            for (int i = 0; i < dstFrames; i++) {
                double srcPos = i / ratio;
                int i0 = (int) Math.floor(srcPos);
                int i1 = Math.min(srcFrames - 1, i0 + 1);
                double t = srcPos - i0;
                for (int c = 0; c < ch; c++) {
                    float a = src[Math.min(src.length - 1, i0 * ch + c)];
                    float b = src[Math.min(src.length - 1, i1 * ch + c)];
                    dst[i * ch + c] = (float) ((1 - t) * a + t * b);
                }
            }
            return new AudioData(dst, sr, ch);
        }
        @Override public String suggestedName() { return "audio_resample(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class AudioTrimExpr extends Expression {
        final Expression child; final float start, end;
        AudioTrimExpr(Expression child, float start, float end) {
            this.child = child; this.start = start; this.end = end;
        }
        @Override public Object eval(int row, DataFrame df) {
            AudioData aud = asAudio(child.eval(row, df));
            if (aud == null) return null;
            try { return aud.trim(start, end); } catch (Exception e) { return null; }
        }
        @Override public String suggestedName() { return "audio_trim(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class AudioMfccExpr extends Expression {
        final Expression child; final int n;
        AudioMfccExpr(Expression child, int n) { this.child = child; this.n = n; }
        @Override public Object eval(int row, DataFrame df) {
            AudioData aud = asAudio(child.eval(row, df));
            if (aud == null) return null;
            try { return aud.mfcc(n); } catch (Exception e) { return null; }
        }
        @Override public String suggestedName() { return "audio_mfcc(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    // ================================================================
    // Video namespace
    // ================================================================

    public static final class VideoNameSpace {
        private final Expression parent;
        VideoNameSpace(Expression parent) { this.parent = parent; }

        public Expression decode() { return new VideoUnaryExpr(parent, VideoOp.DECODE); }
        public Expression extractFrames() { return extractFrames(1.0); }
        public Expression extractFrames(double fps) { return new VideoExtractFramesExpr(parent, fps); }
        public Expression frameAt(double second) { return new VideoFrameAtExpr(parent, second); }
        public Expression resize(int w, int h) { return new VideoResizeExpr(parent, w, h); }
        public Expression trim(double startSec, double endSec) { return new VideoTrimExpr(parent, startSec, endSec); }
        public Expression audioExtract() { return new VideoUnaryExpr(parent, VideoOp.AUDIO); }
        public Expression metadata() { return new VideoUnaryExpr(parent, VideoOp.METADATA); }
        public Expression keyframes() { return new VideoUnaryExpr(parent, VideoOp.KEYFRAMES); }
        public Expression sceneDetect() { return new VideoUnaryExpr(parent, VideoOp.SCENE); }
        /** Embed via registered model (e.g. {@code "videomae-base"} / CLIP). */
        public Expression toEmbedding() {
            return AiFunctions.embedVideo(parent, "hash-video");
        }
        public Expression toEmbedding(String modelId) {
            return AiFunctions.embedVideo(parent, modelId);
        }
    }

    enum VideoOp { DECODE, AUDIO, METADATA, KEYFRAMES, SCENE }

    static final class VideoUnaryExpr extends Expression {
        final Expression child; final VideoOp op;
        VideoUnaryExpr(Expression child, VideoOp op) { this.child = child; this.op = op; }
        @Override public Object eval(int row, DataFrame df) {
            Object raw = child.eval(row, df);
            VideoData vid = asVideo(raw);
            if (vid == null && op != VideoOp.DECODE) return null;
            try {
                return switch (op) {
                    case DECODE -> {
                        if (vid != null && (vid.getFrames() == null || vid.getFrames().isEmpty())
                                && vid.getPath() != null) {
                            try { yield VideoData.loadFromFile(vid.getPath()); }
                            catch (Exception e) { yield vid; }
                        }
                        yield vid;
                    }
                    case AUDIO -> vid.getAudioTrack();
                    case METADATA -> {
                        Map<String, Object> m = new LinkedHashMap<>();
                        m.put("duration", vid.getDuration());
                        m.put("fps", vid.getFps());
                        m.put("width", vid.getWidth());
                        m.put("height", vid.getHeight());
                        m.put("frame_count", vid.getFrameCount());
                        m.put("format", vid.getFormat());
                        yield m;
                    }
                    case KEYFRAMES -> {
                        List<VideoData.KeyFrame> kfs = vid.extractKeyFramesUniform(Math.max(1, vid.getFrameCount() / 10));
                        List<Double> ts = new ArrayList<>();
                        if (kfs != null) for (VideoData.KeyFrame kf : kfs) {
                            try {
                                java.lang.reflect.Field tf = kf.getClass().getDeclaredField("timestamp");
                                tf.setAccessible(true);
                                ts.add(((Number) tf.get(kf)).doubleValue());
                            } catch (Exception e) {
                                ts.add((double) ts.size());
                            }
                        }
                        yield ts;
                    }
                    case SCENE -> {
                        List<VideoData.KeyFrame> scenes = vid.detectSceneChanges(0.3);
                        List<Double> ts = new ArrayList<>();
                        if (scenes != null) for (int i = 0; i < scenes.size(); i++) ts.add((double) i);
                        yield ts;
                    }
                };
            } catch (Exception e) {
                return null;
            }
        }
        @Override public String suggestedName() { return "video_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class VideoExtractFramesExpr extends Expression {
        final Expression child; final double fps;
        VideoExtractFramesExpr(Expression child, double fps) { this.child = child; this.fps = fps; }
        @Override public Object eval(int row, DataFrame df) {
            VideoData vid = asVideo(child.eval(row, df));
            if (vid == null) return null;
            List<ImageData> frames = vid.getFrames();
            if (frames == null || frames.isEmpty()) return List.of();
            double srcFps = vid.getFps() > 0 ? vid.getFps() : 30.0;
            double step = Math.max(1.0, srcFps / Math.max(0.1, fps));
            List<ImageData> out = new ArrayList<>();
            for (double i = 0; i < frames.size(); i += step) {
                out.add(frames.get((int) i));
            }
            return out;
        }
        @Override public String suggestedName() { return "video_frames(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class VideoFrameAtExpr extends Expression {
        final Expression child; final double second;
        VideoFrameAtExpr(Expression child, double second) { this.child = child; this.second = second; }
        @Override public Object eval(int row, DataFrame df) {
            VideoData vid = asVideo(child.eval(row, df));
            if (vid == null || vid.getFrames() == null || vid.getFrames().isEmpty()) return null;
            double fps = vid.getFps() > 0 ? vid.getFps() : 30.0;
            int idx = (int) Math.round(second * fps);
            idx = Math.max(0, Math.min(vid.getFrames().size() - 1, idx));
            return vid.getFrames().get(idx);
        }
        @Override public String suggestedName() { return "video_frame_at(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class VideoResizeExpr extends Expression {
        final Expression child; final int w, h;
        VideoResizeExpr(Expression child, int w, int h) { this.child = child; this.w = w; this.h = h; }
        @Override public Object eval(int row, DataFrame df) {
            VideoData vid = asVideo(child.eval(row, df));
            if (vid == null) return null;
            try { return vid.resize(w, h); } catch (Exception e) { return null; }
        }
        @Override public String suggestedName() { return "video_resize(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class VideoTrimExpr extends Expression {
        final Expression child; final double start, end;
        VideoTrimExpr(Expression child, double start, double end) {
            this.child = child; this.start = start; this.end = end;
        }
        @Override public Object eval(int row, DataFrame df) {
            VideoData vid = asVideo(child.eval(row, df));
            if (vid == null) return null;
            try { return vid.trim(start, end); } catch (Exception e) { return null; }
        }
        @Override public String suggestedName() { return "video_trim(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    // ================================================================
    // Tensor / embedding namespace
    // ================================================================

    public static final class TensorNameSpace {
        private final Expression parent;
        TensorNameSpace(Expression parent) { this.parent = parent; }

        public Expression reshape(int... shape) { return new TensorReshapeExpr(parent, shape); }
        public Expression flatten() { return new TensorUnaryExpr(parent, TensorOp.FLATTEN); }
        public Expression transpose() { return transpose(null); }
        public Expression transpose(int[] axes) { return new TensorTransposeExpr(parent, axes); }
        public Expression matmul(Object other) { return new TensorBinaryExpr(parent, Expression.toExpr(other), TensorBinOp.MATMUL); }
        public Expression l2Norm() { return new TensorUnaryExpr(parent, TensorOp.L2_NORM); }
        public Expression dot(Object other) { return new TensorBinaryExpr(parent, Expression.toExpr(other), TensorBinOp.DOT); }
        public Expression cosineSim(Object other) { return new TensorBinaryExpr(parent, Expression.toExpr(other), TensorBinOp.COSINE); }
        public Expression slice(int start, int end) { return slice(start, end, 0); }
        public Expression slice(int start, int end, int axis) { return new TensorSliceExpr(parent, start, end, axis); }
        public Expression concat(Object other) { return concat(other, 0); }
        public Expression concat(Object other, int axis) {
            return new TensorBinaryExpr(parent, Expression.toExpr(other), TensorBinOp.CONCAT, axis);
        }
        public Expression mean() { return new TensorUnaryExpr(parent, TensorOp.MEAN); }
        public Expression sum() { return new TensorUnaryExpr(parent, TensorOp.SUM); }
        public Expression max() { return new TensorUnaryExpr(parent, TensorOp.MAX); }
        public Expression min() { return new TensorUnaryExpr(parent, TensorOp.MIN); }
        public Expression toNumpy() { return new TensorUnaryExpr(parent, TensorOp.TO_NUMPY); }
    }

    enum TensorOp { FLATTEN, L2_NORM, MEAN, SUM, MAX, MIN, TO_NUMPY }
    enum TensorBinOp { MATMUL, DOT, COSINE, CONCAT }

    static final class TensorUnaryExpr extends Expression {
        final Expression child; final TensorOp op;
        TensorUnaryExpr(Expression child, TensorOp op) { this.child = child; this.op = op; }
        @Override public Object eval(int row, DataFrame df) {
            Object raw = child.eval(row, df);
            float[] vec = asFloatVector(raw);
            TensorData td = asTensor(raw);
            try {
                return switch (op) {
                    case FLATTEN -> {
                        if (td != null) yield new TensorData(td.getData(), new int[]{td.size()});
                        if (vec != null) yield new TensorData(vec, new int[]{vec.length});
                        yield null;
                    }
                    case L2_NORM -> {
                        if (vec == null) yield null;
                        double sum = 0;
                        for (float v : vec) sum += v * v;
                        double n = Math.sqrt(sum);
                        if (n == 0) yield Arrays.copyOf(vec, vec.length);
                        float[] out = new float[vec.length];
                        for (int i = 0; i < vec.length; i++) out[i] = (float) (vec[i] / n);
                        if (raw instanceof EmbeddingData)
                            yield new EmbeddingData(out, ((EmbeddingData) raw).getModelName());
                        yield out;
                    }
                    case MEAN -> {
                        if (vec == null || vec.length == 0) yield null;
                        double s = 0; for (float v : vec) s += v;
                        yield s / vec.length;
                    }
                    case SUM -> {
                        if (vec == null) yield null;
                        double s = 0; for (float v : vec) s += v;
                        yield s;
                    }
                    case MAX -> {
                        if (vec == null || vec.length == 0) yield null;
                        float m = Float.NEGATIVE_INFINITY;
                        for (float v : vec) if (v > m) m = v;
                        yield (double) m;
                    }
                    case MIN -> {
                        if (vec == null || vec.length == 0) yield null;
                        float m = Float.POSITIVE_INFINITY;
                        for (float v : vec) if (v < m) m = v;
                        yield (double) m;
                    }
                    case TO_NUMPY -> vec;
                };
            } catch (Exception e) {
                return null;
            }
        }
        @Override public String suggestedName() { return "tensor_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class TensorReshapeExpr extends Expression {
        final Expression child; final int[] shape;
        TensorReshapeExpr(Expression child, int[] shape) {
            this.child = child; this.shape = shape == null ? new int[]{ -1 } : shape.clone();
        }
        @Override public Object eval(int row, DataFrame df) {
            TensorData td = asTensor(child.eval(row, df));
            if (td == null) {
                float[] v = asFloatVector(child.eval(row, df));
                if (v == null) return null;
                td = new TensorData(v, new int[]{v.length});
            }
            try { return td.reshape(shape); } catch (Exception e) { return null; }
        }
        @Override public String suggestedName() { return "tensor_reshape(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class TensorTransposeExpr extends Expression {
        final Expression child; final int[] axes;
        TensorTransposeExpr(Expression child, int[] axes) { this.child = child; this.axes = axes; }
        @Override public Object eval(int row, DataFrame df) {
            TensorData td = asTensor(child.eval(row, df));
            if (td == null) return null;
            int[] shape = td.getShape();
            if (shape == null || shape.length == 1) return td;
            if (shape.length == 2) {
                // 2D transpose
                float[] data = td.getData();
                int r = shape[0], c = shape[1];
                float[] out = new float[data.length];
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < c; j++)
                        out[j * r + i] = data[i * c + j];
                return new TensorData(out, new int[]{c, r});
            }
            // higher-dim: reverse axes by default
            int[] newShape = new int[shape.length];
            for (int i = 0; i < shape.length; i++) newShape[i] = shape[shape.length - 1 - i];
            try { return td.reshape(newShape); } catch (Exception e) { return td; }
        }
        @Override public String suggestedName() { return "tensor_transpose(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class TensorSliceExpr extends Expression {
        final Expression child; final int start, end, axis;
        TensorSliceExpr(Expression child, int start, int end, int axis) {
            this.child = child; this.start = start; this.end = end; this.axis = axis;
        }
        @Override public Object eval(int row, DataFrame df) {
            float[] vec = asFloatVector(child.eval(row, df));
            if (vec == null) return null;
            int s = Math.max(0, start);
            int e = Math.min(vec.length, end);
            if (e < s) return new float[0];
            return Arrays.copyOfRange(vec, s, e);
        }
        @Override public String suggestedName() { return "tensor_slice(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class TensorBinaryExpr extends Expression {
        final Expression left, right;
        final TensorBinOp op;
        final int axis;
        TensorBinaryExpr(Expression left, Expression right, TensorBinOp op) {
            this(left, right, op, 0);
        }
        TensorBinaryExpr(Expression left, Expression right, TensorBinOp op, int axis) {
            this.left = left; this.right = right; this.op = op; this.axis = axis;
        }
        @Override public Object eval(int row, DataFrame df) {
            float[] a = asFloatVector(left.eval(row, df));
            float[] b = asFloatVector(right.eval(row, df));
            if (a == null || b == null) return null;
            return switch (op) {
                case DOT -> {
                    int n = Math.min(a.length, b.length);
                    double s = 0;
                    for (int i = 0; i < n; i++) s += a[i] * b[i];
                    yield s;
                }
                case COSINE -> {
                    int n = Math.min(a.length, b.length);
                    double dot = 0, na = 0, nb = 0;
                    for (int i = 0; i < n; i++) {
                        dot += a[i] * b[i];
                        na += a[i] * a[i];
                        nb += b[i] * b[i];
                    }
                    double denom = Math.sqrt(na) * Math.sqrt(nb);
                    yield denom == 0 ? 0.0 : dot / denom;
                }
                case CONCAT -> {
                    float[] out = new float[a.length + b.length];
                    System.arraycopy(a, 0, out, 0, a.length);
                    System.arraycopy(b, 0, out, a.length, b.length);
                    yield out;
                }
                case MATMUL -> {
                    // treat as row-vector · col-vector → outer if lengths differ, else dot as 1x1
                    if (a.length == b.length) {
                        // element-wise as fallback matvec when no shape info: outer product flattened NxN if small
                        if (a.length <= 64) {
                            float[] out = new float[a.length * b.length];
                            for (int i = 0; i < a.length; i++)
                                for (int j = 0; j < b.length; j++)
                                    out[i * b.length + j] = a[i] * b[j];
                            yield new TensorData(out, new int[]{a.length, b.length});
                        }
                        double s = 0;
                        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
                        yield s;
                    }
                    yield null;
                }
            };
        }
        @Override public String suggestedName() {
            return "tensor_" + op.name().toLowerCase() + "(" + left.suggestedName() + ")";
        }
        @Override public Set<String> referencedColumns() {
            Set<String> s = new LinkedHashSet<>(left.referencedColumns());
            s.addAll(right.referencedColumns());
            return s;
        }
    }

    // ================================================================
    // Text namespace
    // ================================================================

    public static final class TextNameSpace {
        private final Expression parent;
        TextNameSpace(Expression parent) { this.parent = parent; }

        public Expression clean() { return new TextUnaryExpr(parent, TextOp.CLEAN); }
        public Expression tokenize() { return new TextUnaryExpr(parent, TextOp.TOKENIZE); }
        public Expression sentenceSplit() { return new TextUnaryExpr(parent, TextOp.SENTENCES); }
        public Expression removeStopwords() { return removeStopwords("en"); }
        public Expression removeStopwords(String lang) { return new TextStopExpr(parent, lang); }
        public Expression lemmatize() { return new TextUnaryExpr(parent, TextOp.LEMMATIZE); }
        public Expression summarize() { return new TextUnaryExpr(parent, TextOp.SUMMARIZE); }
        public Expression piiMask() { return new TextUnaryExpr(parent, TextOp.PII_MASK); }
        public Expression toEmbedding() { return toEmbedding(64); }
        public Expression toEmbedding(int dim) { return new TextEmbedExpr(parent, dim); }
        /** Embed via registered model (e.g. {@code "bge-small-zh"}). */
        public Expression toEmbedding(String modelId) {
            return AiFunctions.embedText(parent, modelId);
        }
    }

    enum TextOp { CLEAN, TOKENIZE, SENTENCES, LEMMATIZE, SUMMARIZE, PII_MASK }

    static final class TextUnaryExpr extends Expression {
        final Expression child; final TextOp op;
        TextUnaryExpr(Expression child, TextOp op) { this.child = child; this.op = op; }
        @Override public Object eval(int row, DataFrame df) {
            String s = asText(child.eval(row, df));
            if (s == null) return null;
            return switch (op) {
                case CLEAN -> s.replaceAll("[\\p{Cntrl}&&[^\n\t]]", "")
                        .replaceAll("[^\\p{L}\\p{N}\\s\\p{P}]", " ")
                        .replaceAll("\\s+", " ")
                        .trim();
                case TOKENIZE -> Arrays.asList(s.trim().split("\\s+"));
                case SENTENCES -> Arrays.asList(s.split("(?<=[.!?。！？])\\s*"));
                case LEMMATIZE -> s.toLowerCase(Locale.ROOT)
                        .replaceAll("ing\\b", "")
                        .replaceAll("ed\\b", "")
                        .replaceAll("ies\\b", "y")
                        .replaceAll("s\\b", "");
                case SUMMARIZE -> {
                    String[] sents = s.split("(?<=[.!?。！？])\\s*");
                    yield sents.length == 0 ? s : sents[0].trim();
                }
                case PII_MASK -> s
                        .replaceAll("\\b\\d{11}\\b", "[PHONE]")
                        .replaceAll("\\b\\d{15,18}\\b", "[ID]")
                        .replaceAll("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "[EMAIL]");
            };
        }
        @Override public String suggestedName() { return "text_" + op.name().toLowerCase() + "(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class TextStopExpr extends Expression {
        final Expression child; final String lang;
        private static final Set<String> EN = Set.of(
            "a","an","the","and","or","but","in","on","at","to","for","of","is","are","was","were","be","been",
            "have","has","had","do","does","did","will","would","could","should","may","might","must","shall",
            "i","you","he","she","it","we","they","this","that","these","those","with","as","by","from");
        private static final Set<String> ZH = Set.of(
            "的","了","在","是","我","有","和","就","不","人","都","一","一个","上","也","很","到","说","要","去","你","会","着","没有","看","好","自己","这");
        TextStopExpr(Expression child, String lang) { this.child = child; this.lang = lang; }
        @Override public Object eval(int row, DataFrame df) {
            String s = asText(child.eval(row, df));
            if (s == null) return null;
            Set<String> stops = "zh".equalsIgnoreCase(lang) ? ZH : EN;
            String[] toks = s.split("\\s+");
            StringBuilder sb = new StringBuilder();
            for (String t : toks) {
                if (!stops.contains(t.toLowerCase(Locale.ROOT))) {
                    if (sb.length() > 0) sb.append(' ');
                    sb.append(t);
                }
            }
            return sb.toString();
        }
        @Override public String suggestedName() { return "text_rm_stop(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }

    static final class TextEmbedExpr extends Expression {
        final Expression child; final int dim;
        TextEmbedExpr(Expression child, int dim) { this.child = child; this.dim = dim; }
        @Override public Object eval(int row, DataFrame df) {
            String s = asText(child.eval(row, df));
            if (s == null) return null;
            float[] emb = new float[dim];
            // simple hashing trick bag-of-chars embedding
            for (int i = 0; i < s.length(); i++) {
                int h = Character.hashCode(s.charAt(i));
                emb[Math.floorMod(h, dim)] += 1.0f;
                emb[Math.floorMod(h * 31, dim)] += 0.5f;
            }
            double n = 0;
            for (float v : emb) n += v * v;
            n = Math.sqrt(n);
            if (n > 0) for (int i = 0; i < dim; i++) emb[i] /= n;
            return new EmbeddingData(emb, "hash-text");
        }
        @Override public String suggestedName() { return "text_embed(" + child.suggestedName() + ")"; }
        @Override public Set<String> referencedColumns() { return child.referencedColumns(); }
    }
}
