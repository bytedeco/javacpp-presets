package org.bytedeco.pytorch.vision.pillow.codec;

import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.UnidentifiedImageError;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Pure-Java PBM/PGM/PPM (Netpbm) codec — no ImageIO provider required.
 * Supports binary P4/P5/P6 and ASCII P1/P2/P3.
 */
public final class PpmPlugin implements ImagePlugin {

    @Override
    public String format() {
        return "PPM";
    }

    @Override
    public List<String> extensions() {
        return List.of("ppm", "pgm", "pbm", "pnm");
    }

    @Override
    public List<String> mimeTypes() {
        return List.of("image/x-portable-pixmap", "image/x-portable-graymap", "image/x-portable-bitmap");
    }

    @Override
    public boolean accept(byte[] prefix, int length) {
        if (prefix == null || length < 2) return false;
        return prefix[0] == 'P' && prefix[1] >= '1' && prefix[1] <= '6';
    }

    @Override
    public Image open(Path path) throws IOException {
        return open(Files.readAllBytes(path));
    }

    @Override
    public Image open(InputStream in) throws IOException {
        return open(in.readAllBytes());
    }

    @Override
    public Image open(byte[] data) throws IOException {
        Objects.requireNonNull(data, "data");
        if (data.length < 3 || data[0] != 'P') {
            throw new UnidentifiedImageError("not a PPM/PGM/PBM file");
        }
        char magic = (char) data[1];
        Parser p = new Parser(data, 2);
        int width = p.nextInt();
        int height = p.nextInt();
        int maxval = 1;
        if (magic != '1' && magic != '4') {
            maxval = p.nextInt();
        }
        if (width <= 0 || height <= 0) {
            throw new IOException("invalid PPM size " + width + "x" + height);
        }
        if (maxval <= 0 || maxval > 65535) {
            throw new IOException("invalid maxval " + maxval);
        }
        return switch (magic) {
            case '1' -> readAsciiBitmap(p, width, height);
            case '2' -> readAsciiGray(p, width, height, maxval);
            case '3' -> readAsciiRgb(p, width, height, maxval);
            case '4' -> readBinaryBitmap(data, p.pos, width, height);
            case '5' -> readBinaryGray(data, p.pos, width, height, maxval);
            case '6' -> readBinaryRgb(data, p.pos, width, height, maxval);
            default -> throw new UnidentifiedImageError("unsupported netpbm magic P" + magic);
        };
    }

    private static Image readAsciiBitmap(Parser p, int w, int h) throws IOException {
        ImagingBuffer buf = new ImagingBuffer("1", w, h);
        for (int i = 0; i < w * h; i++) {
            int v = p.nextInt();
            buf.data()[i] = (byte) (v != 0 ? 0 : 255); // PBM 1=black
        }
        Image im = Image.fromBuffer(buf, "PPM", Map.of("maxval", 1));
        im.setFormat("PPM");
        return im;
    }

    private static Image readAsciiGray(Parser p, int w, int h, int maxval) throws IOException {
        ImagingBuffer buf = new ImagingBuffer("L", w, h);
        for (int i = 0; i < w * h; i++) {
            int v = p.nextInt();
            buf.data()[i] = (byte) scaleTo8(v, maxval);
        }
        Image im = Image.fromBuffer(buf, "PPM", Map.of("maxval", maxval));
        im.setFormat("PPM");
        return im;
    }

    private static Image readAsciiRgb(Parser p, int w, int h, int maxval) throws IOException {
        ImagingBuffer buf = new ImagingBuffer("RGB", w, h);
        byte[] d = buf.data();
        for (int i = 0; i < w * h * 3; i++) {
            d[i] = (byte) scaleTo8(p.nextInt(), maxval);
        }
        Image im = Image.fromBuffer(buf, "PPM", Map.of("maxval", maxval));
        im.setFormat("PPM");
        return im;
    }

    private static Image readBinaryBitmap(byte[] data, int pos, int w, int h) throws IOException {
        ImagingBuffer buf = new ImagingBuffer("1", w, h);
        int rowBytes = (w + 7) / 8;
        int need = pos + rowBytes * h;
        if (data.length < need) {
            throw new IOException("truncated P4 data");
        }
        byte[] d = buf.data();
        int di = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int byteIndex = pos + y * rowBytes + (x / 8);
                int bit = 7 - (x % 8);
                boolean black = ((data[byteIndex] >> bit) & 1) == 1;
                d[di++] = (byte) (black ? 0 : 255);
            }
        }
        Image im = Image.fromBuffer(buf, "PPM", Map.of());
        im.setFormat("PPM");
        return im;
    }

    private static Image readBinaryGray(byte[] data, int pos, int w, int h, int maxval) throws IOException {
        ImagingBuffer buf = new ImagingBuffer("L", w, h);
        int bpp = maxval > 255 ? 2 : 1;
        int need = pos + w * h * bpp;
        if (data.length < need) {
            throw new IOException("truncated P5 data");
        }
        byte[] d = buf.data();
        if (bpp == 1 && maxval == 255) {
            System.arraycopy(data, pos, d, 0, w * h);
        } else if (bpp == 1) {
            for (int i = 0; i < w * h; i++) {
                d[i] = (byte) scaleTo8(data[pos + i] & 0xff, maxval);
            }
        } else {
            for (int i = 0; i < w * h; i++) {
                int v = ((data[pos + i * 2] & 0xff) << 8) | (data[pos + i * 2 + 1] & 0xff);
                d[i] = (byte) scaleTo8(v, maxval);
            }
        }
        Image im = Image.fromBuffer(buf, "PPM", Map.of("maxval", maxval));
        im.setFormat("PPM");
        return im;
    }

    private static Image readBinaryRgb(byte[] data, int pos, int w, int h, int maxval) throws IOException {
        ImagingBuffer buf = new ImagingBuffer("RGB", w, h);
        int bpp = maxval > 255 ? 2 : 1;
        int need = pos + w * h * 3 * bpp;
        if (data.length < need) {
            throw new IOException("truncated P6 data");
        }
        byte[] d = buf.data();
        if (bpp == 1 && maxval == 255) {
            System.arraycopy(data, pos, d, 0, w * h * 3);
        } else if (bpp == 1) {
            for (int i = 0; i < w * h * 3; i++) {
                d[i] = (byte) scaleTo8(data[pos + i] & 0xff, maxval);
            }
        } else {
            for (int i = 0; i < w * h * 3; i++) {
                int v = ((data[pos + i * 2] & 0xff) << 8) | (data[pos + i * 2 + 1] & 0xff);
                d[i] = (byte) scaleTo8(v, maxval);
            }
        }
        Image im = Image.fromBuffer(buf, "PPM", Map.of("maxval", maxval));
        im.setFormat("PPM");
        return im;
    }

    private static int scaleTo8(int v, int maxval) {
        if (maxval == 255) return Math.max(0, Math.min(255, v));
        return Math.max(0, Math.min(255, (v * 255 + maxval / 2) / maxval));
    }

    @Override
    public void save(Image image, Path path, Map<String, Object> options) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (OutputStream out = Files.newOutputStream(path)) {
            String name = path.getFileName() == null ? "x.ppm" : path.getFileName().toString().toLowerCase(Locale.ROOT);
            String hint = name.endsWith(".pbm") ? "PBM" : name.endsWith(".pgm") ? "PGM" : "PPM";
            save(image, out, hint, options);
        }
    }

    @Override
    public void save(Image image, OutputStream out, String formatHint, Map<String, Object> options)
            throws IOException {
        Objects.requireNonNull(image, "image");
        Objects.requireNonNull(out, "out");
        String hint = formatHint == null ? "PPM" : formatHint.toUpperCase(Locale.ROOT);
        boolean binary = options == null || !Boolean.FALSE.equals(options.get("binary"));
        if ("PBM".equals(hint) || image.mode().equals("1")) {
            writeP4orP1(image.convert("1"), out, binary);
            return;
        }
        if ("PGM".equals(hint) || image.mode().equals("L")) {
            writeP5orP2(image.mode().equals("L") ? image : image.convert("L"), out, binary);
            return;
        }
        Image rgb = image.mode().equals("RGB") ? image : image.convert("RGB");
        writeP6orP3(rgb, out, binary);
    }

    private static void writeP4orP1(Image im, OutputStream out, boolean binary) throws IOException {
        int w = im.width(), h = im.height();
        if (binary) {
            out.write(("P4\n" + w + " " + h + "\n").getBytes(StandardCharsets.US_ASCII));
            int rowBytes = (w + 7) / 8;
            byte[] row = new byte[rowBytes];
            for (int y = 0; y < h; y++) {
                ArraysFill(row, (byte) 0);
                for (int x = 0; x < w; x++) {
                    int v = im.getpixel(x, y)[0];
                    if (v < 128) { // black
                        row[x / 8] |= (byte) (1 << (7 - (x % 8)));
                    }
                }
                out.write(row);
            }
        } else {
            StringBuilder sb = new StringBuilder();
            sb.append("P1\n").append(w).append(' ').append(h).append('\n');
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    if (x > 0) sb.append(' ');
                    sb.append(im.getpixel(x, y)[0] < 128 ? 1 : 0);
                }
                sb.append('\n');
            }
            out.write(sb.toString().getBytes(StandardCharsets.US_ASCII));
        }
    }

    private static void writeP5orP2(Image im, OutputStream out, boolean binary) throws IOException {
        int w = im.width(), h = im.height();
        if (binary) {
            out.write(("P5\n" + w + " " + h + "\n255\n").getBytes(StandardCharsets.US_ASCII));
            out.write(im.tobytes());
        } else {
            StringBuilder sb = new StringBuilder();
            sb.append("P2\n").append(w).append(' ').append(h).append("\n255\n");
            byte[] d = im.tobytes();
            for (int i = 0; i < d.length; i++) {
                if (i > 0) sb.append(i % w == 0 ? '\n' : ' ');
                sb.append(d[i] & 0xff);
            }
            sb.append('\n');
            out.write(sb.toString().getBytes(StandardCharsets.US_ASCII));
        }
    }

    private static void writeP6orP3(Image im, OutputStream out, boolean binary) throws IOException {
        int w = im.width(), h = im.height();
        if (binary) {
            out.write(("P6\n" + w + " " + h + "\n255\n").getBytes(StandardCharsets.US_ASCII));
            out.write(im.tobytes());
        } else {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            bos.write(("P3\n" + w + " " + h + "\n255\n").getBytes(StandardCharsets.US_ASCII));
            byte[] d = im.tobytes();
            StringBuilder line = new StringBuilder();
            for (int i = 0; i < d.length; i++) {
                if (line.length() > 60) {
                    line.append('\n');
                    bos.write(line.toString().getBytes(StandardCharsets.US_ASCII));
                    line.setLength(0);
                }
                if (line.length() > 0) line.append(' ');
                line.append(d[i] & 0xff);
            }
            line.append('\n');
            bos.write(line.toString().getBytes(StandardCharsets.US_ASCII));
            bos.writeTo(out);
        }
    }

    private static void ArraysFill(byte[] a, byte v) {
        java.util.Arrays.fill(a, v);
    }

    /** Minimal token parser skipping whitespace and comments. */
    private static final class Parser {
        final byte[] data;
        int pos;

        Parser(byte[] data, int pos) {
            this.data = data;
            this.pos = pos;
        }

        int nextInt() throws IOException {
            skip();
            if (pos >= data.length) throw new IOException("unexpected EOF in PPM header/body");
            int start = pos;
            if (data[pos] == '-' || data[pos] == '+') pos++;
            while (pos < data.length && data[pos] >= '0' && data[pos] <= '9') pos++;
            if (start == pos) throw new IOException("expected int at " + pos);
            return Integer.parseInt(new String(data, start, pos - start, StandardCharsets.US_ASCII));
        }

        void skip() {
            while (pos < data.length) {
                byte c = data[pos];
                if (c == '#') {
                    while (pos < data.length && data[pos] != '\n') pos++;
                    continue;
                }
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    pos++;
                    continue;
                }
                break;
            }
            // after header numbers, binary section may start immediately after single whitespace
        }
    }
}
