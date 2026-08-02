package org.bytedeco.pytorch.vision.pillow.core;

import org.bytedeco.pytorch.vision.pillow.enums.Resampling;

import java.util.Objects;

/**
 * Resize filters: NEAREST / BOX / BILINEAR / HAMMING / BICUBIC / LANCZOS.
 * Pure Java; not a thin wrapper over {@link java.awt.image.AffineTransformOp} for LANCZOS.
 */
public final class Resample {

    private Resample() {}

    public static ImagingBuffer resize(ImagingBuffer src, int outW, int outH, Resampling filter) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(filter, "filter");
        if (outW <= 0 || outH <= 0) {
            throw new IllegalArgumentException("output size must be positive");
        }
        if (src.width() == outW && src.height() == outH) {
            return src.copy();
        }
        return switch (filter) {
            case NEAREST -> nearest(src, outW, outH);
            case BOX -> box(src, outW, outH);
            case BILINEAR -> bilinear(src, outW, outH);
            case HAMMING -> general(src, outW, outH, FilterKernel.HAMMING, 1.0);
            case BICUBIC -> general(src, outW, outH, FilterKernel.BICUBIC, 2.0);
            case LANCZOS -> general(src, outW, outH, FilterKernel.LANCZOS, 3.0);
        };
    }

    private static ImagingBuffer nearest(ImagingBuffer src, int outW, int outH) {
        ImagingBuffer out = new ImagingBuffer(src.modeInfo(), outW, outH);
        if (src.getPalette() != null) {
            out.setPalette(src.getPalette());
        }
        double xScale = (double) src.width() / outW;
        double yScale = (double) src.height() / outH;
        for (int y = 0; y < outH; y++) {
            int sy = Math.min(src.height() - 1, (int) Math.floor(y * yScale));
            for (int x = 0; x < outW; x++) {
                int sx = Math.min(src.width() - 1, (int) Math.floor(x * xScale));
                out.putpixel(x, y, src.getpixel(sx, sy));
            }
        }
        return out;
    }

    private static ImagingBuffer box(ImagingBuffer src, int outW, int outH) {
        // box = average of contributing source pixels (good for downscale)
        if (outW >= src.width() && outH >= src.height()) {
            return bilinear(src, outW, outH);
        }
        ImagingBuffer out = new ImagingBuffer(src.modeInfo(), outW, outH);
        int bands = src.bands();
        boolean byteMode = src.modeInfo().isByteMode();
        if (!byteMode) {
            return nearest(src, outW, outH);
        }
        double xScale = (double) src.width() / outW;
        double yScale = (double) src.height() / outH;
        for (int y = 0; y < outH; y++) {
            int y0 = (int) Math.floor(y * yScale);
            int y1 = Math.min(src.height(), (int) Math.ceil((y + 1) * yScale));
            if (y1 <= y0) y1 = Math.min(src.height(), y0 + 1);
            for (int x = 0; x < outW; x++) {
                int x0 = (int) Math.floor(x * xScale);
                int x1 = Math.min(src.width(), (int) Math.ceil((x + 1) * xScale));
                if (x1 <= x0) x1 = Math.min(src.width(), x0 + 1);
                long[] sum = new long[bands];
                int count = 0;
                for (int sy = y0; sy < y1; sy++) {
                    for (int sx = x0; sx < x1; sx++) {
                        int[] p = src.getpixel(sx, sy);
                        for (int b = 0; b < bands; b++) {
                            sum[b] += p[b];
                        }
                        count++;
                    }
                }
                int[] pix = new int[bands];
                if (count == 0) count = 1;
                for (int b = 0; b < bands; b++) {
                    pix[b] = (int) ((sum[b] + count / 2) / count);
                }
                out.putpixel(x, y, pix);
            }
        }
        return out;
    }

    private static ImagingBuffer bilinear(ImagingBuffer src, int outW, int outH) {
        ImagingBuffer out = new ImagingBuffer(src.modeInfo(), outW, outH);
        if (!src.modeInfo().isByteMode()) {
            return nearest(src, outW, outH);
        }
        int bands = src.bands();
        double xScale = (src.width() == 1) ? 0 : (double) (src.width() - 1) / Math.max(1, outW - 1);
        double yScale = (src.height() == 1) ? 0 : (double) (src.height() - 1) / Math.max(1, outH - 1);
        // use half-pixel mapping closer to Pillow for non-1 sizes
        if (outW > 1) {
            xScale = (double) src.width() / outW;
        }
        if (outH > 1) {
            yScale = (double) src.height() / outH;
        }
        for (int y = 0; y < outH; y++) {
            double sy = (y + 0.5) * yScale - 0.5;
            int y0 = (int) Math.floor(sy);
            int y1 = y0 + 1;
            double fy = sy - y0;
            if (y0 < 0) {
                y0 = 0;
                y1 = 0;
                fy = 0;
            }
            if (y1 >= src.height()) {
                y1 = src.height() - 1;
                y0 = y1;
                fy = 0;
            }
            for (int x = 0; x < outW; x++) {
                double sx = (x + 0.5) * xScale - 0.5;
                int x0 = (int) Math.floor(sx);
                int x1 = x0 + 1;
                double fx = sx - x0;
                if (x0 < 0) {
                    x0 = 0;
                    x1 = 0;
                    fx = 0;
                }
                if (x1 >= src.width()) {
                    x1 = src.width() - 1;
                    x0 = x1;
                    fx = 0;
                }
                int[] p00 = src.getpixel(x0, y0);
                int[] p10 = src.getpixel(x1, y0);
                int[] p01 = src.getpixel(x0, y1);
                int[] p11 = src.getpixel(x1, y1);
                int[] pix = new int[bands];
                for (int b = 0; b < bands; b++) {
                    double top = p00[b] * (1 - fx) + p10[b] * fx;
                    double bot = p01[b] * (1 - fx) + p11[b] * fx;
                    pix[b] = ImagingBuffer.clamp8((int) Math.round(top * (1 - fy) + bot * fy));
                }
                out.putpixel(x, y, pix);
            }
        }
        return out;
    }

    @FunctionalInterface
    private interface FilterKernel {
        double weight(double x);

        FilterKernel BICUBIC = x -> {
            x = Math.abs(x);
            if (x <= 1) {
                return (1.5 * x - 2.5) * x * x + 1;
            }
            if (x < 2) {
                return ((-0.5 * x + 2.5) * x - 4) * x + 2;
            }
            return 0;
        };

        FilterKernel LANCZOS = x -> {
            x = Math.abs(x);
            if (x == 0) return 1;
            if (x >= 3) return 0;
            double pix = Math.PI * x;
            return 3 * Math.sin(pix) * Math.sin(pix / 3) / (pix * pix);
        };

        FilterKernel HAMMING = x -> {
            x = Math.abs(x);
            if (x == 0) return 1;
            if (x >= 1) return 0;
            double pix = Math.PI * x;
            return Math.sin(pix) / pix * (0.54 + 0.46 * Math.cos(pix));
        };
    }

    private static ImagingBuffer general(ImagingBuffer src, int outW, int outH, FilterKernel kernel, double support) {
        if (!src.modeInfo().isByteMode()) {
            return nearest(src, outW, outH);
        }
        // separable: horizontal then vertical
        ImagingBuffer tmp = resampleAxis(src, outW, src.height(), true, kernel, support);
        return resampleAxis(tmp, outW, outH, false, kernel, support);
    }

    private static ImagingBuffer resampleAxis(ImagingBuffer src, int outW, int outH, boolean horizontal,
                                             FilterKernel kernel, double support) {
        ImagingBuffer out = new ImagingBuffer(src.modeInfo(), outW, outH);
        int bands = src.bands();
        int inW = src.width();
        int inH = src.height();
        if (horizontal) {
            double scale = (double) inW / outW;
            double filterScale = Math.max(scale, 1.0);
            double radius = support * filterScale;
            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {
                    double center = (x + 0.5) * scale;
                    int left = Math.max(0, (int) Math.floor(center - radius));
                    int right = Math.min(inW - 1, (int) Math.ceil(center + radius));
                    double[] acc = new double[bands];
                    double wsum = 0;
                    for (int sx = left; sx <= right; sx++) {
                        double w = kernel.weight((sx + 0.5 - center) / filterScale);
                        if (w == 0) continue;
                        int[] p = src.getpixel(sx, y);
                        for (int b = 0; b < bands; b++) {
                            acc[b] += p[b] * w;
                        }
                        wsum += w;
                    }
                    int[] pix = new int[bands];
                    if (wsum <= 1e-12) {
                        int sx = Math.min(inW - 1, Math.max(0, (int) Math.floor(center)));
                        pix = src.getpixel(sx, y);
                    } else {
                        for (int b = 0; b < bands; b++) {
                            pix[b] = ImagingBuffer.clamp8((int) Math.round(acc[b] / wsum));
                        }
                    }
                    out.putpixel(x, y, pix);
                }
            }
        } else {
            double scale = (double) inH / outH;
            double filterScale = Math.max(scale, 1.0);
            double radius = support * filterScale;
            for (int y = 0; y < outH; y++) {
                double center = (y + 0.5) * scale;
                int top = Math.max(0, (int) Math.floor(center - radius));
                int bot = Math.min(inH - 1, (int) Math.ceil(center + radius));
                for (int x = 0; x < outW; x++) {
                    double[] acc = new double[bands];
                    double wsum = 0;
                    for (int sy = top; sy <= bot; sy++) {
                        double w = kernel.weight((sy + 0.5 - center) / filterScale);
                        if (w == 0) continue;
                        int[] p = src.getpixel(x, sy);
                        for (int b = 0; b < bands; b++) {
                            acc[b] += p[b] * w;
                        }
                        wsum += w;
                    }
                    int[] pix = new int[bands];
                    if (wsum <= 1e-12) {
                        int sy = Math.min(inH - 1, Math.max(0, (int) Math.floor(center)));
                        pix = src.getpixel(x, sy);
                    } else {
                        for (int b = 0; b < bands; b++) {
                            pix[b] = ImagingBuffer.clamp8((int) Math.round(acc[b] / wsum));
                        }
                    }
                    out.putpixel(x, y, pix);
                }
            }
        }
        return out;
    }
}
