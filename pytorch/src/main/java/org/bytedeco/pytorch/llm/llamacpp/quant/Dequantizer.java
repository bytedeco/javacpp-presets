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

package org.bytedeco.pytorch.llm.llamacpp.quant;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * GGML block dequantizers (Q4_0 / Q4_1 / Q8_0) → float32.
 * Layout matches ggml reference (32-element blocks).
 *
 * <p>Q4_0 block (18 bytes): {@code f16 scale | 16 bytes nibbles (32 x 4-bit)}.
 * Q4_1 block (20 bytes): {@code f16 scale | f16 min | 16 bytes nibbles}.
 * Q8_0 block (34 bytes): {@code f16 scale | 32 x int8}.
 */
public final class Dequantizer {

    public static final int QK4_0 = 32;
    public static final int QK4_1 = 32;
    public static final int QK8_0 = 32;
    public static final int BLOCK_Q4_0 = 18;
    public static final int BLOCK_Q4_1 = 20;
    public static final int BLOCK_Q8_0 = 34;

    private Dequantizer() {}

    public static float[] dequant(byte[] data, long nElements, GgmlQuantType type) {
        if (data == null) throw new IllegalArgumentException("data");
        int n = (int) Math.min(nElements, Integer.MAX_VALUE);
        return switch (type) {
            case F32 -> decodeF32(data, n);
            case F16, BF16 -> decodeF16(data, n); // bf16 approx via f16 path for smoke
            case Q4_0 -> dequantQ4_0(data, n);
            case Q4_1 -> dequantQ4_1(data, n);
            case Q8_0 -> dequantQ8_0(data, n);
            case I8 -> decodeI8(data, n);
            default -> throw new UnsupportedOperationException("dequant not implemented for " + type);
        };
    }

    public static float[] dequantQ4_0(byte[] data, int nElements) {
        int nBlocks = (nElements + QK4_0 - 1) / QK4_0;
        float[] out = new float[nElements];
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int o = 0;
        for (int b = 0; b < nBlocks && o < nElements; b++) {
            if (bb.remaining() < BLOCK_Q4_0) break;
            float d = fp16ToFloat(bb.getShort() & 0xffff);
            byte[] qs = new byte[16];
            bb.get(qs);
            for (int i = 0; i < 16 && o < nElements; i++) {
                int q = qs[i] & 0xff;
                int x0 = (q & 0x0f) - 8;
                out[o++] = x0 * d;
                if (o >= nElements) break;
                int x1 = (q >> 4) - 8;
                out[o++] = x1 * d;
            }
        }
        return out;
    }

    public static float[] dequantQ4_1(byte[] data, int nElements) {
        int nBlocks = (nElements + QK4_1 - 1) / QK4_1;
        float[] out = new float[nElements];
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int o = 0;
        for (int b = 0; b < nBlocks && o < nElements; b++) {
            if (bb.remaining() < BLOCK_Q4_1) break;
            float d = fp16ToFloat(bb.getShort() & 0xffff);
            float m = fp16ToFloat(bb.getShort() & 0xffff);
            byte[] qs = new byte[16];
            bb.get(qs);
            for (int i = 0; i < 16 && o < nElements; i++) {
                int q = qs[i] & 0xff;
                out[o++] = (q & 0x0f) * d + m;
                if (o >= nElements) break;
                out[o++] = (q >> 4) * d + m;
            }
        }
        return out;
    }

    public static float[] dequantQ8_0(byte[] data, int nElements) {
        int nBlocks = (nElements + QK8_0 - 1) / QK8_0;
        float[] out = new float[nElements];
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int o = 0;
        for (int b = 0; b < nBlocks && o < nElements; b++) {
            if (bb.remaining() < BLOCK_Q8_0) break;
            float d = fp16ToFloat(bb.getShort() & 0xffff);
            for (int i = 0; i < QK8_0 && o < nElements; i++) {
                out[o++] = bb.get() * d;
            }
        }
        return out;
    }

    /** Build a synthetic Q4_0 payload from float values (for unit tests). */
    public static byte[] quantizeQ4_0(float[] values) {
        int n = values.length;
        int nBlocks = (n + QK4_0 - 1) / QK4_0;
        ByteBuffer bb = ByteBuffer.allocate(nBlocks * BLOCK_Q4_0).order(ByteOrder.LITTLE_ENDIAN);
        for (int b = 0; b < nBlocks; b++) {
            int base = b * QK4_0;
            float amax = 0;
            for (int i = 0; i < QK4_0; i++) {
                int idx = base + i;
                float v = idx < n ? values[idx] : 0;
                amax = Math.max(amax, Math.abs(v));
            }
            float d = amax / 7f;
            if (d == 0) d = 1e-6f;
            bb.putShort(floatToFp16(d));
            for (int i = 0; i < 16; i++) {
                int i0 = base + i;
                int i1 = base + i + 16;
                int q0 = i0 < n ? Math.max(0, Math.min(15, Math.round(values[i0] / d) + 8)) : 8;
                int q1 = i1 < n ? Math.max(0, Math.min(15, Math.round(values[i1] / d) + 8)) : 8;
                bb.put((byte) ((q1 << 4) | (q0 & 0x0f)));
            }
        }
        return bb.array();
    }

    private static float[] decodeF32(byte[] data, int n) {
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[n];
        for (int i = 0; i < n && bb.remaining() >= 4; i++) out[i] = bb.getFloat();
        return out;
    }

    private static float[] decodeF16(byte[] data, int n) {
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[n];
        for (int i = 0; i < n && bb.remaining() >= 2; i++) {
            out[i] = fp16ToFloat(bb.getShort() & 0xffff);
        }
        return out;
    }

    private static float[] decodeI8(byte[] data, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n && i < data.length; i++) out[i] = data[i];
        return out;
    }

    /** IEEE-ish fp16 → float32. */
    public static float fp16ToFloat(int h) {
        int s = (h >>> 15) & 1;
        int e = (h >>> 10) & 0x1f;
        int f = h & 0x3ff;
        int out;
        if (e == 0) {
            if (f == 0) out = s << 31;
            else {
                // subnormal
                while ((f & 0x400) == 0) { f <<= 1; e--; }
                e++;
                f &= 0x3ff;
                out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
            }
        } else if (e == 31) {
            out = (s << 31) | 0x7f800000 | (f << 13);
        } else {
            out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
        }
        return Float.intBitsToFloat(out);
    }

    public static short floatToFp16(float v) {
        int bits = Float.floatToIntBits(v);
        int s = (bits >>> 16) & 0x8000;
        int e = ((bits >>> 23) & 0xff) - 127 + 15;
        int f = bits & 0x7fffff;
        if (e <= 0) {
            if (e < -10) return (short) s;
            f = (f | 0x800000) >> (1 - e);
            return (short) (s | (f >> 13));
        } else if (e >= 31) {
            return (short) (s | 0x7c00);
        }
        return (short) (s | (e << 10) | (f >> 13));
    }
}
