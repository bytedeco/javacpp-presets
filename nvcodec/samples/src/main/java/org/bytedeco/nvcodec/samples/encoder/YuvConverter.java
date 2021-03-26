package org.bytedeco.nvcodec.samples.encoder;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;

import static org.bytedeco.javacpp.Pointer.*;

public class YuvConverter {
    private BytePointer quad;
    private int width;
    private int height;

    private int bitDepth;
    private final int byteDepth;

    public YuvConverter(int width, int height, int bitDepth) {
        this.width = width;
        this.height = height;
        this.bitDepth = bitDepth;
        this.byteDepth = bitDepth / 8;

        this.quad = new BytePointer((long) width * height / 4 * this.byteDepth);
    }

    public void planarToUVInterleaved(Pointer frame) {
        this.planarToUVInterleaved(frame, 0);
    }

    public void planarToUVInterleaved(Pointer frame, int pitch) {
        if (pitch == 0) {
            pitch = this.width;
        }

        Pointer puv = frame.getPointer((long) pitch * this.height);

        if (pitch == this.width) {
            memcpy(this.quad, puv, ((long) this.width * this.height / 4) * this.byteDepth);
        } else {
            for (int i = 0; i < this.height / 2; i++) {
                memcpy(this.quad.getPointer(((long) this.width / 2 * i) * this.byteDepth), puv.getPointer(((long) pitch / 2 * i) * this.byteDepth), (long) this.width / 2 * this.byteDepth);
            }
        }

        Pointer pv = puv.getPointer(((long) (pitch / 2) * (height / 2) * this.byteDepth));


        for (int y = 0; y < this.height / 2; y++) {
            for (int x = 0; x < this.width / 2; x++) {
                memcpy(pv.getPointer(((long) y * pitch + x * 2) * this.byteDepth), this.quad.getPointer(((long) y * this.width / 2 + x) * this.byteDepth), this.byteDepth);

                memcpy(pv.getPointer(((long) y * pitch + x * 2 + 1) * this.byteDepth), this.quad.getPointer(((long) y * pitch / 2 + x) * this.byteDepth), this.byteDepth);
            }
        }

    }

    public void UVInterleavedToPlanar(BytePointer frame) {
        this.UVInterleavedToPlanar(frame, 0);
    }

    public void UVInterleavedToPlanar(BytePointer frame, int pitch) {
        if (pitch == 0) {
            pitch = this.width;
        }

        BytePointer puv = frame.getPointer((long) pitch * height * this.byteDepth);
        BytePointer pu = new BytePointer(puv);
        BytePointer pv = puv.getPointer((long) pitch * height / 4 * this.byteDepth);

        byte[] tempBuffer = new byte[this.byteDepth];

        for (int y = 0; y < this.height / 2; y++) {
            for (int x = 0; x < this.width / 2; x++) {
                memcpy(pu.getPointer(((long) y * pitch / 2 + x) * this.byteDepth), puv.getPointer(((long) y * pitch + x * 2) * this.byteDepth), this.byteDepth);
                memcpy(this.quad.getPointer(((long) y * width / 2 + x) * this.byteDepth), puv.getPointer(((long) y * pitch + x * 2 + 1) * this.byteDepth), this.byteDepth);
            }
        }

        if (pitch == this.width) {
            memcpy(pv, this.quad, ((long) this.width * this.height / 4) * this.byteDepth);
        } else {
            for (int i = 0; i < this.height / 2; i++) {
                memcpy(pv.getPointer(((long) pitch / 2 * i) * this.byteDepth), this.quad.getPointer(((long) this.width / 2 * i) * this.byteDepth), (long) this.width / 2 * this.byteDepth);
            }
        }
    }
}
