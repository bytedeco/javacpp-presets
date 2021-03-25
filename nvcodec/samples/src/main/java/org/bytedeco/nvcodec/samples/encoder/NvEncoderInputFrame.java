package org.bytedeco.nvcodec.samples.encoder;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.nvcodec.global.nvencodeapi.*;

public class NvEncoderInputFrame {
    private Pointer inputPointer;
    private int[] chromaOffsets;
    private int numChromaPlanes;
    private int pitch;
    private int chromaPitch;
    private int bufferFormat;
    private int resourceType;

    public NvEncoderInputFrame() {
        this.chromaOffsets = new int[2];
    }

    public Pointer getInputPointer() {
        return inputPointer;
    }

    public void setInputPointer(Pointer inputPointer) {
        this.inputPointer = inputPointer;
    }

    public int[] getChromaOffsets() {
        return chromaOffsets;
    }

    public void setChromaOffsets(int[] chromaOffsets) {
        this.chromaOffsets = chromaOffsets;
    }

    public int getNumChromaPlanes() {
        return numChromaPlanes;
    }

    public void setNumChromaPlanes(int numChromaPlanes) {
        this.numChromaPlanes = numChromaPlanes;
    }

    public int getPitch() {
        return pitch;
    }

    public void setPitch(int pitch) {
        this.pitch = pitch;
    }

    public int getChromaPitch() {
        return chromaPitch;
    }

    public void setChromaPitch(int chromaPitch) {
        this.chromaPitch = chromaPitch;
    }

    public int getBufferFormat() {
        return bufferFormat;
    }

    public void setBufferFormat(int bufferFormat) {
        this.bufferFormat = bufferFormat;
    }

    public int getResourceType() {
        return resourceType;
    }

    public void setResourceType(int resourceType) {
        this.resourceType = resourceType;
    }
}
