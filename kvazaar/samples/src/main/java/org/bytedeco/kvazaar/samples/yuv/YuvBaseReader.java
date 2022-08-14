/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.bytedeco.kvazaar.samples.yuv;

import java.io.IOException;
import java.nio.channels.SeekableByteChannel;

/**
 *
 * @author bradh
 */
public abstract class YuvBaseReader implements YuvReader {
    

    protected int frameWidth;
    protected int frameHeight;
    protected int frameRateNumerator;
    protected int frameRateDenominator;
    protected InterlacingMode interlacing;
    protected PixelAspectRatio pixelAspectRatio;
    protected ColourSpace colourSpace = ColourSpace.YUV420;

    @Override
    public abstract YuvFrame getNextFrame() throws IOException;

    @Override
    public int getFrameWidth() {
        return frameWidth;
    }

    @Override
    public int getFrameHeight() {
        return frameHeight;
    }

    @Override
    public int getFrameRateNumerator() {
        return frameRateNumerator;
    }

    @Override
    public int getFrameRateDenominator() {
        return frameRateDenominator;
    }

    @Override
    public InterlacingMode getInterlacing() {
        return interlacing;
    }

    @Override
    public PixelAspectRatio getPixelAspectRatio() {
        return pixelAspectRatio;
    }

    @Override
    public ColourSpace getColourSpace() {
        return colourSpace;
    }

    protected int getFrameSizeBytes() {
        switch (this.colourSpace) {
            case YUV420:
                return frameWidth * frameHeight * 3 / 2;
            case YUV422:
                return frameWidth * frameHeight * 2;
            case YUV444:
                return frameWidth * frameHeight * 3;
            case UNKNOWN:
            default:
                throw new AssertionError(this.colourSpace.name());
        }
    }
    
        protected int getLumaPlaneSizeInBytes() {
            return frameWidth * frameHeight;
    }
        
                protected int getChromaPlaneSizeInBytes() {
        switch (this.colourSpace) {
            case YUV420:
                return frameWidth * frameHeight / 4;
            case YUV422:
                return frameWidth * frameHeight / 2;
            case YUV444:
                return frameWidth * frameHeight;
            case UNKNOWN:
            default:
                throw new AssertionError(this.colourSpace.name());
        }
    }


    
}
