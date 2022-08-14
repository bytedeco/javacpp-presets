package org.bytedeco.kvazaar.samples.yuv;

import java.io.IOException;

/**
 * YUV Reader.
 */
public interface YuvReader {

    ColourSpace getColourSpace();

    YuvFrame getNextFrame() throws IOException;

    int getFrameHeight();

    int getFrameRateDenominator();

    int getFrameRateNumerator();

    int getFrameWidth();

    InterlacingMode getInterlacing();

    PixelAspectRatio getPixelAspectRatio();

    boolean hasMoreFrames() throws IOException;

    void readHeader() throws IOException;
    
}
