package org.bytedeco.kvazaar.samples.yuv;

/**
 * Colour Space options.
 */
public enum ColourSpace {
    YUV420("420", 2, 2),
    YUV422("422", 2, 1),
    YUV444("444", 1, 1),
    UNKNOWN("", 0, 0);
    
        private final String code;
        private final int horizontalSubsampling;
        private final int verticalSubsampling;

    private ColourSpace(String code, int horizontalSubsampling, int verticalSubsampling) {
        this.code = code;
        this.horizontalSubsampling = horizontalSubsampling;
        this.verticalSubsampling = verticalSubsampling;
    }

    public String getEncodedValue() {
        return code;
    }

    public int getHorizontalSubsampling() {
        return horizontalSubsampling;
    }

    public int getVerticalSubsampling() {
        return verticalSubsampling;
    }
    
        public static ColourSpace lookup(String code) {
        for (ColourSpace mode: values()) {
            if (mode.getEncodedValue().equals(code)){
                return mode;
            }
        }
        return UNKNOWN;
    }
}
