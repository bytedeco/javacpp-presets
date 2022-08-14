package org.bytedeco.kvazaar.samples.yuv;


/**
 * Pixel aspect ratio.
 */
public enum PixelAspectRatio {
    Unknown("0:0"),
    Square("1:1"),
    NTSC_SVCD("4:3"),
    NTSC_DVD("4:5"),
    NTSC_DVD_WIDE("32:27");
    
    private final String ratioCode;

    private PixelAspectRatio(String ratioCode) {
        this.ratioCode = ratioCode;
    }

    public String getRatioCode() {
        return ratioCode;
    }
    
        public static PixelAspectRatio lookup(String code) {
        for (PixelAspectRatio mode: values()) {
            if (mode.getRatioCode().equals(code)){
                return mode;
            }
        }
        return Unknown;
    }
    
}
