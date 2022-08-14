package org.bytedeco.kvazaar.samples.yuv;

/**
 * Pixel interlacing mode.
 */
public enum InterlacingMode {
    /**
     * Unknown mode.
     * 
     * <p>This is not valid, and should not be created.
     */
    Unknown('?'),
    /**
     * Progressive.
     */
    Progressive ('p'),
    /**
     * Top field first.
     */
    TopFirst('t'),
    /**
     * Bottom field first.
     */
    BottomFirst('b'),
    /**
     * Mixed modes.
     * 
     * <p>This is usually associated with per-frame interlacing.
     */
    Mixed('m');
    
    private final char modeCode;

    private InterlacingMode(char modeCode) {
        this.modeCode = modeCode;
    }

    public char getModeCode() {
        return modeCode;
    }
    
    public static InterlacingMode lookup(char code) {
        for (InterlacingMode mode: values()) {
            if (mode.getModeCode()== code) {
                return mode;
            }
        }
        return Unknown;
    }
    
    public static InterlacingMode lookup(String code) {
        return lookup(code.charAt(0));
    }
}
