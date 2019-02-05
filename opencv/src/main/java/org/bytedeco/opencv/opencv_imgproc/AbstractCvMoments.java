package org.bytedeco.opencv.opencv_imgproc;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_imgproc.class)
public abstract class AbstractCvMoments extends Pointer {
    public AbstractCvMoments(Pointer p) { super(p); }

    public static ThreadLocal<CvMoments> createThreadLocal() {
        return new ThreadLocal<CvMoments>() {
            @Override protected CvMoments initialValue() {
                return new CvMoments();
            }
        };
    }
}
