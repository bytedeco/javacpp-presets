package org.bytedeco.javacpp.opencv_imgproc;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = opencv_imgproc_presets.class)
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
