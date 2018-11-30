package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = opencv_core_presets.class)
public abstract class AbstractCvFont extends Pointer {
    public AbstractCvFont(Pointer p) { super(p); }

//        public AbstractCvFont(int font_face, double hscale, double vscale,
//            double shear, int thickness, int line_type) {
//            allocate();
//            cvInitFont(this, font_face, hscale, vscale, shear, thickness, line_type);
//        }
//        public AbstractCvFont(int font_face, double scale, int thickness) {
//            allocate();
//            cvInitFont(this, font_face, scale, scale, 0, thickness, CV_AA);
//        }
}
