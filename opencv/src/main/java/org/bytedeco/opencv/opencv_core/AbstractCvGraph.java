package org.bytedeco.opencv.opencv_core;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Properties;

import static org.bytedeco.opencv.global.opencv_core.*;

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public abstract class AbstractCvGraph extends CvSet {
    public AbstractCvGraph(Pointer p) { super(p); }

    public static CvGraph create(int graph_flags, int header_size, int vtx_size,
            int edge_size, CvMemStorage storage) {
        return cvCreateGraph(graph_flags, header_size, vtx_size, edge_size, storage);
    }
}
