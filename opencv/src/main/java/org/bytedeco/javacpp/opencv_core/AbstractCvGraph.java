package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;

import static org.bytedeco.javacpp.opencv_core.opencv_core.cvCreateGraph;

public abstract class AbstractCvGraph extends CvSet {
      public AbstractCvGraph(Pointer p) { super(p); }

      public static CvGraph create(int graph_flags, int header_size, int vtx_size,
              int edge_size, CvMemStorage storage) {
          return cvCreateGraph(graph_flags, header_size, vtx_size, edge_size, storage);
      }
  }
