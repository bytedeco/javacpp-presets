package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;

import static org.bytedeco.javacpp.opencv_core.opencv_core.cvCreateSet;

public abstract class AbstractCvSet extends CvSeq {
      public AbstractCvSet(Pointer p) { super(p); }

      public static CvSet create(int set_flags, int header_size, int elem_size,
              CvMemStorage storage) {
          return cvCreateSet(set_flags, header_size, elem_size, storage);
      }
  }
