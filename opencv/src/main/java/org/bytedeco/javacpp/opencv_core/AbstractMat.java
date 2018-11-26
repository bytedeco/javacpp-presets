package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;

import static org.bytedeco.javacpp.opencv_core.opencv_core.*;

public abstract class AbstractMat extends AbstractArray {
      public AbstractMat(Pointer p) { super(p); }

      public abstract void create(int rows, int cols, int type);
      public abstract void release();
      public abstract int type();
      public abstract int depth();
      public abstract int channels();
      public abstract int rows();
      public abstract int cols();
      public abstract BytePointer data();
      public abstract int size(int i);
      public abstract int step(int i);

      @Override public int arrayChannels() { return channels(); }
      @Override public int arrayDepth() {
          switch (depth()) {
              case CV_8U : return IPL_DEPTH_8U;
              case CV_8S : return IPL_DEPTH_8S;
              case CV_16U: return IPL_DEPTH_16U;
              case CV_16S: return IPL_DEPTH_16S;
              case CV_32S: return IPL_DEPTH_32S;
              case CV_32F: return IPL_DEPTH_32F;
              case CV_64F: return IPL_DEPTH_64F;
              default: assert (false);
          }
          return -1;
      }
      @Override public int arrayOrigin() { return 0; }
      @Override public void arrayOrigin(int origin) { }
      @Override public int arrayWidth() { return cols(); }
      @Override public int arrayHeight() { return rows(); }
      @Override public IplROI arrayROI() { return null; }
      @Override public int arraySize() { return step(0)*size(0); }
      @Override public BytePointer arrayData() { return data(); }
      @Override public int arrayStep() { return step(0); }

      public static final Mat EMPTY = null;
  }
