package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.ValueGetter;

@Name("IplImage*")
  public class IplImageArray extends CvArrArray {
      public IplImageArray(IplImage ... array) { this(array.length); put(array); position(0); }
      public IplImageArray(long size) { allocateArray(size); }
      public IplImageArray(Pointer p) { super(p); }
      private native void allocateArray(long size);

      @Override public IplImageArray position(long position) {
          return (IplImageArray)super.position(position);
      }
      @Override public IplImageArray put(CvArr ... array) {
          return (IplImageArray)super.put(array);
      }
      @Override @ValueGetter
      public native IplImage get();
      @Override public IplImageArray put(CvArr p) {
          if (p instanceof IplImage) {
              return (IplImageArray)super.put(p);
          } else {
              throw new ArrayStoreException(p.getClass().getName());
          }
      }
  }
