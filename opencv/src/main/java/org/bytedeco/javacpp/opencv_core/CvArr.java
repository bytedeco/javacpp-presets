package org.bytedeco.javacpp.opencv_core;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Opaque;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = opencv_core_presets.class)
@Opaque public class CvArr extends AbstractArray {
    public CvArr(Pointer p) { super(p); }

    @Override public int arrayChannels()          { throw new UnsupportedOperationException(); }
    @Override public int arrayDepth()             { throw new UnsupportedOperationException(); }
    @Override public int arrayOrigin()            { throw new UnsupportedOperationException(); }
    @Override public void arrayOrigin(int origin) { throw new UnsupportedOperationException(); }
    @Override public int arrayWidth()             { throw new UnsupportedOperationException(); }
    @Override public int arrayHeight()            { throw new UnsupportedOperationException(); }
    @Override public IplROI arrayROI()            { throw new UnsupportedOperationException(); }
    @Override public int arraySize()              { throw new UnsupportedOperationException(); }
    @Override public BytePointer arrayData()      { throw new UnsupportedOperationException(); }
    @Override public int arrayStep()              { throw new UnsupportedOperationException(); }
}
