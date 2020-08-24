// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.arrow;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.arrow.global.arrow.*;


@Name("arrow::IntervalScalar<arrow::DayTimeIntervalType>") @Properties(inherit = org.bytedeco.arrow.presets.arrow.class)
public class BaseDayTimeIntervalScalar extends BaseBaseDayTimeIntervalScalar {
    static { Loader.load(); }

  
  
    public BaseDayTimeIntervalScalar(@ByVal DayTimeIntervalType.DayMilliseconds value, @SharedPtr @Cast({"", "std::shared_ptr<arrow::DataType>"}) DataType type) { super((Pointer)null); allocate(value, type); }
    private native void allocate(@ByVal DayTimeIntervalType.DayMilliseconds value, @SharedPtr @Cast({"", "std::shared_ptr<arrow::DataType>"}) DataType type);
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BaseDayTimeIntervalScalar(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BaseDayTimeIntervalScalar(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public BaseDayTimeIntervalScalar position(long position) {
        return (BaseDayTimeIntervalScalar)super.position(position);
    }
    @Override public BaseDayTimeIntervalScalar getPointer(long i) {
        return new BaseDayTimeIntervalScalar(this).position(position + i);
    }


  public BaseDayTimeIntervalScalar(@ByVal @Cast("arrow::IntervalScalar<arrow::DayTimeIntervalType>::ValueType*") DayTimeIntervalType.DayMilliseconds value) { super((Pointer)null); allocate(value); }
  private native void allocate(@ByVal @Cast("arrow::IntervalScalar<arrow::DayTimeIntervalType>::ValueType*") DayTimeIntervalType.DayMilliseconds value);
  public BaseDayTimeIntervalScalar() { super((Pointer)null); allocate(); }
  private native void allocate();
}