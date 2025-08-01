// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.onnx;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.onnx.global.onnx.*;
  // namespace internal

// Represents one field in an UnknownFieldSet.
@Namespace("google::protobuf") @Properties(inherit = org.bytedeco.onnx.presets.onnx.class)
public class UnknownField extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public UnknownField() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public UnknownField(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public UnknownField(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public UnknownField position(long position) {
        return (UnknownField)super.position(position);
    }
    @Override public UnknownField getPointer(long i) {
        return new UnknownField((Pointer)this).offsetAddress(i);
    }

  /** enum google::protobuf::UnknownField::Type */
  public static final int
    TYPE_VARINT = 0,
    TYPE_FIXED32 = 1,
    TYPE_FIXED64 = 2,
    TYPE_LENGTH_DELIMITED = 3,
    TYPE_GROUP = 4;

  // The field's field number, as seen on the wire.
  public native int number();

  // The field type.
  public native @Cast("google::protobuf::UnknownField::Type") int type();

  // Accessors -------------------------------------------------------
  // Each method works only for UnknownFields of the corresponding type.

  public native @Cast("uint64_t") long varint();
  public native @Cast("uint32_t") int fixed32();
  public native @Cast("uint64_t") long fixed64();
  public native @StdString BytePointer length_delimited();
  public native @Const @ByRef UnknownFieldSet group();

  public native void set_varint(@Cast("uint64_t") long value);
  public native void set_fixed32(@Cast("uint32_t") int value);
  public native void set_fixed64(@Cast("uint64_t") long value);
  public native void set_length_delimited(@StdString BytePointer value);
  public native void set_length_delimited(@StdString String value);
  public native @StdString @Cast({"char*", "std::string*"}) BytePointer mutable_length_delimited();
  public native UnknownFieldSet mutable_group();

  public native @Cast("size_t") long GetLengthDelimitedSize();


  // If this UnknownField contains a pointer, delete it.
  public native void Delete();

  // Make a deep copy of any pointers in this UnknownField.
  public native void DeepCopy(@Const @ByRef UnknownField other);

  // Set the wire type of this UnknownField. Should only be used when this
  // UnknownField is being created.
  public native void SetType(@Cast("google::protobuf::UnknownField::Type") int type);

  public native @Cast("uint32_t") int number_(); public native UnknownField number_(int setter);
  public native @Cast("uint32_t") int type_(); public native UnknownField type_(int setter);
    @Name("data_.varint_") public native @Cast("uint64_t") long data__varint_(); public native UnknownField data__varint_(long setter);
    @Name("data_.fixed32_") public native @Cast("uint32_t") int data__fixed32_(); public native UnknownField data__fixed32_(int setter);
    @Name("data_.fixed64_") public native @Cast("uint64_t") long data__fixed64_(); public native UnknownField data__fixed64_(long setter);
    @Name("data_.group_") public native UnknownFieldSet data__group_(); public native UnknownField data__group_(UnknownFieldSet setter);
}
