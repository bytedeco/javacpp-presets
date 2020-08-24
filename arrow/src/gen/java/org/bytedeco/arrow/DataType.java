// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.arrow;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.arrow.global.arrow.*;


/** \brief Base class for all data types
 * 
 *  Data types in this library are all *logical*. They can be expressed as
 *  either a primitive physical type (bytes or bits of some fixed size), a
 *  nested type consisting of other data types, or another data type (e.g. a
 *  timestamp encoded as an int64).
 * 
 *  Simple datatypes may be entirely described by their Type::type id, but
 *  complex datatypes are usually parametric. */
@Namespace("arrow") @NoOffset @Properties(inherit = org.bytedeco.arrow.presets.arrow.class)
public class DataType extends Fingerprintable {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DataType(Pointer p) { super(p); }


  /** \brief Return whether the types are equal
   * 
   *  Types that are logically convertible from one to another (e.g. List<UInt8>
   *  and Binary) are NOT equal. */
  public native @Cast("bool") boolean Equals(@Const @ByRef DataType other, @Cast("bool") boolean check_metadata/*=false*/);
  public native @Cast("bool") boolean Equals(@Const @ByRef DataType other);

  /** \brief Return whether the types are equal */

  public native @Deprecated @SharedPtr @Cast({"", "std::shared_ptr<arrow::Field>"}) Field child(int i);

  /** Returns the the child-field at index i. */
  public native @SharedPtr @Cast({"", "std::shared_ptr<arrow::Field>"}) Field field(int i);

  public native @Const @Deprecated @ByRef FieldVector children();

  /** \brief Returns the children fields associated with this type. */
  public native @Const @ByRef FieldVector fields();

  public native @Deprecated int num_children();

  /** \brief Returns the number of children fields associated with this type. */
  public native int num_fields();

  public native @ByVal Status Accept(TypeVisitor visitor);

  /** \brief A string representation of the type, including any children */
  public native @StdString String ToString();

  /** \brief Return hash value (excluding metadata in child fields) */
  
  ///
  public native @Cast("size_t") long Hash();

  /** \brief A string name of the type, omitting any child fields
   * 
   *  \note Experimental API
   *  @since 0.7.0 */
  
  ///
  public native @StdString String name();

  /** \brief Return the data type layout.  Children are not included.
   * 
   *  \note Experimental API */
  public native @ByVal DataTypeLayout layout();

  /** \brief Return the type category */
  public native @Cast("arrow::Type::type") int id();
}