// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletSoftBody;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;
import org.bytedeco.bullet.BulletCollision.*;
import static org.bytedeco.bullet.global.BulletCollision.*;
import org.bytedeco.bullet.BulletDynamics.*;
import static org.bytedeco.bullet.global.BulletDynamics.*;

import static org.bytedeco.bullet.global.BulletSoftBody.*;
  //for placement new
// #endif          //BT_USE_PLACEMENT_NEW

/**The btAlignedObjectArray template class uses a subset of the stl::vector interface for its methods
 * It is developed to replace stl::vector to avoid portability issues, including STL alignment issues to add SIMD/SSE data */
@Name("btAlignedObjectArray<LagrangeMultiplier>") @NoOffset @Properties(inherit = org.bytedeco.bullet.presets.BulletSoftBody.class)
public class LagrangeMultiplierArray extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LagrangeMultiplierArray(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public LagrangeMultiplierArray(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public LagrangeMultiplierArray position(long position) {
        return (LagrangeMultiplierArray)super.position(position);
    }
    @Override public LagrangeMultiplierArray getPointer(long i) {
        return new LagrangeMultiplierArray((Pointer)this).offsetAddress(i);
    }

	public native @ByRef @Name("operator =") LagrangeMultiplierArray put(@Const @ByRef LagrangeMultiplierArray other);
	public LagrangeMultiplierArray() { super((Pointer)null); allocate(); }
	private native void allocate();

	/**Generally it is best to avoid using the copy constructor of an btAlignedObjectArray, and use a (const) reference to the array instead. */
	public LagrangeMultiplierArray(@Const @ByRef LagrangeMultiplierArray otherArray) { super((Pointer)null); allocate(otherArray); }
	private native void allocate(@Const @ByRef LagrangeMultiplierArray otherArray);

	/** return the number of elements in the array */
	public native int size();

	public native @ByRef LagrangeMultiplier at(int n);

	public native @ByRef @Name("operator []") LagrangeMultiplier get(int n);

	/**clear the array, deallocated memory. Generally it is better to use array.resize(0), to reduce performance overhead of run-time memory (de)allocations. */
	public native void clear();

	public native void pop_back();

	/**resize changes the number of elements in the array. If the new size is larger, the new elements will be constructed using the optional second argument.
	 * when the new number of elements is smaller, the destructor will be called, but memory will not be freed, to reduce performance overhead of run-time memory (de)allocations. */
	public native void resizeNoInitialize(int newsize);

	public native void resize(int newsize, @Const @ByRef(nullValue = "LagrangeMultiplier()") LagrangeMultiplier fillData);
	public native void resize(int newsize);
	public native @ByRef LagrangeMultiplier expandNonInitializing();

	public native @ByRef LagrangeMultiplier expand(@Const @ByRef(nullValue = "LagrangeMultiplier()") LagrangeMultiplier fillValue);
	public native @ByRef LagrangeMultiplier expand();

	public native void push_back(@Const @ByRef LagrangeMultiplier _Val);

	/** return the pre-allocated (reserved) elements, this is at least as large as the total number of elements,see size() and reserve() */
	public native @Name("capacity") int _capacity();

	public native void reserve(int _Count);

	/**heap sort from http://www.csse.monash.edu.au/~lloyd/tildeAlgDS/Sort/Heap/ */ /*downHeap*/

	public native void swap(int index0, int index1);

	/**non-recursive binary search, assumes sorted array */
	

	

	// If the key is not in the array, return -1 instead of 0,
	// since 0 also means the first element in the array.
	

	public native void removeAtIndex(int index);
	

	//PCK: whole function
	public native void initializeFromBuffer(Pointer buffer, int size, int _capacity);

	public native void copyFromArray(@Const @ByRef LagrangeMultiplierArray otherArray);
}
