// Targeted by JavaCPP version 1.5.8-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.Bullet3OpenCL;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.Bullet3Common.*;
import static org.bytedeco.bullet.global.Bullet3Common.*;
import org.bytedeco.bullet.Bullet3Collision.*;
import static org.bytedeco.bullet.global.Bullet3Collision.*;
import org.bytedeco.bullet.Bullet3Dynamics.*;
import static org.bytedeco.bullet.global.Bullet3Dynamics.*;

import static org.bytedeco.bullet.global.Bullet3OpenCL.*;
  //for b3SortData (perhaps move it?)
@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.Bullet3OpenCL.class)
public class b3BoundSearchCL extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public b3BoundSearchCL(Pointer p) { super(p); }

	/** enum b3BoundSearchCL::Option */
	public static final int
		BOUND_LOWER = 0,
		BOUND_UPPER = 1,
		COUNT = 2;

	public native @ByRef cl_context m_context(); public native b3BoundSearchCL m_context(cl_context setter);
	public native @ByRef cl_device_id m_device(); public native b3BoundSearchCL m_device(cl_device_id setter);
	public native @ByRef cl_command_queue m_queue(); public native b3BoundSearchCL m_queue(cl_command_queue setter);

	public native @ByRef cl_kernel m_lowerSortDataKernel(); public native b3BoundSearchCL m_lowerSortDataKernel(cl_kernel setter);
	public native @ByRef cl_kernel m_upperSortDataKernel(); public native b3BoundSearchCL m_upperSortDataKernel(cl_kernel setter);
	public native @ByRef cl_kernel m_subtractKernel(); public native b3BoundSearchCL m_subtractKernel(cl_kernel setter);

	public native b3Int4OCLArray m_constbtOpenCLArray(); public native b3BoundSearchCL m_constbtOpenCLArray(b3Int4OCLArray setter);
	public native b3UnsignedIntOCLArray m_lower(); public native b3BoundSearchCL m_lower(b3UnsignedIntOCLArray setter);
	public native b3UnsignedIntOCLArray m_upper(); public native b3BoundSearchCL m_upper(b3UnsignedIntOCLArray setter);

	public native b3FillCL m_filler(); public native b3BoundSearchCL m_filler(b3FillCL setter);

	public b3BoundSearchCL(@ByVal cl_context context, @ByVal cl_device_id device, @ByVal cl_command_queue queue, int size) { super((Pointer)null); allocate(context, device, queue, size); }
	private native void allocate(@ByVal cl_context context, @ByVal cl_device_id device, @ByVal cl_command_queue queue, int size);

	//	src has to be src[i].m_key <= src[i+1].m_key
	public native void execute(@ByRef b3SortDataOCLArray src, int nSrc, @ByRef b3UnsignedIntOCLArray dst, int nDst, @Cast("b3BoundSearchCL::Option") int option/*=b3BoundSearchCL::BOUND_LOWER*/);
	public native void execute(@ByRef b3SortDataOCLArray src, int nSrc, @ByRef b3UnsignedIntOCLArray dst, int nDst);

	public native void executeHost(@ByRef b3SortDataArray src, int nSrc, @ByRef b3UnsignedIntArray dst, int nDst, @Cast("b3BoundSearchCL::Option") int option/*=b3BoundSearchCL::BOUND_LOWER*/);
	public native void executeHost(@ByRef b3SortDataArray src, int nSrc, @ByRef b3UnsignedIntArray dst, int nDst);
}