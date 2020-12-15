import org.bytedeco.javacpp.*;

import org.bytedeco.opencl.*;
import static org.bytedeco.opencl.global.OpenCL.*;

public class VecAdd {
    static final int MEM_SIZE = 128;
    static final int MAX_SOURCE_SIZE = 0x100000;

    public static void main(String[] args) {
        _cl_platform_id platform_id = new _cl_platform_id(null);
        _cl_device_id device_id = new _cl_device_id(null);
        _cl_context context = new _cl_context(null);
        _cl_command_queue command_queue = new _cl_command_queue(null);
        _cl_mem memobj = new _cl_mem(null);
        _cl_program program = new _cl_program(null);
        _cl_kernel kernel = new _cl_kernel(null);
        IntPointer ret_num_devices = new IntPointer(1);
        IntPointer ret_num_platforms = new IntPointer(1);
        IntPointer ret_pointer = new IntPointer(1);
        int ret;

        FloatPointer mem = new FloatPointer(MEM_SIZE);
        String source_str = "__kernel void vecAdd(__global float* a) {"
                          + "    int gid = get_global_id(0);"
                          + "    a[gid] += a[gid];"
                          + "}";

        /* Initialize Data */
        for (int i = 0; i < MEM_SIZE; i++) {
            mem.put(i, i);
        }

        /* Get platform/device information */
        ret = clGetPlatformIDs(1, platform_id, ret_num_platforms);
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, device_id, ret_num_devices);

        /* Create OpenCL Context */
        context = clCreateContext(null, 1, device_id, null, null, ret_pointer);

        /* Create Command Queue */
        command_queue = clCreateCommandQueue(context, device_id, 0, ret_pointer);

        /* Create memory buffer*/
        memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * Loader.sizeof(FloatPointer.class), null, ret_pointer);

        /* Transfer data to memory buffer */
        ret = clEnqueueWriteBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * Loader.sizeof(FloatPointer.class), mem, 0, (PointerPointer)null, null);

        /* Create Kernel program from the read in source */
        program = clCreateProgramWithSource(context, 1, new PointerPointer(source_str), new SizeTPointer(1).put(source_str.length()), ret_pointer);

        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, device_id, null, null, null);

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "vecAdd", ret_pointer);

        /* Set OpenCL kernel argument */
        ret = clSetKernelArg(kernel, 0, Loader.sizeof(PointerPointer.class), new PointerPointer(1).put(memobj));

        SizeTPointer global_work_size = new SizeTPointer(MEM_SIZE, 0, 0);
        SizeTPointer local_work_size = new SizeTPointer(MEM_SIZE, 0, 0);

        /* Execute OpenCL kernel */
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, null, global_work_size, local_work_size, 0, (PointerPointer)null, null);

        /* Transfer result from the memory buffer */
        ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * Loader.sizeof(FloatPointer.class), mem, 0, (PointerPointer)null, null);

        /* Display result */
        for (int i = 0; i < MEM_SIZE; i++) {
            System.out.println("mem[" + i + "] : " + mem.get(i));
        }

        /* Finalization */
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(memobj);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

        System.exit(0);
    }
}
