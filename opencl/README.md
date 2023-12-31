JavaCPP Presets for OpenCL 
==========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/opencl/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/opencl) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/opencl.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![opencl](https://github.com/bytedeco/javacpp-presets/workflows/opencl/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Aopencl)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * OpenCL 3.0.15  https://www.khronos.org/registry/OpenCL/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/opencl/apidocs/


Sample Usage
------------
Here is a simple example of OpenCL ported to Java from this C source code:

 * https://us.fixstars.com/products/opencl/book/OpenCLProgrammingBook/online-offline-compilation/

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `VecAdd.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.opencl</groupId>
    <artifactId>vecadd</artifactId>
    <version>1.5.9</version>
    <properties>
        <exec.mainClass>VecAdd</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>opencl-platform</artifactId>
            <version>3.0-1.5.9</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `VecAdd.java` source file
```java
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
```
