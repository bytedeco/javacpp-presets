/*
 * Copyright (C) 2022 Andrey Krainyak
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



// This #ifdef disables these stubs during native code compilation,
// as the stubs should be used for generation of the Java-side
// code only, and the native code should use the original definitions
// provided by clew's header.
#ifdef XXXXXXXXXX

using BuildProgramCallback = void (*)(cl_program, void*);
cl_int clBuildProgram(cl_program /* program */, cl_uint /* num_devices */, const cl_device_id * /* device_list */, const char * /* options */, BuildProgramCallback, void * /* user_data */);

using CreateContextCallback = void (*)(const char*, const void*, size_t, void*);
cl_context clCreateContext(const cl_context_properties * /* properties */, cl_uint /* num_devices */, const cl_device_id * /* devices */, CreateContextCallback, void * /* user_data */, cl_int * /* errcode_ret */);
cl_context clCreateContextFromType(const cl_context_properties * /* properties */, cl_device_type /* device_type */, CreateContextCallback, void * /* user_data */, cl_int * /* errcode_ret */);

using EnqueueNativeKernelCallback = void (*)(void*);
cl_int clEnqueueNativeKernel(cl_command_queue /* command_queue */, EnqueueNativeKernelCallback, void * /* args */, size_t /* cb_args */, cl_uint /* num_mem_objects */, const cl_mem * /* mem_list */, const void ** /* args_mem_loc */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);

using EventCallback = void (*)(cl_event, cl_int, void*);
cl_int clSetEventCallback(cl_event /* event */, cl_int /* command_exec_callback_type */, EventCallback, void * /* user_data */);

using MemObjectDestructorCallback = void (*)(cl_mem, void*);
cl_int clSetMemObjectDestructorCallback(cl_mem /* memobj */, MemObjectDestructorCallback, void * /*user_data */);

/***************************************************************************
              Generated with bullet/gen-clew-stubs script
***************************************************************************/

cl_mem clCreateBuffer(cl_context /* context */, cl_mem_flags /* flags */, size_t /* size */, void * /* host_ptr */, cl_int * /* errcode_ret */);
cl_command_queue clCreateCommandQueue(cl_context /* context */, cl_device_id /* device */, cl_command_queue_properties /* properties */, cl_int * /* errcode_ret */);
cl_mem clCreateImage2D(cl_context /* context */, cl_mem_flags /* flags */, const cl_image_format * /* image_format */, size_t /* image_width */, size_t /* image_height */, size_t /* image_row_pitch */, void * /* host_ptr */, cl_int * /* errcode_ret */);
cl_mem clCreateImage3D(cl_context /* context */, cl_mem_flags /* flags */, const cl_image_format * /* image_format */, size_t /* image_width */, size_t /* image_height */, size_t /* image_depth */, size_t /* image_row_pitch */, size_t /* image_slice_pitch */, void * /* host_ptr */, cl_int * /* errcode_ret */);
cl_kernel clCreateKernel(cl_program /* program */, const char * /* kernel_name */, cl_int * /* errcode_ret */);
cl_int clCreateKernelsInProgram(cl_program /* program */, cl_uint /* num_kernels */, cl_kernel * /* kernels */, cl_uint * /* num_kernels_ret */);
cl_program clCreateProgramWithBinary(cl_context /* context */, cl_uint /* num_devices */, const cl_device_id * /* device_list */, const size_t * /* lengths */, const unsigned char ** /* binaries */, cl_int * /* binary_status */, cl_int * /* errcode_ret */);
cl_program clCreateProgramWithSource(cl_context /* context */, cl_uint /* count */, const char ** /* strings */, const size_t * /* lengths */, cl_int * /* errcode_ret */);
cl_sampler clCreateSampler(cl_context /* context */, cl_bool /* normalized_coords */, cl_addressing_mode /* addressing_mode */, cl_filter_mode /* filter_mode */, cl_int * /* errcode_ret */);
cl_mem clCreateSubBuffer(cl_mem /* buffer */, cl_mem_flags /* flags */, cl_buffer_create_type /* buffer_create_type */, const void * /* buffer_create_info */, cl_int * /* errcode_ret */);
cl_event clCreateUserEvent(cl_context /* context */, cl_int * /* errcode_ret */);
cl_int clEnqueueBarrier(cl_command_queue /* command_queue */);
cl_int clEnqueueCopyBuffer(cl_command_queue /* command_queue */, cl_mem /* src_buffer */, cl_mem /* dst_buffer */, size_t /* src_offset */, size_t /* dst_offset */, size_t /* cb */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueCopyBufferRect(cl_command_queue /* command_queue */, cl_mem /* src_buffer */, cl_mem /* dst_buffer */, const size_t * /* src_origin */, const size_t * /* dst_origin */, const size_t * /* region */, size_t /* src_row_pitch */, size_t /* src_slice_pitch */, size_t /* dst_row_pitch */, size_t /* dst_slice_pitch */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueCopyBufferToImage(cl_command_queue /* command_queue */, cl_mem /* src_buffer */, cl_mem /* dst_image */, size_t /* src_offset */, const size_t * /* dst_origin[3] */, const size_t * /* region[3] */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueCopyImage(cl_command_queue /* command_queue */, cl_mem /* src_image */, cl_mem /* dst_image */, const size_t * /* src_origin[3] */, const size_t * /* dst_origin[3] */, const size_t * /* region[3] */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueCopyImageToBuffer(cl_command_queue /* command_queue */, cl_mem /* src_image */, cl_mem /* dst_buffer */, const size_t * /* src_origin[3] */, const size_t * /* region[3] */, size_t /* dst_offset */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
void clEnqueueMapBuffer(cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_map */, cl_map_flags /* map_flags */, size_t /* offset */, size_t /* cb */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */, cl_int * /* errcode_ret */);
void clEnqueueMapImage(cl_command_queue /* command_queue */, cl_mem /* image */, cl_bool /* blocking_map */, cl_map_flags /* map_flags */, const size_t * /* origin[3] */, const size_t * /* region[3] */, size_t * /* image_row_pitch */, size_t * /* image_slice_pitch */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */, cl_int * /* errcode_ret */);
cl_int clEnqueueMarker(cl_command_queue /* command_queue */, cl_event * /* event */);
cl_int clEnqueueNDRangeKernel(cl_command_queue /* command_queue */, cl_kernel /* kernel */, cl_uint /* work_dim */, const size_t * /* global_work_offset */, const size_t * /* global_work_size */, const size_t * /* local_work_size */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueReadBuffer(cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_read */, size_t /* offset */, size_t /* cb */, void * /* ptr */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueReadBufferRect(cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_read */, const size_t * /* buffer_origin */, const size_t * /* host_origin */, const size_t * /* region */, size_t /* buffer_row_pitch */, size_t /* buffer_slice_pitch */, size_t /* host_row_pitch */, size_t /* host_slice_pitch */, void * /* ptr */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueReadImage(cl_command_queue /* command_queue */, cl_mem /* image */, cl_bool /* blocking_read */, const size_t * /* origin[3] */, const size_t * /* region[3] */, size_t /* row_pitch */, size_t /* slice_pitch */, void * /* ptr */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueTask(cl_command_queue /* command_queue */, cl_kernel /* kernel */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueUnmapMemObject(cl_command_queue /* command_queue */, cl_mem /* memobj */, void * /* mapped_ptr */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueWaitForEvents(cl_command_queue /* command_queue */, cl_uint /* num_events */, const cl_event * /* event_list */);
cl_int clEnqueueWriteBuffer(cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_write */, size_t /* offset */, size_t /* cb */, const void * /* ptr */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueWriteBufferRect(cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_write */, const size_t * /* buffer_origin */, const size_t * /* host_origin */, const size_t * /* region */, size_t /* buffer_row_pitch */, size_t /* buffer_slice_pitch */, size_t /* host_row_pitch */, size_t /* host_slice_pitch */, const void * /* ptr */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clEnqueueWriteImage(cl_command_queue /* command_queue */, cl_mem /* image */, cl_bool /* blocking_write */, const size_t * /* origin[3] */, const size_t * /* region[3] */, size_t /* input_row_pitch */, size_t /* input_slice_pitch */, const void * /* ptr */, cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */, cl_event * /* event */);
cl_int clFinish(cl_command_queue /* command_queue */);
cl_int clFlush(cl_command_queue /* command_queue */);
cl_int clGetCommandQueueInfo(cl_command_queue /* command_queue */, cl_command_queue_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetContextInfo(cl_context /* context */, cl_context_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetDeviceIDs(cl_platform_id /* platform */, cl_device_type /* device_type */, cl_uint /* num_entries */, cl_device_id * /* devices */, cl_uint * /* num_devices */);
cl_int clGetDeviceInfo(cl_device_id /* device */, cl_device_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetEventInfo(cl_event /* event */, cl_event_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetEventProfilingInfo(cl_event /* event */, cl_profiling_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
void clGetExtensionFunctionAddress(const char * /* func_name */);
cl_int clGetImageInfo(cl_mem /* image */, cl_image_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetKernelInfo(cl_kernel /* kernel */, cl_kernel_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetKernelWorkGroupInfo(cl_kernel /* kernel */, cl_device_id /* device */, cl_kernel_work_group_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetMemObjectInfo(cl_mem /* memobj */, cl_mem_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetPlatformIDs(cl_uint /* num_entries */, cl_platform_id * /* platforms */, cl_uint * /* num_platforms */);
cl_int clGetPlatformInfo(cl_platform_id /* platform */, cl_platform_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetProgramBuildInfo(cl_program /* program */, cl_device_id /* device */, cl_program_build_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetProgramInfo(cl_program /* program */, cl_program_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetSamplerInfo(cl_sampler /* sampler */, cl_sampler_info /* param_name */, size_t /* param_value_size */, void * /* param_value */, size_t * /* param_value_size_ret */);
cl_int clGetSupportedImageFormats(cl_context /* context */, cl_mem_flags /* flags */, cl_mem_object_type /* image_type */, cl_uint /* num_entries */, cl_image_format * /* image_formats */, cl_uint * /* num_image_formats */);
cl_int clReleaseCommandQueue(cl_command_queue /* command_queue */);
cl_int clReleaseContext(cl_context /* context */);
cl_int clReleaseEvent(cl_event /* event */);
cl_int clReleaseKernel(cl_kernel /* kernel */);
cl_int clReleaseMemObject(cl_mem /* memobj */);
cl_int clReleaseProgram(cl_program /* program */);
cl_int clReleaseSampler(cl_sampler /* sampler */);
cl_int clRetainCommandQueue(cl_command_queue /* command_queue */);
cl_int clRetainContext(cl_context /* context */);
cl_int clRetainEvent(cl_event /* event */);
cl_int clRetainKernel(cl_kernel /* kernel */);
cl_int clRetainMemObject(cl_mem /* memobj */);
cl_int clRetainProgram(cl_program /* program */);
cl_int clRetainSampler(cl_sampler /* sampler */);
// cl_int clSetCommandQueueProperty(cl_command_queue /* command_queue */, cl_command_queue_properties /* properties */, cl_bool /* enable */, cl_command_queue_properties * /* old_properties */);
cl_int clSetKernelArg(cl_kernel /* kernel */, cl_uint /* arg_index */, size_t /* arg_size */, const void * /* arg_value */);
cl_int clSetUserEventStatus(cl_event /* event */, cl_int /* execution_status */);
cl_int clUnloadCompiler(void);
cl_int clWaitForEvents(cl_uint /* num_events */, const cl_event * /* event_list */);

#endif // XXXXXXXXXX
