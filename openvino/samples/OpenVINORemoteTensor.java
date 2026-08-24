import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.openvino.ov_core_t;
import org.bytedeco.openvino.ov_remote_context_t;
import org.bytedeco.openvino.ov_shape_t;
import org.bytedeco.openvino.ov_tensor_t;

import static org.bytedeco.openvino.global.openvino.*;

/** Compile-time smoke sample for the non-variadic OpenCL interop overloads. */
public final class OpenVINORemoteTensor {
    private OpenVINORemoteTensor() {
    }

    public static ov_remote_context_t createContext(ov_core_t core, Pointer clContext, Pointer clQueue) {
        ov_remote_context_t context = new ov_remote_context_t();
        check(ov_core_create_context(core, "GPU", 6, context,
                string(ov_property_key_intel_gpu_context_type()), "OCL",
                string(ov_property_key_intel_gpu_ocl_context()), clContext,
                string(ov_property_key_intel_gpu_ocl_queue()), clQueue));
        return context;
    }

    public static ov_tensor_t wrapBuffer(ov_remote_context_t context, ov_shape_t shape, Pointer clMem) {
        ov_tensor_t tensor = new ov_tensor_t();
        check(ov_remote_context_create_tensor(context, U8, shape, 4, tensor,
                string(ov_property_key_intel_gpu_shared_mem_type()), "OCL_BUFFER",
                string(ov_property_key_intel_gpu_mem_handle()), clMem));
        return tensor;
    }

    private static String string(BytePointer value) {
        return value.getString();
    }

    private static void check(int status) {
        if (status != OK) {
            throw new IllegalStateException(ov_get_error_info(status).getString());
        }
    }
}
