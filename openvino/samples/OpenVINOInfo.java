import org.bytedeco.javacpp.Loader;
import org.bytedeco.openvino.ov_core_t;
import org.bytedeco.openvino.ov_available_devices_t;

import static org.bytedeco.openvino.global.openvino.*;

public class OpenVINOInfo {
    public static void main(String[] args) {
        String jniLibraryPath = Loader.load(org.bytedeco.openvino.global.openvino.class);
        System.out.println("OpenVINO JavaCPP JNI library loaded from: " + jniLibraryPath);

        ov_core_t core = new ov_core_t((org.bytedeco.javacpp.Pointer)null);
        int status = ov_core_create(core);
        if (status != OK) {
            throw new IllegalStateException("ov_core_create failed: " + ov_get_error_info(status).getString());
        }

        try {
            ov_available_devices_t devices = new ov_available_devices_t();
            status = ov_core_get_available_devices(core, devices);
            if (status != OK) {
                throw new IllegalStateException("ov_core_get_available_devices failed: " + ov_get_error_info(status).getString());
            }
            try {
                for (int i = 0; i < devices.size(); i++) {
                    System.out.println("OpenVINO device: " + devices.devices(i).getString());
                }
            } finally {
                ov_available_devices_free(devices);
            }
        } finally {
            ov_core_free(core);
        }
    }
}
