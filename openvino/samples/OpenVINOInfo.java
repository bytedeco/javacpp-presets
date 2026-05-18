import org.bytedeco.javacpp.Loader;
import org.bytedeco.openvino.ov_core_t;

public class OpenVINOInfo {
    public static void main(String[] args) {
        String jniLibraryPath = Loader.load(org.bytedeco.openvino.global.openvino.class);
        System.out.println("OpenVINO JavaCPP JNI library loaded from: " + jniLibraryPath);

        ov_core_t core = new ov_core_t();
        System.out.println("Allocated ov_core_t wrapper successfully: " + core);

        System.out.println("OpenVINO preset load verification complete.");
    }
}
