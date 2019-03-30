import org.bytedeco.javacpp.*;
import org.bytedeco.cpu_features.*;
import static org.bytedeco.cpu_features.global.cpu_features.*;

public class SimpleExample {

    static X86Features features = GetX86Info().features();

    static void Compute() {
        if (features.aes() != 0 && features.sse4_2() != 0) {
            System.out.println("Run optimized code.");
        } else {
            System.out.println("Run standard code.");
        }
    }

    public static void main(String args[]) {
        Compute();
    }
}
