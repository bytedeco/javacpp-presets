package samples.demo.layer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class GNNTester {
    // 简单的断言工具
    public static void check(boolean condition, String message) {
        if (!condition) {
            System.err.println("❌ [失败] " + message);
            System.exit(1);
        } else {
            System.out.println("✅ [通过] " + message);
        }
    }

    // 校验 Shape
    public static void assertShape(Tensor t, long... expected) {
        long[] actual = t.sizes().vec().get();
        check(actual.length == expected.length, "维度数量不匹配");
        for (int i = 0; i < actual.length; i++) {
            check(actual[i] == expected[i], "维度 " + i + " 预期 " + expected[i] + " 实际 " + actual[i]);
        }
    }
}