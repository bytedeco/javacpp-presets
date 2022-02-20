import java.nio.file.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.onnx.*;
import static org.bytedeco.onnx.global.onnx.*;

public class LoadModel {
    public static void main(String[] args) throws Exception {
        OpSchemaVector allSchemas = OpSchemaRegistry.get_all_schemas();
        System.out.println(allSchemas.size());

        byte[] bytes = Files.readAllBytes(Paths.get(args.length > 0 ? args[0] : "examples/resources/single_relu.onnx"));

        ModelProto model = new ModelProto();
        ParseProtoFromBytes(model, new BytePointer(bytes), bytes.length);

        check_model(model);

        InferShapes(model);

//        StringVector passes = new StringVector("eliminate_nop_transpose", "eliminate_nop_pad", "fuse_consecutive_transposes", "fuse_transpose_into_gemm");
//        Optimize(model, passes);

        check_model(model);

        ConvertVersion(model, 8);

        System.out.println(model.graph().input_size());
    }
}
