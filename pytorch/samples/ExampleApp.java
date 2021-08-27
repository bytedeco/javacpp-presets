// © Copyright 2021, PyTorch.

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

// Ported from C++ code at https://pytorch.org/tutorials/advanced/cpp_export.html
public class ExampleApp {
    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        if (args.length != 1) {
            System.err.println("usage: java ExampleApp <path-to-exported-script-module>");
            System.exit(-1);
        }

        JitModule module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = load(args[0]);
        } catch (Exception e) {
            System.err.println("error loading the model");
            throw e;
        }

        System.out.println("ok");

        // Create a vector of inputs.
        IValueVector inputs = new IValueVector();
        inputs.push_back(new IValue(ones(1, 3, 224, 224)));

        // Execute the model and turn its output into a tensor.
        Tensor output = module.forward(inputs).toTensor();
        print(output.slice(/*dim=*/1, /*start=*/new LongOptional(0), /*end=*/new LongOptional(5), /*step=*/1));
    }
}
