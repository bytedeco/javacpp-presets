/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import org.bytedeco.javacpp.*;
import org.bytedeco.tensorflowlite.*;
import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>
public class Minimal {
    static void TFLITE_MINIMAL_CHECK(boolean x) {
      if (!x) {
        System.err.print("Error at ");
        Thread.dumpStack();
        System.exit(1);
      }
    }

    public static void main(String[] args) {
      if (args.length != 1) {
        System.err.println("minimal <tflite model>");
        System.exit(1);
      }
      String filename = args[0];

      // Load model
      FlatBufferModel model = FlatBufferModel.BuildFromFile(filename);
      TFLITE_MINIMAL_CHECK(model != null && !model.isNull());

      // Build the interpreter with the InterpreterBuilder.
      // Note: all Interpreters should be built with the InterpreterBuilder,
      // which allocates memory for the Interpreter and does various set up
      // tasks so that the Interpreter can read the provided model.
      BuiltinOpResolver resolver = new BuiltinOpResolver();
      InterpreterBuilder builder = new InterpreterBuilder(model, resolver);
      Interpreter interpreter = new Interpreter((Pointer)null);
      builder.apply(interpreter);
      TFLITE_MINIMAL_CHECK(interpreter != null && !interpreter.isNull());

      // Allocate tensor buffers.
      TFLITE_MINIMAL_CHECK(interpreter.AllocateTensors() == kTfLiteOk);
      System.out.println("=== Pre-invoke Interpreter State ===");
      PrintInterpreterState(interpreter);

      // Fill input buffers
      // TODO(user): Insert code to fill input tensors.
      // Note: The buffer of the input tensor with index `i` of type T can
      // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

      // Run inference
      TFLITE_MINIMAL_CHECK(interpreter.Invoke() == kTfLiteOk);
      System.out.println("\n\n=== Post-invoke Interpreter State ===");
      PrintInterpreterState(interpreter);

      // Read output buffers
      // TODO(user): Insert getting data out code.
      // Note: The buffer of the output tensor with index `i` of type T can
      // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

      System.exit(0);
    }
}
