// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import com.google.gson.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.tritonserver.tritondevelopertoolsserver.*;
import static org.bytedeco.tritonserver.global.tritondevelopertoolsserver.*;

public class SimpleCPP {
    // Helper functions
    static void FAIL(String MSG) {
      System.err.println("Failure: " + MSG);
      System.exit(1);
    }

    static void Usage(String msg) {
      if (msg != null) {
        System.err.println(msg);
      }

      System.err.println("Usage: java " + SimpleCPP.class.getSimpleName() +" [options]");
      System.err.println("\t-v Enable verbose logging");
      System.err.println("\t-r [model repository absolute path]");
      System.exit(1);
    }

    static void
    CompareResult(
        String output0_name, String output1_name,
        IntPointer input0, IntPointer input1, IntPointer output0,
        IntPointer output1) {
      for (int i = 0; i < 16; ++i) {
        System.out.println(input0.get(i) + " + " + input1.get(i) + " = "
                         + output0.get(i));
        System.out.println(input0.get(i) + " - " + input1.get(i) + " = "
                         + output1.get(i));

        if ((input0.get(i) + input1.get(i)) != output0.get(i)) {
          FAIL("incorrect sum in " + output0_name);
        }
        if ((input0.get(i) - input1.get(i)) != output1.get(i)) {
          FAIL("incorrect difference in " + output1_name);
        }
      }
    }

    static void
    GenerateInputData(
        IntPointer[] input0_data, IntPointer[] input1_data) {
      input0_data[0] = new IntPointer(16);
      input1_data[0] = new IntPointer(16);
      for (int i = 0; i < 16; ++i) {
        input0_data[0].put(i, 2);
        input1_data[0].put(i, 1 * i);
      }
    }

    static int RunInference(int verbose_level, String model_repository_path, String model_name) {
      StringVector model_repository_paths = new StringVector(model_repository_path);
      ServerOptions options = new ServerOptions(model_repository_paths);
      LoggingOptions logging_options = options.logging_();
      logging_options.SetVerboseLevel(verbose_level);
      options.SetLoggingOptions(logging_options);

      GenericTritonServer server = GenericTritonServer.Create(options);
      StringSet loaded_models = server.LoadedModels();
      System.out.println("Loaded_models count : " + loaded_models.size());

      InferOptions infer_options = new InferOptions(model_name);
      GenericInferRequest request = GenericInferRequest.Create(infer_options);

      BytePointer input0_data;
      BytePointer input1_data;
      IntPointer[] p0 = {null}, p1 = {null};
      GenerateInputData(p0, p1);
      input0_data = p0[0].getPointer(BytePointer.class);
      input1_data = p1[0].getPointer(BytePointer.class);

      LongPointer shape0 = new LongPointer(2);
      LongPointer shape1 = new LongPointer(2);
      shape0.put(0, 1);
      shape0.put(1, 16);
      shape1.put(0, 1);
      shape1.put(1, 16);
      Tensor tensor0 = new Tensor(input0_data, 4 * 16, 8, shape0, 0, 1);
      Tensor tensor1 = new Tensor(input1_data, 4 * 16, 8, shape1, 0, 1);
      request.AddInput("INPUT0", tensor0);
      request.AddInput("INPUT1", tensor1);
      GenericInferResult result = server.Infer(request);

            Tensor output = result.Output("OUTPUT0");
      BytePointer buffer = output.buffer_();

      System.out.println("buffer to string : " + buffer.toString());
      System.out.println("output val at index 0 : " + buffer.getInt(0));
      System.out.println("output val at index 1 : " + buffer.getInt(1 * 4));
      System.out.println("output val at index 2 : " + buffer.getInt(2 * 4));
      System.out.println("output val at index 3 : " + buffer.getInt(3 * 4));
      System.out.println("output val at index 4 : " + buffer.getInt(4 * 4));
      System.out.println("output val at index 5 : " + buffer.getInt(5 * 4));
      System.out.println("output val at index 6 : " + buffer.getInt(6 * 4));
      System.out.println("output val at index 7 : " + buffer.getInt(7 * 4));
      return 0;
    }

    public static void
    main(String[] args) throws Exception {
      String model_repository_path = "./models";
      int verbose_level = 0;

      // Parse commandline...
      for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
          case "-r":
          model_repository_path = args[++i];
          break;
          case "-v":
          verbose_level = 1;
          break;
          case "-?":
          Usage(null);
          break;
        }
      }

      // We use a simple model that takes 2 input tensors of 16 strings
      // each and returns 2 output tensors of 16 strings each. The input
      // strings must represent integers. One output tensor is the
      // element-wise sum of the inputs and one output is the element-wise
      // difference.
      String model_name = "simple";
      if (model_repository_path == null) {
        Usage("-r must be used to specify model repository path");
      }

      RunInference(verbose_level, model_repository_path, model_name);

      System.exit(0);
    }
}
