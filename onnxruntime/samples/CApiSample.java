// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.onnxruntime.*;
import static org.bytedeco.onnxruntime.global.onnxruntime.*;

public class CApiSample {

    static final OrtApi g_ort = OrtGetApiBase().GetApi().call(ORT_API_VERSION);

    //*****************************************************************************
    // helper function to check for status
    static void CheckStatus(OrtStatus status) {
        if (status != null && !status.isNull()) {
          String msg = g_ort.GetErrorMessage().call(status).getString();
          System.err.println(msg);
          g_ort.ReleaseStatus().call(status);
          System.exit(1);
        }
    }

    public static void main(String[] args) throws Exception {
      //*************************************************************************
      // initialize  enviroment...one enviroment per process
      // enviroment maintains thread pools and other state info
      PointerPointer<OrtEnv> envs = new PointerPointer<OrtEnv>(1);
      CheckStatus(g_ort.CreateEnv().call(ORT_LOGGING_LEVEL_WARNING, new BytePointer("test"), envs));
      OrtEnv env = envs.get(OrtEnv.class);

      // initialize session options if needed
      PointerPointer<OrtSessionOptions> sessions_options = new PointerPointer<OrtSessionOptions>(1);
      CheckStatus(g_ort.CreateSessionOptions().call(sessions_options));
      OrtSessionOptions session_options = sessions_options.get(OrtSessionOptions.class);
      g_ort.SetIntraOpNumThreads().call(session_options, 1);

      // Sets graph optimization level
      g_ort.SetSessionGraphOptimizationLevel().call(session_options, ORT_ENABLE_BASIC);

      // Optionally add more execution providers via session_options
      // E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
      // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
      OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, 1);

      //*************************************************************************
      // create session and load model into memory
      // using squeezenet version 1.3
      // URL = https://github.com/onnx/models/tree/master/squeezenet
      PointerPointer<OrtSession> sessions = new PointerPointer<OrtSession>(1);
      String s = args.length > 0 ? args[0] : "squeezenet.onnx";
      Pointer model_path = Loader.getPlatform().startsWith("windows") ? new CharPointer(s) : new BytePointer(s);

      System.out.println("Using Onnxruntime C API");
      CheckStatus(g_ort.CreateSession().call(env, model_path, session_options, sessions));
      OrtSession session = sessions.get(OrtSession.class);

      //*************************************************************************
      // print model input layer (node names, types, shape etc.)
      SizeTPointer num_input_nodes = new SizeTPointer(1);
      OrtStatus status;
      PointerPointer<OrtAllocator> allocators = new PointerPointer<OrtAllocator>(1);
      CheckStatus(g_ort.GetAllocatorWithDefaultOptions().call(allocators));
      OrtAllocator allocator = allocators.get(OrtAllocator.class);

      // print number of model input nodes
      status = g_ort.SessionGetInputCount().call(session, num_input_nodes);
      PointerPointer input_node_names = new PointerPointer(num_input_nodes.get());
      LongPointer input_node_dims = null;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

      System.out.println("Number of inputs = " + num_input_nodes.get());

      // iterate over all input nodes
      for (long i = 0; i < num_input_nodes.get(); i++) {
        // print input node names
        PointerPointer<BytePointer> input_names = new PointerPointer<BytePointer>(1);
        status = g_ort.SessionGetInputName().call(session, i, allocator, input_names);
        BytePointer input_name = input_names.get(BytePointer.class);
        System.out.println("Input " + i + " : name=" + input_name.getString());
        input_node_names.put(i, input_name);

        // print input node types
        PointerPointer<OrtTypeInfo> typeinfos = new PointerPointer<OrtTypeInfo>(1);
        status = g_ort.SessionGetInputTypeInfo().call(session, i, typeinfos);
        OrtTypeInfo typeinfo = typeinfos.get(OrtTypeInfo.class);
        PointerPointer<OrtTensorTypeAndShapeInfo> tensor_infos = new PointerPointer<OrtTensorTypeAndShapeInfo>(1);
        CheckStatus(g_ort.CastTypeInfoToTensorInfo().call(typeinfo, tensor_infos));
        OrtTensorTypeAndShapeInfo tensor_info = tensor_infos.get(OrtTensorTypeAndShapeInfo.class);
        IntPointer type = new IntPointer(1);
        CheckStatus(g_ort.GetTensorElementType().call(tensor_info, type));
        System.out.println("Input " + i + " : type=" + type.get());

        // print input shapes/dims
        SizeTPointer num_dims = new SizeTPointer(1);
        CheckStatus(g_ort.GetDimensionsCount().call(tensor_info, num_dims));
        System.out.println("Input " + i + " : num_dims=" + num_dims.get());
        input_node_dims = new LongPointer(num_dims.get());
        g_ort.GetDimensions().call(tensor_info, input_node_dims, num_dims.get());
        for (long j = 0; j < num_dims.get(); j++)
          System.out.println("Input " + i + " : dim " + j + "=" + input_node_dims.get(j));

        g_ort.ReleaseTypeInfo().call(typeinfo);
      }

      // Results should be...
      // Number of inputs = 1
      // Input 0 : name = data_0
      // Input 0 : type = 1
      // Input 0 : num_dims = 4
      // Input 0 : dim 0 = 1
      // Input 0 : dim 1 = 3
      // Input 0 : dim 2 = 224
      // Input 0 : dim 3 = 224

      //*************************************************************************
      // Similar operations to get output node information.
      // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
      // OrtSessionGetOutputTypeInfo() as shown above.

      //*************************************************************************
      // Score the model using sample data, and inspect values

      long input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                               // use OrtGetTensorShapeElementCount() to get official size!

      FloatPointer input_tensor_values = new FloatPointer(input_tensor_size);
      PointerPointer output_node_names = new PointerPointer("softmaxout_1");

      // initialize input data with values in [0.0, 1.0]
      FloatIndexer idx = FloatIndexer.create(input_tensor_values);
      for (long i = 0; i < input_tensor_size; i++)
        idx.put(i, (float)i / (input_tensor_size + 1));

      // create input tensor object from data values
      PointerPointer<OrtMemoryInfo> memory_infos = new PointerPointer<OrtMemoryInfo>(1);
      CheckStatus(g_ort.CreateCpuMemoryInfo().call(OrtArenaAllocator, OrtMemTypeDefault, memory_infos));
      OrtMemoryInfo memory_info = memory_infos.get(OrtMemoryInfo.class);
      PointerPointer<OrtValue> input_tensors = new PointerPointer<OrtValue>(1).put(0, null);
      CheckStatus(g_ort.CreateTensorWithDataAsOrtValue().call(memory_info, input_tensor_values, input_tensor_size * Float.SIZE / 8, input_node_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_tensors));
      OrtValue input_tensor = input_tensors.get(OrtValue.class);
      IntPointer is_tensor = new IntPointer(1);
      CheckStatus(g_ort.IsTensor().call(input_tensor, is_tensor));
      assert is_tensor.get() != 0;

      // score model & input tensor, get back output tensor
      PointerPointer<OrtValue> output_tensors = new PointerPointer<OrtValue>(1).put(0, null);
      CheckStatus(g_ort.Run().call(session, null, input_node_names, input_tensors, 1, output_node_names, 1, output_tensors));
      OrtValue output_tensor = output_tensors.get(OrtValue.class);
      CheckStatus(g_ort.IsTensor().call(output_tensor, is_tensor));
      assert is_tensor.get() != 0;

      // Get pointer to output tensor float values
      PointerPointer<FloatPointer> floatarrs = new PointerPointer<FloatPointer>(1);
      CheckStatus(g_ort.GetTensorMutableData().call(output_tensor, floatarrs));
      FloatPointer floatarr = floatarrs.get(FloatPointer.class);
      assert Math.abs(floatarr.get(0) - 0.000045) < 1e-6;

      // score the model, and print scores for first 5 classes
      for (int i = 0; i < 5; i++)
        System.out.println("Score for class [" + i + "] =  " + floatarr.get(i));

      // Results should be as below...
      // Score for class[0] = 0.000045
      // Score for class[1] = 0.003846
      // Score for class[2] = 0.000125
      // Score for class[3] = 0.001180
      // Score for class[4] = 0.001317

      g_ort.ReleaseMemoryInfo().call(memory_info);
      g_ort.ReleaseValue().call(output_tensor);
      g_ort.ReleaseValue().call(input_tensor);
      g_ort.ReleaseSession().call(session);
      g_ort.ReleaseSessionOptions().call(session_options);
      g_ort.ReleaseEnv().call(env);
      System.out.println("Done!");
      System.exit(0);
    }
}
