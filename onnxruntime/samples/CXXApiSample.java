// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

import java.nio.file.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.onnxruntime.*;
import static org.bytedeco.onnxruntime.global.onnxruntime.*;

public class CXXApiSample {

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

      Env env = new Env(ORT_LOGGING_LEVEL_WARNING, new BytePointer("test"));

      // initialize session options if needed
      SessionOptions session_options = new SessionOptions();
      session_options.SetIntraOpNumThreads(1);

      // Sets graph optimization level
      session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

      // Optionally add more execution providers via session_options
      // E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
      // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

      OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options.asOrtSessionOptions(), 1);
      //*************************************************************************
      // create session and load model into memory
      // using squeezenet version 1.3
      // URL = https://github.com/onnx/models/tree/master/squeezenet

      String model_path = args.length > 0 ? args[0] : "squeezenet.onnx";
      Session session = new Session(env, model_path, session_options); 
 
      System.out.println("Using Onnxruntime C++ API");

      //*************************************************************************
      // print model input layer (node names, types, shape etc.)
      Long num_input_nodes = session.GetInputCount();
 
      AllocatorWithDefaultOptions allocator = new AllocatorWithDefaultOptions();

      // print number of model input nodes
 
      PointerPointer input_node_names = new PointerPointer(num_input_nodes);
      LongPointer input_node_dims = null;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

      System.out.println("Number of inputs = " + num_input_nodes);

      // iterate over all input nodes
      for (long i = 0; i < num_input_nodes; i++) {
        // print input node names
        BytePointer input_name = session.GetInputName(i, allocator.asOrtAllocator()); 
        System.out.println("Input " + i + " : name=" + input_name.getString());
        input_node_names.put(i, input_name);

        // print input node types
        TypeInfo type_info = session.GetInputTypeInfo(i); 

        OrtTypeInfo ort_type_info = type_info.asOrtTypeInfo(); 
        
	//Using C API here because GetTensorTypeAndShapeInfo() isn't there 
	PointerPointer<OrtTensorTypeAndShapeInfo> tensor_infos = new PointerPointer<OrtTensorTypeAndShapeInfo>(1);
        CheckStatus(g_ort.CastTypeInfoToTensorInfo().call(ort_type_info, tensor_infos));
        OrtTensorTypeAndShapeInfo ort_tensor_info = tensor_infos.get(OrtTensorTypeAndShapeInfo.class);
        IntPointer type = new IntPointer(1);
        CheckStatus(g_ort.GetTensorElementType().call(ort_tensor_info, type));
        System.out.println("Input " + i + " : type=" + type.get());

	TensorTypeAndShapeInfo tensor_info = new TensorTypeAndShapeInfo(ort_tensor_info);

	//Back to C++ API
        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();


        System.out.println("Input " + i + " : num_dims=" + input_node_dims.capacity());
        for (long j = 0; j < input_node_dims.capacity(); j++)
          System.out.println("Input " + i + " : dim " + j + "=" + input_node_dims.get(j));


	g_ort.ReleaseTypeInfo().call(ort_type_info);
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

      MemoryInfo memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      OrtMemoryInfo ort_memory_info = memory_info.asOrtMemoryInfo(); 
     
      //Value::CreateTensor in C++ API only takes C OrtMemoryInfo, not C++ MemoryInfo 
      Value input_tensor = Value.CreateTensor(ort_memory_info, input_tensor_values, input_tensor_size * Float.SIZE / 8, input_node_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
 
      boolean is_tens = input_tensor.IsTensor();
      System.out.println(is_tens);
      assert is_tens;
     
      ValueVector output_tensor = session.Run(new RunOptions(), input_node_names, input_tensor, 1, output_node_names, 1); 
     
      boolean is_tensor = output_tensor.get(0).IsTensor(); 
      assert output_tensor.size()==1 && is_tensor;

      // Get pointer to output tensor float values
      FloatPointer floatarr = output_tensor.get(0).GetTensorMutableDataFloat();
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

      g_ort.ReleaseMemoryInfo().call(ort_memory_info);
      //g_ort.ReleaseValue().call(ort_output_tensor);
      //g_ort.ReleaseValue().call(input_tensor);
      //g_ort.ReleaseSession().call(session);
      //g_ort.ReleaseSessionOptions().call(session_options);
      //g_ort.ReleaseEnv().call(env);
      System.out.println("Done!");
      System.exit(0);
    }
}
