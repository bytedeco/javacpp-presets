import java.io.*;
import java.util.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.BytePointer;

import org.bytedeco.cuda.cudart.*;
import org.bytedeco.tritonserver.tritonserver.*;
import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.tensorrt.global.tritonserver.*;

public class Simple {

	static void FAIL(String msg)
    {
        System.err.println("Cuda failure: " + msg);
        System.exit(1);
    }

	static void FAIL_IF_ERR(TRITONSERVER_Error X, String MSG)
    {
        TRITONSERVER_Error err_ = X;
		if (err__ != null) {
		    System.err.println("error: " + MSG
				+ ":" + TRITONSERVER_ErrorCodeString(err__) + " - "
				+ TRITONSERVER_ErrorMessage(err__));
			TRITONSERVER_ErrorDelete(err__);    
            System.exit(1);	
		}
    }
    
    static void FAIL_IF_CUDA_ERR(cudaError_t X, String MSG)
	{
        cudaError_t err__ = X;                                               
        if (err__ != cudaSuccess) {
			System.err.println("error: " + MSG
				+ ":" + TRITONSERVER_ErrorCodeString(err__) + " - "
				+ cudaGetErrorString(err__));
			System.exit(1);                                                                 
        }                                                                      
    }

    boolean enforce_memory_type = false;
    TRITONSERVER_MemoryType requested_memory_type;

	final boolean triton_enable_gpu = false;
	if (triton_enable_gpu)
    {   
        public static class cuda_data_deleter extends FunctionPointer {
        	public void call(Pointer data) {
				if (data != null) {
                	cudaPointerAttributes attr;
                	auto cuerr = cudaPointerGetAttributes(attr, data);
                	if (cuerr != cudaSuccess) {
                    	//std::cerr << "error: failed to get CUDA pointer attribute of " << data
                    	//    << ": " << cudaGetErrorString(cuerr) << std::endl;
                    	//jack: how to print "Pointer data" here, %what?
			        	System.err.printf("error: failed to get CUDA pointer attribute of %?: %s\n", data, cudaGetErrorString(cuerr));
                	}
                	if (attr.type == cudaMemoryTypeDevice) {
                    	cuerr = cudaFree(data);
                	} else if (attr.type == cudaMemoryTypeHost) {
                    	cuerr = cudaFreeHost(data);
                	}
                	if (cuerr != cudaSuccess) {
                    	//std::cerr << "error: failed to release CUDA pointer " << data << ": "
                    	//    << cudaGetErrorString(cuerr) << std::endl;
                    	//jack: how to print "Pointer data" here, %what?
			        	System.err.printf("error: failed to release CUDA pointer %?: %s\n", data, cudaGetErrorString(cuerr)); ??
                	}
            	}

        	}
    	}
        
    }			

    void Usage(String[] args, String msg = String) 
    {
        if (!msg.isEmpty()) {
            System.err.printf("%s\n", msg);
        }

        System.err.printf("Usage: %s [options]\n", argv[0].get());
        System.err.printf("\t-m <\"system\"|\"pinned\"|gpu>\n");
		System.err.printf("Enforce the memory type for input and output tensors.\n");
		System.err.printf("If not specified, inputs will be in system memory and outputs\n");
		System.err.printf("will be based on the model's preferred type.\n");
        System.err.printf("\t-v Enable verbose logging\n");
        System.err.printf("\t-r [model repository absolute path]\n");
        System.err.printf("\t-c Enable web camera input.\n");

        System.exit(1);
    }

	TRITONSERVER_Error ResponseAlloc(TRITONSERVER_ResponseAllocator allocator,
		char tensor_name, long byte_size, TRITONSERVER_MemoryType preferred_memory_type,
		long preferred_memory_type_id, Pointer userp, PointerPointer buffer,
		PointerPointer buffer_userp, TRITONSERVER_MemoryType actual_memory_type,
		long actual_memory_type_id)
	{
		// Initially attempt to make the actual memory type and id that we
		// allocate be the same as preferred memory type
		actual_memory_type = preferred_memory_type;
		actual_memory_type_id = preferred_memory_type_id;
	
		// If 'byte_size' is zero just return 'buffer' == nullptr, we don't
		// need to do any other book-keeping.
		if (byte_size == 0) {
			buffer = null;
			buffer_userp = null;
			System.out.printf("allocated %d %s\n", byte_size, tensor_name);
		} else {
			Pointer allocated_ptr = null;
			if (enforce_memory_type) {
				actual_memory_type = requested_memory_type;
			}
	
			switch (actual_memory_type) {
				if (triton_enable_gpu)
				{
					case TRITONSERVER_MEMORY_CPU_PINNED: {
						int err = cudaSetDevice(actual_memory_type_id);
						if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
							(err != cudaErrorInsufficientDriver)) {
							return TRITONSERVER_ErrorNew(
									   TRITONSERVER_ERROR_INTERNAL,
									   new BytePointer("unable to recover current CUDA device: cudaGetErrorString(err)"));		           
						}
	
						err = cudaHostAlloc(allocated_ptr, byte_size, cudaHostAllocPortable);
						if (err != cudaSuccess) {
							return TRITONSERVER_ErrorNew(
									   TRITONSERVER_ERROR_INTERNAL,
									   new BytePointer("cudaHostAlloc failed: cudaGetErrorString(err)"));
						}
						break;
					}
	
					case TRITONSERVER_MEMORY_GPU: {
						int err = cudaSetDevice(actual_memory_type_id);
						if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
							(err != cudaErrorInsufficientDriver)) {
							return TRITONSERVER_ErrorNew(
									   TRITONSERVER_ERROR_INTERNAL,
									   new BytePointer("unable to recover current CUDA device: cudaGetErrorString(err)"));
						}
	
						err = cudaMalloc(allocated_ptr, byte_size);
						if (err != cudaSuccess) {
							return TRITONSERVER_ErrorNew(
								       TRITONSERVER_ERROR_INTERNAL,
								       new BytePointer("cudaMalloc failed: cudaGetErrorString(err)"));
						}
						break;
					}
				}   
					
			    // Use CPU memory if the requested memory type is unknown
				// (default case).
				case TRITONSERVER_MEMORY_CPU:
				default: {
					actual_memory_type = TRITONSERVER_MEMORY_CPU;
					allocated_ptr = new byte[byte_size];
					break;
				}
			}   
	
			// Pass the tensor name with buffer_userp so we can show it when
			// releasing the buffer.
			if (allocated_ptr != null) {
				buffer = allocated_ptr;
				buffer_userp = new String(tensor_name);
				System.out.printf("allocated %d bytes in %s for result tensor %s\n", byte_size, 
					TRITONSERVER_MemoryTypeString(actual_memory_type), tensor_name);
			}
		}
	
		return null;  // Success
	}	 
	
	TRITONSERVER_Error ResponseRelease(TRITONSERVER_ResponseAllocator allocator,
		Pointer buffer, Pointer buffer_userp, long byte_size, TRITONSERVER_MemoryType memory_type,
		long memory_type_id)
	{
		String name = null;
		if (buffer_userp != null) {
			name = (String)(buffer_userp);
		} else {
			name = new String("<unknown>");
		}
	
		System.out.printf("Releasing buffer of size %d in %s for result %s\n", byte_size, 
			TRITONSERVER_MemoryTypeString(memory_type), name);
		switch (memory_type) {
			case TRITONSERVER_MEMORY_CPU:
				//jack: for c++ free, I just use "= null", is this correct?
				//free(buffer);
				buffer = null;
				break;
			
		   if (triton_enable_gpu){
			case TRITONSERVER_MEMORY_CPU_PINNED: {
				int err = cudaSetDevice(memory_type_id);
				if (err == cudaSuccess) {
					err = cudaFreeHost(buffer);
				}
				if (err != cudaSuccess) {
					System.err.printf("error: failed to cudaFree: %s.\n", cudaGetErrorString(err));
				}
				break;
			}
			case TRITONSERVER_MEMORY_GPU: {
				int err = cudaSetDevice(memory_type_id);
				if (err == cudaSuccess) {
					err = cudaFree(buffer);
				}
				if (err != cudaSuccess) {
					System.err.printf("error: failed to cudaFree: %s.\n", cudaGetErrorString(err));
				}
				break;
			}
		   }
		   
			default:
				System.err.printf("error: unexpected buffer allocated in CUDA managed memory.\n");
				break;
		}
	
		name = null;
	
		return null;  // Success
	}

	void 
	InferRequestComplete(
		 TRITONSERVER_InferenceRequest request, int flags, Pointer userp)
	{
		 // We reuse the request so we don't delete it here.
	}
		 
	void
	InferResponseComplete(
		TRITONSERVER_InferenceResponse response, long flags, Pointer userp)
	{
		 if (response != null) {
		 // Send 'response' to the future.
		 //jack: how to do with std::promise? and which java object can do with .set_value?
			std::promise<TRITONSERVER_InferenceResponse*>* p =
			reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
			p->set_value(response);
			p = null;
		 }
	}

	void 
	InferResponseComplete(
		TRITONSERVER_InferenceResponse response, int flags, Pointer userp)
	{
		if (response != null) {
			// Send 'response' to the future.
			//jack: how to do with std::promise, set_value can be replaced by which java func? for reinterpret_cast, should be replaced by which one?
			std::promise<TRITONSERVER_InferenceResponse*>* p =
				reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
			p->set_value(response);
			p = null;
		}
	}
		
	TRITONSERVER_Error
	ParseModelMetadata(
		const rapidjson::Document& model_metadata, boolean is_int,
		boolean is_torch_model)
	{
		String seen_data_type;
		for (const auto& input : model_metadata["inputs"].GetArray()) {
		    if (strcmp(input["datatype"].GetString(), "INT32") &&
				strcmp(input["datatype"].GetString(), "FP32")) {
			    return TRITONSERVER_ErrorNew(
				  TRITONSERVER_ERROR_UNSUPPORTED,
				  new BytePointer("simple lib example only supports model with data type INT32 or FP32"));
			}
			if (seen_data_type.isEmpty()) {
			    seen_data_type = input["datatype"].GetString();
			} else if (strcmp(seen_data_type.c_str(), input["datatype"].GetString())) {
			  return TRITONSERVER_ErrorNew(
				  TRITONSERVER_ERROR_INVALID_ARG,
				  new BytePointer("the inputs and outputs of 'simple' model must have the data type"));
			}
		}
		for (const auto& output : model_metadata["outputs"].GetArray()) {
			if (strcmp(output["datatype"].GetString(), "INT32") &&
				strcmp(output["datatype"].GetString(), "FP32")) {
			    return TRITONSERVER_ErrorNew(
				  TRITONSERVER_ERROR_UNSUPPORTED,
				  new BytePointer("simple lib example only supports model with data type INT32 or FP32"));
			} else if (strcmp(seen_data_type.c_str(), output["datatype"].GetString())) {
			    return TRITONSERVER_ErrorNew(
				  TRITONSERVER_ERROR_INVALID_ARG,
				  new BytePointer("the inputs and outputs of 'simple' model must have the data type"));
			}
		}
		//jack: check about c_str and strcmp
		is_int = (strcmp(seen_data_type.c_str(), "INT32") == 0);
		is_torch_model =
			(strcmp(model_metadata["platform"].GetString(), "pytorch_libtorch") == 0);
		return null;
	}

	//jack: how to do with template? how to do with resize?
	template <typename T>
	void
	GenerateInputData(
			std::vector<char>* input0_data, std::vector<char>* input1_data)
	{
		 input0_data->resize(16 * sizeof(T));
		 input1_data->resize(16 * sizeof(T));
		 for (size_t i = 0; i < 16; ++i) {
			 ((T*)input0_data->data())[i] = i;
			 ((T*)input1_data->data())[i] = 1;
		 }
	}
		
	template <typename T>
	void
	CompareResult(
		String output0_name, String output1_name,
		Pointer input0, Pointer input1, Pointer output0,
		Pointer output1)
	{
		 for (size_t i = 0; i < 16; ++i) {
			std::cout << ((T*)input0)[i] << " + " << ((T*)input1)[i] << " = "
					  << ((T*)output0)[i] << std::endl;
			std::cout << ((T*)input0)[i] << " - " << ((T*)input1)[i] << " = "
					  << ((T*)output1)[i] << std::endl;
		
			if ((((T*)input0)[i] + ((T*)input1)[i]) != ((T*)output0)[i]) {
			  FAIL("incorrect sum in " + output0_name);
			}
			if ((((T*)input0)[i] - ((T*)input1)[i]) != ((T*)output1)[i]) {
			  FAIL("incorrect difference in " + output1_name);
			}
		 }
	}
		
	void
	Check(
		TRITONSERVER_InferenceResponse response,
		char[] input0_data, char[] input1_data,
		String output0, String output1,
		long expected_byte_size,
		TRITONSERVER_DataType expected_datatype, boolean is_int)
	{
	//jack: how to do with unordered_map? 
	    std::unordered_map<std::string, std::vector<char>> output_data;
		
		long output_count;
		FAIL_IF_ERR(
			TRITONSERVER_InferenceResponseOutputCount(response, output_count),
			"getting number of response outputs");
		if (output_count != 2) {
			FAIL("expecting 2 response outputs, got " + String(output_count));
		}
		
		for (long idx = 0; idx < output_count; ++idx) {
			BytePointer cname;
			TRITONSERVER_DataType datatype;
		//jack: is there PointerLong? int64 should be long, right?
			const int64_t* shape;
			long dim_count;
			Pointer base;
			long byte_size;
			TRITONSERVER_MemoryType memory_type;
			long memory_type_id;
			Pointer userp;
		
			FAIL_IF_ERR(
				TRITONSERVER_InferenceResponseOutput(
					response, idx, cname, datatype, shape, dim_count, base,
					byte_size, memory_type, memory_type_id, userp),
				"getting output info");
		
			if (cname == null) {
			  FAIL("unable to get output name");
			}
		
			String name(cname);
			if ((name != output0) && (name != output1)) {
			  FAIL("unexpected output '" + name + "'");
			}
		//jack: when the above shape issue fixed, will change this to some position stuff 
			if ((dim_count != 2) || (shape[0] != 1) || (shape[1] != 16)) {
			  FAIL("unexpected shape for '" + name + "'");
			}
		
			if (datatype != expected_datatype) {
			  FAIL(
				  "unexpected datatype '" +
				  String(TRITONSERVER_DataTypeString(datatype)) + "' for '" +
				  name + "'");
			}
		
			if (byte_size != expected_byte_size) {
			  FAIL(
				  "unexpected byte-size, expected " +
				  String(expected_byte_size) + ", got " +
				  String(byte_size) + " for " + name);
			}
		
			if (enforce_memory_type && (memory_type != requested_memory_type)) {
			  FAIL(
				  "unexpected memory type, expected to be allocated in " +
				  String(TRITONSERVER_MemoryTypeString(requested_memory_type)) +
				  ", got " + String(TRITONSERVER_MemoryTypeString(memory_type)) +
				  ", id " + String(memory_type_id) + " for " + name);
			}
		
			// We make a copy of the data here... which we could avoid for
			// performance reasons but ok for this simple example.
			//jack: change this when unordered_map is fixed
			char[] odata = output_data[name];
			//jack: how to do with std::vector func of assign?
			switch (memory_type) {
			  case TRITONSERVER_MEMORY_CPU: {
				std::cout << name << " is stored in system memory" << std::endl;
				const char* cbase = reinterpret_cast<const char*>(base);
				odata.assign(cbase, cbase + byte_size);
				break;
			  }
		
			  case TRITONSERVER_MEMORY_CPU_PINNED: {
				std::cout << name << " is stored in pinned memory" << std::endl;
				const char* cbase = reinterpret_cast<const char*>(base);
				odata.assign(cbase, cbase + byte_size);
				break;
			  }
		
		if (triton_enable_gpu)
		{
			  case TRITONSERVER_MEMORY_GPU: {
				std::cout << name << " is stored in GPU memory" << std::endl;
				odata.reserve(byte_size);
				FAIL_IF_CUDA_ERR(
					cudaMemcpy(&odata[0], base, byte_size, cudaMemcpyDeviceToHost),
					"getting " + name + " data from GPU memory");
				break;
			  }
		}
		
			  default:
				FAIL("unexpected memory type");
			}
		  }
		
		  if (is_int) {
			CompareResult<int32_t>(
				output0, output1, &input0_data[0], &input1_data[0],
				output_data[output0].data(), output_data[output1].data());
		  } else {
			CompareResult<float>(
				output0, output1, &input0_data[0], &input1_data[0],
				output_data[output0].data(), output_data[output1].data());
		  }
	}
		
	
	public static void main(String[] args)
	{
		String model_repository_path;
		int verbose_level = 0;
		
		// Parse commandline...
		//jack: how to do arg check in java, any reference?
		int opt;
		  while ((opt = getopt(argc, argv, "vm:r:")) != -1) {
			switch (opt) {
			  case 'm': {
				enforce_memory_type = true;
				if (!strcmp(optarg, "system")) {
				  requested_memory_type = TRITONSERVER_MEMORY_CPU;
				} else if (!strcmp(optarg, "pinned")) {
				  requested_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
				} else if (!strcmp(optarg, "gpu")) {
				  requested_memory_type = TRITONSERVER_MEMORY_GPU;
				} else {
				  Usage(
					  argv,
					  "-m must be used to specify one of the following types:"
					  " <\"system\"|\"pinned\"|gpu>");
				}
				break;
			  }
			  case 'r':
				model_repository_path = optarg;
				break;
			  case 'v':
				verbose_level = 1;
				break;
			  case '?':
				Usage(argv);
				break;
			}
		  }
		
		  if (model_repository_path.isEmpty()) {
			Usage(argv, "-r must be used to specify model repository path");
		  }
		if (triton_enable_gpu)
		{
		  if (enforce_memory_type && requested_memory_type != TRITONSERVER_MEMORY_CPU) {
			Usage(argv, "-m can only be set to \"system\" without enabling GPU");
		  }
		}
				
		  // Check API version.
	    long api_version_major, api_version_minor;
		FAIL_IF_ERR(
			TRITONSERVER_ApiVersion(api_version_major, api_version_minor),
			"getting Triton API version");
		if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
			(TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
			FAIL("triton server API version mismatch");
		}
		
		// Create the server...
		TRITONSERVER_ServerOptions server_options = null;
		FAIL_IF_ERR(
		    TRITONSERVER_ServerOptionsNew(server_options),
			"creating server options");
		FAIL_IF_ERR(
			TRITONSERVER_ServerOptionsSetModelRepositoryPath(
				  server_options, model_repository_path.c_str()),
			"setting model repository path");
		FAIL_IF_ERR(
			TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
			"setting verbose logging level");
		FAIL_IF_ERR(
			TRITONSERVER_ServerOptionsSetBackendDirectory(
				server_options, "/opt/tritonserver/backends"),
			"setting backend directory");
		FAIL_IF_ERR(
			TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
				server_options, "/opt/tritonserver/repoagents"),
			"setting repository agent directory");
		FAIL_IF_ERR(
			TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
			"setting strict model configuration");
		if (triton_enable_gpu)
		{
		  double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
		}
		else
		{
		  double min_compute_capability = 0;
		}	
		  FAIL_IF_ERR(
			  TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
				  server_options, min_compute_capability),
			  "setting minimum supported CUDA compute capability");
		
		TRITONSERVER_Server server_ptr = null;
		FAIL_IF_ERR(
			TRITONSERVER_ServerNew(server_ptr, server_options), "creating server");
		FAIL_IF_ERR(
			TRITONSERVER_ServerOptionsDelete(server_options),
			"deleting server options");
		//jack: how to do with shared_ptr here?
		std::shared_ptr<TRITONSERVER_Server> server(
			server_ptr, TRITONSERVER_ServerDelete);
		
		// Wait until the server is both live and ready.
		long health_iters = 0;
		while (true) {
		    boolean live, ready;
			FAIL_IF_ERR(
			//jack: how to do with get func of shared_ptr?
				TRITONSERVER_ServerIsLive(server.get(), &live),
				"unable to get server liveness");
			FAIL_IF_ERR(
				TRITONSERVER_ServerIsReady(server.get(), &ready),
				"unable to get server readiness");
			System.out.println("Server Health: live" + ", ready");
			
			if (live && ready) {
			  break;
			}
		
			if (++health_iters >= 10) {
			  FAIL("failed to find healthy inference server");
			}
		
			Thread.sleep(500);
		}
		
		// Print status of the server.
		{
		    TRITONSERVER_Message server_metadata_message;
			FAIL_IF_ERR(
				TRITONSERVER_ServerMetadata(server.get(), server_metadata_message),
				"unable to get server metadata message");
			Pointer buffer;
			long byte_size;
			FAIL_IF_ERR(
				TRITONSERVER_MessageSerializeToJson(
					server_metadata_message, buffer, byte_size),
				"unable to serialize server metadata message");
		
			System.out.println("Server Status: ");
			System.out.println(String(buffer, byte_size));
		
			FAIL_IF_ERR(
				TRITONSERVER_MessageDelete(server_metadata_message),
				"deleting status metadata");
		  }
		  //jack: is this right??
		  String model_name = "simple";
		
		  // Wait for the model to become available.
		  boolean is_torch_model = false;
		  boolean is_int = true;
		  boolean is_ready = false;
		  health_iters = 0;
		  while (!is_ready) {
			FAIL_IF_ERR(
				TRITONSERVER_ServerModelIsReady(
					server.get(), model_name.c_str(), 1, &is_ready),
				"unable to get model readiness");
			if (!is_ready) {
			  if (++health_iters >= 10) {
				FAIL("model failed to be ready in 10 iterations");
			  }
			  Thread.sleep(500);
			  continue;
			}
		
			TRITONSERVER_Message model_metadata_message;
			FAIL_IF_ERR(
				TRITONSERVER_ServerModelMetadata(
					server.get(), model_name.c_str(), 1, model_metadata_message),
				"unable to get model metadata message");
			Pointer buffer;
			long byte_size;
			FAIL_IF_ERR(
				TRITONSERVER_MessageSerializeToJson(
					model_metadata_message, buffer, byte_size),
				"unable to serialize model status protobuf");
		
			rapidjson::Document model_metadata;
			model_metadata.Parse(buffer, byte_size);
			if (model_metadata.HasParseError()) {
			  FAIL(
				  "error: failed to parse model metadata from JSON: " +
				  String(GetParseError_En(model_metadata.GetParseError())) +
				  " at " + String(model_metadata.GetErrorOffset()));
			}
		
			FAIL_IF_ERR(
				TRITONSERVER_MessageDelete(model_metadata_message),
				"deleting status protobuf");
		    //jack: how to do with strcmp?
			if (strcmp(model_metadata["name"].GetString(), model_name.c_str())) {
			  FAIL("unable to find metadata for model");
			}
		
			boolean found_version = false;
			if (model_metadata.HasMember("versions")) {
			  //jack: how to set type for auto here?
			  for (const auto& version : model_metadata["versions"].GetArray()) {
				if (strcmp(version.GetString(), "1") == 0) {
				  found_version = true;
				  break;
				}
			  }
			}
			if (!found_version) {
			  FAIL("unable to find version 1 status for model");
			}
		
			FAIL_IF_ERR(
				ParseModelMetadata(model_metadata, is_int, is_torch_model),
				"parsing model metadata");
		  }
		
		  // Create the allocator that will be used to allocate buffers for
		  // the result tensors.
		  TRITONSERVER_ResponseAllocator allocator = null;
		  FAIL_IF_ERR(
			  TRITONSERVER_ResponseAllocatorNew(
				  allocator, ResponseAlloc, ResponseRelease, null /* start_fn */),
			  "creating response allocator");
		
		  // Inference
		  TRITONSERVER_InferenceRequest irequest = null;
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestNew(
				  irequest, server.get(), model_name.c_str(), -1 /* model_version */),
			  "creating inference request");
		
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestSetId(irequest, "my_request_id"),
			  "setting ID for the request");
		
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestSetReleaseCallback(
				  irequest, InferRequestComplete, null /* request_release_userp */),
			  "setting request release callback");
		
		  // Inputs
		  //jack: dont know how to do with this
		  auto input0 = is_torch_model ? "INPUT__0" : "INPUT0";
		  auto input1 = is_torch_model ? "INPUT__1" : "INPUT1";
		  //jack: how to do this with long []?
		  std::vector<int64_t> input0_shape({1, 16});
		  std::vector<int64_t> input1_shape({1, 16});
		
		  TRITONSERVER_DataType datatype =
			  (is_int) ? TRITONSERVER_TYPE_INT32 : TRITONSERVER_TYPE_FP32;
		
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestAddInput(
				  irequest, input0, datatype, &input0_shape[0], input0_shape.size()),
			  "setting input 0 meta-data for the request");
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestAddInput(
				  irequest, input1, datatype, &input1_shape[0], input1_shape.size()),
			  "setting input 1 meta-data for the request");
		  //jack: how to set this auto?
		  auto output0 = is_torch_model ? "OUTPUT__0" : "OUTPUT0";
		  auto output1 = is_torch_model ? "OUTPUT__1" : "OUTPUT1";
		
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
			  "requesting output 0 for the request");
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output1),
			  "requesting output 1 for the request");
		
		  // Create the data for the two input tensors. Initialize the first
		  // to unique values and the second to all ones.
		  std::vector<char> input0_data;
		  std::vector<char> input1_data;
		  char[] input0_data;
		  char[] input1_data;
		  //jack: will do this if template is fixed
		  if (is_int) {
			GenerateInputData<int32_t>(&input0_data, &input1_data);
		  } else {
			GenerateInputData<float>(&input0_data, &input1_data);
		  }
		  //jack: how to do size of char[]?
		  size_t input0_size = input0_data.size();
		  size_t input1_size = input1_data.size();
		
		  const void* input0_base = &input0_data[0];
		  const void* input1_base = &input1_data[0];
		
		if (triton_enable_gpu)
		{
		  //jack: how to do with this?
		  std::unique_ptr<void, decltype(cuda_data_deleter)> input0_gpu(
			  nullptr, cuda_data_deleter);
		  std::unique_ptr<void, decltype(cuda_data_deleter)> input1_gpu(
			  nullptr, cuda_data_deleter);
		  boolean use_cuda_memory =
			  (enforce_memory_type &&
			   (requested_memory_type != TRITONSERVER_MEMORY_CPU));
		  if (use_cuda_memory) {
			FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
			if (requested_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
			  Pointer dst;
			  FAIL_IF_CUDA_ERR(
				  cudaMalloc(dst, input0_size),
				  "allocating GPU memory for INPUT0 data");
			  input0_gpu.reset(dst);
			  FAIL_IF_CUDA_ERR(
				  cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToDevice),
				  "setting INPUT0 data in GPU memory");
			  FAIL_IF_CUDA_ERR(
				  cudaMalloc(&dst, input1_size),
				  "allocating GPU memory for INPUT1 data");
			  input1_gpu.reset(dst);
			  FAIL_IF_CUDA_ERR(
				  cudaMemcpy(dst, &input1_data[0], input1_size, cudaMemcpyHostToDevice),
				  "setting INPUT1 data in GPU memory");
			} else {
			  Pointer dst;
			  FAIL_IF_CUDA_ERR(
				  cudaHostAlloc(dst, input0_size, cudaHostAllocPortable),
				  "allocating pinned memory for INPUT0 data");
			  input0_gpu.reset(dst);
			  FAIL_IF_CUDA_ERR(
				  cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToHost),
				  "setting INPUT0 data in pinned memory");
			  FAIL_IF_CUDA_ERR(
				  cudaHostAlloc(dst, input1_size, cudaHostAllocPortable),
				  "allocating pinned memory for INPUT1 data");
			  input1_gpu.reset(dst);
			  FAIL_IF_CUDA_ERR(
				  cudaMemcpy(dst, &input1_data[0], input1_size, cudaMemcpyHostToHost),
				  "setting INPUT1 data in pinned memory");
			}
		  }
		
		  input0_base = use_cuda_memory ? input0_gpu.get() : &input0_data[0];
		  input1_base = use_cuda_memory ? input1_gpu.get() : &input1_data[0];
		}
		
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestAppendInputData(
				  irequest, input0, input0_base, input0_size, requested_memory_type,
				  0 /* memory_type_id */),
			  "assigning INPUT0 data");
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestAppendInputData(
				  irequest, input1, input1_base, input1_size, requested_memory_type,
				  0 /* memory_type_id */),
			  "assigning INPUT1 data");
		
		  // Perform inference...
		  {
		    //jack: how to do with std::promise
			auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
			//jack: how to do with std::future
			std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();
		
			FAIL_IF_ERR(
				TRITONSERVER_InferenceRequestSetResponseCallback(
					irequest, allocator, null /* response_allocator_userp */,
					InferResponseComplete, reinterpret_cast<void*>(p)),
				"setting response callback");
		
			FAIL_IF_ERR(
				TRITONSERVER_ServerInferAsync(
					server.get(), irequest, null /* trace */),
				"running inference");
		
			// Wait for the inference to complete.
			TRITONSERVER_InferenceResponse completed_response = completed.get();
		
			FAIL_IF_ERR(
				TRITONSERVER_InferenceResponseError(completed_response),
				"response status");
		
			Check(
				completed_response, input0_data, input1_data, output0, output1,
				input0_size, datatype, is_int);
		
			FAIL_IF_ERR(
				TRITONSERVER_InferenceResponseDelete(completed_response),
				"deleting inference response");
		  }
		
		  // Modify some input data in place and then reuse the request
		  // object. For simplicity we only do this when the input tensors are
		  // in non-pinned system memory.
		  if (!enforce_memory_type ||
			  (requested_memory_type == TRITONSERVER_MEMORY_CPU)) {
			if (is_int) {
			//jack: how to do with reinterpret_cast?
			  int32_t* input0_base = reinterpret_cast<int32_t*>(&input0_data[0]);
			  input0_base[0] = 27;
			} else {
			  float* input0_base = reinterpret_cast<float*>(&input0_data[0]);
			  input0_base[0] = 27.0;
			}
		    //jack: promise and future
			auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
			std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();
		
			// Using a new promise so have to re-register the callback to set
			// the promise as the userp.
			FAIL_IF_ERR(
				TRITONSERVER_InferenceRequestSetResponseCallback(
					irequest, allocator, null /* response_allocator_userp */,
					InferResponseComplete, reinterpret_cast<void*>(p)),
				"setting response callback");
		
			FAIL_IF_ERR(
				TRITONSERVER_ServerInferAsync(
					server.get(), irequest, null /* trace */),
				"running inference");
		
			// Wait for the inference to complete.
			TRITONSERVER_InferenceResponse completed_response = completed.get();
			FAIL_IF_ERR(
				TRITONSERVER_InferenceResponseError(completed_response),
				"response status");
		
			Check(
				completed_response, input0_data, input1_data, output0, output1,
				input0_size, datatype, is_int);
		
			FAIL_IF_ERR(
				TRITONSERVER_InferenceResponseDelete(completed_response),
				"deleting inference response");
		  }
		
		  // Remove input data and then add back different data.
		  {
			FAIL_IF_ERR(
				TRITONSERVER_InferenceRequestRemoveAllInputData(irequest, input0),
				"removing INPUT0 data");
			FAIL_IF_ERR(
				TRITONSERVER_InferenceRequestAppendInputData(
					irequest, input0, input1_base, input1_size, requested_memory_type,
					0 /* memory_type_id */),
				"assigning INPUT1 data to INPUT0");
		
			auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
			std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();
		
			// Using a new promise so have to re-register the callback to set
			// the promise as the userp.
			FAIL_IF_ERR(
				TRITONSERVER_InferenceRequestSetResponseCallback(
					irequest, allocator, null /* response_allocator_userp */,
					InferResponseComplete, reinterpret_cast<void*>(p)),
				"setting response callback");
		
			FAIL_IF_ERR(
				TRITONSERVER_ServerInferAsync(
					server.get(), irequest, null /* trace */),
				"running inference");
		
			// Wait for the inference to complete.
			TRITONSERVER_InferenceResponse completed_response = completed.get();
			FAIL_IF_ERR(
				TRITONSERVER_InferenceResponseError(completed_response),
				"response status");
		
			// Both inputs are using input1_data...
			Check(
				completed_response, input1_data, input1_data, output0, output1,
				input0_size, datatype, is_int);
		
			FAIL_IF_ERR(
				TRITONSERVER_InferenceResponseDelete(completed_response),
				"deleting inference response");
		  }
		
		  FAIL_IF_ERR(
			  TRITONSERVER_InferenceRequestDelete(irequest),
			  "deleting inference request");
		
		  FAIL_IF_ERR(
			  TRITONSERVER_ResponseAllocatorDelete(allocator),
			  "deleting response allocator");
		
		  System.exit(0);
	}
    


	
}	


