import java.io.*;
import java.util.*;
import org.bytedeco.javacpp.*;

import org.bytedeco.cuda.cudart.*;
import org.bytedeco.tensorrt.nvinfer.*;
import org.bytedeco.tensorrt.nvparsers.*;
import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.tensorrt.global.nvinfer.*;
import static org.bytedeco.tensorrt.global.nvparsers.*;

public class SampleGoogleNet {
    static void CHECK(int status)
    {
        if (status != 0)
        {
            System.out.println("Cuda failure: " + status);
            System.exit(6);
        }
    }

    // Logger for GIE info/warning/errors
    static class Logger extends ILogger
    {
        @Override public void log(Severity severity, String msg)
        {
            severity = severity.intern();

            // suppress verbose-level messages
            if (severity == Severity.kVERBOSE) return;

            switch (severity)
            {
                case kINTERNAL_ERROR: System.err.print("INTERNAL_ERROR: "); break;
                case kERROR: System.err.print("ERROR: "); break;
                case kWARNING: System.err.print("WARNING: "); break;
                case kINFO: System.err.print("INFO: "); break;
                case kVERBOSE: System.err.print("VERBOSE: "); break;
                default: System.err.print("UNKNOWN: "); break;
            }
            System.err.println(msg);
        }
    }
    static Logger gLogger = new Logger();

    static String locateFile(String input, String[] directories)
    {
        String file = "";
        int MAX_DEPTH = 10;
        boolean found = false;
        for (String dir : directories)
        {
            file = dir + input;
            for (int i = 0; i < MAX_DEPTH && !found; i++)
            {
                File checkFile = new File(file);
                found = checkFile.exists();
                if (found) break;
                file = "../" + file;
            }
            if (found) break;
            file = "";
        }

        if (file.isEmpty())
            System.err.println("Could not find a file due to it not existing in the data directory.");
        return file;
    }

    // stuff we know about the network and the caffe input/output blobs

    static int BATCH_SIZE = 4;
    static int TIMING_ITERATIONS = 1000;

    static String INPUT_BLOB_NAME = "data";
    static String OUTPUT_BLOB_NAME = "prob";


    static String locateFile(String input)
    {
        String[] dirs = {"data/samples/googlenet/", "data/googlenet/"};
        return locateFile(input, dirs);
    }

    static class Profiler extends IProfiler
    {
        LinkedHashMap<String, Float> mProfile = new LinkedHashMap<String, Float>();

        @Override public void reportLayerTime(String layerName, float ms)
        {
            Float time = mProfile.get(layerName);
            mProfile.put(layerName, (time != null ? time : 0) + ms);
        }

        public void printLayerTimes()
        {
            float totalTime = 0;
            for (Map.Entry<String,Float> e : mProfile.entrySet())
            {
                System.out.printf("%-40.40s %4.3fms\n", e.getKey(), e.getValue() / TIMING_ITERATIONS);
                totalTime += e.getValue();
            }
            System.out.printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
        }

    }
    static Profiler gProfiler = new Profiler();

    static void caffeToGIEModel(String deployFile,     // name for caffe prototxt
                         String modelFile,             // name for model 
                         String[] outputs,             // network outputs
                         int maxBatchSize,             // batch size - NB must be at least as large as the batch we want to run with)
                         IHostMemory[] gieModelStream)
    {
        // create API root class - must span the lifetime of the engine usage
        IBuilder builder = createInferBuilder(gLogger);
        INetworkDefinition network = builder.createNetworkV2(0);
        IBuilderConfig config = builder.createBuilderConfig();

        // parse the caffe model to populate the network, then set the outputs
        ICaffeParser parser = createCaffeParser();

        boolean useFp16 = builder.platformHasFastFp16();

        DataType modelDataType = useFp16 ? DataType.kHALF : DataType.kFLOAT; // create a 16-bit model if it's natively supported
        IBlobNameToTensor blobNameToTensor =
            parser.parse(locateFile(deployFile),                // caffe deploy file
                                     locateFile(modelFile),     // caffe model file
                                     network,                   // network definition that the parser will populate
                                     modelDataType);

        assert blobNameToTensor != null;
        // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate    
        for (String s : outputs)
            network.markOutput(blobNameToTensor.find(s));

        // Build the engine
        builder.setMaxBatchSize(maxBatchSize);
        config.setMaxWorkspaceSize(16 << 20);

        // set up the network for paired-fp16 format if available
        if(useFp16)
            config.setFlag(BuilderFlag.kFP16);

        ICudaEngine engine = builder.buildEngineWithConfig(network, config);
        assert engine != null;

        // we don't need the network any more, and we can destroy the parser
        network.destroy();
        parser.destroy();

        // serialize the engine, then close everything down
        gieModelStream[0] = engine.serialize();
        engine.destroy();
        builder.destroy();
        shutdownProtobufLibrary();
    }

    static void timeInference(ICudaEngine engine, int batchSize)
    {
        // input and output buffer pointers that we pass to the engine - the engine requires exactly ICudaEngine::getNbBindings(),
        // of these, but in this case we know that there is exactly one input and one output.
        assert engine.getNbBindings() == 2;
        PointerPointer buffers = new PointerPointer(2);

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than ICudaEngine::getNbBindings()
        int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

        // allocate GPU buffers
        Dims3 inputDims = new Dims3(engine.getBindingDimensions(inputIndex)), outputDims = new Dims3(engine.getBindingDimensions(outputIndex));
        long inputSize = batchSize * inputDims.d(0) * inputDims.d(1) * inputDims.d(2) * Float.SIZE / 8;
        long outputSize = batchSize * outputDims.d(0) * outputDims.d(1) * outputDims.d(2) * Float.SIZE / 8;

        CHECK(cudaMalloc(buffers.position(inputIndex), inputSize));
        CHECK(cudaMalloc(buffers.position(outputIndex), outputSize));

        IExecutionContext context = engine.createExecutionContext();
        context.setProfiler(gProfiler);

        // zero the input buffer
        CHECK(cudaMemset(buffers.position(inputIndex).get(), 0, inputSize));

        for (int i = 0; i < TIMING_ITERATIONS;i++)
            context.execute(batchSize, buffers.position(0));

        // release the context and buffers
        context.destroy();
        CHECK(cudaFree(buffers.position(inputIndex).get()));
        CHECK(cudaFree(buffers.position(outputIndex).get()));
    }


    public static void main(String[] args)
    {
        System.out.println("Building and running a GPU inference engine for GoogleNet, N=4...");

        // parse the caffe model and the mean file
        IHostMemory[] gieModelStream = { null };
        caffeToGIEModel("googlenet.prototxt", "googlenet.caffemodel", new String[] { OUTPUT_BLOB_NAME }, BATCH_SIZE, gieModelStream);

        // create an engine
        IRuntime infer = createInferRuntime(gLogger);
        ICudaEngine engine = infer.deserializeCudaEngine(gieModelStream[0].data(), gieModelStream[0].size(), null);

        System.out.println("Bindings after deserializing:"); 
        for (int bi = 0; bi < engine.getNbBindings(); bi++) { 
            if (engine.bindingIsInput(bi)) { 
                System.out.printf("Binding %d (%s): Input.\n",  bi, engine.getBindingName(bi));
            } else { 
                System.out.printf("Binding %d (%s): Output.\n", bi, engine.getBindingName(bi));
            } 
        }

        // run inference with null data to time network performance
        timeInference(engine, BATCH_SIZE);

        engine.destroy();
        infer.destroy();

        gProfiler.printLayerTimes();

        System.out.println("Done.");

        System.exit(0);
    }
}
