import java.io.File;
import java.util.Random;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import org.bytedeco.tvm.*;
import org.bytedeco.tvm.Module;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;
import static org.bytedeco.tvm.global.tvm_runtime.*;

/**
 * Example showing how to import, optimize, and deploy a BERT model based on:
 * @see <a href="https://gist.github.com/icemelon9/860d3d2c9566d6f69fa8112840dd95c1">Optimize the BERT model on CPUs</a>
 * @see <a href="https://github.com/apache/tvm/tree/v0.7/apps/howto_deploy/">How to Deploy TVM Modules</a>
 */
public class DeployBERT {

    public static void OptimizeBERT() throws Exception {
        // Extract to JavaCPP's cache the Clang compiler as required by TVM on Windows
        String clang = Loader.load(org.bytedeco.llvm.program.clang.class).replace('\\', '/');
        String clangPath = clang.substring(0, clang.lastIndexOf('/'));

        // Extract to JavaCPP's cache CPython and obtain the path to the executable file
        String python = Loader.load(org.bytedeco.cpython.python.class);

        // Install in JavaCPP's cache GluonNLP and MXNet to download and import BERT model
        new ProcessBuilder(python, "-m", "pip", "install", "gluonnlp", "mxnet", "pytest").inheritIO().start().waitFor();

        // Initialize the embedded Python interpreter inside the same process as the JVM
        Pointer program = Py_DecodeLocale(DeployBERT.class.getSimpleName(), null);
        if (program == null) {
            System.err.println("Fatal error: cannot get class name");
            System.exit(1);
        }
        Py_SetProgramName(program);

        // Add TVM and its dependencies to Python path using C API to embed script in Java
        Py_Initialize(org.bytedeco.tvm.presets.tvm.cachePackages());
        PySys_SetArgv(1, new PointerPointer(1).put(program));
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        // Run Python script in embedded interpreter
        PyRun_StringFlags("\"\"\"optimize_bert.py\"\"\"\n"
                + "import time\n"
                + "import argparse\n"
                + "import numpy as np\n"
                + "import mxnet as mx\n"
                + "import gluonnlp as nlp\n"
                + "import tvm\n"
                + "import tvm.testing\n"
                + "from tvm import relay\n"
                + "import tvm.contrib.graph_runtime as runtime\n"
                + "import os\n"

                + "os.environ[\"PATH\"] += os.pathsep + \"" + clangPath + "\"\n"

                + "def timer(thunk, repeat=1, number=10, dryrun=3, min_repeat_ms=1000):\n"
                + "    \"\"\"Helper function to time a function\"\"\"\n"
                + "    for i in range(dryrun):\n"
                + "        thunk()\n"
                + "    ret = []\n"
                + "    for _ in range(repeat):\n"
                + "        while True:\n"
                + "            beg = time.time()\n"
                + "            for _ in range(number):\n"
                + "                thunk()\n"
                + "            end = time.time()\n"
                + "            lat = (end - beg) * 1e3\n"
                + "            if lat >= min_repeat_ms:\n"
                + "                break\n"
                + "            number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))\n"
                + "        ret.append(lat / number)\n"
                + "    return ret\n"

                + "parser = argparse.ArgumentParser(description=\"Optimize BERT-base model from GluonNLP\")\n"
                + "parser.add_argument(\"-b\", \"--batch\", type=int, default=1,\n"
                + "                    help=\"Batch size (default: 1)\")\n"
                + "parser.add_argument(\"-l\", \"--length\", type=int, default=128,\n"
                + "                    help=\"Sequence length (default: 128)\")\n"
                + "args = parser.parse_args()\n"
                + "batch = args.batch\n"
                + "seq_length = args.length\n"

                + "# Instantiate a BERT classifier using GluonNLP\n"
                + "model_name = 'bert_12_768_12'\n"
                + "dataset = 'book_corpus_wiki_en_uncased'\n"
                + "mx_ctx = mx.cpu()\n"
                + "bert, _ = nlp.model.get_model(\n"
                + "    name=model_name,\n"
                + "    ctx=mx_ctx,\n"
                + "    dataset_name=dataset,\n"
                + "    pretrained=False,\n"
                + "    use_pooler=True,\n"
                + "    use_decoder=False,\n"
                + "    use_classifier=False)\n"
                + "model = nlp.model.BERTClassifier(bert, dropout=0.1, num_classes=2)\n"
                + "model.initialize(ctx=mx_ctx)\n"
                + "model.hybridize(static_alloc=True)\n"

                + "# Prepare input data\n"
                + "dtype = \"float32\"\n"
                + "inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(dtype)\n"
                + "token_types = np.random.uniform(size=(batch, seq_length)).astype(dtype)\n"
                + "valid_length = np.asarray([seq_length] * batch).astype(dtype)\n"

                + "# Convert to MXNet NDArray and run the MXNet model\n"
                + "inputs_nd = mx.nd.array(inputs, ctx=mx_ctx)\n"
                + "token_types_nd = mx.nd.array(token_types, ctx=mx_ctx)\n"
                + "valid_length_nd = mx.nd.array(valid_length, ctx=mx_ctx)\n"
                + "mx_out = model(inputs_nd, token_types_nd, valid_length_nd)\n"
                + "mx_out.wait_to_read()\n"

                + "# Benchmark the MXNet latency\n"
                + "res = timer(lambda: model(inputs_nd, token_types_nd, valid_length_nd).wait_to_read(),\n"
                + "            repeat=3,\n"
                + "            dryrun=5,\n"
                + "            min_repeat_ms=1000)\n"
                + "print(f\"MXNet latency for batch {batch} and seq length {seq_length}: {np.mean(res):.2f} ms\")\n"

                + "######################################\n"
                + "# Optimize the BERT model using TVM\n"
                + "######################################\n"

                + "# First, Convert the MXNet model into TVM Relay format\n"
                + "shape_dict = {\n"
                + "    'data0': (batch, seq_length),\n"
                + "    'data1': (batch, seq_length),\n"
                + "    'data2': (batch,)\n"
                + "}\n"
                + "mod, params = relay.frontend.from_mxnet(model, shape_dict)\n"

                + "# Compile the imported model\n"
                + "target = \"llvm -libs=mkl\"\n"
                + "with relay.build_config(opt_level=3, required_pass=[\"FastMath\"]):\n"
                + "    graph, lib, cparams = relay.build(mod, target, params=params)\n"

                + "# Create the executor and set the parameters and inputs\n"
                + "ctx = tvm.cpu()\n"
                + "rt = runtime.create(graph, lib, ctx)\n"
                + "rt.set_input(**cparams)\n"
                + "rt.set_input(data0=inputs, data1=token_types, data2=valid_length)\n"

                + "# Run the executor and validate the correctness\n"
                + "rt.run()\n"
                + "out = rt.get_output(0)\n"
                + "tvm.testing.assert_allclose(out.asnumpy(), mx_out.asnumpy(), rtol=1e-3, atol=1e-3)\n"

                + "# Benchmark the TVM latency\n"
                + "ftimer = rt.module.time_evaluator(\"run\", ctx, repeat=3, min_repeat_ms=1000)\n"
                + "prof_res = np.array(ftimer().results) * 1000\n"
                + "print(f\"TVM latency for batch {batch} and seq length {seq_length}: {np.mean(prof_res):.2f} ms\")\n"

                + "# Export the model to a native library\n"
                + "with tvm.transform.PassContext(opt_level=3, required_pass=[\"FastMath\"]):\n"
                + "    compiled_lib = relay.build(mod, \"llvm -libs=mkl\", params=params)\n"
                + "compiled_lib.export_library(\"lib/libbert.so\")\n",

                Py_file_input, globals, globals, null);

        if (PyErr_Occurred() != null) {
            System.err.println("Python error occurred");
            PyErr_Print();
            System.exit(-1);
        }
        PyMem_RawFree(program);
    }

    public static void DeployBERTRuntime() {
        System.out.println("Running BERT runtime...");
        // load in the library
        DLDevice ctx = new DLDevice().device_type(kDLCPU).device_id(0);
        Module mod_factory = Module.LoadFromFile("lib/libbert.so");
        // create the BERT runtime module
        TVMValue values = new TVMValue(2);
        IntPointer codes = new IntPointer(2);
        TVMArgsSetter setter = new TVMArgsSetter(values, codes);
        setter.apply(0, ctx);
        TVMRetValue rv = new TVMRetValue();
        mod_factory.GetFunction("default").CallPacked(new TVMArgs(values, codes, 1), rv);
        Module gmod = rv.asModule();
        PackedFunc set_input = gmod.GetFunction("set_input");
        PackedFunc get_output = gmod.GetFunction("get_output");
        PackedFunc run = gmod.GetFunction("run");

        // Use the C++ API to create some random sequence
        int batch = 1, seq_length = 128;
        DLDataType dtype = new DLDataType().code((byte)kDLFloat).bits((byte)32).lanes((short)1);
        NDArray inputs = NDArray.Empty(new ShapeTuple(batch, seq_length), dtype, ctx);
        NDArray token_types = NDArray.Empty(new ShapeTuple(batch, seq_length), dtype, ctx);
        NDArray valid_length = NDArray.Empty(new ShapeTuple(batch), dtype, ctx);
        NDArray output = NDArray.Empty(new ShapeTuple(batch, 2), dtype, ctx);
        FloatPointer inputs_data = new FloatPointer(inputs.accessDLTensor().data()).capacity(batch * seq_length);
        FloatPointer token_types_data = new FloatPointer(token_types.accessDLTensor().data()).capacity(batch * seq_length);
        FloatPointer valid_length_data = new FloatPointer(valid_length.accessDLTensor().data()).capacity(batch);
        FloatPointer output_data = new FloatPointer(output.accessDLTensor().data()).capacity(batch * 2);
        FloatIndexer inputs_idx = FloatIndexer.create(inputs_data, new long[]{batch, seq_length});
        FloatIndexer token_types_idx = FloatIndexer.create(token_types_data, new long[]{batch, seq_length});
        FloatIndexer valid_length_idx = FloatIndexer.create(valid_length_data, new long[]{batch});
        FloatIndexer output_idx = FloatIndexer.create(output_data, new long[]{batch, 2});

        Random random = new Random();
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < seq_length; ++j) {
                inputs_idx.put(i, j, random.nextInt(2000));
                token_types_idx.put(i, j, random.nextFloat());
            }
            valid_length_idx.put(i, seq_length);
        }

        // set the right input
        setter.apply(0, new BytePointer("data0"));
        setter.apply(1, inputs);
        set_input.CallPacked(new TVMArgs(values, codes, 2), rv);
        setter.apply(0, new BytePointer("data1"));
        setter.apply(1, token_types);
        set_input.CallPacked(new TVMArgs(values, codes, 2), rv);
        setter.apply(0, new BytePointer("data2"));
        setter.apply(1, valid_length);
        set_input.CallPacked(new TVMArgs(values, codes, 2), rv);
        // run the code
        run.CallPacked(new TVMArgs(values, codes, 0), rv);
        // get the output
        setter.apply(0, 0);
        setter.apply(1, output);
        get_output.CallPacked(new TVMArgs(values, codes, 2), rv);

        System.out.println(output_idx);
    }

    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        new File("lib").mkdir();
        OptimizeBERT();
        DeployBERTRuntime();
    }
}
