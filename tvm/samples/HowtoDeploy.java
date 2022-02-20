/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import org.bytedeco.tvm.*;
import org.bytedeco.tvm.Module;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;
import static org.bytedeco.tvm.global.tvm_runtime.*;

public class HowtoDeploy {

    static void PrepareTestLibs() throws Exception {
        String clang = Loader.load(org.bytedeco.llvm.program.clang.class).replace('\\', '/');
        String clangPath = clang.substring(0, clang.lastIndexOf('/'));

        Py_Initialize(org.bytedeco.tvm.presets.tvm.cachePackages());
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        PyRun_StringFlags("\"\"\"Script to prepare test_addone.so\"\"\"\n"
                + "import tvm\n"
                + "import numpy as np\n"
                + "from tvm import te\n"
                + "from tvm import relay\n"
                + "import os\n"

                + "def prepare_test_libs(base_path):\n"
                + "    n = te.var(\"n\")\n"
                + "    A = te.placeholder((n,), name=\"A\")\n"
                + "    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name=\"B\")\n"
                + "    s = te.create_schedule(B.op)\n"
                + "    # Compile library as dynamic library\n"
                + "    fadd_dylib = tvm.build(s, [A, B], \"llvm\", name=\"addone\")\n"
                + "    dylib_path = os.path.join(base_path, \"test_addone_dll.so\")\n"
                + "    fadd_dylib.export_library(dylib_path)\n"

                + "    # Compile library in system library mode\n"
                + "    fadd_syslib = tvm.build(s, [A, B], \"llvm --system-lib\", name=\"addonesys\")\n"
                + "    syslib_path = os.path.join(base_path, \"test_addone_sys.o\")\n"
                + "    fadd_syslib.save(syslib_path)\n"

                + "def prepare_graph_lib(base_path):\n"
                + "    x = relay.var(\"x\", shape=(2, 2), dtype=\"float32\")\n"
                + "    y = relay.var(\"y\", shape=(2, 2), dtype=\"float32\")\n"
                + "    params = {\"y\": np.ones((2, 2), dtype=\"float32\")}\n"
                + "    mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))\n"
                + "    # build a module\n"
                + "    compiled_lib = relay.build(mod, tvm.target.create(\"llvm\"), params=params)\n"
                + "    # export it as a shared library\n"
                + "    # If you are running cross compilation, you can also consider export\n"
                + "    # to tar and invoke host compiler later.\n"
                + "    dylib_path = os.path.join(base_path, \"test_relay_add.so\")\n"
                + "    compiled_lib.export_library(dylib_path)\n"

                + "if __name__ == \"__main__\":\n"
                + "    lib_path = os.path.join(os.getcwd(), \"lib\")\n"
                + "    os.makedirs(lib_path, exist_ok = True)\n"
                + "    os.environ[\"PATH\"] += os.pathsep + \"" + clangPath + "\"\n"
                + "    prepare_test_libs(lib_path)\n"
                + "    prepare_graph_lib(lib_path)\n",

                Py_file_input, globals, globals, null);

        if (PyErr_Occurred() != null) {
            System.err.println("Python error occurred");
            PyErr_Print();
            System.exit(-1);
        }
    }

    static void Verify(Module mod, String fname) {
        // Get the function from the module.
        PackedFunc f = mod.GetFunction(fname);
        assert f != null;
        // Allocate the DLPack data structures.
        //
        // Note that we use TVM runtime API to allocate the DLTensor in this example.
        // TVM accept DLPack compatible DLTensors, so function can be invoked
        // as long as we pass correct pointer to DLTensor array.
        //
        // For more information please refer to dlpack.
        // One thing to notice is that DLPack contains alignment requirement for
        // the data pointer and TVM takes advantage of that.
        // If you plan to use your customized data container, please
        // make sure the DLTensor you pass in meet the alignment requirement.
        //
        DLTensor x = new DLTensor(null);
        DLTensor y = new DLTensor(null);
        int ndim = 1;
        int dtype_code = kDLFloat;
        int dtype_bits = 32;
        int dtype_lanes = 1;
        int device_type = kDLCPU;
        int device_id = 0;
        long[] shape = {10};
        TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, x);
        TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, y);
        FloatPointer xdata = new FloatPointer(x.data());
        FloatPointer ydata = new FloatPointer(y.data());
        for (long i = 0; i < shape[0]; ++i) {
            xdata.put(i, i);
        }
        // Invoke the function
        // PackedFunc is a function that can be invoked via positional argument.
        // The signature of the function is specified in tvm.build
        TVMValue values = new TVMValue(2);
        IntPointer codes = new IntPointer(2);
        TVMArgsSetter setter = new TVMArgsSetter(values, codes);
        setter.apply(0, x);
        setter.apply(1, y);
        f.CallPacked(new TVMArgs(values, codes, 2), null);
        // Print out the output
        for (long i = 0; i < shape[0]; ++i) {
            float yi = ydata.get(i);
            System.out.println(yi);
            assert yi == i + 1.0f;
        }
        System.out.println("Finish verification...");
        TVMArrayFree(x);
        TVMArrayFree(y);
    }

    static void DeploySingleOp() {
        // Normally we can directly
        Module mod_dylib = Module.LoadFromFile("lib/test_addone_dll.so");
        System.out.println("Verify dynamic loading from test_addone_dll.so");
        Verify(mod_dylib, "addone");
        // For libraries that are directly packed as system lib and linked together with the app
        // We can directly use GetSystemLib to get the system wide library.
        System.out.println("Verify load function from system lib");
        TVMRetValue rv = new TVMRetValue();
        Registry.Get("runtime.SystemLib").CallPacked(new TVMArgs((TVMValue)null, (IntPointer)null, 0), rv);
        Module mod_syslib = rv.asModule();
        // Verify(mod_syslib, "addonesys");
    }

    static void DeployGraphExecutor() {
        System.out.println("Running graph executor...");
        // load in the library
        DLDevice dev = new DLDevice().device_type(kDLCPU).device_id(0);
        Module mod_factory = Module.LoadFromFile("lib/test_relay_add.so");
        // create the graph executor module
        TVMValue values = new TVMValue(2);
        IntPointer codes = new IntPointer(2);
        TVMArgsSetter setter = new TVMArgsSetter(values, codes);
        setter.apply(0, dev);
        TVMRetValue rv = new TVMRetValue();
        mod_factory.GetFunction("default").CallPacked(new TVMArgs(values, codes, 1), rv);
        Module gmod = rv.asModule();
        PackedFunc set_input = gmod.GetFunction("set_input");
        PackedFunc get_output = gmod.GetFunction("get_output");
        PackedFunc run = gmod.GetFunction("run");

        // Use the C++ API
        NDArray x = NDArray.Empty(new ShapeTuple(2, 2), new DLDataType().code((byte)kDLFloat).bits((byte)32).lanes((short)1), dev);
        NDArray y = NDArray.Empty(new ShapeTuple(2, 2), new DLDataType().code((byte)kDLFloat).bits((byte)32).lanes((short)1), dev);
        FloatPointer xdata = new FloatPointer(x.accessDLTensor().data());
        FloatPointer ydata = new FloatPointer(y.accessDLTensor().data());

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                xdata.put(i * 2 + j, i * 2 + j);
            }
        }
        // set the right input
        setter.apply(0, new BytePointer("x"));
        setter.apply(1, x);
        set_input.CallPacked(new TVMArgs(values, codes, 2), rv);
        // run the code
        run.CallPacked(new TVMArgs(values, codes, 0), rv);
        // get the output
        setter.apply(0, 0);
        setter.apply(1, y);
        get_output.CallPacked(new TVMArgs(values, codes, 2), rv);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                float yi = ydata.get(i * 2 + j);
                System.out.println(yi);
                assert yi == i * 2 + j + 1;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        PrepareTestLibs();
        DeploySingleOp();
        DeployGraphExecutor();
        System.exit(0);
    }
}
