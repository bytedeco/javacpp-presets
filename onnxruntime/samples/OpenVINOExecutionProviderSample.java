// Copyright (C) 2026 JavaCPP Presets contributors
// Licensed either under the Apache License, Version 2.0, or (at your option)
// under the terms of the GNU General Public License as published by
// the Free Software Foundation (subject to the "Classpath" exception),
// either version 2, or any later version.

import java.util.Arrays;
import java.util.Base64;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.onnxruntime.Env;
import org.bytedeco.onnxruntime.MemoryInfo;
import org.bytedeco.onnxruntime.RunOptions;
import org.bytedeco.onnxruntime.Session;
import org.bytedeco.onnxruntime.SessionOptions;
import org.bytedeco.onnxruntime.StringStringMap;
import org.bytedeco.onnxruntime.Value;
import org.bytedeco.onnxruntime.ValueVector;

import static org.bytedeco.onnxruntime.global.onnxruntime.*;

public class OpenVINOExecutionProviderSample {
    private static final String IDENTITY_MODEL_BASE64 =
            "CAgSFmphdmFjcHAtcHJlc2V0cy1zYW1wbGU6VgoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRIOaWRlbnRpdHlfZ3JhcGhaEwoFaW5wdXQSCgoICAESBAoCCANiFAoGb3V0cHV0EgoKCAgBEgQKAggDQgIQDQ==";

    public static void main(String[] args) throws Exception {
        String deviceType = args.length > 0 ? args[0] : "CPU";

        // Load the OpenVINO preset first so that ONNX Runtime's OpenVINO provider can resolve
        // the OpenVINO and TBB native libraries bundled by the openvino-platform dependency.
        Loader.load(org.bytedeco.openvino.global.openvino.class);
        Loader.load(org.bytedeco.onnxruntime.global.onnxruntime.class);

        System.out.println("Available ONNX Runtime providers: " + GetAvailableProviders());

        Env env = new Env(ORT_LOGGING_LEVEL_WARNING, "openvino-ep-sample");
        SessionOptions sessionOptions = new SessionOptions();
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        StringStringMap openvinoOptions = new StringStringMap();
        openvinoOptions.put(new BytePointer("device_type"), deviceType);
        sessionOptions.AppendExecutionProvider_OpenVINO_V2(openvinoOptions);
        System.out.println("Appended OpenVINO execution provider with device_type=" + deviceType);

        byte[] modelBytes = Base64.getDecoder().decode(IDENTITY_MODEL_BASE64);
        BytePointer modelData = new BytePointer(modelBytes);
        Session session = new Session(env, modelData, modelBytes.length, sessionOptions);

        float[] input = new float[] {1.25f, -2.5f, 3.75f};
        FloatPointer inputData = new FloatPointer(input);
        long[] shape = new long[] {3};
        LongPointer shapeData = new LongPointer(shape);
        MemoryInfo memoryInfo = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Value inputTensor = Value.CreateTensorFloat(memoryInfo.asOrtMemoryInfo(), inputData, input.length, shapeData, shape.length);

        BytePointer inputName = new BytePointer("input");
        BytePointer outputName = new BytePointer("output");
        PointerPointer inputNames = new PointerPointer(1).put(0, inputName);
        PointerPointer outputNames = new PointerPointer(1).put(0, outputName);
        ValueVector outputs = session.Run(new RunOptions(), inputNames, inputTensor, 1, outputNames, 1);
        float[] output = new float[input.length];
        outputs.get(0).GetTensorMutableDataFloat().get(output);

        System.out.println("Input:  " + Arrays.toString(input));
        System.out.println("Output: " + Arrays.toString(output));
        if (!Arrays.equals(input, output)) {
            throw new AssertionError("The OpenVINO EP identity model output did not match the input");
        }
        System.out.println("OpenVINO EP identity inference succeeded.");
    }
}
