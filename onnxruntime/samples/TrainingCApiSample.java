import org.bytedeco.javacpp.*;
import org.bytedeco.onnxruntime.*;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.onnxruntime.global.onnxruntime.*;

public class TrainingCApiSample {
    private static void checkStatus(OrtApi ortApi, OrtStatus ortStatus) {
        if (ortStatus != null && !ortStatus.isNull()) {
            BytePointer errorMessagePointer = ortApi.GetErrorMessage().call(ortStatus);
            String errorMessage = (errorMessagePointer != null) ? errorMessagePointer.getString() : "Unknown ORT error";
            throw new RuntimeException(errorMessage);
        }
    }

    private static final class TensorMetadata {
        final int elementType;
        final long[] shapeDimensions;

        TensorMetadata(int elementType, long[] shapeDimensions) {
            this.elementType = elementType;
            this.shapeDimensions = shapeDimensions;
        }
    }

    private static String readTrainingModelInputName(OrtApi ortApi,
                                                     OrtTrainingApi ortTrainingApi,
                                                     OrtTrainingSession ortTrainingSession,
                                                     OrtAllocator ortAllocator,
                                                     long trainingInputIndex) {
        BytePointer inputNamePointer = new BytePointer();
        checkStatus(ortApi, ortTrainingApi.TrainingSessionGetTrainingModelInputName(
                ortTrainingSession, trainingInputIndex, ortAllocator, inputNamePointer
        ));
        return inputNamePointer.getString();
    }

    private static String readTrainingModelOutputName(OrtApi ortApi,
                                                      OrtTrainingApi ortTrainingApi,
                                                      OrtTrainingSession ortTrainingSession,
                                                      OrtAllocator ortAllocator,
                                                      long trainingOutputIndex) {
        BytePointer outputNamePointer = new BytePointer();
        checkStatus(ortApi, ortTrainingApi.TrainingSessionGetTrainingModelOutputName(
                ortTrainingSession, trainingOutputIndex, ortAllocator, outputNamePointer
        ));
        return outputNamePointer.getString();
    }

    private static TensorMetadata readInferenceSessionInputTensorMetadata(OrtApi ortApi,
                                                                          OrtSession ortSessionForIntrospection,
                                                                          long inputIndex) {
        OrtTypeInfo ortTypeInfo = new OrtTypeInfo();
        checkStatus(ortApi, ortApi.SessionGetInputTypeInfo(ortSessionForIntrospection, inputIndex, ortTypeInfo));

        IntPointer onnxTypePointer = new IntPointer(1);
        checkStatus(ortApi, ortApi.GetOnnxTypeFromTypeInfo(ortTypeInfo, onnxTypePointer));
        int onnxType = onnxTypePointer.get(0);
        if (onnxType != ONNX_TYPE_TENSOR) {
            ortApi.ReleaseTypeInfo(ortTypeInfo);
            throw new IllegalStateException("Model input at index " + inputIndex + " is not a tensor. ONNX type = " + onnxType);
        }

        OrtTensorTypeAndShapeInfo ortTensorTypeAndShapeInfo = new OrtTensorTypeAndShapeInfo();
        checkStatus(ortApi, ortApi.CastTypeInfoToTensorInfo(ortTypeInfo, ortTensorTypeAndShapeInfo));

        IntPointer tensorElementTypePointer = new IntPointer(1);
        checkStatus(ortApi, ortApi.GetTensorElementType(ortTensorTypeAndShapeInfo, tensorElementTypePointer));
        int elementType = tensorElementTypePointer.get(0);

        SizeTPointer dimensionsCountPointer = new SizeTPointer(1);
        checkStatus(ortApi, ortApi.GetDimensionsCount(ortTensorTypeAndShapeInfo, dimensionsCountPointer));
        long dimensionsCount = dimensionsCountPointer.get(0);

        if (dimensionsCount < 0 || dimensionsCount > 1024) {
            ortApi.ReleaseTypeInfo(ortTypeInfo);
            throw new IllegalStateException("Unreasonable dimensionsCount=" + dimensionsCount + " for inputIndex=" + inputIndex);
        }

        LongPointer dimensionsPointer = new LongPointer(dimensionsCount);
        checkStatus(ortApi, ortApi.GetDimensions(ortTensorTypeAndShapeInfo, dimensionsPointer, dimensionsCount));

        long[] shapeDimensions = new long[(int) dimensionsCount];
        for (int dimensionIndex = 0; dimensionIndex < shapeDimensions.length; dimensionIndex++) {
            shapeDimensions[dimensionIndex] = dimensionsPointer.get(dimensionIndex);
        }

        ortApi.ReleaseTypeInfo(ortTypeInfo);
        return new TensorMetadata(elementType, shapeDimensions);
    }

    private static long byteSizeForElementType(int onnxTensorElementType) {
        if (onnxTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return 4L;
        }
        if (onnxTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return 8L;
        }
        if (onnxTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            return 4L;
        }
        throw new IllegalStateException("Unsupported tensor element type: " + onnxTensorElementType);
    }

    private static long[] resolveShapeWithBatchSize(long[] shapeDimensions, long resolvedBatchSize) {
        long[] resolvedShapeDimensions = new long[shapeDimensions.length];
        for (int dimensionIndex = 0; dimensionIndex < shapeDimensions.length; dimensionIndex++) {
            long dimensionValue = shapeDimensions[dimensionIndex];
            if (dimensionIndex == 0 && dimensionValue <= 0) {
                resolvedShapeDimensions[dimensionIndex] = resolvedBatchSize;
            } else if (dimensionValue <= 0) {
                throw new IllegalStateException("Cannot auto-resolve dynamic dimension at index " + dimensionIndex + " (value=" + dimensionValue + ").");
            } else {
                resolvedShapeDimensions[dimensionIndex] = dimensionValue;
            }
        }
        return resolvedShapeDimensions;
    }

    private static long computeElementCount(long[] resolvedShapeDimensions) {
        long elementCount = 1;
        for (long resolvedShapeDimension : resolvedShapeDimensions) {
            elementCount = Math.multiplyExact(elementCount, resolvedShapeDimension);
        }
        return elementCount;
    }

    private static Pointer allocateAndFillBufferForElementType(int onnxTensorElementType, long elementCount) {
        if (onnxTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            FloatPointer floatPointer = new FloatPointer(elementCount);
            for (long elementIndex = 0; elementIndex < elementCount; elementIndex++) {
                floatPointer.put(elementIndex, (float) Math.random());
            }
            return floatPointer;
        }

        if (onnxTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            LongPointer longPointer = new LongPointer(elementCount);
            for (long elementIndex = 0; elementIndex < elementCount; elementIndex++) {
                longPointer.put(elementIndex, elementIndex % 10L);
            }
            return longPointer;
        }

        if (onnxTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            IntPointer intPointer = new IntPointer(elementCount);
            for (long elementIndex = 0; elementIndex < elementCount; elementIndex++) {
                intPointer.put(elementIndex, (int) (elementIndex % 10L));
            }
            return intPointer;
        }

        throw new IllegalStateException("Unsupported tensor element type: " + onnxTensorElementType);
    }

    private static OrtValue createTensorWithResolvedShape(OrtApi ortApi,
                                                          OrtMemoryInfo ortMemoryInfo,
                                                          int onnxTensorElementType,
                                                          long[] resolvedShapeDimensions,
                                                          Pointer typedDataBufferPointer,
                                                          long elementCount) {
        LongPointer shapePointer = new LongPointer(resolvedShapeDimensions.length);
        for (int dimensionIndex = 0; dimensionIndex < resolvedShapeDimensions.length; dimensionIndex++) {
            shapePointer.put(dimensionIndex, resolvedShapeDimensions[dimensionIndex]);
        }

        long totalByteSize = Math.multiplyExact(elementCount, byteSizeForElementType(onnxTensorElementType));

        OrtValue ortValue = new OrtValue();
        checkStatus(ortApi, ortApi.CreateTensorWithDataAsOrtValue(
                ortMemoryInfo,
                typedDataBufferPointer,
                totalByteSize,
                shapePointer,
                resolvedShapeDimensions.length,
                onnxTensorElementType,
                ortValue
        ));
        return ortValue;
    }

    private static float readSingleFloatFromTensor(OrtApi ortApi, OrtValue ortValue) {
        OrtTensorTypeAndShapeInfo ortTensorTypeAndShapeInfo = new OrtTensorTypeAndShapeInfo();
        checkStatus(ortApi, ortApi.GetTensorTypeAndShape(ortValue, ortTensorTypeAndShapeInfo));

        IntPointer tensorElementTypePointer = new IntPointer(1);
        checkStatus(ortApi, ortApi.GetTensorElementType(ortTensorTypeAndShapeInfo, tensorElementTypePointer));
        int tensorElementType = tensorElementTypePointer.get(0);
        if (tensorElementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw new IllegalStateException("Loss output is not float. elementType=" + tensorElementType);
        }

        PointerPointer<Pointer> tensorDataPointerPointer = new PointerPointer<>(1);
        checkStatus(ortApi, ortApi.GetTensorMutableData(ortValue, tensorDataPointerPointer));

        Pointer tensorDataPointer = tensorDataPointerPointer.get(Pointer.class, 0);
        FloatPointer floatPointer = new FloatPointer(tensorDataPointer);
        return floatPointer.get(0);
    }

    private static void releaseOutputOrtValuesAndNullSlots(OrtApi ortApi,
                                                           PointerPointer<OrtValue> trainingOutputOrtValuePointerArray,
                                                           int trainingModelOutputCount) {
        for (int outputIndex = 0; outputIndex < trainingModelOutputCount; outputIndex++) {
            Pointer outputOrtValuePointer = trainingOutputOrtValuePointerArray.get(outputIndex);
            if (outputOrtValuePointer != null && !outputOrtValuePointer.isNull()) {
                OrtValue outputOrtValue = new OrtValue(outputOrtValuePointer);
                ortApi.ReleaseValue(outputOrtValue);
                trainingOutputOrtValuePointerArray.put(outputIndex, null);
            }
        }
    }

    public static void main(String[] args) {
        OrtLoggingFunction ortLoggingFunction = new OrtLoggingFunction() {
            @Override
            public void call(Pointer parameter, int severity, BytePointer category, BytePointer logIdentifier, BytePointer codeLocation, BytePointer message) {
                System.out.println(message.getString());
            }
        };

        // You can download prebuilt artifacts for training from the URL above
        // https://github.com/bytedeco/binaries/releases/download/1.5.13/onnxruntime_training_artifacts.zip
        String checkpointPath = "train_models/checkpoint";
        String trainingModelPath = "train_models/training_model.onnx";
        String evaluationModelPath = "train_models/eval_model.onnx";
        String optimizerModelPath = "train_models/optimizer_model.onnx";

        CharPointer checkpointPathPointer = new CharPointer(checkpointPath);
        CharPointer trainingModelPathPointer = new CharPointer(trainingModelPath);
        CharPointer evaluationModelPathPointer = new CharPointer(evaluationModelPath);
        CharPointer optimizerModelPathPointer = new CharPointer(optimizerModelPath);

        //noinspection resource
        OrtApi ortApi = OrtGetApiBase().GetApi().call(ORT_API_VERSION);
        OrtTrainingApi ortTrainingApi = ortApi.GetTrainingApi().call(ORT_API_VERSION);

        OrtEnv ortEnvironment = new OrtEnv();
        checkStatus(ortApi, ortApi.CreateEnvWithCustomLogger(ortLoggingFunction, null, ORT_LOGGING_LEVEL_VERBOSE, "LOG", ortEnvironment));

        OrtAllocator ortAllocator = new OrtAllocator();
        checkStatus(ortApi, ortApi.GetAllocatorWithDefaultOptions(ortAllocator));

        OrtSessionOptions ortSessionOptions = new OrtSessionOptions();
        checkStatus(ortApi, ortApi.CreateSessionOptions(ortSessionOptions));

        OrtCheckpointState ortCheckpointState = new OrtCheckpointState();
        checkStatus(ortApi, ortTrainingApi.LoadCheckpoint(checkpointPathPointer, ortCheckpointState));

        OrtTrainingSession ortTrainingSession = new OrtTrainingSession();
        checkStatus(ortApi, ortTrainingApi.CreateTrainingSession(
                ortEnvironment,
                ortSessionOptions,
                ortCheckpointState,
                trainingModelPathPointer,
                evaluationModelPathPointer,
                optimizerModelPathPointer,
                ortTrainingSession
        ));

        OrtSession ortTrainingModelIntrospectionSession = new OrtSession();
        checkStatus(ortApi, ortApi.CreateSession(ortEnvironment, trainingModelPathPointer, ortSessionOptions, ortTrainingModelIntrospectionSession));

        OrtRunOptions ortRunOptions = new OrtRunOptions();
        checkStatus(ortApi, ortApi.CreateRunOptions(ortRunOptions));

        OrtMemoryInfo ortCpuMemoryInfo = new OrtMemoryInfo();
        checkStatus(ortApi, ortApi.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, ortCpuMemoryInfo));

        SizeTPointer trainingModelInputCountPointer = new SizeTPointer(1);
        checkStatus(ortApi, ortTrainingApi.TrainingSessionGetTrainingModelInputCount(ortTrainingSession, trainingModelInputCountPointer));
        long trainingModelInputCount = trainingModelInputCountPointer.get(0);

        SizeTPointer trainingModelOutputCountPointer = new SizeTPointer(1);
        checkStatus(ortApi, ortTrainingApi.TrainingSessionGetTrainingModelOutputCount(ortTrainingSession, trainingModelOutputCountPointer));
        long trainingModelOutputCount = trainingModelOutputCountPointer.get(0);

        System.out.println("Training model input count = " + trainingModelInputCount);
        System.out.println("Training model output count = " + trainingModelOutputCount);

        List<Integer> trainingModelInputElementTypes = new ArrayList<>();
        List<long[]> trainingModelInputShapeDimensions = new ArrayList<>();

        for (long trainingInputIndex = 0; trainingInputIndex < trainingModelInputCount; trainingInputIndex++) {
            String trainingModelInputName = readTrainingModelInputName(ortApi, ortTrainingApi, ortTrainingSession, ortAllocator, trainingInputIndex);
            TensorMetadata tensorMetadata = readInferenceSessionInputTensorMetadata(ortApi, ortTrainingModelIntrospectionSession, trainingInputIndex);

            trainingModelInputElementTypes.add(tensorMetadata.elementType);
            trainingModelInputShapeDimensions.add(tensorMetadata.shapeDimensions);

            System.out.println("Training input[" + trainingInputIndex + "] name = " + trainingModelInputName);
            System.out.println("Training input[" + trainingInputIndex + "] element type = " + tensorMetadata.elementType);
        }

        for (long trainingOutputIndex = 0; trainingOutputIndex < trainingModelOutputCount; trainingOutputIndex++) {
            String trainingModelOutputName = readTrainingModelOutputName(ortApi, ortTrainingApi, ortTrainingSession, ortAllocator, trainingOutputIndex);
            System.out.println("Training output[" + trainingOutputIndex + "] name = " + trainingModelOutputName);
        }

        long resolvedBatchSize = 64;

        List<OrtValue> pinnedInputOrtValuesForRelease = new ArrayList<>();

        PointerPointer<OrtValue> trainingInputOrtValuePointerArray = new PointerPointer<>(trainingModelInputCount);

        for (int trainingInputIndex = 0; trainingInputIndex < (int) trainingModelInputCount; trainingInputIndex++) {
            int elementType = trainingModelInputElementTypes.get(trainingInputIndex);
            long[] originalShapeDimensions = trainingModelInputShapeDimensions.get(trainingInputIndex);

            long[] resolvedShapeDimensions = resolveShapeWithBatchSize(originalShapeDimensions, resolvedBatchSize);
            long elementCount = computeElementCount(resolvedShapeDimensions);

            Pointer typedDataBufferPointer = allocateAndFillBufferForElementType(elementType, elementCount);

            OrtValue inputOrtValue = createTensorWithResolvedShape(
                    ortApi,
                    ortCpuMemoryInfo,
                    elementType,
                    resolvedShapeDimensions,
                    typedDataBufferPointer,
                    elementCount
            );

            pinnedInputOrtValuesForRelease.add(inputOrtValue);
            trainingInputOrtValuePointerArray.put(trainingInputIndex, inputOrtValue);
        }

        PointerPointer<OrtValue> trainingOutputOrtValuePointerArray = new PointerPointer<>(trainingModelOutputCount);
        int trainingModelOutputCountInteger = (int) trainingModelOutputCount;

        for (int iterationIndex = 0; iterationIndex < 100; iterationIndex++) {
            for (int outputIndex = 0; outputIndex < trainingModelOutputCountInteger; outputIndex++) {
                trainingOutputOrtValuePointerArray.put(outputIndex, (Pointer) null);
            }

            checkStatus(ortApi, ortTrainingApi.TrainStep(
                    ortTrainingSession,
                    ortRunOptions,
                    trainingModelInputCount,
                    trainingInputOrtValuePointerArray,
                    trainingModelOutputCount,
                    trainingOutputOrtValuePointerArray
            ));

            Pointer lossOrtValuePointer = trainingOutputOrtValuePointerArray.get(0);
            if (lossOrtValuePointer == null || lossOrtValuePointer.isNull()) {
                releaseOutputOrtValuesAndNullSlots(ortApi, trainingOutputOrtValuePointerArray, trainingModelOutputCountInteger);
                throw new IllegalStateException("TrainStep output[0] is null. Cannot read loss.");
            }

            float lossValue = readSingleFloatFromTensor(ortApi, new OrtValue(lossOrtValuePointer));
            System.out.println("iteration=" + iterationIndex + " loss=" + lossValue);

            checkStatus(ortApi, ortTrainingApi.OptimizerStep(ortTrainingSession, ortRunOptions));
            checkStatus(ortApi, ortTrainingApi.LazyResetGrad(ortTrainingSession));

            releaseOutputOrtValuesAndNullSlots(ortApi, trainingOutputOrtValuePointerArray, trainingModelOutputCountInteger);
        }

        for (int outputIndex = 0; outputIndex < (int) trainingModelOutputCount; outputIndex++) {
            Pointer outputOrtValuePointer = trainingOutputOrtValuePointerArray.get(outputIndex);
            if (outputOrtValuePointer != null && !outputOrtValuePointer.isNull()) {
                OrtValue outputOrtValue = new OrtValue(outputOrtValuePointer);
                ortApi.ReleaseValue(outputOrtValue);
                trainingOutputOrtValuePointerArray.put(outputIndex, null);
            }
        }

        for (OrtValue ortValue : pinnedInputOrtValuesForRelease) {
            ortApi.ReleaseValue(ortValue);
        }

        ortApi.ReleaseRunOptions(ortRunOptions);
        ortApi.ReleaseSession(ortTrainingModelIntrospectionSession);
        ortTrainingApi.ReleaseTrainingSession(ortTrainingSession);
        ortTrainingApi.ReleaseCheckpointState(ortCheckpointState);
        ortApi.ReleaseSessionOptions(ortSessionOptions);
        ortApi.ReleaseEnv(ortEnvironment);
    }
}