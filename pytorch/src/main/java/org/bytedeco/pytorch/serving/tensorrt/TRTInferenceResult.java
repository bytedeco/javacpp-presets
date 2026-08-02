package org.bytedeco.pytorch.serving.tensorrt;

import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtNotFoundException;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Result of a single {@link TRTEngine#infer} call.
 *
 * <p>Python samples typically return a dict of output name → NumPy array;
 * this type is the Java equivalent.
 */
public final class TRTInferenceResult {
    private final Map<String, TrtTensor> outputs;

    public TRTInferenceResult(Map<String, TrtTensor> outputs) {
        Objects.requireNonNull(outputs, "outputs");
        this.outputs = Collections.unmodifiableMap(new LinkedHashMap<>(outputs));
    }

    public Map<String, TrtTensor> outputs() {
        return outputs;
    }

    public TrtTensor get(String name) {
        TrtTensor t = outputs.get(name);
        if (t == null) {
            throw new TrtNotFoundException("output tensor not found: " + name);
        }
        return t;
    }

    @Override
    public String toString() {
        return "InferenceResult{outputs=" + outputs.keySet() + '}';
    }
}
