package org.bytedeco.pytorch.serving.tritonserver;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonException;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * One inference response produced for a request.
 *
 * <p>Corresponds to Python {@code tritonserver.InferenceResponse}.
 */
public final class TritonInferenceResponse {
    private final TritonModel tritonModel;
    private final String requestId;
    private final Map<String, Object> parameters;
    private final Map<String, TritonTensor> outputs;
    private final TritonException error;
    private final boolean finalResponse;
    /** Internal queue sentinel: not a real response. */
    private final boolean sentinel;

    public TritonInferenceResponse(
            TritonModel tritonModel,
            String requestId,
            Map<String, Object> parameters,
            Map<String, TritonTensor> outputs,
            TritonException error,
            boolean finalResponse) {
        this(tritonModel, requestId, parameters, outputs, error, finalResponse, false);
    }

    /** Package + internal: full constructor including queue sentinel flag. */
public TritonInferenceResponse(
            TritonModel tritonModel,
            String requestId,
            Map<String, Object> parameters,
            Map<String, TritonTensor> outputs,
            TritonException error,
            boolean finalResponse,
            boolean sentinel) {
        this.tritonModel = tritonModel;
        this.requestId = requestId == null ? "" : requestId;
        this.parameters = parameters == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(parameters));
        this.outputs = outputs == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(outputs));
        this.error = error;
        this.finalResponse = finalResponse;
        this.sentinel = sentinel;
    }

    /** End-of-stream marker for {@link TRTResponseIterator} blocking queues. */
    public static TritonInferenceResponse sentinel() {
        return new TritonInferenceResponse(null, "", Map.of(), Map.of(), null, true, true);
    }

    /** Synthetic error response when the complete callback itself fails. */
    public static TritonInferenceResponse errorOnly(TritonModel tritonModel, TritonException error) {
        return new TritonInferenceResponse(
                tritonModel, "", Map.of(), Map.of(), Objects.requireNonNull(error), true, false);
    }

    public TritonModel model() {
        return tritonModel;
    }

    public String requestId() {
        return requestId;
    }

    public Map<String, Object> parameters() {
        return parameters;
    }

    public Map<String, TritonTensor> outputs() {
        return outputs;
    }

    /** Response-level error, or {@code null} on success. */
    public TritonException error() {
        return error;
    }

    public boolean hasError() {
        return error != null;
    }

    /** True if this is the last response for the request (decoupled-aware). */
    public boolean isFinal() {
        return finalResponse;
    }

    /** True if this is the internal end-of-stream queue sentinel (not a real response). */
    public boolean isSentinel() {
        return sentinel;
    }

    @Override
    public String toString() {
        if (sentinel) {
            return "InferenceResponse{sentinel}";
        }
        return "InferenceResponse{requestId='" + requestId
                + "', outputs=" + outputs.keySet()
                + ", error=" + error
                + ", final=" + finalResponse
                + "}";
    }
}
