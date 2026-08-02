package org.bytedeco.pytorch.serving.tritonserver;

/**
 * Facade entry points for the high-level Triton Java API.
 *
 * <p>Mirrors Python {@code import tritonserver} plus common patterns.
 * Uses bytedeco {@code org.bytedeco.tritonserver} 2.70.0-1.5.14-SNAPSHOT.
 *
 * <pre>{@code
 * Server server = Server.builder()
 *     .modelRepository("/models")
 *     .build();
 * server.start();
 *
 * Model model = server.model("simple");
 * InferenceResponse response = model.infer(...);
 * server.stop();
 * }</pre>
 */
public final class TritonServer {
    private TritonServer() {}

    /** Create a Server with default Options. */
    public static TServer create() {
        return new TServer(TritonOption.builder().build());
    }

    /** Create a Server with custom Options. */
    public static TServer create(TritonOption tritonOptions) {
        return new TServer(tritonOptions);
    }

    /** Create Server with model repository path (fluent style). */
    public static TServer withModelRepository(String repo) {
        return create(TritonOption.builder().modelRepository(repo).build());
    }

    /** Load native libraries if not already (called automatically). */
    public static void loadNative() {
        // bytedeco loads automatically via static init of classes
    }
}