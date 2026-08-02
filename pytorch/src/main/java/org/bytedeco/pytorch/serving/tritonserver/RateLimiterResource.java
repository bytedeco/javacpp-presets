package org.bytedeco.pytorch.serving.tritonserver;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
/**
 * Resource count for rate limiting.
 *
 * <p>Python {@code RateLimiterResource} / {@code TRITONSERVER_ServerOptionsAddRateLimiterResource}.
 */
public final class RateLimiterResource {
    private final String name;
    private final int count;
    private final int device;

    public RateLimiterResource(String name, int count, int device) {
        if (name == null || name.isEmpty()) {
            throw new TritonInvalidArgumentException("resource name must be non-empty");
        }
        this.name = name;
        this.count = count;
        this.device = device;
    }

    public String name() {
        return name;
    }

    public int count() {
        return count;
    }

    public int device() {
        return device;
    }
}
