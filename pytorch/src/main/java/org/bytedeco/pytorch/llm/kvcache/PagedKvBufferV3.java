package org.bytedeco.pytorch.llm.kvcache;

/**
 * Compatibility alias for older demos that referenced {@code PagedKvBufferV3}.
 */
public class PagedKvBufferV3 extends PagedKvBuffer {
    public PagedKvBufferV3(String sessionId, CoWBlockManager manager, int numLayers) {
        super(sessionId, manager, numLayers);
    }
}
