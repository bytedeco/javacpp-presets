package org.bytedeco.pytorch.distributed;

import org.bytedeco.pytorch.Device;

/**
 * Re-export of {@link org.bytedeco.pytorch.kvcache.PagedKvCache} under the
 * {@code distributed} package (as requested by the porting plan). Prefer the
 * canonical type in {@code org.bytedeco.pytorch.kvcache} for new code so it
 * does not collide with c10d process-group symbols.
 *
 * <p>Features: paged blocks, CoW fork, multi-layer prefix radix reuse,
 * watermark LRU eviction, optional device placement, stats counters.
 */
public class PagedKvCache extends org.bytedeco.pytorch.kvcache.PagedKvCache {

    public PagedKvCache(int numLayers, int numHeads, int headDim, int blockSize, int maxBlocks) {
        super(numLayers, numHeads, headDim, blockSize, maxBlocks);
    }

    public PagedKvCache(int numLayers, int numHeads, int headDim, int blockSize, int maxBlocks,
                        Device device) {
        super(numLayers, numHeads, headDim, blockSize, maxBlocks, device);
    }

    public PagedKvCache(int numLayers, int numHeads, int headDim, int blockSize, int maxBlocks,
                        Device device, double lowWatermark, double highWatermark) {
        super(numLayers, numHeads, headDim, blockSize, maxBlocks, device, lowWatermark, highWatermark);
    }

    /**
     * Convenience: {@code onGpu=true} places blocks on CUDA:0 when available;
     * falls back to CPU if CUDA is not built/visible.
     */
    public PagedKvCache(int numLayers, int numHeads, int headDim, int blockSize, int maxBlocks,
                        boolean onGpu) {
        super(numLayers, numHeads, headDim, blockSize, maxBlocks, onGpu ? tryCuda() : null);
    }

    private static Device tryCuda() {
        try {
            return new Device(org.bytedeco.pytorch.global.torch.DeviceType.CUDA, (byte) 0);
        } catch (Throwable t) {
            return null;
        }
    }
}
