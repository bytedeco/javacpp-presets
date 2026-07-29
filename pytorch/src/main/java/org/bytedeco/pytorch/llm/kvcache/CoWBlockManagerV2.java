package org.bytedeco.pytorch.llm.kvcache;

/**
 * Compatibility alias for older demos that referenced {@code CoWBlockManagerV2}.
 * Delegates to {@link CoWBlockManager} (numHeads=1).
 */
public class CoWBlockManagerV2 extends CoWBlockManager {
    public CoWBlockManagerV2(int totalBlocks, int numLayers, int blockSize, int headDim, int dtypeValue) {
        super(totalBlocks, numLayers, blockSize, 1, headDim, null, resolve(dtypeValue));
    }

    public CoWBlockManagerV2(int totalBlocks, int numLayers, int blockSize, int numHeads, int headDim,
                  org.bytedeco.pytorch.Device device, org.bytedeco.pytorch.global.torch.ScalarType dtype) {
        super(totalBlocks, numLayers, blockSize, numHeads, headDim, device, dtype);
    }

    private static org.bytedeco.pytorch.global.torch.ScalarType resolve(int value) {
        for (org.bytedeco.pytorch.global.torch.ScalarType e :
                org.bytedeco.pytorch.global.torch.ScalarType.values()) {
            if (e.value == value) return e;
        }
        return org.bytedeco.pytorch.global.torch.kFloat();
    }
}
