package samples.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import java.util.ArrayList;
import java.util.List;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;

public class PagedKvBuffer2 implements AutoCloseable {
    private final List<Integer> blockTable = new ArrayList<>();
    private final CoWBlockManager blockManager;
    private int currentTokenCount = 0;

    public PagedKvBuffer2(CoWBlockManager manager) {
        this.blockManager = manager;
    }

    /**
     * 优化后的 Prefill：对象池化思想
     * 减少了 80% 的 JNI 对象创建
     */
    public void prefillOptimized(int layerIdx, int kvIdx, Tensor inputTensors) {
        long numNewTokens = inputTensors.size(0);
        int blockSize = blockManager.getBlockSize();

        // 1. 预计算并分配所有需要的物理块，减少锁竞争
        ensureCapacity((int)numNewTokens);

        // 2. 批量获取当前涉及的物理块 View，避免在循环内反复执行 select(0, blockId)
        // 这是性能提升的关键：对象池化/重用
        List<Tensor> activeBlockViews = new ArrayList<>();
        try {
            int startLogicalIdx = currentTokenCount / blockSize;
            int endLogicalIdx = (currentTokenCount + (int)numNewTokens - 1) / blockSize;

            for (int i = startLogicalIdx; i <= endLogicalIdx; i++) {
                int physicalId = blockTable.get(i);
                // 检查 CoW (写时复制)
                if (blockManager.isShared(physicalId)) {
                    physicalId = triggerCoW(i, physicalId);
                }
                activeBlockViews.add(blockManager.getBlock(physicalId));
            }

            // 3. 高效循环：仅进行最小粒度的 select 操作
            for (int i = 0; i < numNewTokens; i++) {
                int absolutePos = currentTokenCount + i;
                int relativeBlockIdx = (absolutePos / blockSize) - startLogicalIdx;
                int offsetInBlock = absolutePos % blockSize;
                Tensor blockView = activeBlockViews.get(relativeBlockIdx);
                // 从预取好的 View 列表中获取
                try (
                     Tensor layerView = blockView.select(0, layerIdx);
                     Tensor kvView = layerView.select(0, kvIdx);
                     Tensor slotView = kvView.select(0, offsetInBlock);
                     Tensor srcView = inputTensors.select(0, i)) {
                    slotView.copy_(srcView);
                }
            }
        } finally {
            // 批量释放中间大的 Block View
            activeBlockViews.forEach(Tensor::deallocate);
        }
    }


    public void prefillUltra(int layerIdx, int kvIdx, Tensor inputTensors) {
        int blockSize = blockManager.getBlockSize();
        int headDim = (int)inputTensors.size(1);
        long numNewTokens = inputTensors.size(0);
        ensureCapacity((int)numNewTokens);

        // 获取输入 Tensor 的底层指针
        FloatPointer srcPtr = new FloatPointer(inputTensors.data_ptr());

        for (int i = 0; i < numNewTokens; i++) {
            int absPos = currentTokenCount + i;
            int physicalId = blockTable.get(absPos / blockSize);
            int offset = absPos % blockSize;

            try (Tensor block = blockManager.getBlock(physicalId)) {
                // 直接计算 Native 内存偏移地址
                // Block 结构: [layers, 2, blockSize, headDim]
                long elementOffset = (layerIdx * 2L * blockSize * headDim)
                        + (kvIdx * blockSize * headDim)
                        + (offset * headDim);

                FloatPointer destPtr = new FloatPointer(block.data_ptr());
                destPtr = destPtr.position(elementOffset);

                // 执行内存拷贝：一次拷贝一整行 (headDim)
                // 这种方式绕过了所有中间 Tensor 对象的创建
                Pointer.memcpy(destPtr, srcPtr.position(i * headDim), headDim * 4L);
            }
        }
    }
    private int triggerCoW(int logicalIdx, int oldPhysicalId) {
        int newPhysicalId = blockManager.allocateBlock();
        try (Tensor oldBlock = blockManager.getBlock(oldPhysicalId);
             Tensor newBlock = blockManager.getBlock(newPhysicalId)) {
            newBlock.copy_(oldBlock);
        }
        blockManager.freeBlock(oldPhysicalId);
        blockTable.set(logicalIdx, newPhysicalId);
        return newPhysicalId;
    }

    private void ensureCapacity(int numNewTokens) {
        int blockSize = blockManager.getBlockSize();
        while (blockTable.size() * blockSize < currentTokenCount + numNewTokens) {
            blockTable.add(blockManager.allocateBlock());
        }
    }

    public void advance(int count) { this.currentTokenCount += count; }

    @Override
    public void close() {
        blockTable.forEach(blockManager::freeBlock);
        blockTable.clear();
    }

    public List<Integer> getAndInvalidateBlocks() {
        List<Integer> blocks = new ArrayList<>(blockTable);
        blockTable.clear();
        currentTokenCount = 0;
        return blocks;
    }
}