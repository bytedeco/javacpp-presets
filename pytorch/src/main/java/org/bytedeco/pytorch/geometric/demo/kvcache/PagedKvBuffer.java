package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Tensor;
import java.util.ArrayList;
import java.util.List;

public class PagedKvBuffer implements AutoCloseable {
    private final String sessionId;
    private final PagedBlockManager blockManager;
    private final List<Integer> blockTable; // 逻辑页到物理页的映射
    private int currentTokenCount = 0;

    public PagedKvBuffer(String sessionId, PagedBlockManager blockManager) {
        this.sessionId = sessionId;
        this.blockManager = blockManager;
        this.blockTable = new ArrayList<>();
    }

    /**
     * Prefill 阶段优化写入
     * @param layerIdx 层索引
     * @param kvIdx 0=K, 1=V
     * @param inputTensors 输入的批量张量 [num_input_tokens, head_dim]
     */
    public void prefill(int layerIdx, int kvIdx, Tensor inputTensors) {
        long numNewTokens = inputTensors.size(0);
        int blockSize = blockManager.getBlockSize();

        for (int i = 0; i < numNewTokens; i++) {
            // 逻辑与单个 append 类似，但在循环内复用逻辑
            if ((currentTokenCount + i) % blockSize == 0) {
                // 只有在 layer 0, kv 0 时触发物理分配（保证所有层同步）
                if (layerIdx == 0 && kvIdx == 0) {
                    blockTable.add(blockManager.allocateBlock());
                }
            }

            int logicalIdx = (currentTokenCount + i) / blockSize;
            int offset = (currentTokenCount + i) % blockSize;
            int physicalId = blockTable.get(logicalIdx);

            try (Tensor block = blockManager.getBlock(physicalId);
                 Tensor slot = block.select(0, layerIdx).select(0, kvIdx).select(0, offset);
                 Tensor src = inputTensors.select(0, i)) {
                slot.copy_(src);
            }
        }
        // 注意：在外层循环结束（所有层写完）后再更新 currentTokenCount
    }

    public void advanceTokens(int count) {
        this.currentTokenCount += count;
    }
    /**
     * 写入一个新的 Token
     */
    public void appendToken(int layerIdx, int kvIdx, Tensor tokenData) {
        int blockSize = blockManager.getBlockSize();

        // 检查是否需要新 Block
        if (currentTokenCount % blockSize == 0 && (layerIdx == 0 && kvIdx == 0)) {
            blockTable.add(blockManager.allocateBlock());
        }

        int logicalBlockIdx = currentTokenCount / blockSize;
        int offsetInBlock = currentTokenCount % blockSize;
        int physicalBlockId = blockTable.get(logicalBlockIdx);

        // 获取物理块 View 并写入
        // blockPool 结构: [layer, 2, block_size, head_dim]
        try (Tensor block = blockManager.getBlock(physicalBlockId);
             Tensor layerView = block.select(0, layerIdx);
             Tensor kvView = layerView.select(0, kvIdx);
             Tensor slotView = kvView.select(0, offsetInBlock)) {
            slotView.copy_(tokenData);
        }
    }
    
    

    public void finishToken() {
        currentTokenCount++;
    }

    /**
     * 获取当前 Session 的所有物理块 ID (供算子使用)
     */
    public int[] getBlockIds() {
        return blockTable.stream().mapToInt(i -> i).toArray();
    }

    @Override
    public void close() {
        blockTable.forEach(blockManager::freeBlock);
        blockTable.clear();
    }
}
