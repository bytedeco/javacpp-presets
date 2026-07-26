package org.bytedeco.pytorch.geometric.demo.kvcache;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class CoWBlockManagerV3 extends CoWBlockManagerV2 {
    // 核心：内容哈希 -> 物理块 ID 的映射
    private final ConcurrentHashMap<Long, Integer> contentCache = new ConcurrentHashMap<>();
    // 引用计数：防止共享块被误驱逐
    private final AtomicInteger[] refCounts;

    public CoWBlockManagerV3(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
        super(totalBlocks, layers, blockSize, headDim, dtype);
        this.refCounts = new AtomicInteger[totalBlocks];
        for (int i = 0; i < totalBlocks; i++) {
            refCounts[i] = new AtomicInteger(0);
        }
    }

    /**
     * 尝试命中缓存的分配方法
     */
    public int getOrAllocateBlock(long contentHash) {
        // 1. 尝试从缓存中获取
        Integer cachedBlockId = contentCache.get(contentHash);
        if (cachedBlockId != null) {
            refCounts[cachedBlockId].incrementAndGet();
            return cachedBlockId; // 缓存命中！TPS 翻倍的关键
        }

        // 2. 缓存未命中，执行标准分配（可能触发 LRU）
        List<Integer> allocated = this.allocateBlocks(1, "internal", null);
        int newBlockId = allocated.get(0);

        // 3. 建立映射
        contentCache.put(contentHash, newBlockId);
        refCounts[newBlockId].set(1);
        return newBlockId;
    }

    /**
     * 重写释放逻辑：支持引用计数
     */
    public void releaseBlock(int blockId, long contentHash) {
        if (refCounts[blockId].decrementAndGet() == 0) {
            // 只有没人用了，才真正从缓存中移除并放回池子
            contentCache.remove(contentHash);
            returnBlockToPool(blockId);
        }
    }
    protected void returnBlockToPool(int blockId) {
        refCounts[blockId].set(0);
        freePool.add(blockId);
    }
}
