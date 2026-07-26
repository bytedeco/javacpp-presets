package org.bytedeco.pytorch.geometric.demo.kvcache;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class CoWBlockManagerV4 extends CoWBlockManagerV2 {
    // 内容哈希 -> 物理块 ID
    private final ConcurrentHashMap<Long, Integer> prefixCache = new ConcurrentHashMap<>();
    // 物理块 ID -> 引用计数
    private final AtomicInteger[] refCounts;
    // 专门用于存放引用计数为 1 的块，便于快速回收
    private final LinkedHashSet<Integer> reclaimableUniqueBlocks = new LinkedHashSet<>();
    private final ReentrantLock cacheLock = new ReentrantLock();

    public CoWBlockManagerV4(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
        super(totalBlocks, layers, blockSize, headDim, dtype);
        this.refCounts = new AtomicInteger[totalBlocks];
        for (int i = 0; i < totalBlocks; i++) {
            refCounts[i] = new AtomicInteger(0);
        }
    }

    /**
     * 带缓存命中的块获取
     */
    public int getOrAllocateBlock(long contentHash, String sessionId) {
        cacheLock.lock();
        try {
            // 1. 检查缓存命中
            Integer cachedId = prefixCache.get(contentHash);
            if (cachedId != null) {
                refCounts[cachedId].incrementAndGet();
                // 既然有人共享，它就从“待回收独占列表”中移出
                reclaimableUniqueBlocks.remove(cachedId);
                return cachedId;
            }

            // 2. 缓存未命中，申请新块
            int blockId = fetchNextAvailableBlock();

            // 3. 初始化缓存元数据
            prefixCache.put(contentHash, blockId);
            refCounts[blockId].set(1);
            // 新分配的块暂时是独占的，加入待回收列表
            reclaimableUniqueBlocks.add(blockId);

            return blockId;
        } finally {
            cacheLock.unlock();
        }
    }

    /**
     * 策略核心：两阶段回收逻辑
     */
    private int fetchNextAvailableBlock() {
        // 第一阶段：从自由池直接拿
        Integer id = pollFreePool();
        if (id != null) return id;

        // 第二阶段：从独占且最久未使用的块中“抢占”
        synchronized (reclaimableUniqueBlocks) {
            Iterator<Integer> it = reclaimableUniqueBlocks.iterator();
            if (it.hasNext()) {
                int victimId = it.next();
                it.remove();

                // 彻底从缓存映射中清理该块
                invalidateBlockMetadata(victimId);
                return victimId;
            }
        }

        // 第三阶段：如果依然没有，触发强制 Session 级别驱逐 (V2 的逻辑)
        if (evictOldestSession("system")) {
            return pollFreePool();
        }

        throw new RuntimeException("GPU Memory Critical: No unique blocks left to reclaim.");
    }

    /**
     * 从自由池中弹出一个干净的块
     */
    private Integer pollFreePool() {
//        PriorityQueue<Object> freePool;
        Integer id = freePool.poll();
        if (id != null) {
            // 初始化元数据
            refCounts[id].set(0);
            return id;
        }
        return null;
    }
    private void invalidateBlockMetadata(int blockId) {
        // 这里的开销极低，因为只是从 Map 中移除索引
        prefixCache.values().removeIf(v -> v == blockId);
        refCounts[blockId].set(0);
    }

    /**
     * V2 逻辑的底层钩子：将块放回池子
     */
    protected void returnBlockToPool(int blockId) {
        refCounts[blockId].set(0);
        freePool.add(blockId);
    }
    
}