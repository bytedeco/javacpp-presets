package org.bytedeco.pytorch.geometric.demo.kvcache;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class CoWBlockManagerV5 extends CoWBlockManagerV2 {
    // 正向索引：Content Hash -> 物理块 ID
    private final ConcurrentHashMap<Long, Integer> prefixCache = new ConcurrentHashMap<>();
    // 反向索引：物理块 ID -> Content Hash (解决卡死的关键：实现 O(1) 清理)
    private final ConcurrentHashMap<Integer, Long> blockToHash = new ConcurrentHashMap<>();

    private final AtomicInteger[] refCounts;
    // 独占块列表（按分配顺序排列，作为驱逐的第一梯队）
    private final LinkedHashSet<Integer> reclaimableUniqueBlocks = new LinkedHashSet<>();

    // 锁分级：cacheLock 保护哈希映射，父类 globalLock 保护物理池
    private final ReentrantLock cacheLock = new ReentrantLock();

    public CoWBlockManagerV5(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
        super(totalBlocks, layers, blockSize, headDim, dtype);
        this.refCounts = new AtomicInteger[totalBlocks];
        for (int i = 0; i < totalBlocks; i++) {
            refCounts[i] = new AtomicInteger(0);
        }
    }

    /**
     * 带缓存命中的块分配
     */
    public int getOrAllocateBlock(long contentHash, String sessionId, PagedKvBufferV3 buffer) {
        cacheLock.lock();
        try {
            // 1. 检查缓存命中
            Integer cachedId = prefixCache.get(contentHash);
            if (cachedId != null) {
                refCounts[cachedId].incrementAndGet();
                synchronized (reclaimableUniqueBlocks) {
                    reclaimableUniqueBlocks.remove(cachedId);
                }
                return cachedId;
            }

            // 2. 缓存未命中，获取新块
            int blockId = fetchAvailableBlock(sessionId, buffer);

            // 3. 建立双向映射
            prefixCache.put(contentHash, blockId);
            blockToHash.put(blockId, contentHash);
            refCounts[blockId].set(1);

            synchronized (reclaimableUniqueBlocks) {
                reclaimableUniqueBlocks.add(blockId);
            }
            return blockId;
        } finally {
            cacheLock.unlock();
        }
    }

    int fetchAvailableBlock(String sessionId, PagedKvBufferV3 buffer) {
        // 第一阶段：从自由池获取 (O(1))
        Integer id = freePool.poll();
        if (id != null) return id;

        // 第二阶段：从独占缓存块中抢占 (O(1))
        synchronized (reclaimableUniqueBlocks) {
            Iterator<Integer> it = reclaimableUniqueBlocks.iterator();
            if (it.hasNext()) {
                int victimId = it.next();
                it.remove();

                // 彻底抹除旧元数据：由于有反向索引，不再需要遍历 removeIf
                Long oldHash = blockToHash.remove(victimId);
                if (oldHash != null) prefixCache.remove(oldHash);

                refCounts[victimId].set(0);
                return victimId;
            }
        }

        // 第三阶段：强制驱逐整个 Session (V2 逻辑)
        if (evictOldestSession(sessionId)) {
            Integer retryId = freePool.poll();
            if (retryId != null) return retryId;
        }

        throw new RuntimeException("Memory exhausted: failed to allocate or evict blocks.");
    }

    @Override
    public void releaseSession(String sessionId) {
        // 释放逻辑也需要同步更新引用计数
        super.releaseSession(sessionId);
    }
}
