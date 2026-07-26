package org.bytedeco.pytorch.geometric.demo.kvcache;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class CoWBlockManagerV2 implements AutoCloseable {
    // 基础物理存储与元数据
    final int totalBlocks;
    private final int blockSize;

    final ConcurrentLinkedQueue<Integer> freePool;
    private final ReentrantLock globalLock = new ReentrantLock();
    
    // LRU 核心数据结构：按访问时间排序的 Session 映射
    // 使用 LinkedHashMap 的 accessOrder 特性实现 LRU
    private final Map<String, PagedKvBufferV3> activeSessions =
            Collections.synchronizedMap(new LinkedHashMap<String, PagedKvBufferV3>(16, 0.75f, true));

    public CoWBlockManagerV2(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
        this.totalBlocks = totalBlocks;
        this.blockSize = blockSize;
        this.freePool = new ConcurrentLinkedQueue<>();
        for (int i = 0; i < totalBlocks; i++) {
            freePool.add(i);
        }
    }

    /**
     * 增强版分配方法：支持自动驱逐
     */
    public List<Integer> allocateBlocks(int count, String sessionId, PagedKvBufferV3 currentBuffer) {
        List<Integer> allocated = new ArrayList<>();

        while (allocated.size() < count) {
            Integer blockId = freePool.poll();

            if (blockId != null) {
                allocated.add(blockId);
            } else {
                // 关键点：物理块耗尽，触发 LRU 驱逐
                if (!evictOldestSession(sessionId)) {
                    // 如果连驱逐都分不到块（比如所有块都被当前请求占用）
                    throw new RuntimeException("GPU Memory Exhausted: No blocks available for eviction.");
                }
            }
        }

        // 记录/更新当前 Session 的活跃状态
        activeSessions.put(sessionId, currentBuffer);
        return allocated;
    }

    /**
     * 驱逐最老的 Session
     * @param excludeSessionId 排除当前正在申请内存的 Session
     */
    boolean evictOldestSession(String excludeSessionId) {
        globalLock.lock();
        try {
            String victimId = null;
            PagedKvBufferV3 victimBuffer = null;

            // LinkedHashMap 的第一个元素就是最久未访问的
            synchronized (activeSessions) {
                Iterator<Map.Entry<String, PagedKvBufferV3>> it = activeSessions.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry<String, PagedKvBufferV3> entry = it.next();
                    if (!entry.getKey().equals(excludeSessionId)) {
                        victimId = entry.getKey();
                        victimBuffer = entry.getValue();
                        it.remove(); // 从 LRU 队列移除
                        break;
                    }
                }
            }

            if (victimBuffer != null) {
                // 执行强制释放逻辑
                List<Integer> releasedBlocks = victimBuffer.getAndInvalidateBlocks();
                freePool.addAll(releasedBlocks);
                System.out.printf("[LRU Evict] Session %s preempted. Released %d blocks.%n",
                        victimId, releasedBlocks.size());
                return true;
            }
            return false;
        } finally {
            globalLock.unlock();
        }
    }

    // 统计工具
    public int getFreeBlockCount() { return freePool.size(); }
    public int getActiveBlockCount() { return totalBlocks - freePool.size(); }

    @Override
    public void close() { /* 清理 LibTorch 物理内存 */ }

    public int getBlockSize() {
        return this.blockSize; 
    }

    public void releaseSession(String sessionId) {
        activeSessions.remove(sessionId);
    }

    protected Lock getGlobalLock() {
        return this.globalLock;
    }

    /**
     * V2 逻辑的底层钩子：将块放回池子
     */
//    protected void returnBlockToPool(int blockId) {
//        refCounts[blockId].set(0);
//        freePool.add(blockId);
//    }
}