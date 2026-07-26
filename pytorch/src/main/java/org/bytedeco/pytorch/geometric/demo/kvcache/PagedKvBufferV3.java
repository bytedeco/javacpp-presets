package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Tensor;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class PagedKvBufferV3 implements AutoCloseable {
    private final String sessionId;
    private final CoWBlockManagerV2 manager;
    private final int numLayers;
    private final List<Integer>[] kBlockMaps;
    private final List<Integer>[] vBlockMaps;

    private final ReentrantReadWriteLock stateLock = new ReentrantReadWriteLock();
    private final AtomicBoolean isInvalidated = new AtomicBoolean(false);

    @SuppressWarnings("unchecked")
    public PagedKvBufferV3(String sessionId, CoWBlockManagerV2 manager, int numLayers) {
        this.sessionId = sessionId;
        this.manager = manager;
        this.numLayers = numLayers;
        this.kBlockMaps = new ArrayList[numLayers];
        this.vBlockMaps = new ArrayList[numLayers];
        for (int i = 0; i < numLayers; i++) {
            // 修复点：显式创建可变列表
            kBlockMaps[i] = new ArrayList<>();
            vBlockMaps[i] = new ArrayList<>();
        }
    }

    public void prefillUltra(int layer, int kvType, Tensor input) {
        stateLock.readLock().lock();
        try {
            if (isInvalidated.get()) return;

            int numTokens = (int) input.size(0);
            int blockSize = manager.getBlockSize();
            int neededBlocks = (numTokens + blockSize - 1) / blockSize;

            // 申请块
            List<Integer> newBlocks = manager.allocateBlocks(neededBlocks, sessionId, this);
            if (kvType == 0) kBlockMaps[layer].addAll(newBlocks);
            else vBlockMaps[layer].addAll(newBlocks);

            // 模拟写入 MPS...
        } finally {
            stateLock.readLock().unlock();
        }
    }

    public List<Integer> getAndInvalidateBlocks() {
        stateLock.writeLock().lock();
        try {
            isInvalidated.set(true);
            List<Integer> allBlocks = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                allBlocks.addAll(kBlockMaps[i]);
                allBlocks.addAll(vBlockMaps[i]);
                kBlockMaps[i].clear();
                vBlockMaps[i].clear();
            }
            return allBlocks;
        } finally {
            stateLock.writeLock().unlock();
        }
    }

    /**
     * Radix Tree 版 Prefill
     * @param hashes 该层输入按 Block 分块后的哈希列表
     */
//    public void prefillWithRadix(int layer, int kvType, List<Long> hashes, Tensor input) {
//        stateLock.readLock().lock();
//        try {
//            if (isInvalidated.get()) return;
//
//            // 调用 V6 的路径匹配方法
//            List<Integer> blockIds = ((CoWBlockManagerV6)manager).matchAndAllocatePath(hashes, sessionId, this);
//
//            if (kvType == 0) kBlockMaps[layer].addAll(blockIds);
//            else vBlockMaps[layer].addAll(blockIds);
//
//            // TODO: 底层 MPS 拷贝逻辑
//        } finally {
//            stateLock.readLock().unlock();
//        }
//    }
    @Override public void close() { manager.releaseSession(sessionId); }

    public CharSequence getSessionId() {
        return sessionId;
    }

    public int getKBlockCount(int layer) {
        return kBlockMaps[layer].size();
    }

    public int getVBlockCount(int layer) {
        return vBlockMaps[layer].size();
    }
}
//public class PagedKvBufferV3 implements AutoCloseable {
//    private final String sessionId;
//    private final CoWBlockManagerV2 manager;
//    private final int numLayers;
//
//    // 核心存储：Layer -> KV(0为K, 1为V) -> List of Physical Block IDs
//    private final List<List<Integer>>[] layerBlockMaps;
//
//    // 状态控制：防止在驱逐时发生并发读写
//    private final ReentrantReadWriteLock stateLock = new ReentrantReadWriteLock();
//    private final AtomicBoolean isInvalidated = new AtomicBoolean(false);
//
//    private int currentPosition = 0;
//
//    @SuppressWarnings("unchecked")
//    public PagedKvBufferV3(String sessionId, CoWBlockManagerV2 manager, int numLayers) {
//        this.sessionId = sessionId;
//        this.manager = manager;
//        this.numLayers = numLayers;
//        this.layerBlockMaps = new ArrayList[numLayers];
//        for (int i = 0; i < numLayers; i++) {
//            layerBlockMaps[i] = Arrays.asList(new ArrayList<>(), new ArrayList<>());
//        }
//    }
//
//    /**
//     * 执行 Prefill：如果块不足，内部会触发 manager 的驱逐逻辑
//     */
//    public void prefillUltra(int layer, int kvType, Tensor input) {
//        stateLock.readLock().lock();
//        try {
//            if (isInvalidated.get()) throw new RuntimeException("Session " + sessionId + " has been evicted.");
//
//            int numTokens = (int) input.size(0);
//            int blockSize = manager.getBlockSize();
//            int neededBlocks = (numTokens + blockSize - 1) / blockSize;
//
//            // 动态按需向 Manager 申请块（Manager 内部可能会因为空间不足触发 LRU 驱逐其他 Session）
//            List<Integer> newBlocks = manager.allocateBlocks(neededBlocks, sessionId, this);
//            layerBlockMaps[layer].get(kvType).addAll(newBlocks);
//
//            // 此处应调用你的底层 LibTorch 拷贝算子
//            // copy_to_paged_cache(input, layerBlockMaps[layer].get(kvType), ...);
//
//        } finally {
//            stateLock.readLock().unlock();
//        }
//    }
//
//    /**
//     * 【核心方法】由 Manager 的驱逐线程调用
//     * 封锁当前 Buffer 并提取所有占有的块 ID 以便回收
//     */
//    public List<Integer> getAndInvalidateBlocks() {
//        stateLock.writeLock().lock();
//        try {
//            isInvalidated.set(true);
//            List<Integer> allBlocks = new ArrayList<>();
//            for (int i = 0; i < numLayers; i++) {
//                allBlocks.addAll(layerBlockMaps[i].get(0));
//                allBlocks.addAll(layerBlockMaps[i].get(1));
//                layerBlockMaps[i].get(0).clear();
//                layerBlockMaps[i].get(1).clear();
//            }
//            return allBlocks;
//        } finally {
//            stateLock.writeLock().unlock();
//        }
//    }
//
//    public void advance(int count) {
//        this.currentPosition += count;
//    }
//
//    @Override
//    public void close() {
//        // 主动关闭时，也要从 Manager 的活跃 Session 列表中移除
//        manager.releaseSession(sessionId);
//    }
//}