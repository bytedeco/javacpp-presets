package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.*;

public class CoWBlockManagerV6 extends CoWBlockManagerV2 {
    private final int actualBlockSize;
    private final Condition diskFullCondition;
    private final ReentrantLock treeLock = new ReentrantLock();
    // FIX: 跟踪每个 Session 持有的 Radix 节点，以便在 Session 释放时减少引用计数
    private final ConcurrentHashMap<String, List<RadixNode>> sessionNodes = new ConcurrentHashMap<>();

    public int getOrAllocateBlock(long currentHash, String sid, PagedKvBufferV3 kv) {
        treeLock.lock();
        try {
            RadixNode current = root;
            RadixNode next = current.children.get(currentHash);
            RadixNode targetNode;
            if (next != null) {
                targetNode = next;
//                next.refCount.incrementAndGet();
//                return next.blockId;
            } else {
                // 申请新块：如果 freePool 为空，此方法现在会阻塞并尝试驱逐
                int bId = allocateWithRetry(sid, kv);
                RadixNode newNode = new RadixNode(currentHash, bId);
                newNode.refCount.set(0);
                current.children.put(currentHash, newNode);
                targetNode = newNode;
//                return bId;
            }
            // 增加引用计数
            targetNode.refCount.incrementAndGet();

            // FIX: 记录该 Session 使用了这个节点
            sessionNodes.computeIfAbsent(sid, k -> new CopyOnWriteArrayList<>()).add(targetNode);

            return targetNode.blockId;
        } finally {
            treeLock.unlock();
        }
    }

    // Radix Tree 节点结构
    static class RadixNode {
        final long hash;
        final int blockId;
        final ConcurrentHashMap<Long, RadixNode> children = new ConcurrentHashMap<>();
        final AtomicInteger refCount = new AtomicInteger(0);
        RadixNode(long hash, int blockId) { this.hash = hash; this.blockId = blockId; }
    }

    private final RadixNode root = new RadixNode(-1, -1);

    public CoWBlockManagerV6(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
        super(totalBlocks, layers, blockSize, headDim, dtype);
        this.actualBlockSize = blockSize;
        // 使用父类 globalLock 的 Condition 实现阻塞分配
        this.diskFullCondition = super.getGlobalLock().newCondition();
    }

    /**
     * 核心改进：阻塞式路径匹配
     * 如果内存满了，线程会进入等待状态，直到其他 Session 被驱逐释放出块
     */
    public List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer) {
        List<Integer> result = new ArrayList<>();
        RadixNode current = root;

        treeLock.lock(); // 确保路径操作的原子性
        try{
            for (Long h : pathHashes) {
                RadixNode next = current.children.get(h);
                RadixNode targetNode;
                if (next != null) {
                    targetNode = next;
//                next.refCount.incrementAndGet();
//                result.add(next.blockId);
//                current = next;
                } else {
                    // 申请新块：如果 freePool 为空，此方法现在会阻塞并尝试驱逐
                    int bId = allocateWithRetry(sessionId, buffer);
                    RadixNode newNode = new RadixNode(h, bId);
                    newNode.refCount.set(1);
                    current.children.put(h, newNode);
//                result.add(bId);
//                current = newNode;
                    targetNode = newNode;
                }
                targetNode.refCount.incrementAndGet();
                sessionNodes.computeIfAbsent(sessionId, k -> new CopyOnWriteArrayList<>()).add(targetNode);

                result.add(targetNode.blockId);
                current = targetNode;
            }        } finally {
            treeLock.unlock();
        }
   
        return result;
    }

    private int allocateWithRetry(String sessionId, PagedKvBufferV3 buffer) {
        super.getGlobalLock().lock();
        try {
            while (true) {
                Integer id = freePool.poll();
                if (id != null) return id;

                // 尝试驱逐
                if (evictOldestSession(sessionId)) {
                    continue; // 驱逐成功，重新尝试从 freePool 拿
                }

                // 驱逐失败（所有块都在忙），阻塞等待 100ms
                if (!diskFullCondition.await(100, TimeUnit.MILLISECONDS)) {
                    // 超时后仍无进展，才真正报错
                    throw new RuntimeException("GPU Memory Timeout: System saturated.");
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } finally {
            super.getGlobalLock().unlock();
        }
    }

    @Override
    boolean evictOldestSession(String excludeId) {
        boolean success = super.evictOldestSession(excludeId);
        if (success) {
            diskFullCondition.signalAll(); // 唤醒正在等待块的线程
        }
        return success;
    }


    // FIX: 重写释放逻辑，清理 V6 特有的 Radix Tree 引用
    @Override
    public void releaseSession(String sessionId) {
        List<RadixNode> nodes = sessionNodes.remove(sessionId);
        if (nodes != null) {
            for (RadixNode node : nodes) {
                // 扣减引用计数，表示该 Session 不再锁定这些节点
                // 当 refCount 降为 0 时，表示节点处于“可回收”状态（虽然物理 Block 可能还在，直到被 LRU 驱逐）
                node.refCount.decrementAndGet();
            }
        }
        super.releaseSession(sessionId);
    }
}
//public class CoWBlockManagerV6 extends CoWBlockManagerV5 {
//
//    // 内部类：树节点
//    static class RadixNode {
//        final long hash;
//        final int blockId;
//        final ConcurrentHashMap<Long, RadixNode> children = new ConcurrentHashMap<>();
//        final AtomicInteger refCount = new AtomicInteger(1);
//
//        RadixNode(long hash, int blockId) {
//            this.hash = hash;
//            this.blockId = blockId;
//        }
//    }
//
//    private final RadixNode root = new RadixNode(-1, -1); // 虚拟根节点
//    private final ReentrantLock treeLock = new ReentrantLock();
//
//    public CoWBlockManagerV6(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
//        super(totalBlocks, layers, blockSize, headDim, dtype);
//    }
//
//    /**
//     * 核心方法：多级路径匹配与分配
//     * @param blockHashes 整个 Prompt 按块划分后的 Hash 列表
//     */
//    public List<Integer> matchOrAllocatePath(List<Long> blockHashes, String sessionId, PagedKvBufferV3 buffer) {
//        treeLock.lock();
//        try {
//            List<Integer> resultBlockIds = new ArrayList<>();
//            RadixNode currentNode = root;
//
//            for (Long h : blockHashes) {
//                RadixNode nextNode = currentNode.children.get(h);
//
//                if (nextNode != null) {
//                    // 1. 命中缓存级别
//                    nextNode.refCount.incrementAndGet();
//                    resultBlockIds.add(nextNode.blockId);
//                    currentNode = nextNode;
//                } else {
//                    // 2. 失配，需要分配新块并挂载到树上
//                    int newBlockId = fetchAvailableBlock(sessionId, buffer);
//                    RadixNode newNode = new RadixNode(h, newBlockId);
//
//                    currentNode.children.put(h, newNode);
//                    resultBlockIds.add(newBlockId);
//                    currentNode = newNode;
//                }
//            }
//            return resultBlockIds;
//        } finally {
//            treeLock.unlock();
//        }
//    }
//
//    /**
//     * 增强版驱逐：Radix Tree 必须支持从叶子节点开始回收
//     */
//    @Override
//    protected boolean evictOldestSession(String excludeId) {
//        // 在 Radix Tree 中，驱逐不仅是释放块，还要清理树的路径
//        // 优先回收 refCount 为 1 且属于最老 Session 的叶子节点
//        return super.evictOldestSession(excludeId);
//    }
//}