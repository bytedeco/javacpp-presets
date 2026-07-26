package org.bytedeco.pytorch.geometric.demo.kvcache;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.*;

public class CoWBlockManagerV7 extends CoWBlockManagerV2 {
    private final int actualBlockSize;
    private final Condition diskFullCondition;
    private final ReentrantLock treeLock = new ReentrantLock();
    // 跟踪每个 Session 持有的 Radix 节点
    private final ConcurrentHashMap<String, List<RadixNode>> sessionNodes = new ConcurrentHashMap<>();

    // 内部类：树节点
    static class RadixNode {
        final long hash;
        final int blockId;
        final ConcurrentHashMap<Long, RadixNode> children = new ConcurrentHashMap<>();
        final AtomicInteger refCount = new AtomicInteger(0);
        RadixNode(long hash, int blockId) { this.hash = hash; this.blockId = blockId; }
    }

    private final RadixNode root = new RadixNode(-1, -1);

    public CoWBlockManagerV7(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
        super(totalBlocks, layers, blockSize, headDim, dtype);
        this.actualBlockSize = blockSize;
        this.diskFullCondition = super.getGlobalLock().newCondition();
    }

    public int getOrAllocateBlock(long currentHash, String sid, PagedKvBufferV3 kv) {
        treeLock.lock();
        try {
            RadixNode current = root;
            RadixNode next = current.children.get(currentHash);
            RadixNode targetNode;

            if (next != null) {
                targetNode = next;
            } else {
                int bId = allocateWithRetry(sid, kv);
                RadixNode newNode = new RadixNode(currentHash, bId);
                // FIX: 初始为 0，因为后面会统一 increment
                newNode.refCount.set(0);
                current.children.put(currentHash, newNode);
                targetNode = newNode;
            }

            targetNode.refCount.incrementAndGet();
            sessionNodes.computeIfAbsent(sid, k -> new CopyOnWriteArrayList<>()).add(targetNode);

            return targetNode.blockId;
        } finally {
            treeLock.unlock();
        }
    }

    public List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer) {
        List<Integer> result = new ArrayList<>();
        RadixNode current = root;

        treeLock.lock();
        try {
            for (Long h : pathHashes) {
                RadixNode next = current.children.get(h);
                RadixNode targetNode;

                if (next != null) {
                    targetNode = next;
                } else {
                    int bId = allocateWithRetry(sessionId, buffer);
                    RadixNode newNode = new RadixNode(h, bId);
                    // FIX: 这里必须设为 0，之前设为 1 会导致双重计费 (1 + increment = 2)
                    newNode.refCount.set(0);
                    current.children.put(h, newNode);
                    targetNode = newNode;
                }

                targetNode.refCount.incrementAndGet();
                sessionNodes.computeIfAbsent(sessionId, k -> new CopyOnWriteArrayList<>()).add(targetNode);

                result.add(targetNode.blockId);
                current = targetNode;
            }
        } finally {
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
                    continue;
                }

                // 等待
                if (!diskFullCondition.await(100, TimeUnit.MILLISECONDS)) {
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
            diskFullCondition.signalAll();
        }
        return success;
    }

    @Override
    public void releaseSession(String sessionId) {
        List<RadixNode> nodes = sessionNodes.remove(sessionId);
        if (nodes != null) {
            for (RadixNode node : nodes) {
                node.refCount.decrementAndGet();
            }
        }
        super.releaseSession(sessionId);
    }
}

