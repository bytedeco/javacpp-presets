package org.bytedeco.pytorch.geometric.demo.kvcache;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.*;

public class CoWBlockManagerV8 extends CoWBlockManagerV2 {
    private final int actualBlockSize;
    private final Condition diskFullCondition;
    private final ReentrantLock treeLock = new ReentrantLock();
    public static final LongAdder EVICT_COUNT = new LongAdder();
    public static final LongAdder WAIT_COUNT = new LongAdder();
    private final double lowWatermark = 0.10; // 10% 触发清理
    private final double highWatermark = 0.20; // 清理到 20% 停止

    // 统计指标
    public final LongAdder totalRequests = new LongAdder();
    public final LongAdder cacheHitBlocks = new LongAdder();

    // 记录那些 refCount 为 0 但仍在树中的“幽灵节点”
    private final Deque<RadixNode> ghostCache = new ConcurrentLinkedDeque<>();

    // 跟踪每个 Session 持有的 Radix 节点
    private final ConcurrentHashMap<String, List<RadixNode>> sessionNodes = new ConcurrentHashMap<>();

    public void releaseBlocks(String sessionId) {
        List<RadixNode> nodes = sessionNodes.remove(sessionId);
        if (nodes != null) {
            super.getGlobalLock().lock();
            try {
                for (RadixNode node : nodes) {
                    // 只有当最后一个引用消失时，才物理释放
                    if (node.refCount.decrementAndGet() == 0) {
                        // 放入幽灵缓存，等待后续剪枝
                        ghostCache.addLast(node);
                    }
                }
                diskFullCondition.signalAll();
            } finally {
                super.getGlobalLock().unlock();
            }
        }
    }

    public long[] getPhysicalBlockIds(String sessionId) {
        List<RadixNode> nodes = sessionNodes.get(sessionId);
        if (nodes == null) {
            return new long[0];
        }
        long[] blockIds = new long[nodes.size()];
        for (int i = 0; i < nodes.size(); i++) {
            blockIds[i] = nodes.get(i).blockId;
        }
        return blockIds;
    }

    public void allocateBlockss(int length, int blockSize) {
        int blocksNeeded = (int) Math.ceil((double) length / blockSize);
        for (int i = 0; i < blocksNeeded; i++) {
            Integer id = freePool.poll();
            if (id != null) {
                // 分配成功
                
            } else {
                throw new RuntimeException("GPU Memory Exhausted: No blocks available for allocation.");
            }
        }
    }
    
    

    // 内部类：树节点
    static class RadixNode {
        final long hash;
        final int blockId;
        final ConcurrentHashMap<Long, RadixNode> children = new ConcurrentHashMap<>();
        final AtomicInteger refCount = new AtomicInteger(0);
        RadixNode(long hash, int blockId) { this.hash = hash; this.blockId = blockId; }
    }

    private final RadixNode root = new RadixNode(-1, -1);

    public CoWBlockManagerV8(int totalBlocks, int layers, int blockSize, int headDim, int dtype) {
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
        // 检查水位，如果太低则触发自动剪枝
        checkAndPrune();
        return result;
    }
    private int allocateWithRetry(String sessionId, PagedKvBufferV3 buffer) {
        super.getGlobalLock().lock();
        try {
            int retryCount = 0;
            while (true) {
                Integer id = freePool.poll();
                if (id != null) return id;

                // 1. 尝试驱逐
                if (evictOldestSession(sessionId)) {
                    continue;
                }
                WAIT_COUNT.increment(); // 记录阻塞
                // 2. 指数级退让等待 (Exponential Backoff)
                // 面对万级虚拟线程，100ms 太短，我们需要更具弹性的等待周期
                retryCount++;
                long waitTime = Math.min(100 + (retryCount * 100), 2000);

                if (!diskFullCondition.await(waitTime, TimeUnit.MILLISECONDS)) {
                    // 只有在重试多次且依然无法获取内存时，才抛出 Fatal
                    if (retryCount > 10) {
                        throw new RuntimeException("GPU Memory Timeout: System Saturated after " + retryCount + " retries");
                    }
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } finally {
            super.getGlobalLock().unlock();
        }
    }

    private void checkAndPrune() {
        int free = getFreeBlockCount();
        int total = getTotalBlocks();

        // 如果水位低于 10%，开始剪枝
        if (free < total * lowWatermark) {
            super.getGlobalLock().lock();
            try {
                while (getFreeBlockCount() < total * highWatermark && !ghostCache.isEmpty()) {
                    RadixNode victim = ghostCache.pollFirst();
                    if (victim != null && victim.refCount.get() == 0) {
                        // 真正的物理回池
                        freePool.add(victim.blockId);
                        // 注意：这里需要从树中解除绑定逻辑，通常通过父节点索引实现
                    }
                }
            } finally {
                super.getGlobalLock().unlock();
            }
        }
    }

    private int getTotalBlocks() {
        return super.totalBlocks;
    }

    private int allocateWithRetry2(String sessionId, PagedKvBufferV3 buffer) {
        super.getGlobalLock().lock();
        try {
            while (true) {
                Integer id = freePool.poll();
                if (id != null) return id;

                if (evictOldestSession(sessionId)) {
                    continue;
                }
                WAIT_COUNT.increment(); // 记录阻塞
                if (!diskFullCondition.await(500, TimeUnit.MILLISECONDS)) {
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
        // 1. 调用父类执行驱逐逻辑（释放物理块）
        // 注意：父类实现可能在返回前已经释放了锁
        boolean success = super.evictOldestSession(excludeId);

        // 2. 如果驱逐成功，唤醒那些因为“资源耗尽”而阻塞在 allocateWithRetry 中的线程
        if (success) {
            // FIX: 显式获取锁。
            EVICT_COUNT.increment(); // 记录驱逐
            // 这里的 Lock 是 ReentrantLock，如果当前线程（如 allocateWithRetry）已经持有，则重入是安全的。
            // 如果当前线程（如父类的 allocateBlocks）未持有，则这里获取锁避免 IllegalMonitorStateException。
            super.getGlobalLock().lock();
            try {
                diskFullCondition.signalAll();
            } finally {
                super.getGlobalLock().unlock();
            }
        }
        return success;
    }

//    @Override
//    public void releaseSession(String sessionId) {
//        List<RadixNode> nodes = sessionNodes.remove(sessionId);
//        if (nodes != null) {
//            for (RadixNode node : nodes) {
//                node.refCount.decrementAndGet();
//            }
//        }
//        super.releaseSession(sessionId);
//    }
//    @Override
//    public void releaseSession(String sessionId) {
//        List<RadixNode> nodes = sessionNodes.remove(sessionId);
//        if (nodes != null) {
//            super.getGlobalLock().lock();
//            try {
//                for (RadixNode node : nodes) {
//                    // 只有当最后一个引用消失时，才物理释放
//                    if (node.refCount.decrementAndGet() == 0) {
//                        // 从树中移除该节点（可选：可以保留做 Cache，直到内存真正不足）
//                        // 这里我们选择立即回池以通过压测
//                        freePool.add(node.blockId);
//                        // 递归清理逻辑：如果父节点也没有子节点且 refCount 为 0，也可回收
//                    }
//                }
//                diskFullCondition.signalAll();
//            } finally {
//                super.getGlobalLock().unlock();
//            }
//        }
//        // 注意：不再调用 super.releaseSession，因为 V8/V9 已经接管了所有块的生命周期
//    }

    @Override
    public void releaseSession(String sessionId) {
        // 1. 获取该 Session 持有的所有树节点
        List<RadixNode> nodes = getSessionNodes().remove(sessionId);

        if (nodes != null) {
            super.getGlobalLock().lock();
            try {
                for (RadixNode node : nodes) {
                    // 2. 减少引用计数
                    int remainingRefs = node.refCount.decrementAndGet();

                    // 3. 引用归零：说明没有任何活跃请求在使用这个块
                    if (remainingRefs == 0) {
                        // 物理放回自由池，供 allocateWithRetry 竞争
                        freePool.add(node.blockId);

                        // 4. (进阶) 可选：在此处从树中移除该节点，或者保留作为候选 Cache
                        // 建议保留在树中，通过下一次 evictOldestSession 逻辑清理
                    }
                }
                // 5. 唤醒所有因等待显存而阻塞的线程
                getDiskFullCondition().signalAll();
            } finally {
                super.getGlobalLock().unlock();
            }
        }
        // 注意：不再调用 super.releaseSession(sessionId)，
        // 因为 V9 已经接管了基于引用计数的物理块生命周期管理
    }

    // Getter 实现
    public ConcurrentHashMap<String, List<RadixNode>> getSessionNodes() {
        return sessionNodes;
    }

    public Condition getDiskFullCondition() {
        return diskFullCondition;
    }

}
