package org.bytedeco.pytorch.geometric.demo.kvcache;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Stack;


public class CoWBlockManager extends PagedBlockManager {
    // 记录每个物理块被多少个 Session 共享
    private final ConcurrentHashMap<Integer, AtomicInteger> refCounts = new ConcurrentHashMap<>();

    public CoWBlockManager(int maxBlocks, int numLayers, int blockSize, int headDim, int scalarType) {
        super(maxBlocks, numLayers, blockSize, headDim, scalarType);
    }
    

    @Override
    public synchronized int allocateBlock() {
        int blockId = super.allocateBlock();
        refCounts.put(blockId, new AtomicInteger(1));
        return blockId;
    }

    /**
     * 增加引用计数（当一个 Session 派生出子请求时）
     */
    public void incrementRef(int blockId) {
        refCounts.get(blockId).incrementAndGet();
    }

    /**
     * 减少引用计数，当计数归零时真正释放
     */
    @Override
    public synchronized void freeBlock(int blockId) {
        if (refCounts.get(blockId).decrementAndGet() == 0) {
            refCounts.remove(blockId);
            super.freeBlock(blockId);
        }
    }

    public boolean isShared(int blockId) {
        return refCounts.get(blockId).get() > 1;
    }

    public Object getActiveBlockCount() {
        return refCounts.size();
    }
}