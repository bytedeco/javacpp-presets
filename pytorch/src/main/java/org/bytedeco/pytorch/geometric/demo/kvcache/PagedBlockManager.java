package org.bytedeco.pytorch.geometric.demo.kvcache;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import java.util.Stack;

public class PagedBlockManager implements AutoCloseable {
    private final Tensor blockPool; // 总物理块池 [max_blocks, num_layers, 2, block_size, head_dim]
    private final Stack<Integer> freeBlocks;
    private final int blockSize;

    public PagedBlockManager(int maxBlocks, int numLayers, int blockSize, int headDim, int scalarType) {
        this.blockSize = blockSize;
        long[] shape = {maxBlocks, numLayers, 2, blockSize, headDim};
        torch.ScalarType st = torch.ScalarType.Undefined;
        for (torch.ScalarType e : torch.ScalarType.values()) {
            if (e.value == scalarType) {
                st = e;
                break;
            }
        }
        TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(st));

        // 分配全局大块内存
        this.blockPool = torch.zeros(shape, options);
        this.freeBlocks = new Stack<>();
        for (int i = maxBlocks - 1; i >= 0; i--) {
            freeBlocks.push(i);
        }
    }

    public synchronized int allocateBlock() {
        if (freeBlocks.isEmpty()) throw new RuntimeException("Out of KV Cache Memory!");
        return freeBlocks.pop();
    }

    public synchronized void freeBlock(int blockId) {
        freeBlocks.push(blockId);
    }

    public Tensor getBlock(int blockId) {
        return blockPool.select(0, blockId);
    }

    public int getBlockSize() { return blockSize; }

    @Override
    public void close() {
        blockPool.deallocate();
    }
}