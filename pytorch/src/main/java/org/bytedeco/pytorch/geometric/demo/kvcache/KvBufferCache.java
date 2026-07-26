package org.bytedeco.pytorch.geometric.demo.kvcache;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.pytorch.global.torch.*;

public class KvBufferCache implements Closeable {
//    private static final Logger logger = LoggerFactory.getLogger(KvBufferCache.class);
    private final ConcurrentMap<String, KvBuffer> kvBufferCache = new ConcurrentHashMap<>();

    // 对应 PyTorch 的 ScalarType (kFloat32, kBFloat16)
    private final int scalarType;
    private final int kvLength;
    private final int numLayers;
    private final int contextLength;

    public KvBufferCache(int scalarType, int numLayers, int contextLength, int kvLength) {
        this.scalarType = scalarType;
        this.numLayers = numLayers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
    }

    public KvBuffer getKvBuffer(String session) {
        return kvBufferCache.computeIfAbsent(session, s -> new KvBuffer(s));
    }

    @Override
    public void close() {
        kvBufferCache.values().forEach(KvBuffer::close);
        kvBufferCache.clear();
    }

    public class KvBuffer implements AutoCloseable {
        private final String session;
        private final AtomicInteger currentPos = new AtomicInteger(0);

        // 使用 LibTorch 张量存储: [layers, 2, context_length, kv_length]
        // 2 代表 Key 和 Value
        private Tensor fullCache;

        KvBuffer(String session) {
            this.session = session;
            long[] shape = { numLayers, 2, contextLength, kvLength };
//            TensorOptions options = new TensorOptions()
//                    .dtype(new ScalarTypeOptional(scalarType))
//                    .layout(new LayoutOptional(kStrided())) // 明确 layout
//                    .device(new DeviceOptional(kCPU));
            // 在堆外分配内存
            ScalarType st = ScalarType.Undefined;
            for (ScalarType e : ScalarType.values()) {
                if (e.value == scalarType) {
                    st = e;
                    break;
                }
            }
            this.fullCache = torch.zeros(shape, new TensorOptions().dtype(new ScalarTypeOptional(st)));
//            logger.info("Allocated LibTorch KV Cache for session {}: {} MB",
//                    session, (numLayers * 2L * contextLength * kvLength * getElementSize()) / 1024 / 1024);
        }

        private int getElementSize() {
            return scalarType == ScalarType.Float.value ? 4 : 2;
        }

        /**
         * 获取特定位置的 K 或 V 张量切片
         * @param layerIndex 层索引
         * @param position 上下文位置
         * @param kvIndex 0 为 Key, 1 为 Value
         */
//        public Tensor getTensorAt(int layerIndex, int position, int kvIndex) {
//            // 使用 LibTorch 的 select 操作进行高效切片 (无数据拷贝)
//            // 等价于 fullCache[layerIndex, kvIndex, position]
//            return fullCache.select(0, layerIndex)
//                    .select(0, kvIndex)
//                    .select(0, position);
//        }

        public Tensor getTensorAt(int layerIndex, int position, int kvIndex) {
            // 初始 fullCache 形状: [numLayers, 2, contextLength, kvLength]
            // 我们依次在对应的轴上进行选择
            return fullCache.select(0, (long)layerIndex)   // 选层 (轴0)
                    .select(0, (long)kvIndex)      // 选K或V (在剩下的 Tensor 中轴0是原来的轴1)
                    .select(0, (long)position);    // 选位置 (在剩下的 Tensor 中轴0是原来的轴2)
        }
//
//        public Tensor getTensorsUpTo(int layerIndex, int kvIndex, int upperBound) {
//            // 等价于 Python: fullCache[layerIndex, kvIndex, 0:upperBound, :]
//            return fullCache.select(0, (long)layerIndex)   // 选层
//                    .select(0, (long)kvIndex)      // 选K/V
//                    .slice(0, 0, (long)upperBound, 1); // 在当前第0轴（原第2轴）切片
//        }
//        public Tensor getTensorAt(int layerIndex, int position, int kvIndex) {
//            // 初始形状: [numLayers, 2, contextLength, kvLength]
//            return fullCache.select(0, (long)layerIndex)   // 选层，剩余 [2, contextLength, kvLength]
//                    .select(0, (long)kvIndex)      // 选K或V，剩余 [contextLength, kvLength]
//                    .select(0, (long)position);     // 选位置，剩余 [kvLength]
//        }

        /**
         * 获取截止到当前位置的所有 K 或 V (用于 Attention 计算)
         */
        public Tensor getTensorsUpTo(int layerIndex, int kvIndex, int upperBound) {
            // 等价于 fullCache[layerIndex, kvIndex, 0:upperBound]
            return fullCache.select(0, layerIndex)
                    .select(0, kvIndex)
                    .slice(0, new LongOptional(0), new LongOptional(upperBound) , 1);
        }

        public void append2(int layerIndex, int kvIndex, Tensor newData) {
            int pos = currentPos.get();
            // 将新的 token embedding 拷贝进缓存相应位置
            Tensor target = getTensorAt(layerIndex, kvIndex, pos);
//            target.put_(newData);
            target.copy_(newData);
        }
        public void append(int layerIndex, int kvIndex, Tensor newData) {
            int pos = currentPos.get();
            // 将新的 token embedding 拷贝进缓存相应位置
            // 修正参数顺序：getTensorAt(layer, position, kv_index)
            Tensor target = getTensorAt(layerIndex, pos, kvIndex);
//            target.put_(newData);
            target.copy_(newData);
        }

        public int getCurrentPosition() { return currentPos.get(); }
        public void incrementPosition() { currentPos.incrementAndGet(); }

        @Override
        public void close() {
            if (fullCache != null) {
                fullCache.deallocate(); // 显式释放堆外内存
                fullCache = null;
            }
        }
    }
}
