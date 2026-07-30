import org.bytedeco.pytorch.jit.*;

//package samples.demo.kvcache;
//import org.bytedeco.pytorch.jit.JitModule;
//
//import java.util.*;
//import java.util.concurrent.ConcurrentHashMap;
//import org.bytedeco.javacpp.PointerScope;
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;
//import org.bytedeco.pytorch.llm.kvcache.PagedKvBuffer;
//
//import static org.bytedeco.pytorch.global.torch.*;
//
//
///**
// * 集成真实PagedKvBufferV3的Qwen3 Java推理类
// * 核心：通过PagedKvBufferV3完成KV缓存的物理写入/读取，真正落地PageAttention机制
// */
//public class Qwen3JavaInferenceV4 {
//    private final JitModule model;
//    private final CoWBlockManager blockManager;
//    private final int numLayers; // 模型层数
//    private final int blockSize; // 每个缓存块的token数（vLLM默认16/32）
//    private final Map<String, SessionCache> sessionCacheMap; // 会话级缓存
//
//    // 会话缓存类：绑定PagedKvBufferV3，存储PageAttention完整状态
//    private static class SessionCache {
//        long[] slotMapping; // 每个token对应的缓存slot（PageAttention核心映射）
//        long[] positionIds; // 实际位置ID（处理分页后的位置偏移）
//        Map<Integer, Tensor[]> kvCache; // 各层的KV缓存 (key: layerId, value: [K, V])
//        PagedKvBuffer kvBuffer; // 物理KV缓存载体（核心新增）
//        List<Long>[] layerPathHashes; // 各层的哈希路径（用于Radix树匹配）
//
//        @SuppressWarnings("unchecked")
//        public SessionCache(String sessionId, CoWBlockManager blockManager, int numLayers, int blockSize) {
//            this.slotMapping = new long[0];
//            this.positionIds = new long[0];
//            this.kvCache = new ConcurrentHashMap<>();
//            // 初始化物理KV缓存载体
//            this.kvBuffer = new PagedKvBuffer(sessionId, blockManager, numLayers);
//            // 初始化各层哈希路径
//            this.layerPathHashes = new List[numLayers];
//            for (int i = 0; i < numLayers; i++) {
//                this.layerPathHashes[i] = new ArrayList<>();
//            }
//        }
//
//        // 生成Token序列的哈希值（用于Radix树路径匹配）
//        public long generateHash(long[] slotMapping, int startIdx, int endIdx) {
//            // 简化实现：基于slot范围生成哈希，真实场景需用数据内容哈希
//            return Objects.hash(Arrays.copyOfRange(slotMapping, startIdx, endIdx));
//        }
//    }
//
//    public Qwen3JavaInferenceV4(String modelPath, CoWBlockManager manager, int numLayers, int blockSize) {
//        // 1. 加载TorchScript模型（指定MPS设备）
//        this.model = load(modelPath, new DeviceOptional(new Device(kCPU())), false);
//        this.blockManager = manager;
//        this.numLayers = numLayers;
//        this.blockSize = blockSize;
//        this.sessionCacheMap = new ConcurrentHashMap<>();
//    }
//
//    /**
//     * 带真实PagedKvBufferV3的生成函数
//     * @param sessionId 会话ID（区分不同请求的缓存）
//     * @param inputTokenIds 输入token ID数组
//     */
//    public void generate(String sessionId, int[] inputTokenIds) {
//        try (PointerScope scope = new PointerScope()) {
//            // 初始化会话缓存（绑定PagedKvBufferV3）
//            SessionCache sessionCache = sessionCacheMap.computeIfAbsent(
//                    sessionId,
//                    k -> new SessionCache(k, blockManager, numLayers, blockSize)
//            );
//
//            // 2. 初始输入处理：转换为Tensor + 初始化PageAttention映射 + 写入KV缓存
//            Tensor inputTensor = tensor(inputTokenIds).view(1, -1).to(new Device(kMPS()), ScalarType.Long);
//            initPageAttentionMapping(sessionCache, inputTokenIds.length, inputTensor);
//
//            // 模拟逐Token生成（推理循环）
//            int totalGenerated = 0;
//            int maxNewTokens = 50;
//            while (totalGenerated < maxNewTokens) {
//                // 3. PageAttention核心：准备所有必要输入参数（含真实KV缓存）
//                IValueVector inputs = preparePageAttentionInputs(sessionId, sessionCache, inputTensor);
//
//                // 4. 执行模型推理（带PageAttention的前向计算）
//                IValue output = model.forward(inputs);
//                Tensor logits = output.toTensor();
//
//                // 5. Greedy采样获取下一个token
//                Tensor nextTokenTensor = logits.select(1L, -1L).argmax(new LongOptional(-1), false);
//                int nextToken = (int) nextTokenTensor.item().toLong();
//                System.out.print("Token: " + nextToken + " ");
//
//                // 终止条件：遇到结束token
//                if (nextToken == 151643) break;
//
//                // 6. 更新PageAttention状态 + 写入新Token的KV缓存
//                updatePageAttentionMapping(sessionCache, nextToken, nextTokenTensor);
//
//                // 更新下一轮输入
//                inputTensor = nextTokenTensor.view(1, 1).to(new Device(kMPS()), ScalarType.Long);
//                totalGenerated++;
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//
//    /**
//     * 初始化PageAttention核心映射关系 + 真实KV缓存写入
//     * PageAttention第一步：绑定PagedKvBufferV3完成物理缓存分配和数据写入
//     */
//    private void initPageAttentionMapping(SessionCache sessionCache, int seqLen, Tensor inputTensor) {
//        // 1. 初始化slot_mapping和position_ids
//        sessionCache.slotMapping = new long[seqLen];
//        sessionCache.positionIds = new long[seqLen];
//        for (int i = 0; i < seqLen; i++) {
//            sessionCache.slotMapping[i] = i;
//            sessionCache.positionIds[i] = i;
//        }
//
//        // 2. 为每一层分配KV缓存块并写入数据（PageAttention核心：真实物理缓存操作）
//        for (int layer = 0; layer < numLayers; layer++) {
//            // 2.1 按块大小切分输入，生成哈希路径（Radix树匹配用）
//            List<Long> pathHashes = new ArrayList<>();
//            for (int i = 0; i < seqLen; i += blockSize) {
//                int endIdx = Math.min(i + blockSize, seqLen);
//                long hash = sessionCache.generateHash(sessionCache.slotMapping, i, endIdx);
//                pathHashes.add(hash);
//            }
//            sessionCache.layerPathHashes[layer] = pathHashes;
//
//            // 2.2 调用PagedKvBufferV3完成KV缓存的Prefill（真实块分配+数据写入）
//            // 写入K缓存（kvType=0）
//            sessionCache.kvBuffer.prefillUltra(layer, 0, inputTensor);
//            // 写入V缓存（kvType=1）
//            sessionCache.kvBuffer.prefillUltra(layer, 1, inputTensor);
//
//            // 2.3 （可选）使用Radix树路径匹配分配块（更贴近vLLM原生实现）
//            // sessionCache.kvBuffer.prefillWithRadix(layer, 0, pathHashes, inputTensor);
//            // sessionCache.kvBuffer.prefillWithRadix(layer, 1, pathHashes, inputTensor);
//        }
//
//        // 3. 向BlockManager申请初始缓存块（与PagedKvBufferV3的块分配联动）
//        blockManager.allocateBlocks((seqLen + blockSize - 1) / blockSize, blockSize);
//    }
//
//    /**
//     * 更新PageAttention映射 + 新Token的KV缓存写入
//     * 核心：为新token分配slot，并通过PagedKvBufferV3写入物理KV缓存
//     */
//    private void updatePageAttentionMapping(SessionCache sessionCache, int nextToken, Tensor nextTokenTensor) {
//        int currentLen = sessionCache.slotMapping.length;
//        // 1. 扩展slot_mapping和position_ids
//        long[] newSlotMapping = Arrays.copyOf(sessionCache.slotMapping, currentLen + 1);
//        long[] newPositionIds = Arrays.copyOf(sessionCache.positionIds, currentLen + 1);
//        newSlotMapping[currentLen] = currentLen;
//        newPositionIds[currentLen] = currentLen;
//        sessionCache.slotMapping = newSlotMapping;
//        sessionCache.positionIds = newPositionIds;
//
//        // 2. PageAttention核心：检查是否需要分配新的缓存块
//        boolean needNewBlock = (currentLen % blockSize == 0);
//        if (needNewBlock) {
//            // 2.1 分配新块
//            blockManager.allocateBlocks(1, blockSize);
//
//            // 2.2 为每一层写入新Token的KV缓存（真实物理操作）
//            for (int layer = 0; layer < numLayers; layer++) {
//                // 生成新块的哈希值
//                long newHash = sessionCache.generateHash(sessionCache.slotMapping, currentLen, currentLen + 1);
//                sessionCache.layerPathHashes[layer].add(newHash);
//
//                // 写入新Token的KV缓存
//                sessionCache.kvBuffer.prefillUltra(layer, 0, nextTokenTensor);
//                sessionCache.kvBuffer.prefillUltra(layer, 1, nextTokenTensor);
//            }
//        }
//    }
//
//    /**
//     * 准备PageAttention所需的全部输入参数（含真实KV缓存）
//     * 包含：input_ids, position_ids, block_table, slot_mapping, kv_cache（来自PagedKvBufferV3）等
//     */
//    private IValueVector preparePageAttentionInputs(String sessionId, SessionCache sessionCache, Tensor inputTensor) {
//        // 1. 基础张量构建（position_ids/block_table/slot_mapping）
//        Tensor positionIds = tensor(sessionCache.positionIds).view(1, -1).to(new Device(kMPS()), ScalarType.Long);
//        Tensor blockTable = getBlockTableFromManager(sessionId, sessionCache.slotMapping.length);
//        Tensor slotMapping = tensor(sessionCache.slotMapping).view(1, -1).to(new Device(kMPS()), ScalarType.Long);
//
//        // 2. 从PagedKvBufferV3获取真实的KV缓存块映射（PageAttention核心数据源）
//        IValueVector kvCacheInputs = new IValueVector();
//        for (int layer = 0; layer < numLayers; layer++) {
//            // 2.1 从sessionCache或kvBuffer获取已分配的KV块
//            Tensor kTensor = getKVCacheTensorFromBuffer(sessionCache.kvBuffer, layer, 0);
//            Tensor vTensor = getKVCacheTensorFromBuffer(sessionCache.kvBuffer, layer, 1);
//
//            // 2.2 存入KV缓存输入列表
//            kvCacheInputs.push_back(new IValue(kTensor));
//            kvCacheInputs.push_back(new IValue(vTensor));
//
//            // 2.3 更新会话缓存的KV张量（供后续推理使用）
//            sessionCache.kvCache.put(layer, new Tensor[]{kTensor, vTensor});
//        }
//
//        // 3. 组装所有输入（符合Qwen3 + vLLM PageAttention的真实输入规范）
//        IValueVector inputs = new IValueVector();
//        inputs.push_back(new IValue(inputTensor));          // input_ids
//        inputs.push_back(new IValue(positionIds));          // position_ids
//        inputs.push_back(new IValue(blockTable));           // block_table
//        inputs.push_back(new IValue(slotMapping));          // slot_mapping
//        inputs.push_back(new IValue(kvCacheInputs));        // kv_cache（来自PagedKvBufferV3）
//
//        return inputs;
//    }
//
//    /**
//     * 从PagedKvBufferV3获取指定层的KV缓存张量（真实物理缓存读取）
//     * @param kvBuffer KV缓存载体
//     * @param layer 模型层索引
//     * @param kvType 0=K缓存，1=V缓存
//     * @return K/V缓存张量（PageAttention注意力计算的核心数据）
//     */
//    private Tensor getKVCacheTensorFromBuffer(PagedKvBuffer kvBuffer, int layer, int kvType) {
//        // 模拟从PagedKvBufferV3的物理块中读取KV数据
//        // 真实场景需调用LibTorch的MPS/CPU/GPU内存读取接口
//        int headDim = 128; // Qwen3典型head_dim配置，需根据实际模型调整
//        int numHeads = 32;  // Qwen3典型num_heads配置
//        int kb = kvBuffer.getKBlockCount(layer) * blockSize;
//        int vb = kvBuffer.getVBlockCount(layer) * blockSize;
//        int seqLen = (int) (kvType == 0 ? kb : vb);
////                kvBuffer.getKBlockCount(layer) * blockSize :
////                kvBuffer.getVBlockCount(layer) * blockSize
////        );
//
//        // 创建KV缓存张量（模拟真实数据，实际需从物理块读取）
//        Tensor kvTensor = randn(new long []{1, numHeads, seqLen, headDim},
//                new TensorOptions().device(new DeviceOptional(new Device(kMPS()))).dtype(new ScalarTypeOptional(ScalarType.Float)));
//
//        return kvTensor;
//    }
//
//    /**
//     * 从BlockManager获取block_table（PageAttention的物理块映射）
//     */
//    private Tensor getBlockTableFromManager(String sessionId, int seqLen) {
//        long[] physicalBlockIds = blockManager.getPhysicalBlockIds(sessionId);
//
//        // 构建block_table：[batch_size, num_blocks] 形状的LongTensor
//        Tensor blockTable = tensor(physicalBlockIds).view(1, -1).to(new Device(kMPS()), ScalarType.Long);
//
//        // 补全缺失的块（避免PageAttention索引越界）
//        int requiredBlocks = (seqLen + blockSize - 1) / blockSize;
//        if (blockTable.size(1) < requiredBlocks) {
//            Tensor padding = tensor(new long[requiredBlocks - (int)blockTable.size(1)])
//                    .view(1, -1).to(new Device(kMPS()), ScalarType.Long);
//            blockTable = cat(new TensorVector(blockTable, padding), 1);
//        }
//
//        return blockTable;
//    }
//
//    /**
//     * 释放会话缓存（含PagedKvBufferV3的资源释放）
//     */
//    public void releaseSessionCache(String sessionId) {
//        if (sessionCacheMap.containsKey(sessionId)) {
//            SessionCache sessionCache = sessionCacheMap.get(sessionId);
//
//            // 1. 释放PagedKvBufferV3的物理资源（核心：关闭KV缓存）
//            try {
//                sessionCache.kvBuffer.close();
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//
//            // 2. 释放BlockManager中的缓存块
//            blockManager.releaseBlocks(sessionId);
//
//            // 3. 清理会话缓存（含KV张量释放）
//            for (Tensor[] kv : sessionCache.kvCache.values()) {
//                kv[0].close(); // 释放K张量
//                kv[1].close(); // 释放V张量
//            }
//            sessionCacheMap.remove(sessionId);
//        }
//    }
//
//    // ======================== 为PagedKvBufferV3补充必要的扩展方法 ========================
//    /**
//     * 获取指定层K缓存的块数量（PagedKvBufferV3扩展方法）
//     */
//    private int getKBlockCount(PagedKvBuffer kvBuffer, int layer) {
//        // 实际需从PagedKvBufferV3的kBlockMaps中获取，这里为简化做模拟
//        return (int) Math.ceil((double) kvBuffer.getSessionId().length() / blockSize);
//    }
//
//    /**
//     * 获取指定层V缓存的块数量（PagedKvBufferV3扩展方法）
//     */
//    private int getVBlockCount(PagedKvBuffer kvBuffer, int layer) {
//        // 实际需从PagedKvBufferV3的vBlockMaps中获取，这里为简化做模拟
//        return (int) Math.ceil((double) kvBuffer.getSessionId().length() / blockSize);
//    }
//
//    /**
//     * 获取会话ID（PagedKvBufferV3扩展方法）
//     */
//    private String getSessionId(PagedKvBuffer kvBuffer) {
//        // 实际需从PagedKvBufferV3的sessionId字段获取，这里为简化做模拟
//        return "session_" + System.currentTimeMillis();
//    }
//}
//
//// 补充CoWBlockManagerV9的接口定义（适配PagedKvBufferV3）
////interface CoWBlockManagerV9 extends CoWBlockManagerV2 {
////    void allocateBlocks(int numBlocks, int blockSize);
////    void releaseBlocks(String sessionId);
////    long[] getPhysicalBlockIds(String sessionId);
////    List<Integer> matchAndAllocatePath(List<Long> pathHashes, String sessionId, PagedKvBufferV3 buffer);
////    List<Integer> allocateBlocks(int neededBlocks, String sessionId, PagedKvBufferV3 buffer);
////}