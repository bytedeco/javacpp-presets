import org.bytedeco.pytorch.jit.*;

//package samples.demo.kvcache;
//import org.bytedeco.pytorch.jit.JitModule;
//
//import org.bytedeco.javacpp.PointerScope;
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;
//
//import static org.bytedeco.pytorch.global.torch.*;
//
//public class Qwen3JavaInference {
////    private final CompilationUnit model;
//    private final JitModule model;
//    private final CoWBlockManager blockManager;
//    private final int numLayers = 32;
//
//    public Qwen3JavaInference(String modelPath, CoWBlockManager manager) {
//        // 1. 加载 TorchScript 模型
//        this.model = load(modelPath, new DeviceOptional(new Device(kMPS())),false);
//        this.blockManager = manager;
//    }
//
//    public void generate(String sessionId, int[] inputTokenIds) {
//        try (PointerScope scope = new PointerScope()) {
//            // 2. 将输入转换为 Tensor [1, seq_len]
//            Tensor inputTensor = tensor(inputTokenIds).view(1, -1).to(new Device(kMPS()),ScalarType.Float);
//
//            // 模拟逐 Token 生成
//            for (int step = 0; step < 50; step++) {
//                // 3. 为每一层准备 Block Table (从你的 Manager 中获取)
//                // 注意：在真实 vLLM 中，这通常是一个 [batch, num_blocks] 的矩阵
//                Tensor blockTable = prepareBlockTable(sessionId);
//
//                // 4. 执行推理
//                // Qwen3 TorchScript 期待输入：input_ids, position_ids, block_tables, slot_mapping 等
//                IValueVector inputs = new IValueVector(new IValue(inputTensor), new IValue(blockTable));
//                Tensor logits = model.forward(inputs).toTensor();
//
//                // 5. 采样（简单起见使用 Greedy Search）
//                Tensor nextTokenTensor = logits.select(1l, -1l).argmax(new LongOptional(-1), false);
//                int nextToken = (int) nextTokenTensor.item().toLong();
//
//                System.out.print("Token: " + nextToken + " "); // 实际应调用 Tokenizer 解码
//
//                if (nextToken == 151643) break; // Qwen3 的 <|endoftext|>
//
//                // 更新下一轮输入
//                inputTensor = nextTokenTensor.view(1, 1);
//            }
//        }
//    }
//
//    private Tensor prepareBlockTable(String sid) {
//        // 这里对接你的 CoWBlockManagerV8
//        // 返回一个包含物理块索引的 LongTensor
//        return tensor(new long[]{102, 105, 200}).to(new Device(kMPS()),ScalarType.Long);
//    }
//}
