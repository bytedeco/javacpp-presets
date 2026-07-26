package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

/**
 * TensorVector 遍历工具类
 * 基于 javacpp-pytorch 的 TensorVector 实现便捷的遍历操作
 */
public class TensorVectorUtils {

    /**
     * 普通遍历 TensorVector
     * @param tensorVector 要遍历的 TensorVector 对象
     * @param consumer 对每个 Tensor 执行的操作（Consumer 函数式接口）
     * @throws NullPointerException 如果 tensorVector 或 consumer 为 null
     */
    public static void forEach(TensorVector tensorVector, Consumer<Tensor> consumer) {
        // 空值校验
        if (tensorVector == null) {
            throw new NullPointerException("TensorVector 不能为 null");
        }
        if (consumer == null) {
            throw new NullPointerException("遍历操作 Consumer 不能为 null");
        }

        // 获取迭代器进行遍历（推荐使用迭代器，符合 C++ vector 遍历习惯）
        TensorVector.Iterator iterator = tensorVector.begin();
        TensorVector.Iterator end = tensorVector.end();

        // 使用 PointerScope 自动管理 Tensor 资源，避免内存泄漏
        try (PointerScope scope = new PointerScope()) {
            while (!iterator.equals(end)) {
                // 获取当前迭代器指向的 Tensor
                Tensor tensor = iterator.get();
                // 执行自定义操作
                consumer.accept(tensor);
                // 迭代器自增
                iterator.increment();
            }
        }
    }

    /**
     * 索引式普通遍历（备选方案，按索引访问）
     * 适合需要知道元素索引的场景
     * @param tensorVector 要遍历的 TensorVector 对象
     * @param indexedConsumer 接收索引和 Tensor 的操作接口
     * @throws NullPointerException 如果 tensorVector 或 indexedConsumer 为 null
     */
    public static void forEachWithIndex(TensorVector tensorVector, IndexedTensorConsumer indexedConsumer) {
        if (tensorVector == null) {
            throw new NullPointerException("TensorVector 不能为 null");
        }
        if (indexedConsumer == null) {
            throw new NullPointerException("IndexedTensorConsumer 不能为 null");
        }

        long size = tensorVector.size();
        // 按索引遍历
        try (PointerScope scope = new PointerScope()) {
            for (long i = 0; i < size; i++) {
                Tensor tensor = tensorVector.get(i);
                indexedConsumer.accept(i, tensor);
            }
        }
    }

    /**
     * 简化的普通遍历（静态方法直接打印示例）
     * 适合快速查看 TensorVector 内容的场景
     * @param tensorVector 要遍历的 TensorVector 对象
     */
    public static void simpleTraverse(TensorVector tensorVector) {
        forEach(tensorVector, tensor -> {
            // 示例：打印 Tensor 的基本信息（可根据需求修改）
            System.out.println("Tensor 信息: " + tensor);
            // 可选：获取 Tensor 的数据类型、形状等
             System.out.println("数据类型: " + tensor.dtype());
             System.out.println("形状: " + Arrays.toString(tensor.shape()));
        });
    }

    /**
     * 函数式接口：支持索引的 Tensor 消费器
     * 用于需要同时获取索引和 Tensor 的遍历场景
     */
    @FunctionalInterface
    public interface IndexedTensorConsumer {
        void accept(long index, Tensor tensor);
    }

    // 测试示例
    public static void main(String[] args) {
        // 1. 创建测试用的 TensorVector
        TensorVector tensorVector = new TensorVector();
        // 添加测试 Tensor
        tensorVector.push_back(torch.zeros(new long[]{2, 3}));
        tensorVector.push_back(torch.ones(new long[]{1, 4}));
        tensorVector.push_back(torch.eye(3));

        // 2. 测试普通遍历（函数式）
        System.out.println("=== 普通函数式遍历 ===");
        TensorVectorUtils.forEach(tensorVector, tensor -> {
            torch.print(tensor);
            System.out.println("当前 Tensor 形状: " + tensor.sizes());
        });

        // 3. 测试带索引的遍历
        System.out.println("\n=== 带索引的遍历 ===");
        TensorVectorUtils.forEachWithIndex(tensorVector, (index, tensor) -> {
            System.out.printf("索引 %d: Tensor 维度数 = %d%n", index, tensor.dim());
            torch.print(tensor);
        });

        // 4. 测试简化遍历（直接打印）
        System.out.println("\n=== 简化遍历（打印） ===");
        TensorVectorUtils.simpleTraverse(tensorVector);

        // 释放资源
        tensorVector.close();
    }


    /**
     * 遍历 TensorVector 并返回包含所有 Tensor 的 Java List
     * 注意：List 中的 Tensor 仍受 PointerScope 管理，若需长期持有需手动管理内存
     * @param tensorVector 要转换的张量向量
     * @return 包含所有 Tensor 的 List<Tensor>
     * @throws NullPointerException 如果 tensorVector 为 null
     */
    public static List<Tensor> toList(TensorVector tensorVector) {
        if (tensorVector == null) {
            throw new NullPointerException("TensorVector 不能为 null");
        }

        List<Tensor> tensorList = new ArrayList<>();
        long size = tensorVector.size();

        // 遍历并添加所有 Tensor 到 List
        try (PointerScope scope = new PointerScope()) {
            for (long i = 0; i < size; i++) {
                Tensor tensor = tensorVector.get(i);
                tensorList.add(tensor);
            }
        }

        return tensorList;
    }

    /**
     * 遍历 TensorVector 并返回包含 Tensor 数据的 Java List（转换为 float 数组）
     * 适合直接获取张量数值，避免直接操作 Tensor 对象
     * @param tensorVector 要转换的张量向量
     * @return 包含每个 Tensor 数据的 List<float[]>
     * @throws NullPointerException 如果 tensorVector 为 null
     */
//    public static List<float[]> toFloatArrayList(TensorVector tensorVector) {
//        if (tensorVector == null) {
//            throw new NullPointerException("TensorVector 不能为 null");
//        }
//
//        List<float[]> dataList = new ArrayList<>();
//        forEach(tensorVector, tensor -> {
//            // 将 Tensor 转换为 float 数组（需确保 Tensor 是 float 类型）
//            float[] data = tensor.data_ptr_float().get((int) tensor.numel());
//            dataList.add(data);
//        });
//
//        return dataList;
//    }
}