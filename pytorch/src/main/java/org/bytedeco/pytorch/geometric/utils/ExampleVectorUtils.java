package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.utils.spacy.Example;
import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * ExampleVector 遍历工具类
 * 基于 javacpp-pytorch 的 ExampleVector（数据样本向量）实现便捷的遍历操作
 * 每个 Example 包含输入张量（data）和标签张量（target）
 */
public class ExampleVectorUtils {

    // ===================== 核心遍历方法（基于迭代器）=====================
    /**
     * 遍历 ExampleVector 中的每个样本（推荐方式）
     * @param exampleVector 要遍历的样本向量
     * @param consumer 处理单个 Example 的消费器
     * @throws NullPointerException 如果参数为 null
     */
    public static void forEach(ExampleVector exampleVector, Consumer<Example> consumer) {
        // 空值校验
        if (exampleVector == null) {
            throw new NullPointerException("ExampleVector 不能为 null");
        }
        if (consumer == null) {
            throw new NullPointerException("Consumer 不能为 null");
        }

        // 使用 PointerScope 自动管理 Tensor/Example 资源，避免内存泄漏
        try (PointerScope scope = new PointerScope()) {
            ExampleVector.Iterator iterator = exampleVector.begin();
            ExampleVector.Iterator end = exampleVector.end();

            while (!iterator.equals(end)) {
                // 获取当前样本
                Example example = iterator.get();
                // 执行自定义操作
                consumer.accept(example);
                // 迭代器自增
                iterator.increment();
            }
        }
    }

    // ===================== 直接遍历输入/标签张量（最常用）=====================
    /**
     * 遍历 ExampleVector，直接获取每个样本的输入张量和标签张量
     * 无需手动解析 Example，简化开发
     * @param exampleVector 要遍历的样本向量
     * @param biConsumer 接收 (inputTensor, targetTensor) 的双参数消费器
     * @throws NullPointerException 如果参数为 null
     */
    public static void forEachTensorPair(ExampleVector exampleVector, BiConsumer<Tensor, Tensor> biConsumer) {
        if (exampleVector == null) {
            throw new NullPointerException("ExampleVector 不能为 null");
        }
        if (biConsumer == null) {
            throw new NullPointerException("BiConsumer 不能为 null");
        }

        try (PointerScope scope = new PointerScope()) {
            ExampleVector.Iterator iterator = exampleVector.begin();
            ExampleVector.Iterator end = exampleVector.end();

            while (!iterator.equals(end)) {
                Example example = iterator.get();
                // 提取 Example 的输入张量（data）和标签张量（target）
                Tensor input = example.data();
                Tensor target = example.target();
                // 传递输入/标签张量给消费器
                biConsumer.accept(input, target);
                iterator.increment();
            }
        }
    }

    // ===================== 带索引的遍历 =====================
    /**
     * 带索引遍历 ExampleVector（支持获取样本位置）
     * @param exampleVector 要遍历的样本向量
     * @param indexedConsumer 接收 (index, example) 的双参数消费器
     * @throws NullPointerException 如果参数为 null
     */
    public static void forEachWithIndex(ExampleVector exampleVector, IndexedExampleConsumer indexedConsumer) {
        if (exampleVector == null) {
            throw new NullPointerException("ExampleVector 不能为 null");
        }
        if (indexedConsumer == null) {
            throw new NullPointerException("IndexedExampleConsumer 不能为 null");
        }

        try (PointerScope scope = new PointerScope()) {
            long size = exampleVector.size();
            for (long i = 0; i < size; i++) {
                Example example = exampleVector.get(i);
                indexedConsumer.accept(i, example);
            }
        }
    }

    /**
     * 带索引遍历，直接获取输入/标签张量
     * @param exampleVector 要遍历的样本向量
     * @param indexedBiConsumer 接收 (index, inputTensor, targetTensor) 的三参数消费器
     * @throws NullPointerException 如果参数为 null
     */
    public static void forEachTensorPairWithIndex(ExampleVector exampleVector, IndexedTensorPairConsumer indexedBiConsumer) {
        if (exampleVector == null) {
            throw new NullPointerException("ExampleVector 不能为 null");
        }
        if (indexedBiConsumer == null) {
            throw new NullPointerException("IndexedTensorPairConsumer 不能为 null");
        }

        try (PointerScope scope = new PointerScope()) {
            long size = exampleVector.size();
            for (long i = 0; i < size; i++) {
                Example example = exampleVector.get(i);
                Tensor input = example.data();
                Tensor target = example.target();
                indexedBiConsumer.accept(i, input, target);
            }
        }
    }

    // ===================== 简化遍历（调试用）=====================
    /**
     * 简化遍历：打印所有样本的输入/标签张量信息（开箱即用）
     * @param exampleVector 要遍历的样本向量
     */
    public static void simpleTraverse(ExampleVector exampleVector) {
        if (exampleVector == null) {
            System.out.println("ExampleVector 为 null");
            return;
        }

        System.out.println("=== ExampleVector 样本内容 ===");
        if (exampleVector.empty()) {
            System.out.println("样本向量为空");
            return;
        }

        forEachTensorPairWithIndex(exampleVector, (index, input, target) -> {
            System.out.printf("样本 %d | 输入张量形状: %s | 标签张量形状: %s%n",
                    index, input.sizes(), target.sizes());
        });
    }

    // ===================== 自定义函数式接口 =====================
    /**
     * 带索引的 Example 消费器（双参数：索引、样本）
     */
    @FunctionalInterface
    public interface IndexedExampleConsumer {
        void accept(long index, Example example);
    }

    /**
     * 带索引的张量对消费器（三参数：索引、输入张量、标签张量）
     */
    @FunctionalInterface
    public interface IndexedTensorPairConsumer {
        void accept(long index, Tensor inputTensor, Tensor targetTensor);
    }

    // ===================== 测试示例 =====================
    public static void main(String[] args) {
        // 1. 创建测试用的 ExampleVector
        ExampleVector exampleVector = new ExampleVector();

        // 创建测试样本（输入张量 + 标签张量）
        Example example1 = new Example(torch.randn(new long[]{1, 10}), torch.tensor(new float[]{0}));
        Example example2 = new Example(torch.randn(new long[]{1, 10}), torch.tensor(new float[]{1}));
        Example example3 = new Example(torch.randn(new long[]{1, 10}), torch.tensor(new float[]{2}));

        // 添加样本到向量
        exampleVector.push_back(example1);
        exampleVector.push_back(example2);
        exampleVector.push_back(example3);

        // 2. 测试基础遍历（Example 对象）
        System.out.println("=== 基础 Example 遍历 ===");
        ExampleVectorUtils.forEach(exampleVector, example -> {
            System.out.printf("输入张量维度: %d | 标签张量维度: %d%n",
                    example.data().dim(), example.target().dim());
        });

        // 3. 测试直接遍历张量对（最常用）
        System.out.println("\n=== 直接遍历输入/标签张量 ===");
        ExampleVectorUtils.forEachTensorPair(exampleVector, (input, target) -> {
            System.out.printf("输入张量数据类型: %s | 标签值: %f%n",
                    input.dtype(), target.item().toFloat());
        });

        // 4. 测试带索引遍历张量对
        System.out.println("\n=== 带索引遍历张量对 ===");
        ExampleVectorUtils.forEachTensorPairWithIndex(exampleVector, (index, input, target) -> {
            System.out.printf("样本 %d | 输入张量元素数: %d | 标签值: %f%n",
                    index, input.numel(), target.item().toFloat());
        });

        // 5. 测试简化遍历（打印所有信息）
        System.out.println("\n=== 简化遍历（完整打印）===");
        ExampleVectorUtils.simpleTraverse(exampleVector);

        // 释放资源
        exampleVector.clear();
        exampleVector.close();
        example1.close();
        example2.close();
        example3.close();
    }


    /**
     * 遍历 ExampleVector 并返回包含所有 Example 的 Java List
     * @param exampleVector 要转换的样本向量
     * @return 包含所有 Example 的 List<Example>
     * @throws NullPointerException 如果 exampleVector 为 null
     */
    public static List<Example> toList(ExampleVector exampleVector) {
        if (exampleVector == null) {
            throw new NullPointerException("ExampleVector 不能为 null");
        }

        List<Example> exampleList = new ArrayList<>();
        forEach(exampleVector, example -> exampleList.add(example));
        return exampleList;
    }

    /**
     * 遍历 ExampleVector 并返回包含所有输入/标签张量对的 Java List
     * 每个元素是长度为 2 的数组：[inputTensor, targetTensor]
     * @param exampleVector 要转换的样本向量
     * @return 包含张量对的 List<Tensor[]>
     * @throws NullPointerException 如果 exampleVector 为 null
     */
    public static List<Tensor[]> toTensorPairList(ExampleVector exampleVector) {
        if (exampleVector == null) {
            throw new NullPointerException("ExampleVector 不能为 null");
        }

        List<Tensor[]> tensorPairList = new ArrayList<>();
        forEachTensorPair(exampleVector, (input, target) -> {
            tensorPairList.add(new Tensor[]{input, target});
        });

        return tensorPairList;
    }

    /**
     * 遍历 ExampleVector 并返回包含样本数据的 Java List（输入转 float 数组）
     * 适合快速获取样本数值，无需操作 Tensor
     * @param exampleVector 要转换的样本向量
     * @return 包含 (输入数组, 标签值) 的 List<Map<String, Object>>
     * @throws NullPointerException 如果 exampleVector 为 null
     */
    public static List<Map<String, Object>> toDataList(ExampleVector exampleVector) {
        if (exampleVector == null) {
            throw new NullPointerException("ExampleVector 不能为 null");
        }

        List<Map<String, Object>> dataList = new ArrayList<>();
        forEachTensorPair(exampleVector, (input, target) -> {
            Map<String, Object> dataMap = new HashMap<>();
            // 输入张量转 float 数组
            dataMap.put("input", input.data_ptr_float().get((int) input.numel()));
            // 标签张量转 float 值（适合标量标签）
            dataMap.put("target", target.item().toFloat());
            dataList.add(dataMap);
        });

        return dataList;
    }
}
