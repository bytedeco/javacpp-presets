package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * StringTensorDict 遍历工具类
 * 基于 javacpp-pytorch 的 StringTensorDict（有序字典）实现便捷的遍历操作
 */
public class StringTensorDictUtils {

    // ===================== 核心遍历方法（基于迭代器）=====================
    /**
     * 遍历 StringTensorDict 的键值对（推荐方式，符合原生迭代逻辑）
     * @param dict 要遍历的有序字典
     * @param consumer 接收 (key, value) 的双参数消费器
     * @throws NullPointerException 如果 dict 或 consumer 为 null
     */
    public static void forEach(StringTensorDict dict, BiConsumer<String, Tensor> consumer) {
        // 空值校验
        if (dict == null) {
            throw new NullPointerException("StringTensorDict 不能为 null");
        }
        if (consumer == null) {
            throw new NullPointerException("BiConsumer 不能为 null");
        }

        // 使用 PointerScope 自动管理资源，避免内存泄漏
        try (PointerScope scope = new PointerScope()) {
            // 获取迭代器（指向 StringTensorDictItem 元素）
            StringTensorDictItemVector.Iterator iterator = dict.begin();
            StringTensorDictItemVector.Iterator end = dict.end();

            while (!iterator.equals(end)) {
                // 获取当前键值对项
                StringTensorDictItem item = iterator.get();
                // 转换为 Java 字符串（避免 BytePointer 内存问题）
                String key = item.key().getString();
                Tensor value = item.value();
                // 执行自定义操作
                consumer.accept(key, value);
                // 迭代器自增
                iterator.increment();
            }
        }
    }

    // ===================== 带索引的遍历（按顺序）=====================
    /**
     * 带索引遍历 StringTensorDict（有序字典保留插入顺序）
     * @param dict 要遍历的有序字典
     * @param indexedConsumer 接收 (index, key, value) 的三参数消费器
     * @throws NullPointerException 如果 dict 或 indexedConsumer 为 null
     */
    public static void forEachWithIndex(StringTensorDict dict, IndexedStringTensorConsumer indexedConsumer) {
        if (dict == null) {
            throw new NullPointerException("StringTensorDict 不能为 null");
        }
        if (indexedConsumer == null) {
            throw new NullPointerException("IndexedStringTensorConsumer 不能为 null");
        }

        try (PointerScope scope = new PointerScope()) {
            long size = dict.size();
            for (long i = 0; i < size; i++) {
                // 按索引获取键值对项
                StringTensorDictItem item = dict.get(i);
                String key = item.key().getString();
                Tensor value = item.value();
                indexedConsumer.accept(i, key, value);
            }
        }
    }

    // ===================== 单独遍历键/值 =====================
    /**
     * 仅遍历 StringTensorDict 的所有键
     * @param dict 要遍历的有序字典
     * @param keyConsumer 处理键的消费器
     */
    public static void forEachKey(StringTensorDict dict, Consumer<String> keyConsumer) {
        if (dict == null || keyConsumer == null) {
            throw new NullPointerException("参数不能为 null");
        }

        try (PointerScope scope = new PointerScope()) {
            // 直接获取所有键的向量进行遍历
            StringVector keys = dict.keys();
            for (long i = 0; i < keys.size(); i++) {
                String key = keys.get(i).getString();
                keyConsumer.accept(key);
            }
        }
    }

    /**
     * 仅遍历 StringTensorDict 的所有值
     * @param dict 要遍历的有序字典
     * @param valueConsumer 处理值的消费器
     */
    public static void forEachValue(StringTensorDict dict, Consumer<Tensor> valueConsumer) {
        if (dict == null || valueConsumer == null) {
            throw new NullPointerException("参数不能为 null");
        }

        try (PointerScope scope = new PointerScope()) {
            // 直接获取所有值的向量，复用之前的 TensorVector 遍历逻辑
            TensorVector values = dict.values();
            TensorVectorUtils.forEach(values, valueConsumer);
        }
    }

    // ===================== 简化遍历（调试用）=====================
    /**
     * 简化遍历：打印所有键值对（开箱即用，适合调试）
     * @param dict 要遍历的有序字典
     */
    public static void simpleTraverse(StringTensorDict dict) {
        if (dict == null) {
            System.out.println("StringTensorDict 为 null");
            return;
        }

        System.out.println("=== StringTensorDict 内容 ===");
        if (dict.is_empty()) {
            System.out.println("字典为空");
            return;
        }

        forEachWithIndex(dict, (index, key, value) -> {
            System.out.printf("索引 %d | 键: %s | 值: %s%n", index, key, value);
        });
    }

    // ===================== 自定义函数式接口 =====================
    /**
     * 带索引的 StringTensorDict 消费器（三参数：索引、键、值）
     */
    @FunctionalInterface
    public interface IndexedStringTensorConsumer {
        void accept(long index, String key, Tensor value);
    }

    // ===================== 测试示例 =====================
    public static void main(String[] args) {
        // 1. 创建测试用的 StringTensorDict
        StringTensorDict dict = new StringTensorDict("TestKey");
        // 插入测试键值对
        dict.insert("input_1", torch.randn(new long[]{2, 3}));
        dict.insert("hidden_1", torch.randn(new long[]{3, 4}));
        dict.insert("output_1", torch.randn(new long[]{4, 1}));

        // 2. 测试基础键值对遍历
        System.out.println("=== 基础键值对遍历 ===");
        StringTensorDictUtils.forEach(dict, (key, value) -> {
            System.out.printf("键: %s | 张量形状: %s%n", key, value.sizes());
        });

        // 3. 测试带索引遍历
        System.out.println("\n=== 带索引遍历 ===");
        StringTensorDictUtils.forEachWithIndex(dict, (index, key, value) -> {
            System.out.printf("索引 %d | 键: %s | 张量维度: %d%n", index, key, value.dim());
        });

        // 4. 测试仅遍历键
        System.out.println("\n=== 仅遍历键 ===");
        StringTensorDictUtils.forEachKey(dict, key -> System.out.println("键: " + key));

        // 5. 测试仅遍历值
        System.out.println("\n=== 仅遍历值 ===");
        StringTensorDictUtils.forEachValue(dict, tensor -> System.out.println("张量数据类型: " + tensor.dtype()));

        // 6. 测试简化遍历（打印所有内容）
        System.out.println("\n=== 简化遍历（完整打印）===");
        StringTensorDictUtils.simpleTraverse(dict);

        // 释放资源
        dict.clear();
        dict.close();
    }


    /**
     * 遍历 StringTensorDict 并返回包含所有键的 Java List
     * @param dict 要转换的有序字典
     * @return 包含所有键的 List<String>
     * @throws NullPointerException 如果 dict 为 null
     */
    public static List<String> keysToList(StringTensorDict dict) {
        if (dict == null) {
            throw new NullPointerException("StringTensorDict 不能为 null");
        }

        List<String> keyList = new ArrayList<>();
        forEachKey(dict, key -> keyList.add(key));
        return keyList;
    }

    /**
     * 遍历 StringTensorDict 并返回包含所有值的 Java List
     * @param dict 要转换的有序字典
     * @return 包含所有 Tensor 的 List<Tensor>
     * @throws NullPointerException 如果 dict 为 null
     */
    public static List<Tensor> valuesToList(StringTensorDict dict) {
        if (dict == null) {
            throw new NullPointerException("StringTensorDict 不能为 null");
        }

        List<Tensor> valueList = new ArrayList<>();
        forEachValue(dict, tensor -> valueList.add(tensor));
        return valueList;
    }

    /**
     * 遍历 StringTensorDict 并返回包含所有键值对的 Java List
     * @param dict 要转换的有序字典
     * @return 包含键值对的 List<Map.Entry<String, Tensor>>
     * @throws NullPointerException 如果 dict 为 null
     */
    public static List<Map.Entry<String, Tensor>> toEntryList(StringTensorDict dict) {
        if (dict == null) {
            throw new NullPointerException("StringTensorDict 不能为 null");
        }

        List<Map.Entry<String, Tensor>> entryList = new ArrayList<>();
        forEach(dict, (key, tensor) -> {
            // 使用 AbstractMap.SimpleEntry 封装键值对
            entryList.add(new AbstractMap.SimpleEntry<>(key, tensor));
        });

        return entryList;
    }
}
