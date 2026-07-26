package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
//package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Scatter {

    /**
     * 实现 scatter_add (一比一还原 pytorch-scatter 逻辑) + 数据异常约束
     *
     * @param src     输入特征数据 [E, C]
     * @param index   索引张量 [E]
     * @param dim     聚合维度 (通常为 0)
     * @param dimSize 输出的第一维大小 (节点总数 N)
     * @return 聚合后的张量 [N, C]
     */
    public static Tensor scatter_add(Tensor src, Tensor index, long dim, long dimSize) {
        // ========== 新增：输入数据异常校验 ==========
        validateInput(src, index, dim, dimSize);

        // 1. 构建输出形状 [N, C]
        long[] srcShape = src.shape();
        long[] outShape = new long[srcShape.length];
        System.arraycopy(srcShape, 0, outShape, 0, srcShape.length);
        outShape[(int) dim] = dimSize;

        // 2. 初始化全 0 输出张量
        Tensor out = torch.zeros(outShape, src.options());

        // 3. 准备 Index 广播
        Tensor expandedIndex = index;
        if (index.dim() < src.dim()) {
            for (int i = 0; i < src.dim() - index.dim(); i++) {
                expandedIndex = expandedIndex.unsqueeze(-1);
            }
        }
        expandedIndex = expandedIndex.expand_as(src);

        // 4. 调用 LibTorch 原生 scatter_add
        out = out.scatter_add(dim, expandedIndex, src);

        // ========== 新增：输出数据异常修复 ==========
        out = fixNumericAnomalies(out);

        return out;
    }

    // 1. 实现 scatter_mean + 数据异常约束
    public static Tensor scatter_mean(Tensor src, Tensor index, long dim, long dimSize) {
        // ========== 新增：输入数据异常校验 ==========
        validateInput(src, index, dim, dimSize);

        // 创建与输出形状一致的 sum 张量
        long[] outShape = src.shape();
        outShape[(int) dim] = dimSize;
        Tensor out = torch.zeros(outShape, src.options());

        // 计算累加和
        out = out.scatter_add(dim, index.unsqueeze(-1).expand_as(src), src);

        // 计算每个 index 出现的次数 (Count)
        Tensor count = torch.zeros(outShape, src.options());
        Tensor ones = torch.ones_like(src);
        count = count.scatter_add(dim, index.unsqueeze(-1).expand_as(src), ones);

        // 防止除以 0
        out = out.divide(count.clamp_min(new Scalar(1.0)));

        // ========== 新增：输出数据异常修复 ==========
        out = fixNumericAnomalies(out);

        return out;
    }

    /**
     * 实现 scatter_max (兼容 JavaCPP amin/amax 签名) + 数据异常约束
     */
    public static Tensor scatter_max(Tensor src, Tensor index, long dim, long dimSize) {
        // ========== 新增：输入数据异常校验 ==========
        validateInput(src, index, dim, dimSize);

        // 获取输出形状
        long[] srcShape = src.shape();
        long[] outShape = new long[srcShape.length];
        System.arraycopy(srcShape, 0, outShape, 0, srcShape.length);
        outShape[(int) dim] = dimSize;

        // 初始化为极小值（调整为更安全的数值，避免溢出）
        Tensor out = torch.full(outShape, new Scalar(-1e9), src.options()); // 替换 -1e38 避免溢出

        for (int i = 0; i < dimSize; i++) {
            // 使用 Scalar 匹配 eq 签名
            Tensor mask = index.eq(new Scalar(i));

            // JavaCPP 中 any().item() 后通常使用 toBool()
            if (mask.any().item().toBool()) {
                // 1. 提取该 index 对应的节点特征
                Tensor maskExpanded = mask.unsqueeze(-1).expand_as(src);
                Tensor maskedSrc = src.masked_select(maskExpanded).view(-1, src.size(-1));

                // ========== 新增：maskedSrc 异常校验 ==========
                maskedSrc = fixNumericAnomalies(maskedSrc);

                // 2. 调用 amax，传入 long[] 类型的 dim 参数
                Tensor maxVal = maskedSrc.amax(new long[]{0}, false);

                // 3. 拷贝回结果张量
                out.select((int) dim, i).copy_(maxVal);

                // 释放临时张量
                maskedSrc.close();
                maxVal.close();
            }
            mask.close();
        }

        // ========== 新增：输出数据异常修复 ==========
        out = fixNumericAnomalies(out);
        // ========== 新增：孤立节点值替换（核心修复 -inf） ==========
        out = replaceIsolatedNodeValues(out, src, dim);

        return out;
    }

    /**
     * 实现 scatter_min (兼容 JavaCPP amin/amax 签名) + 数据异常约束
     */
    public static Tensor scatter_min(Tensor src, Tensor index, long dim, long dimSize) {
        // ========== 新增：输入数据异常校验 ==========
        validateInput(src, index, dim, dimSize);

        long[] srcShape = src.shape();
        long[] outShape = new long[srcShape.length];
        System.arraycopy(srcShape, 0, outShape, 0, srcShape.length);
        outShape[(int) dim] = dimSize;

        // 初始化为极大值（调整为更安全的数值，避免溢出）
        Tensor out = torch.full(outShape, new Scalar(1e9), src.options()); // 替换 1e38 避免溢出

        for (int i = 0; i < dimSize; i++) {
            Tensor mask = index.eq(new Scalar(i));

            if (mask.any().item().toBool()) {
                Tensor maskExpanded = mask.unsqueeze(-1).expand_as(src);
                Tensor maskedSrc = src.masked_select(maskExpanded).view(-1, src.size(-1));

                // ========== 新增：maskedSrc 异常校验 ==========
                maskedSrc = fixNumericAnomalies(maskedSrc);

                // 调用 amin，传入 long[] 类型的 dim 参数
                Tensor minVal = maskedSrc.amin(new long[]{0}, false);

                out.select((int) dim, i).copy_(minVal);

                // 释放临时张量
                maskedSrc.close();
                minVal.close();
            }
            mask.close();
        }

        // ========== 新增：输出数据异常修复 ==========
        out = fixNumericAnomalies(out);
        // ========== 新增：孤立节点值替换 ==========
        out = replaceIsolatedNodeValues(out, src, dim);

        return out;
    }

    /**
     * 增强版 org.bytedeco.pytorch.geometric.utils.Scatter: 支持 sum, mean, max + 数据异常约束
     */
    public static Tensor scatter(Tensor src, Tensor index, long dimSize, String reduce) {
        // ========== 新增：输入数据异常校验 ==========
        validateScatterInput(src, index, dimSize, reduce);

        // 1. 获取 src 的完整维度
        long[] srcShape = src.shape();

        // 2. 构建输出维度: 将第0维 (E) 替换为 dimSize (N)
        long[] outShape = new long[srcShape.length];
        outShape[0] = dimSize;
        if (dimSize < 0) {
            dimSize = index.max().item_long() + 1;
        }

        System.arraycopy(srcShape, 1, outShape, 1, srcShape.length - 1);

        // 3. 初始化输出 Tensor
        Tensor out = torch.zeros(outShape, src.options());

        if (reduce.equals("add") || reduce.equals("sum")) {
            // Sum 聚合: 初始为 0，累加
            out = out.index_add_(0, index, src);

        } else if (reduce.equals("mean")) {
            // Mean 聚合: Sum / Count
            Tensor sum = out.index_add_(0, index, src);

            // 计算 Count
            Tensor count = torch.zeros(new long[]{dimSize}, src.options());
            Tensor ones = torch.ones(new long[]{src.size(0)}, src.options());
            count.index_add_(0, index, ones);

            // 避免除以0
            count = count.clamp_min(new Scalar(1.0));

            // 广播维度以匹配 out
            for (int i = 1; i < outShape.length; i++) {
                count = count.unsqueeze(i);
            }

            out = sum.div(count);

        } else if (reduce.equals("max")) {
            // Max 聚合: 初始为更安全的极小值（避免溢出）
            out.fill_(new Scalar(-1e9)); // 替换 -1.0e38

            // 使用 index_reduce_ 进行 "amax" (argmax reduction)
            out = out.index_reduce_(0, index, src, "amax", false);
        } else {
            throw new UnsupportedOperationException("org.bytedeco.pytorch.geometric.utils.Scatter reduce='" + reduce + "' not implemented");
        }

        // ========== 新增：输出数据异常修复 ==========
        out = fixNumericAnomalies(out);
        // ========== 新增：孤立节点值替换 ==========
        if (reduce.equals("max")) {
            out = replaceIsolatedNodeValues(out, src, 0);
        }

        return out;
    }

    /**
     * 增强版 org.bytedeco.pytorch.geometric.utils.Scatter: 自动适配多维特征 + 数据异常约束
     */
    public static Tensor scatter1(Tensor src, Tensor index, long dimSize, String reduce) {
        // ========== 新增：输入数据异常校验 ==========
        validateScatterInput(src, index, dimSize, reduce);

        // 1. 获取 src 的完整维度
        long[] srcShape = src.shape();

        // 2. 构建输出维度: 将第0维 (E) 替换为 dimSize (N)
        long[] outShape = new long[srcShape.length];
        outShape[0] = dimSize;
        System.arraycopy(srcShape, 1, outShape, 1, srcShape.length - 1);

        // 3. 初始化输出 Tensor
        Tensor out = torch.zeros(outShape, src.options());

        if (reduce.equals("add") || reduce.equals("sum")) {
            out = out.index_add_(0, index, src);
        } else if (reduce.equals("mean")) {
            // Mean = Sum / Count
            Tensor sum = out.index_add_(0, index, src);

            // 4. 计算 Count
            Tensor count = torch.zeros(new long[]{dimSize}, src.options());
            Tensor ones = torch.ones(new long[]{src.size(0)}, src.options());
            count.index_add_(0, index, ones);

            // 避免除以0
            count = count.clamp_min(new Scalar(1.0));

            // 5. 调整 count 维度
            for (int i = 1; i < outShape.length; i++) {
                count = count.unsqueeze(i);
            }

            out = sum.div(count);
        } else if (reduce.equals("max")) {
            throw new UnsupportedOperationException("org.bytedeco.pytorch.geometric.utils.Scatter max not implemented yet");
        }

        // ========== 新增：输出数据异常修复 ==========
        out = fixNumericAnomalies(out);

        return out;
    }

    public static Tensor scatter2(Tensor src, Tensor index, long dimSize, String reduce) {
        // ========== 新增：输入数据异常校验 ==========
        validateScatterInput(src, index, dimSize, reduce);

        // 初始化输出 Tensor，全0
        long features = src.size(1);
        Tensor out = torch.zeros(new long[]{dimSize, features}, src.options());

        if (reduce.equals("add") || reduce.equals("sum")) {
            out = out.index_add_(0, index, src);
        } else if (reduce.equals("mean")) {
            // Mean = Sum / Count
            Tensor sum = out.index_add_(0, index, src);
            Tensor count = torch.zeros(new long[]{dimSize}, src.options());
            Tensor ones = torch.ones(new long[]{src.size(0)}, src.options());
            count.index_add_(0, index, ones);
            // 避免除以0
            count = count.clamp_min(new Scalar(1.0));
            out = sum.div(count.unsqueeze(1));
        } else if (reduce.equals("max")) {
            throw new UnsupportedOperationException("org.bytedeco.pytorch.geometric.utils.Scatter max not implemented yet");
        }

        // ========== 新增：输出数据异常修复 ==========
        out = fixNumericAnomalies(out);

        return out;
    }

    // ====================== 新增：数据异常约束核心方法 ======================

    /**
     * 输入数据基础校验（防止空值/维度错误/索引越界）
     */
    private static void validateInput(Tensor src, Tensor index, long dim, long dimSize) {
        // 1. 空值校验
        if (src == null || index == null) {
            throw new NullPointerException("src 和 index 不能为空");
        }
        // 2. 维度合法性校验
        if (dim < 0 || dim >= src.dim()) {
            throw new IllegalArgumentException("dim 越界: dim=" + dim + ", src维度=" + src.dim());
        }
        // 3. 索引长度匹配校验
        if (index.size(0) != src.size((int) dim)) {
            throw new IllegalArgumentException("index长度与src指定维度不匹配: index=" + index.size(0) + ", src.dim(" + dim + ")=" + src.size((int) dim));
        }
        // 4. 索引值范围校验
        long maxIndex = index.max().item_long();
        if (maxIndex >= dimSize) {
            throw new IllegalArgumentException("index值越界: 最大索引=" + maxIndex + ", dimSize=" + dimSize);
        }
        // 5. 输入数据 NaN/Inf 校验
        if (src.isnan().any().item().toBool()) {
            throw new IllegalArgumentException("src 包含 NaN 值");
        }
        if (src.isinf().any().item().toBool()) {
            throw new IllegalArgumentException("src 包含 Inf 值");
        }
    }

    /**
     * scatter 方法专用输入校验
     */
    private static void validateScatterInput(Tensor src, Tensor index, long dimSize, String reduce) {
        validateInput(src, index, 0, dimSize); // 复用基础校验
        // 聚合方式校验
        if (!reduce.equals("add") && !reduce.equals("sum") && !reduce.equals("mean") && !reduce.equals("max")) {
            throw new UnsupportedOperationException("不支持的聚合方式: " + reduce);
        }
    }

    /**
     * 修复数值异常（NaN/Inf/溢出值）
     */
    private static Tensor fixNumericAnomalies(Tensor tensor) {
        // 1. NaN 替换为 0
        Tensor nanMask = tensor.isnan();
        if (nanMask.any().item().toBool()) {
            tensor = tensor.where(nanMask.logical_not(), torch.zeros_like(tensor));
        }
        // 2. Inf 替换为安全极值
        Tensor infMask = tensor.isinf();
        if (infMask.any().item().toBool()) {
            // 正 Inf 替换为 1e9，负 Inf 替换为 -1e9
            Tensor posInfMask = tensor.gt(new Scalar(1e18));
            Tensor negInfMask = tensor.lt(new Scalar(-1e18));
            tensor = tensor.where(posInfMask.logical_not(), torch.full_like(tensor, new Scalar(1e9)));
            tensor = tensor.where(negInfMask.logical_not(), torch.full_like(tensor, new Scalar(-1e9)));
        }
        // 3. 数值裁剪（防止溢出）
        tensor = tensor.clamp(new ScalarOptional(new Scalar(-1e9)), new ScalarOptional(new Scalar(1e9)));

        // 释放临时张量
        nanMask.close();
        infMask.close();
//        posInfMask.close();
//        negInfMask.close();

        return tensor;
    }

    /**
     * 替换孤立节点的极值为合理值（核心修复 EdgeConv 的 -inf 问题）
     */
    private static Tensor replaceIsolatedNodeValues(Tensor out, Tensor src, long dim) {
        // 1. 检测孤立节点（值为初始极值的节点）
        Tensor minValMask = out.eq(new Scalar(-1e9)); // max聚合的初始值
        Tensor maxValMask = out.eq(new Scalar(1e9));  // min聚合的初始值

        // 2. 孤立节点替换为 src 的均值（合理默认值）
        if (minValMask.any().item().toBool()) {
            Tensor srcMean = src.mean(new long[]{0}, false, new ScalarTypeOptional()); // 计算src的特征均值
            out = out.where(minValMask.logical_not(), srcMean.expand_as(out));
            srcMean.close();
        }
        if (maxValMask.any().item().toBool()) {
            Tensor srcMean = src.mean(new long[]{0}, false, new ScalarTypeOptional());
            out = out.where(maxValMask.logical_not(), srcMean.expand_as(out));
            srcMean.close();
        }

        // 释放临时张量
        minValMask.close();
        maxValMask.close();

        return out;
    }
}
//public class Scatter {
//
//    /**
//     * 实现 scatter_add (一比一还原 pytorch-scatter 逻辑)
//     *
//     * @param src     输入特征数据 [E, C]
//     * @param index   索引张量 [E]
//     * @param dim     聚合维度 (通常为 0)
//     * @param dimSize 输出的第一维大小 (节点总数 N)
//     * @return 聚合后的张量 [N, C]
//     */
//    public static Tensor scatter_add(Tensor src, Tensor index, long dim, long dimSize) {
//        // 1. 构建输出形状 [N, C]
//        long[] srcShape = src.shape();
//        long[] outShape = new long[srcShape.length];
//        System.arraycopy(srcShape, 0, outShape, 0, srcShape.length);
//        outShape[(int) dim] = dimSize;
//
//        // 2. 初始化全 0 输出张量
//        Tensor out = torch.zeros(outShape, src.options());
//
//        // 3. 准备 Index 广播
//        // index 原本可能是 [E]，需要变为 [E, 1] 然后 expand 成 [E, C]
//        // 这样才能与 src [E, C] 的每个元素对应
//        Tensor expandedIndex = index;
//        if (index.dim() < src.dim()) {
//            for (int i = 0; i < src.dim() - index.dim(); i++) {
//                expandedIndex = expandedIndex.unsqueeze(-1);
//            }
//        }
//        expandedIndex = expandedIndex.expand_as(src);
//
//        // 4. 调用 LibTorch 原生 scatter_add
//        // 签名通常为: out.scatter_add(dim, index, src)
//        // 注意：JavaCPP 封装的方法通常会直接修改 out 并返回
//        return out.scatter_add(dim, expandedIndex, src);
//    }
//
//    // 1. 实现 scatter_mean
//    public static Tensor scatter_mean(Tensor src, Tensor index, long dim, long dimSize) {
//        // 创建与输出形状一致的 sum 张量
//        long[] outShape = src.shape();
//        outShape[(int) dim] = dimSize;
//        Tensor out = torch.zeros(outShape, src.options());
//
//        // 计算累加和
//        out = out.scatter_add(dim, index.unsqueeze(-1).expand_as(src), src);
//
//        // 计算每个 index 出现的次数 (Count)
//        Tensor count = torch.zeros(outShape, src.options());
//        Tensor ones = torch.ones_like(src);
//        count = count.scatter_add(dim, index.unsqueeze(-1).expand_as(src), ones);
//
//        // 防止除以 0
//        return out.divide(count.clamp_min(new Scalar(1.0)));
//    }
//
////    // 2. 实现 scatter_max
////    public static Tensor scatter_max2(Tensor src, Tensor index, long dim, long dimSize) {
////        // 初始化为极小值
////        long[] outShape = src.shape();
////        outShape[(int)dim] = dimSize;
////        Tensor out = torch.full(outShape, new Scalar(-1e38), src.options());
////
////        // 注意：LibTorch 原生的 scatter_ 并不直接支持 scatter_max 逻辑（那是 torch-scatter 的功能）
////        // 方案：使用 torch.amax 和 mask 模拟（性能略低）或使用特定的 C++ 算子
////        // 这里提供一个通用的模拟方法：
////        for (int i = 0; i < dimSize; i++) {
////            Tensor mask = index.eq(new Scalar(i));
////            if (mask.any().item().toBool()) {
////                Tensor maskedSrc = src.masked_select(mask.unsqueeze(-1).expand_as(src)).view(-1, src.size(-1));
////                out.select(dim, i).copy_(maskedSrc.amax(0));
////            }
////        }
////        return out;
////    }
//
//    /**
//     * 实现 scatter_max (兼容 JavaCPP amin/amax 签名)
//     */
//    public static Tensor scatter_max(Tensor src, Tensor index, long dim, long dimSize) {
//        // 获取输出形状
//        long[] srcShape = src.shape();
//        long[] outShape = new long[srcShape.length];
//        System.arraycopy(srcShape, 0, outShape, 0, srcShape.length);
//        outShape[(int) dim] = dimSize;
//
//        // 初始化为极小值
//        Tensor out = torch.full(outShape, new Scalar(-1e38), src.options());
//
//        for (int i = 0; i < dimSize; i++) {
//            // 使用 Scalar 匹配 eq 签名
//            Tensor mask = index.eq(new Scalar(i));
//
//            // JavaCPP 中 any().item() 后通常使用 toBool()
//            if (mask.any().item().toBool()) {
//                // 1. 提取该 index 对应的节点特征
//                // mask 需扩展到特征维度 [N, C]
//                Tensor maskExpanded = mask.unsqueeze(-1).expand_as(src);
//                Tensor maskedSrc = src.masked_select(maskExpanded).view(-1, src.size(-1));
//
//                // 2. 调用 amax，传入 long[] 类型的 dim 参数
//                // 根据你提供的签名：amax(long[] dim, boolean keepdim)
//                Tensor maxVal = maskedSrc.amax(new long[]{0}, false);
//
//                // 3. 拷贝回结果张量
//                out.select((int) dim, i).copy_(maxVal);
//            }
//        }
//        return out;
//    }
//
//    /**
//     * 实现 scatter_min (兼容 JavaCPP amin/amax 签名)
//     */
//    public static Tensor scatter_min(Tensor src, Tensor index, long dim, long dimSize) {
//        long[] srcShape = src.shape();
//        long[] outShape = new long[srcShape.length];
//        System.arraycopy(srcShape, 0, outShape, 0, srcShape.length);
//        outShape[(int) dim] = dimSize;
//
//        // 初始化为极大值
//        Tensor out = torch.full(outShape, new Scalar(1e38), src.options());
//
//        for (int i = 0; i < dimSize; i++) {
//            Tensor mask = index.eq(new Scalar(i));
//
//            if (mask.any().item().toBool()) {
//                Tensor maskExpanded = mask.unsqueeze(-1).expand_as(src);
//                Tensor maskedSrc = src.masked_select(maskExpanded).view(-1, src.size(-1));
//
//                // 调用 amin，传入 long[] 类型的 dim 参数
//                Tensor minVal = maskedSrc.amin(new long[]{0}, false);
//
//                out.select((int) dim, i).copy_(minVal);
//            }
//        }
//        return out;
//    }
//
//    // 3. 实现 scatter_min
////    public static Tensor scatter_min2(Tensor src, Tensor index, long dim, long dimSize) {
////        long[] outShape = src.shape();
////        outShape[(int)dim] = dimSize;
////        Tensor out = torch.full(outShape, new Scalar(1e38), src.options());
////
////        for (int i = 0; i < dimSize; i++) {
////            Tensor mask = index.eq(new Scalar(i));
////            if (mask.any().item().toBool()) {
////                Tensor maskedSrc = src.masked_select(mask.unsqueeze(-1).expand_as(src)).view(-1, src.size(-1));
////                out.select(dim, i).copy_(maskedSrc.amin(0));
////            }
////        }
////        return out;
////    }
//    /**
//     * 增强版 org.bytedeco.pytorch.geometric.utils.Scatter: 支持 sum, mean, max
//     * @param src 输入数据 (消息) [E, F1, F2, ...]
//     * @param index 目标索引 [E]
//     * @param dimSize 目标节点的数量 (N)
//     * @param reduce "add", "mean", "max"
//     * @return 聚合后的 Tensor [N, F1, F2, ...]
//     */
//    public static Tensor scatter(Tensor src, Tensor index, long dimSize, String reduce) {
//        // 1. 获取 src 的完整维度
//        long[] srcShape = src.shape();
//
//        // 2. 构建输出维度: 将第0维 (E) 替换为 dimSize (N)
//        long[] outShape = new long[srcShape.length];
//        outShape[0] = dimSize;
//        if (dimSize < 0) {
//            dimSize = index.max().item_long() + 1;
//        }
//        
//        System.arraycopy(srcShape, 1, outShape, 1, srcShape.length - 1);
//
//        // 3. 初始化输出 Tensor
//        Tensor out = torch.zeros(outShape, src.options());
//
//        if (reduce.equals("add") || reduce.equals("sum")) {
//            // Sum 聚合: 初始为 0，累加
//            return out.index_add_(0, index, src);
//
//        } else if (reduce.equals("mean")) {
//            // Mean 聚合: Sum / Count
//            Tensor sum = out.index_add_(0, index, src);
//
//            // 计算 Count
//            Tensor count = torch.zeros(new long[]{dimSize}, src.options());
//            Tensor ones = torch.ones(new long[]{src.size(0)}, src.options());
//            count.index_add_(0, index, ones);
//
//            // 避免除以0
//            count = count.clamp_min(new Scalar(1.0));
//
//            // 广播维度以匹配 out
//            for (int i = 1; i < outShape.length; i++) {
//                count = count.unsqueeze(i);
//            }
//
//            return sum.div(count);
//
//        } else if (reduce.equals("max")) {
//            // Max 聚合: 初始为极小值
//            // 如果初始化为0，那么所有负数特征都会错误地变成0
//            out.fill_(new Scalar(-1.0e38)); // 填充接近 float 负无穷的值
//
//            // 使用 index_reduce_ 进行 "amax" (argmax reduction)
//            // 参数: dim, index, source, reduce_str, include_self
//            return out.index_reduce_(0, index, src, "amax", false);
//        }
//
//        throw new UnsupportedOperationException("org.bytedeco.pytorch.geometric.utils.Scatter reduce='" + reduce + "' not implemented");
//    }
//    /**
//     * 模仿 torch_scatter.scatter
//     * @param src 输入数据 (消息) [E, F]
//     * @param index 目标索引 [E]
//     * @param dimSize 目标节点的数量 (N)
//     * @param reduce "add", "mean", "max"
//     * @return 聚合后的 Tensor [N, F]
//     */
//    /**
//     * 增强版 org.bytedeco.pytorch.geometric.utils.Scatter: 自动适配多维特征
//     * @param src 输入数据 (消息) [E, F1, F2, ...]
//     * @param index 目标索引 [E]
//     * @param dimSize 目标节点的数量 (N)
//     * @param reduce "add", "mean", "max"
//     * @return 聚合后的 Tensor [N, F1, F2, ...]
//     */
//    public static Tensor scatter1(Tensor src, Tensor index, long dimSize, String reduce) {
//        // 1. 获取 src 的完整维度
//        long[] srcShape = src.shape();
//
//        // 2. 构建输出维度: 将第0维 (E) 替换为 dimSize (N)，保持后续维度不变
//        // 例如: src [10, 2, 4] -> out [3, 2, 4]
//        long[] outShape = new long[srcShape.length];
//        outShape[0] = dimSize;
//        System.arraycopy(srcShape, 1, outShape, 1, srcShape.length - 1);
//
//        // 3. 初始化输出 Tensor
//        Tensor out = torch.zeros(outShape, src.options());
//
//        if (reduce.equals("add") || reduce.equals("sum")) {
//            return out.index_add_(0, index, src);
//        } else if (reduce.equals("mean")) {
//            // Mean = Sum / Count
//            Tensor sum = out.index_add_(0, index, src);
//
//            // 4. 计算 Count (注意广播机制)
//            // Count 只需要是 [N]，但为了除法，需要扩展维度到 [N, 1, 1...]
//            Tensor count = torch.zeros(new long[]{dimSize}, src.options());
//            Tensor ones = torch.ones(new long[]{src.size(0)}, src.options());
//            count.index_add_(0, index, ones);
//
//            // 避免除以0
//            count = count.clamp_min(new Scalar(1.0));
//
//            // 5. 调整 count 维度以进行广播除法
//            // 如果 out 是 3D [N, H, C]，count 需要变成 [N, 1, 1]
//            for (int i = 1; i < outShape.length; i++) {
//                count = count.unsqueeze(i);
//            }
//
//            return sum.div(count);
//        } else if (reduce.equals("max")) {
//            throw new UnsupportedOperationException("org.bytedeco.pytorch.geometric.utils.Scatter max not implemented yet");
//        }
//        return out;
//    }
//    public static Tensor scatter2(Tensor src, Tensor index, long dimSize, String reduce) {
//        // 初始化输出 Tensor，全0
//        long features = src.size(1);
//        Tensor out = torch.zeros(new long[]{dimSize, features}, src.options());
//
//        if (reduce.equals("add") || reduce.equals("sum")) {
//            // out.index_add_(0, index, src);
//            return out.index_add_(0, index, src);
//        } else if (reduce.equals("mean")) {
//            // Mean = Sum / Count
//            Tensor sum = out.index_add_(0, index, src);
//            Tensor count = torch.zeros(new long[]{dimSize}, src.options());
//            Tensor ones = torch.ones(new long[]{src.size(0)}, src.options());
//            count.index_add_(0, index, ones);
//            // 避免除以0
//            count = count.clamp_min(new Scalar(1.0));
//            return sum.div(count.unsqueeze(1));
//        } else if (reduce.equals("max")) {
//            // LibTorch 的 scatter_reduce 需要较新版本，或者手动实现
//            // 这里为了简单，假设使用 scatter_add，实际生产需要完善 max 逻辑
//            // 提示: 使用 torch.index_reduce_ (如果版本支持) 或循环 (不推荐)
//            throw new UnsupportedOperationException("org.bytedeco.pytorch.geometric.utils.Scatter max not implemented yet");
//        }
//        return out;
//    }
//}