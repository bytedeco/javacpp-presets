package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;
import java.util.List;

/**
 * RemoveTrainingClasses: 从训练集中移除指定类别的节点
 * 原理：修改 data.train_mask，将属于被移除类别的节点对应的位设为 false
 */
public class RemoveTrainingClasses implements BaseTransform {
    private List<Integer> classesToRemove;

    public RemoveTrainingClasses(List<Integer> classesToRemove) {
        this.classesToRemove = classesToRemove;
    }

    @Override
    public GraphData apply(GraphData data) {
        if (data.get("train_mask") == null || data.y == null) {
            return data;
        }

        // 1. 遍历需要移除的类别
        for (Integer cls : classesToRemove) {
            // 2. 找到属于该类别的节点：data.y == cls
            Tensor classMask = data.y.eq(new Scalar(cls));

            // 3. 将这些节点从训练掩码中剔除：train_mask = train_mask & (~classMask)
            // logical_not 为取反，logical_and 为按位与
            data.put("train_mask", logical_and(data.get("train_mask"), logical_not(classMask)));
        }

        return data;
    }
}