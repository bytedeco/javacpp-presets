package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class RandomRotate implements BaseTransform {
    private final float degrees;
    private final int axis;

    public RandomRotate(float degrees, int axis) {
        this.degrees = degrees;
        this.axis = axis;
    }

    @Override
    public GraphData apply(GraphData data) {
        // 1. 随机采样角度并计算三角函数
        double angle = (Math.random() * 2 - 1) * Math.toRadians(degrees);
        float sin = (float) Math.sin(angle);
        float cos = (float) Math.cos(angle);

        // 2. 根据轴构造一维数组 (Flattened Matrix)
        float[] flatMatrix;
        if (axis == 2) { // 绕 Z 轴旋转
            flatMatrix = new float[]{
                    cos,  sin, 0.0f,
                    -sin,  cos, 0.0f,
                    0.0f, 0.0f, 1.0f
            };
        } else if (axis == 1) { // 绕 Y 轴旋转
            flatMatrix = new float[]{
                    cos,  0.0f, -sin,
                    0.0f, 1.0f,  0.0f,
                    sin,  0.0f,  cos
            };
        } else { // 绕 X 轴旋转
            flatMatrix = new float[]{
                    1.0f, 0.0f, 0.0f,
                    0.0f, cos,  sin,
                    0.0f, -sin, cos
            };
        }

        // 3. 将一维数组转换为 [3, 3] 的 Tensor
        // 使用 tensor(float[], options) 构造，然后 view 成 3x3
        Tensor rotMat = tensor(flatMatrix, data.pos.options()).view(new long[]{3, 3});

        // 4. 应用旋转 (矩阵乘法)
        // data.pos: [N, 3] * rotMat: [3, 3] -> [N, 3]
        data.pos = data.pos.mm(rotMat);

        return data;
    }

    /**
     * 将 Java 的二维数组转换为指定 Options 的 Tensor
     */
    public static Tensor toTensor(float[][] array, TensorOptions options) {
        int rows = array.length;
        int cols = array[0].length;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(array[i], 0, flat, i * cols, cols);
        }
        return tensor(flat, options).view(new long[]{rows, cols});
    }
//    @Override
//    public GraphData call2(GraphData data) {
//        // 1. 随机采样弧度
//        double angle = (Math.random() * 2 - 1) * Math.toDegrees(degrees);
//        float sin = (float) Math.sin(angle);
//        float cos = (float) Math.cos(angle);
//
//        // 2. 构造 3D 旋转矩阵 (假设 axis=2 即绕 Z 轴旋转)
//        float[][] rotArr;
//        if (axis == 2) { // Z-axis
//            rotArr = new float[][]{{cos, sin, 0}, {-sin, cos, 0}, {0, 0, 1}};
//        } else if (axis == 1) { // Y-axis
//            rotArr = new float[][]{{cos, 0, -sin}, {0, 1, 0}, {sin, 0, cos}};
//        } else { // X-axis
//            rotArr = new float[][]{{1, 0, 0}, {0, cos, sin}, {0, -sin, cos}};
//        }
//
//        Tensor rotMat = tensor(rotArr, data.pos.options());
//        data.pos = data.pos.mm(rotMat);
//        return data;
//    }
}