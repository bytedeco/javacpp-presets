package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TensorConverter {

    public static void main(String[] args) {
        // 1. 测试 3D Float
        float[][][] floatData = {
                {{1.0f, 2.0f}, {3.0f, 4.0f}},
                {{5.0f, 6.0f}, {7.0f, 8.0f}}
        };
        Tensor tFloat = TensorConverter.toTensor(floatData);
        System.out.println("Float Tensor:");
        System.out.println(tFloat);
        // 输出应为 [2, 2, 2] 的 tensor

        // 2. 测试 2D Boolean (会自动转为 Bool 类型)
        boolean[][] boolData = {
                {true, false},
                {false, true}
        };
        Tensor tBool = TensorConverter.toTensor(boolData);
        System.out.println("Bool Tensor:");
        System.out.println(tBool);
        System.out.println("Dtype: " + tBool.dtype());
        // 输出应为 [2, 2]，dtype 为 Bool

        // 3. 测试 4D Int
        int[][][][] intData = new int[2][2][2][2];
        // ... 填充数据 ...
        Tensor tInt = TensorConverter.toTensor(intData);
        System.out.println("Int Tensor Shape: " + Arrays.toString(tInt.shape()));
    }
    /**
     * 通用入口：将任意维度的 Java 数组转换为 PyTorch Tensor
     * 支持类型：int, long, float, double, short, byte, boolean
     *
     * @param data 多维数组对象 (e.g. float[][], int[][][][])
     * @return 对应 shape 和 dtype 的 Tensor
     */
    public static Tensor toTensor(Object data) {
        if (data == null || !data.getClass().isArray()) {
            throw new IllegalArgumentException("Input data must be a Java array.");
        }

        // 1. 获取数组的 Shape 和总元素数量
        List<Long> shapeList = new ArrayList<>();
        getShape(data, shapeList);
        long[] shape = shapeList.stream().mapToLong(Long::longValue).toArray();
        int totalElements = 1;
        for (long s : shape) {
            totalElements *= (int) s;
        }

        // 2. 获取底层元素类型
        Class<?> componentType = getRootComponentType(data.getClass());

        // 3. 根据类型进行分发处理
        if (componentType == float.class) {
            float[] flat = new float[totalElements];
            flattenFloat(data, flat, new int[]{0});
            return torch.tensor(new FloatPointer(flat)).view(shape);
        }
        else if (componentType == int.class) {
            int[] flat = new int[totalElements];
            flattenInt(data, flat, new int[]{0});
            return torch.tensor(new IntPointer(flat)).view(shape);
        }
        else if (componentType == double.class) {
            double[] flat = new double[totalElements];
            flattenDouble(data, flat, new int[]{0});
            return torch.tensor(new DoublePointer(flat)).view(shape);
        }
        else if (componentType == long.class) {
            long[] flat = new long[totalElements];
            flattenLong(data, flat, new int[]{0});
            return torch.tensor(new LongPointer(flat)).view(shape);
        }
        else if (componentType == byte.class) {
            byte[] flat = new byte[totalElements];
            flattenByte(data, flat, new int[]{0});
            return torch.tensor(new BytePointer(flat)).view(shape);
        }
        else if (componentType == short.class) {
            short[] flat = new short[totalElements];
            flattenShort(data, flat, new int[]{0});
            return torch.tensor(new ShortPointer(flat)).view(shape);
        }
        else if (componentType == boolean.class) {
            // 特殊处理 Boolean：转换为 Byte (0/1) 并指定 kBool
            byte[] flat = new byte[totalElements];
            flattenBoolean(data, flat, new int[]{0});

            // 关键：创建 Tensor 时指定 dtype 为 kBool
            TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float));
            // 注意：这里先通过 BytePointer 创建，然后告知 PyTorch 它是 Bool
            // JavaCPP 的 torch.tensor(BytePointer) 默认创建 kByte (uint8/int8)
            // 所以我们需要显式使用工厂方法或 cast

            // 方法 A: 先生成 Byte Tensor，再 cast (推荐，比较稳妥)
            Tensor byteTensor = torch.tensor(new BytePointer(flat));
            return byteTensor.to(torch.ScalarType.Bool).view(shape);

            // 方法 B: 直接使用 blob (需要小心内存生命周期，这里为了安全用方法 A 拷贝一次)
        }
        else if (componentType == String.class) {
            throw new UnsupportedOperationException(
                    "Direct String array to Tensor conversion is not supported in LibTorch C++ API. " +
                            "Please tokenize your strings into integers (longs) or encode them as UTF-8 bytes first."
            );
        }
        else {
            throw new IllegalArgumentException("Unsupported data type: " + componentType.getName());
        }
    }

    // ==========================================================
    //                 私有辅助方法：递归展平
    //   使用 int[] index 包装器来在递归中保持计数，避免 ArrayList 装箱
    // ==========================================================

    private static void flattenFloat(Object array, float[] flat, int[] index) {
        if (array instanceof float[]) {
            float[] arr = (float[]) array;
            System.arraycopy(arr, 0, flat, index[0], arr.length);
            index[0] += arr.length;
        } else {
            Object[] objArr = (Object[]) array;
            for (Object o : objArr) flattenFloat(o, flat, index);
        }
    }

    private static void flattenInt(Object array, int[] flat, int[] index) {
        if (array instanceof int[]) {
            int[] arr = (int[]) array;
            System.arraycopy(arr, 0, flat, index[0], arr.length);
            index[0] += arr.length;
        } else {
            Object[] objArr = (Object[]) array;
            for (Object o : objArr) flattenInt(o, flat, index);
        }
    }

    private static void flattenDouble(Object array, double[] flat, int[] index) {
        if (array instanceof double[]) {
            double[] arr = (double[]) array;
            System.arraycopy(arr, 0, flat, index[0], arr.length);
            index[0] += arr.length;
        } else {
            Object[] objArr = (Object[]) array;
            for (Object o : objArr) flattenDouble(o, flat, index);
        }
    }

    private static void flattenLong(Object array, long[] flat, int[] index) {
        if (array instanceof long[]) {
            long[] arr = (long[]) array;
            System.arraycopy(arr, 0, flat, index[0], arr.length);
            index[0] += arr.length;
        } else {
            Object[] objArr = (Object[]) array;
            for (Object o : objArr) flattenLong(o, flat, index);
        }
    }

    private static void flattenShort(Object array, short[] flat, int[] index) {
        if (array instanceof short[]) {
            short[] arr = (short[]) array;
            System.arraycopy(arr, 0, flat, index[0], arr.length);
            index[0] += arr.length;
        } else {
            Object[] objArr = (Object[]) array;
            for (Object o : objArr) flattenShort(o, flat, index);
        }
    }

    private static void flattenByte(Object array, byte[] flat, int[] index) {
        if (array instanceof byte[]) {
            byte[] arr = (byte[]) array;
            System.arraycopy(arr, 0, flat, index[0], arr.length);
            index[0] += arr.length;
        } else {
            Object[] objArr = (Object[]) array;
            for (Object o : objArr) flattenByte(o, flat, index);
        }
    }

    private static void flattenBoolean(Object array, byte[] flat, int[] index) {
        if (array instanceof boolean[]) {
            boolean[] arr = (boolean[]) array;
            for (boolean b : arr) {
                flat[index[0]++] = b ? (byte) 1 : (byte) 0;
            }
        } else {
            Object[] objArr = (Object[]) array;
            for (Object o : objArr) flattenBoolean(o, flat, index);
        }
    }

    // ==========================================================
    //                 私有辅助方法：Shape 计算
    // ==========================================================

    private static void getShape(Object array, List<Long> shape) {
        if (array.getClass().isArray()) {
            shape.add((long) Array.getLength(array));
            if (Array.getLength(array) > 0) {
                // 递归检查第一个元素来确定下一维的长度
                // 假设数组是规则的（即每一行的长度相同）
                getShape(Array.get(array, 0), shape);
            }
        }
    }

    private static Class<?> getRootComponentType(Class<?> arrayClass) {
        Class<?> type = arrayClass;
        while (type.isArray()) {
            type = type.getComponentType();
        }
        return type;
    }
}