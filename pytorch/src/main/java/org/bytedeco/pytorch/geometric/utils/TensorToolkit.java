package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

public class TensorToolkit {

    // Get Multidimensional array Shape
    public static long[] getShape(Object array) {
        if (array == null || !array.getClass().isArray()) return new long[0];
        List<Long> shape = new ArrayList<>();
        Object current = array;
        while (current != null && current.getClass().isArray()) {
            int len = Array.getLength(current);
            shape.add((long) len);
            current = (len > 0) ? Array.get(current, 0) : null;
        }
        return shape.stream().mapToLong(i -> i).toArray();
    }

    // flatten Multidimensional array
    public static Object flatten(Object multiDimArray) {
        if (multiDimArray == null) return null;
        Class<?> componentType = multiDimArray.getClass();
        while (componentType.isArray()) componentType = componentType.getComponentType();

        int size = 1;
        for (long d : getShape(multiDimArray)) size *= d;

        Object result = Array.newInstance(componentType, size);
        fill(multiDimArray, result, new int[]{0});
        return result;
    }

    public static void fill(Object src, Object dest, int[] idx) {
        int len = Array.getLength(src);
        for (int i = 0; i < len; i++) {
            Object el = Array.get(src, i);
            if (el != null && el.getClass().isArray()) fill(el, dest, idx);
            else Array.set(dest, idx[0]++, el);
        }
    }

    /**
     * 辅助方法：将张量形状（SizeTPointer）转为易读字符串
     */
    public static String tensorShapeToString(SizeTPointer sizes) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < sizes.limit(); i++) {
            sb.append(sizes.get(i));
            if (i < sizes.limit() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * 辅助方法：打印张量形状（模拟Python的tensor.shape输出格式）
     */
    public static void printTensorShape(Tensor tensor) {
        LongVector sizes = tensor.sizes().vec();
        System.out.print("[");
        for (int i = 0; i < sizes.size(); i++) {
            System.out.print(sizes.get(i));
            if (i < sizes.size() - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
        sizes.close();
    }

    // 辅助方法：获取张量形状的字符串表示
    private static String getShapeString(Tensor tensor) {
        long[] sizes = tensor.sizes().vec().get();
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < sizes.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(sizes[i]);
        }
        sb.append(")");
        return sb.toString();
    }

    /**
     * 方式1：一对一转换（每个boolean对应一个byte）
     * true → 1, false → 0
     * @param boolArray 待转换的boolean数组
     * @return 转换后的byte数组
     * @throws NullPointerException 如果输入数组为null
     */
    public static byte[] booleanToByteArrays(boolean[] boolArray) {
        // 空值校验
        if (boolArray == null) {
            throw new NullPointerException("boolean数组不能为null");
        }

        byte[] byteArray = new byte[boolArray.length];
        for (int i = 0; i < boolArray.length; i++) {
            // 布尔值转byte：true→1，false→0
            byteArray[i] = boolArray[i] ? (byte) 1 : (byte) 0;
        }
        return byteArray;
    }


    public static float[][] reshape(float[] flatArray, long[] shape) {
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        float[][] result = new float[rows][cols];
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = flatArray[idx++];
            }
        }
        return result;
    }

    public static Tensor ensure2D(Tensor tensor) {
        if (tensor == null) return null;
        if (tensor.dim() == 1) {
            return tensor.view(new long[]{tensor.size(0), 1});
        }
        return tensor;
    }
    
}
