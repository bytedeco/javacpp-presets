package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.ColumnType;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.dataframe.tensor.TensorBridge;

import java.util.*;
import java.util.function.DoubleUnaryOperator;

/**
 * 通用向量数据容器（区别于EmbeddingData/ TensorData）
 * 核心特性：
 * 1. 支持float/double/int三种基础数值类型向量；
 * 2. 提供向量基本运算（加减、点积、L2范数、归一化）；
 * 3. 实现StructuredData接口，支持结构化数据转换；
 * 4. 完善的有效性校验和不可变设计；
 * 5. 适配Arrow格式输出；
 * 6. 与NDArray相互转换，支持NumPy风格操作。
 */
public class VectorData extends AbstractDataValue implements StructuredData {
    private static final long serialVersionUID = 1L;

    // 尝试加载Vector API，如果不可用则使用null
    private static final Object SPECIES;
    static {
        Object species = null;
        try {
            Class<?> doubleVectorClass = Class.forName("jdk.incubator.vector.DoubleVector");
            java.lang.reflect.Field speciesField = doubleVectorClass.getField("SPECIES_PREFERRED");
            species = speciesField.get(null);
        } catch (Exception e) {
            // Vector API 不可用，使用传统实现
            species = null;
        }
        SPECIES = species;
    }

    // 核心数据：不同类型的向量存储（仅一种非空）
    private final float[] floatVector;
    private final double[] doubleVector;
    private final int[] intVector;
    // 向量类型标识（float/double/int）
    private final String vectorType;
    // 向量维度
    private final int dimension;
    // 向量名称（可选，用于标识向量用途）
    private final String vectorName;
    // 缓存：L2范数（避免重复计算）
    private Double normCache;
    // 用于多维向量的形状信息
    private final int[] shape;

    // ========== 构造器：按类型区分 ==========
    /**
     * 构造float类型向量
     */
//    public VectorData(float[] vector){
//       return new  VectorData(vector, "vector");
//    }
    public VectorData(float[] vector, String vectorName) {
        Objects.requireNonNull(vector, "向量数据不能为空");
        if (vector.length == 0) {
            throw new IllegalArgumentException("向量维度不能为0");
        }
        this.floatVector = Arrays.copyOf(vector, vector.length);
        this.doubleVector = null;
        this.intVector = null;
        this.vectorType = "float32";
        this.dimension = vector.length;
        this.vectorName = Optional.ofNullable(vectorName).orElse("unnamed_vector");
        this.shape = new int[]{vector.length};
    }

    /**
     * 构造double类型向量
     */
    public VectorData(double[] vector, String vectorName) {
        Objects.requireNonNull(vector, "向量数据不能为空");
        if (vector.length == 0) {
            throw new IllegalArgumentException("向量维度不能为0");
        }
        this.floatVector = null;
        this.doubleVector = Arrays.copyOf(vector, vector.length);
        this.intVector = null;
        this.vectorType = "float64";
        this.dimension = vector.length;
        this.vectorName = Optional.ofNullable(vectorName).orElse("unnamed_vector");
        this.shape = new int[]{vector.length};
    }

    /**
     * 构造int类型向量
     */
    public VectorData(int[] vector, String vectorName) {
        Objects.requireNonNull(vector, "向量数据不能为空");
        if (vector.length == 0) {
            throw new IllegalArgumentException("向量维度不能为0");
        }
        this.floatVector = null;
        this.doubleVector = null;
        this.intVector = Arrays.copyOf(vector, vector.length);
        this.vectorType = "int32";
        this.dimension = vector.length;
        this.vectorName = Optional.ofNullable(vectorName).orElse("unnamed_vector");
        this.shape = new int[]{vector.length};
    }

    /**
     * 构造多维向量
     */
    public VectorData(double[] data, int[] shape, String vectorName) {
        Objects.requireNonNull(data, "向量数据不能为空");
        Objects.requireNonNull(shape, "形状信息不能为空");
        if (data.length == 0) {
            throw new IllegalArgumentException("向量数据不能为空");
        }

        int totalSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        if (data.length != totalSize) {
            throw new IllegalArgumentException("数据长度与形状不匹配");
        }

        this.floatVector = null;
        this.doubleVector = Arrays.copyOf(data, data.length);
        this.intVector = null;
        this.vectorType = "float64";
        this.dimension = data.length;
        this.vectorName = Optional.ofNullable(vectorName).orElse("unnamed_vector");
        this.shape = Arrays.copyOf(shape, shape.length);
    }

    // ==================== NDArray 转换方法 ====================

    /**
     * 从NDArray创建VectorData
     */
    public static VectorData fromNDArray(NDArray ndarray) {
        return fromNDArray(ndarray, "from_ndarray");
    }

    public static VectorData fromNDArray(NDArray ndarray, String vectorName) {
        Objects.requireNonNull(ndarray, "NDArray不能为空");
        double[] data = ndarray.asDoubleArray();
        int[] ishape = new int[ndarray.shape.length];
        for (int i = 0; i < ndarray.shape.length; i++) ishape[i] = (int) ndarray.shape[i];
        if (ishape.length == 1) {
            return switch (ndarray.dtype) {
                case FLOAT32 -> {
                    float[] f = new float[data.length];
                    for (int i = 0; i < data.length; i++) f[i] = (float) data[i];
                    yield new VectorData(f, vectorName);
                }
                case INT32, INT64 -> {
                    int[] iv = new int[data.length];
                    for (int i = 0; i < data.length; i++) iv[i] = (int) data[i];
                    yield new VectorData(iv, vectorName);
                }
                default -> new VectorData(data, vectorName);
            };
        }
        return new VectorData(data, ishape, vectorName);
    }

    /**
     * 创建全零向量
     *
     * @param size       向量大小
     * @param vectorName 向量名称
     * @return 全零VectorData
     */
    public static VectorData zeros(int size, String vectorName) {
        if (size <= 0) {
            throw new IllegalArgumentException("向量大小必须大于0");
        }
        double[] data = new double[size];
        // 数组默认初始化为0，无需额外操作
        return new VectorData(data, vectorName);
    }

    // ==================== NumPy风格操作方法 ====================

    /**
     * 创建全零向量（默认名称）
     */
    public static VectorData zeros(int size) {
        return zeros(size, "zeros_vector");
    }

    /**
     * 创建多维全零数组
     */
    public static VectorData zeros(int[] shape, String vectorName) {
        int totalSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        double[] data = new double[totalSize];
        return new VectorData(data, shape, vectorName);
    }

    public static VectorData zeros(int[] shape) {
        return zeros(shape, "zeros_array");
    }

    /**
     * 创建全一向量
     */
    public static VectorData ones(int size, String vectorName) {
        if (size <= 0) {
            throw new IllegalArgumentException("向量大小必须大于0");
        }
        double[] data = new double[size];
        Arrays.fill(data, 1.0);
        return new VectorData(data, vectorName);
    }

    public static VectorData ones(int size) {
        return ones(size, "ones_vector");
    }

    /**
     * 创建多维全一数组
     */
    public static VectorData ones(int[] shape, String vectorName) {
        int totalSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        double[] data = new double[totalSize];
        Arrays.fill(data, 1.0);
        return new VectorData(data, shape, vectorName);
    }

    public static VectorData ones(int[] shape) {
        return ones(shape, "ones_array");
    }

    /**
     * 创建单位矩阵（眼睛矩阵）
     */
    public static VectorData eye(int n, String vectorName) {
        if (n <= 0) {
            throw new IllegalArgumentException("矩阵大小必须大于0");
        }
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0; // 对角线元素为1
        }
        return new VectorData(data, new int[]{n, n}, vectorName);
    }

    public static VectorData eye(int n) {
        return eye(n, "identity_matrix");
    }

    /**
     * 创建随机向量（均匀分布 [0, 1)）
     */
    public static VectorData rand(int size, String vectorName) {
        if (size <= 0) {
            throw new IllegalArgumentException("向量大小必须大于0");
        }
        Random random = new Random();
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = random.nextDouble();
        }
        return new VectorData(data, vectorName);
    }

    public static VectorData rand(int size) {
        return rand(size, "random_vector");
    }

    /**
     * 创建多维随机数组
     */
    public static VectorData rand(int[] shape, String vectorName) {
        int totalSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        Random random = new Random();
        double[] data = new double[totalSize];
        for (int i = 0; i < totalSize; i++) {
            data[i] = random.nextDouble();
        }
        return new VectorData(data, shape, vectorName);
    }

    public static VectorData rand(int[] shape) {
        return rand(shape, "random_array");
    }

    /**
     * 创建正态分布随机向量（标准正态分布）
     */
    public static VectorData randn(int size, String vectorName) {
        if (size <= 0) {
            throw new IllegalArgumentException("向量大小必须大于0");
        }
        Random random = new Random();
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = random.nextGaussian();
        }
        return new VectorData(data, vectorName);
    }

    public static VectorData randn(int size) {
        return randn(size, "normal_random_vector");
    }

    /**
     * 创建多维正态分布随机数组
     */
    public static VectorData randn(int[] shape, String vectorName) {
        int totalSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        Random random = new Random();
        double[] data = new double[totalSize];
        for (int i = 0; i < totalSize; i++) {
            data[i] = random.nextGaussian();
        }
        return new VectorData(data, shape, vectorName);
    }

    public static VectorData randn(int[] shape) {
        return randn(shape, "normal_random_array");
    }

    // ==================== NumPy风格初始化方法 ====================

    /**
     * 创建随机整数向量
     */
    public static VectorData randint(int low, int high, int size, String vectorName) {
        if (size <= 0) {
            throw new IllegalArgumentException("向量大小必须大于0");
        }
        if (low >= high) {
            throw new IllegalArgumentException("下界必须小于上界");
        }
        Random random = new Random();
        int[] data = new int[size];
        for (int i = 0; i < size; i++) {
            data[i] = random.nextInt(high - low) + low;
        }
        return new VectorData(data, vectorName);
    }

    public static VectorData randint(int low, int high, int size) {
        return randint(low, high, size, "randint_vector");
    }

    /**
     * 创建多维随机整数数组
     */
    public static VectorData randint(int low, int high, int[] shape, String vectorName) {
        int totalSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        if (low >= high) {
            throw new IllegalArgumentException("下界必须小于上界");
        }
        Random random = new Random();
        double[] data = new double[totalSize];
        for (int i = 0; i < totalSize; i++) {
            data[i] = random.nextInt(high - low) + low;
        }
        return new VectorData(data, shape, vectorName);
    }

    public static VectorData randint(int low, int high, int[] shape) {
        return randint(low, high, shape, "randint_array");
    }

    /**
     * 创建线性序列向量（类似numpy的arange）
     */
    public static VectorData arange(double start, double stop, double step, String vectorName) {
        if (step <= 0) {
            throw new IllegalArgumentException("步长必须大于0");
        }
        if (start >= stop) {
            throw new IllegalArgumentException("起始值必须小于结束值");
        }

        List<Double> values = new ArrayList<>();
        for (double value = start; value < stop; value += step) {
            values.add(value);
        }

        double[] data = values.stream().mapToDouble(Double::doubleValue).toArray();
        return new VectorData(data, vectorName);
    }

    public static VectorData arange(double start, double stop, double step) {
        return arange(start, stop, step, "arange_vector");
    }

    public static VectorData arange(int start, int stop) {
        return arange(start, stop, 1.0, "arange_vector");
    }

    /**
     * 创建等间距向量（类似numpy的linspace）
     */
    public static VectorData linspace(double start, double stop, int num, String vectorName) {
        if (num <= 0) {
            throw new IllegalArgumentException("数量必须大于0");
        }
        if (num == 1) {
            return new VectorData(new double[]{start}, vectorName);
        }

        double[] data = new double[num];
        double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            data[i] = start + i * step;
        }

        return new VectorData(data, vectorName);
    }

    public static VectorData linspace(double start, double stop, int num) {
        return linspace(start, stop, num, "linspace_vector");
    }

    /**
     * 转换为NDArray
     */
    public NDArray toNDArray() {
        double[] data = getAsDoubleArray();
        long[] lshape = new long[shape.length];
        for (int i = 0; i < shape.length; i++) lshape[i] = shape[i];
        // NDArray(double[], long...) always FLOAT64 — sufficient for interchange
        return new NDArray(data, lshape);
    }

    /** Convert to a javacpp-pytorch {@link org.bytedeco.pytorch.Tensor} preserving shape. */
    public org.bytedeco.pytorch.Tensor toTensor() {
        return TensorBridge.toTensor(this);
    }

    /** Build from a torch Tensor (1-D or multi-dim). */
    public static VectorData fromTensor(org.bytedeco.pytorch.Tensor t) {
        return TensorBridge.toVectorData(t);
    }

    public static VectorData fromTensor(org.bytedeco.pytorch.Tensor t, String vectorName) {
        return TensorBridge.toVectorData(t, vectorName);
    }

    /**
     * 获取指定位置的值（多维索引）
     */
    public double get(int... indices) {
        int flatIndex = calculateFlatIndex(indices);
        double[] data = getAsDoubleArray();
        return data[flatIndex];
    }

    /**
     * 设置指定位置的值（返回新的VectorData，保持不可变性）
     */
    public VectorData set(double value, int... indices) {
        double[] newData = getAsDoubleArray();
        int flatIndex = calculateFlatIndex(indices);
        newData[flatIndex] = value;
        return new VectorData(newData, Arrays.copyOf(shape, shape.length), vectorName + "_modified");
    }

    /**
     * 重塑形状（类似numpy.reshape）
     */
    public VectorData reshape(int... newShape) {
        int newTotalSize = Arrays.stream(newShape).reduce(1, (a, b) -> a * b);
        if (newTotalSize != dimension) {
            throw new IllegalArgumentException("新形状的总大小必须与原数据大小一致");
        }

        double[] data = getAsDoubleArray();
        return new VectorData(data, newShape, vectorName + "_reshaped");
    }

    /**
     * 转置（仅支持2D矩阵）
     */
    public VectorData transpose() {
        if (shape.length != 2) {
            throw new UnsupportedOperationException("转置仅支持2D矩阵");
        }

        int rows = shape[0];
        int cols = shape[1];
        double[] data = getAsDoubleArray();
        double[] transposed = new double[data.length];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }

        return new VectorData(transposed, new int[]{cols, rows}, vectorName + "_T");
    }

    /**
     * 元素级别运算
     */
    public VectorData apply(DoubleUnaryOperator function) {
        double[] data = getAsDoubleArray();
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = function.applyAsDouble(data[i]);
        }
        return new VectorData(result, Arrays.copyOf(shape, shape.length), vectorName + "_applied");
    }

    /**
     * 向量加法
     */
    public VectorData add(VectorData other) {
        validateSameShape(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();
        double[] result = new double[thisData.length];

        for (int i = 0; i < thisData.length; i++) {
            result[i] = thisData[i] + otherData[i];
        }

        return new VectorData(result, Arrays.copyOf(shape, shape.length), vectorName + "_add");
    }

    /**
     * 向量减法
     */
    public VectorData subtract(VectorData other) {
        validateSameShape(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();
        double[] result = new double[thisData.length];

        for (int i = 0; i < thisData.length; i++) {
            result[i] = thisData[i] - otherData[i];
        }

        return new VectorData(result, Arrays.copyOf(shape, shape.length), vectorName + "_subtract");
    }

    /**
     * 向量乘法（元素级）
     */
    public VectorData multiply(VectorData other) {
        validateSameShape(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();
        double[] result = new double[thisData.length];

        for (int i = 0; i < thisData.length; i++) {
            result[i] = thisData[i] * otherData[i];
        }

        return new VectorData(result, Arrays.copyOf(shape, shape.length), vectorName + "_multiply");
    }

    /**
     * 向量除法（元素级）
     */
    public VectorData divide(VectorData other) {
        validateSameShape(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();
        double[] result = new double[thisData.length];

        for (int i = 0; i < thisData.length; i++) {
            if (otherData[i] == 0.0) {
                throw new ArithmeticException("除零错误在索引: " + i);
            }
            result[i] = thisData[i] / otherData[i];
        }

        return new VectorData(result, Arrays.copyOf(shape, shape.length), vectorName + "_divide");
    }

    /**
     * 标量运算
     */
    public VectorData add(double scalar) {
        return apply(x -> x + scalar);
    }

    public VectorData multiply(double scalar) {
        return apply(x -> x * scalar);
    }

    /**
     * 聚合运算
     */
    public double sum() {
        double[] data = getAsDoubleArray();
        return Arrays.stream(data).sum();
    }

    public double mean() {
        double[] data = getAsDoubleArray();
        return Arrays.stream(data).average().orElse(0.0);
    }

    public double min() {
        double[] data = getAsDoubleArray();
        return Arrays.stream(data).min().orElse(Double.NaN);
    }

    public double max() {
        double[] data = getAsDoubleArray();
        return Arrays.stream(data).max().orElse(Double.NaN);
    }

    public double std() {
        double[] data = getAsDoubleArray();
        double mean = mean();
        double variance = Arrays.stream(data)
                .map(x -> Math.pow(x - mean, 2))
                .average()
                .orElse(0.0);
        return Math.sqrt(variance);
    }

    /**
     * 矩阵乘法（仅支持2D矩阵）
     */
    public VectorData matmul(VectorData other) {
        if (shape.length != 2 || other.shape.length != 2) {
            throw new IllegalArgumentException("矩阵乘法仅支持2D矩阵");
        }

        int m = shape[0];
        int k = shape[1];
        int n = other.shape[1];

        if (k != other.shape[0]) {
            throw new IllegalArgumentException("矩阵维度不匹配：" + k + " vs " + other.shape[0]);
        }

        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();
        double[] result = new double[m * n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += thisData[i * k + l] * otherData[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        return new VectorData(result, new int[]{m, n}, vectorName + "_matmul");
    }

    // ==================== 相似度计算方法 ====================

    /**
     * 计算余弦相似度
     */
    public double cosineSimilarity(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        double dotProduct = 0.0;
        double thisNorm = 0.0;
        double otherNorm = 0.0;

        for (int i = 0; i < thisData.length; i++) {
            dotProduct += thisData[i] * otherData[i];
            thisNorm += thisData[i] * thisData[i];
            otherNorm += otherData[i] * otherData[i];
        }

        if (thisNorm == 0.0 || otherNorm == 0.0) {
            return 0.0; // 避免除零错误
        }

        return dotProduct / (Math.sqrt(thisNorm) * Math.sqrt(otherNorm));
    }

    /**
     * 计算欧几里得距离
     */
    public double euclideanDistance(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        double sum = 0.0;
        for (int i = 0; i < thisData.length; i++) {
            double diff = thisData[i] - otherData[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    /**
     * 计算曼哈顿距离
     */
    public double manhattanDistance(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        double sum = 0.0;
        for (int i = 0; i < thisData.length; i++) {
            sum += Math.abs(thisData[i] - otherData[i]);
        }

        return sum;
    }

    /**
     * 计算点积
     */
    public double dotProduct(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        double sum = 0.0;
        for (int i = 0; i < thisData.length; i++) {
            sum += thisData[i] * otherData[i];
        }

        return sum;
    }

    /**
     * 计算Jaccard相似度（用于二进制向量）
     */
    public double jaccardSimilarity(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        int intersection = 0;
        int union = 0;

        for (int i = 0; i < thisData.length; i++) {
            boolean thisNonZero = thisData[i] != 0.0;
            boolean otherNonZero = otherData[i] != 0.0;

            if (thisNonZero && otherNonZero) {
                intersection++;
            }
            if (thisNonZero || otherNonZero) {
                union++;
            }
        }

        return union == 0 ? 0.0 : (double) intersection / union;
    }

    /**
     * 计算皮尔逊相关系数
     */
    public double pearsonCorrelation(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        double thisMean = Arrays.stream(thisData).average().orElse(0.0);
        double otherMean = Arrays.stream(otherData).average().orElse(0.0);

        double numerator = 0.0;
        double thisVariance = 0.0;
        double otherVariance = 0.0;

        for (int i = 0; i < thisData.length; i++) {
            double thisDiff = thisData[i] - thisMean;
            double otherDiff = otherData[i] - otherMean;

            numerator += thisDiff * otherDiff;
            thisVariance += thisDiff * thisDiff;
            otherVariance += otherDiff * otherDiff;
        }

        double denominator = Math.sqrt(thisVariance * otherVariance);
        return denominator == 0.0 ? 0.0 : numerator / denominator;
    }

    /**
     * 计算汉明距离（用于二进制向量）
     */
    public int hammingDistance(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        int distance = 0;
        for (int i = 0; i < thisData.length; i++) {
            if (thisData[i] != otherData[i]) {
                distance++;
            }
        }

        return distance;
    }

    /**
     * 计算切比雪夫距离（无穷范数）
     */
    public double chebyshevDistance(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        double maxDiff = 0.0;
        for (int i = 0; i < thisData.length; i++) {
            double diff = Math.abs(thisData[i] - otherData[i]);
            maxDiff = Math.max(maxDiff, diff);
        }

        return maxDiff;
    }

    /**
     * 计算KL散度（Kullback-Leibler散度）
     * 注意：要求向量元素为非负且和为1（概率分布）
     */
    public double klDivergence(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        double kl = 0.0;
        for (int i = 0; i < thisData.length; i++) {
            if (thisData[i] > 0 && otherData[i] > 0) {
                kl += thisData[i] * Math.log(thisData[i] / otherData[i]);
            }
        }

        return kl;
    }

    /**
     * 计算JS散度（Jensen-Shannon散度）
     */
    public double jsDivergence(VectorData other) {
        validateSameDimension(other);
        double[] thisData = getAsDoubleArray();
        double[] otherData = other.getAsDoubleArray();

        // 计算平均分布
        double[] avgData = new double[thisData.length];
        for (int i = 0; i < thisData.length; i++) {
            avgData[i] = (thisData[i] + otherData[i]) / 2.0;
        }

        VectorData avgVector = new VectorData(avgData, "avg_vector");

        double kl1 = this.klDivergence(avgVector);
        double kl2 = other.klDivergence(avgVector);

        return (kl1 + kl2) / 2.0;
    }

    /**
     * 向量归一化（L2范数归一化）
     */
    public VectorData normalize() {
        double norm = calculateNorm();
        if (norm == 0.0) {
            return this; // 零向量无法归一化
        }

        return multiply(1.0 / norm);
    }

    /**
     * 单位化（使向量长度为1）
     */
    public VectorData unit() {
        return normalize();
    }

    /**
     * 计算与另一个向量的夹角（弧度）
     */
    public double angle(VectorData other) {
        double similarity = cosineSimilarity(other);
        // 限制值在[-1, 1]范围内以避免数值误差
        similarity = Math.max(-1.0, Math.min(1.0, similarity));
        return Math.acos(similarity);
    }

    /**
     * 计算与另一个向量的夹角（度数）
     */
    public double angleDegrees(VectorData other) {
        return Math.toDegrees(angle(other));
    }

    /**
     * 检查两个向量是否正交
     */
    public boolean isOrthogonal(VectorData other, double tolerance) {
        double dotProd = dotProduct(other);
        return Math.abs(dotProd) < tolerance;
    }

    public boolean isOrthogonal(VectorData other) {
        return isOrthogonal(other, 1e-10);
    }

    /**
     * 计算向量投影
     */
    public VectorData projectOnto(VectorData other) {
        double dotProd = this.dotProduct(other);
        double otherNormSquared = other.dotProduct(other);

        if (otherNormSquared == 0.0) {
            throw new IllegalArgumentException("无法投影到零向量");
        }

        double scalar = dotProd / otherNormSquared;
        return other.multiply(scalar);
    }

    // ==================== 辅助方法 ====================

    /**
     * 计算平坦索引（多维索引转一维索引）
     */
    private int calculateFlatIndex(int[] indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("索引维度不匹配");
        }

        int flatIndex = 0;
        int stride = 1;

        for (int i = shape.length - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("索引超出范围: " + indices[i]);
            }
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }

        return flatIndex;
    }

    /**
     * 验证形状是否相同
     */
    private void validateSameShape(VectorData other) {
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException(
                    String.format("形状不匹配: %s vs %s",
                            Arrays.toString(this.shape), Arrays.toString(other.shape)));
        }
    }

    /**
     * 获取统一的double数组表示
     */
    public double[] getAsDoubleArray() {
        switch (vectorType) {
            case "float32":
                double[] doubleFromFloat = new double[floatVector.length];
                for (int i = 0; i < floatVector.length; i++) {
                    doubleFromFloat[i] = floatVector[i];
                }
                return doubleFromFloat;
            case "float64":
                return Arrays.copyOf(doubleVector, doubleVector.length);
            case "int32":
                double[] doubleFromInt = new double[intVector.length];
                for (int i = 0; i < intVector.length; i++) {
                    doubleFromInt[i] = intVector[i];
                }
                return doubleFromInt;
            default:
                throw new UnsupportedOperationException("不支持的向量类型: " + vectorType);
        }
    }

    /**
     * 返回向量的 double 数组表示（无拷贝或浅拷贝，取决于内部存储）。
     */
    public double[] toDoubleArray() {
        if (doubleVector != null) {
            return Arrays.copyOf(doubleVector, doubleVector.length);
        } else if (floatVector != null) {
            double[] arr = new double[floatVector.length];
            for (int i = 0; i < floatVector.length; i++) arr[i] = floatVector[i];
            return arr;
        } else if (intVector != null) {
            double[] arr = new double[intVector.length];
            for (int i = 0; i < intVector.length; i++) arr[i] = intVector[i];
            return arr;
        } else {
            return new double[0];
        }
    }

    /**
     * 返回一维向量的长度（元素个数）。
     */
    public int size() {
        if (doubleVector != null) return doubleVector.length;
        if (floatVector != null) return floatVector.length;
        if (intVector != null) return intVector.length;
        return 0;
    }

    /**
     * 获取向量名称，便于调试与上层标注。
     */
    public int getVectorSize() {
        return this.dimension;
    }

    public int[] getShape() {
        return this.shape;
    }

    // ========== 核心向量运算方法 ==========
    /**
     * 计算L2范数（欧几里得范数）
     */
    public double calculateNorm() {
        // 缓存命中，直接返回
        if (normCache != null) {
            return normCache;
        }

        double sum = 0.0;
        switch (vectorType) {
            case "float32":
                for (float v : floatVector) sum += v * v;
                break;
            case "float64":
                for (double v : doubleVector) sum += v * v;
                break;
            case "int32":
                for (int v : intVector) sum += v * v;
                break;
        }
        normCache = Math.sqrt(sum);
        return normCache;
    }

    /**
     * 向量加法（返回新对象，原对象不变）
     */
    public VectorData addVector(VectorData other) {
        // 校验维度和类型
        validateSameTypeAndDim(other);

        switch (vectorType) {
            case "float32":
                float[] sumFloat = new float[dimension];
                for (int i = 0; i < dimension; i++) {
                    sumFloat[i] = this.floatVector[i] + other.floatVector[i];
                }
                return new VectorData(sumFloat, vectorName + "_add");
            case "float64":
                double[] sumDouble = new double[dimension];
                for (int i = 0; i < dimension; i++) {
                    sumDouble[i] = this.doubleVector[i] + other.doubleVector[i];
                }
                return new VectorData(sumDouble, vectorName + "_add");
            case "int32":
                int[] sumInt = new int[dimension];
                for (int i = 0; i < dimension; i++) {
                    sumInt[i] = this.intVector[i] + other.intVector[i];
                }
                return new VectorData(sumInt, vectorName + "_add");
            default:
                throw new UnsupportedOperationException("不支持的向量类型：" + vectorType);
        }
    }

    // ========== 辅助方法 ==========
    /**
     * 校验向量类型和维度是否一致 validateSameTypeAndDim(other)
     */
    private void validateSameDimension(VectorData other) {
        if (this.dimension != other.dimension) {
            throw new IllegalArgumentException("向量维度不匹配");
        }
        if (!this.vectorType.equals(other.vectorType)) {
            throw new IllegalArgumentException("向量类型不匹配");
        }
    }

    private void validateSameTypeAndDim(VectorData other) {
        if (this.dimension != other.dimension) {
            throw new IllegalArgumentException("向量维度不匹配");
        }
        if (!this.vectorType.equals(other.vectorType)) {
            throw new IllegalArgumentException("向量类型不匹配");
        }
    }

    // ========== Getter（返回副本，避免外部修改） ==========
    public float[] getFloatVector() {
        return floatVector == null ? null : Arrays.copyOf(floatVector, floatVector.length);
    }

    public double[] getDoubleVector() {
        return doubleVector == null ? null : Arrays.copyOf(doubleVector, doubleVector.length);
    }

    public int[] getIntVector() {
        return intVector == null ? null : Arrays.copyOf(intVector, intVector.length);
    }

    public String getVectorType() {
        return vectorType;
    }


    @Override
    public int getSize() {
        // 实现StructuredData接口：向量维度即为大小
        return dimension;
    }

    public String getVectorName() {
        return vectorName;
    }

    // ========== 实现AbstractLanceData抽象方法 ==========
    @Override
    public String getDataType() {
        return ColumnType.VECTOR.name(); // 关联ColumnType.VECTOR枚举
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回包含所有核心信息的Map
        Map<String, Object> arrowData = new HashMap<>();
        arrowData.put("vectorType", vectorType);
        arrowData.put("dimension", dimension);
        arrowData.put("vectorName", vectorName);
        arrowData.put("norm", calculateNorm());

        // 根据类型放入对应向量数据
        switch (vectorType) {
            case "float32":
                arrowData.put("data", floatVector);
                break;
            case "float64":
                arrowData.put("data", doubleVector);
                break;
            case "int32":
                arrowData.put("data", intVector);
                break;
        }
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        // 简短描述：类型+维度+名称
        return String.format("type=%s, dim=%d, name=%s",
                vectorType, dimension, vectorName);
    }

    // ========== 重写有效性校验 ==========
    @Override
    public boolean isValid() {
        // 基础校验 + 向量专属校验
        return super.isValid()
                && dimension > 0
                && vectorType != null
                && (floatVector != null || doubleVector != null || intVector != null)
                // 确保只有一种向量数据非空
                && (floatVector != null ? 1 : 0)
                + (doubleVector != null ? 1 : 0)
                + (intVector != null ? 1 : 0) == 1;
    }

    // ========== 实现StructuredData接口 ==========
    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("vectorType", vectorType);
        map.put("dimension", dimension);
        map.put("vectorName", vectorName);
        map.put("norm", calculateNorm());

        // 放入对应类型的向量数据（副本）
        switch (vectorType) {
            case "float32":
                map.put("vector", getFloatVector());
                break;
            case "float64":
                map.put("vector", getDoubleVector());
                break;
            case "int32":
                map.put("vector", getIntVector());
                break;
        }
        return map;
    }

    // ========== 重写通用方法 ==========
    @Override
    public String toString() {
        // 向量预览（最多显示前5个元素）
        String vectorPreview = getVectorPreview();
        return String.format("VectorData[type=%s, dim=%d, name=%s, vector=%s]",
                vectorType, dimension, vectorName, vectorPreview);
    }

//    @Override
//    public String toString() {
//        // 向量转CSV友好格式：用逗号分隔值
//        String vectorPreview = getVectorPreview();
//        return Arrays.stream(floatVector)
//                .mapToObj(String::valueOf)
//                .collect(Collectors.joining(","));
//    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        VectorData that = (VectorData) o;
        return dimension == that.dimension
                && Objects.equals(vectorType, that.vectorType)
                && Objects.equals(vectorName, that.vectorName)
                && Arrays.equals(floatVector, that.floatVector)
                && Arrays.equals(doubleVector, that.doubleVector)
                && Arrays.equals(intVector, that.intVector);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(vectorType, dimension, vectorName);
        result = 31 * result + Arrays.hashCode(floatVector);
        result = 31 * result + Arrays.hashCode(doubleVector);
        result = 31 * result + Arrays.hashCode(intVector);
        return result;
    }

    /**
     * 向量预览（最多显示前5个元素）
     */
    private String getVectorPreview() {
        List<Object> preview = new ArrayList<>();
        int previewSize = Math.min(dimension, 5);

        switch (vectorType) {
            case "float32":
                for (int i = 0; i < previewSize; i++) {
                    preview.add(floatVector[i]);
                }
                break;
            case "float64":
                for (int i = 0; i < previewSize; i++) {
                    preview.add(doubleVector[i]);
                }
                break;
            case "int32":
                for (int i = 0; i < previewSize; i++) {
                    preview.add(intVector[i]);
                }
                break;
        }

        if (dimension > 5) {
            preview.add("...");
        }
        return preview.toString();
    }

    @Override
    public Number getNumericValue(){
        return null;
    }
}












