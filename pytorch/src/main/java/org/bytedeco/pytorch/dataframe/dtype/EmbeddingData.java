package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.ColumnType;
import org.bytedeco.pytorch.dataframe.tensor.TensorBridge;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 嵌入向量容器（最终修复版）
 * 核心改进：
 * 1. 新增 isNormalizedFlag 标记当前对象是否为归一化向量；
 * 2. 归一化后新对象直接标记为已归一化，原对象不修改状态；
 * 3. 分离“归一化计算缓存”和“当前对象状态”。
 */
public class EmbeddingData extends AbstractDataValue implements StructuredData{
    private static final long serialVersionUID = 1L;

    // 核心数据：当前对象存储的向量（原始/归一化）
    private final float[] vector;
    // 向量维度
    private final int dimension;
    // 模型名称
    private final String modelName;
    // 关键：标记当前对象是否为归一化向量（解决核心问题）
    private final boolean isNormalizedFlag;
    // 缓存：原向量的归一化结果（仅原对象使用）
    private float[] normalizedVectorCache;

    // 构造器1：创建原始向量对象（默认未归一化）
    public EmbeddingData(float[] vector, String modelName) {
        this.vector = Arrays.copyOf(Objects.requireNonNull(vector), vector.length);
        this.dimension = vector.length;
        this.modelName = modelName;
        this.isNormalizedFlag = false; // 原始向量默认未归一化
        if (dimension == 0) {
            throw new IllegalArgumentException("嵌入向量维度不能为0");
        }
    }

    // 构造器2：创建归一化向量对象（私有，仅内部normalize()使用）
    private EmbeddingData(float[] normalizedVector, String modelName, boolean isNormalized) {
        this.vector = Arrays.copyOf(normalizedVector, normalizedVector.length);
        this.dimension = normalizedVector.length;
        this.modelName = modelName;
        this.isNormalizedFlag = isNormalized; // 直接标记为已归一化
        this.normalizedVectorCache = null; // 归一化对象无需缓存
    }

    /**
     * 归一化（L2归一化）：
     * 1. 原对象计算并缓存归一化向量（不修改自身状态）；
     * 2. 返回新的“已归一化”对象；
     */
    public EmbeddingData normalize() {
        // 若已缓存归一化向量，直接返回新对象
        if (normalizedVectorCache != null) {
            return new EmbeddingData(normalizedVectorCache, modelName, true);
        }
        // 计算L2归一化
        float sum = 0.0f;
        for (float v : vector) sum += v * v;
        float norm = (float) Math.sqrt(sum);
        if (norm == 0) {
            normalizedVectorCache = Arrays.copyOf(vector, vector.length);
            return new EmbeddingData(normalizedVectorCache, modelName, true);
        }
        // 计算归一化向量并缓存（原对象缓存，不修改自身状态）
        normalizedVectorCache = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            normalizedVectorCache[i] = vector[i] / norm;
        }
        // 返回新的“已归一化”对象
        return new EmbeddingData(normalizedVectorCache, modelName, true);
    }

    /**
     * 计算余弦相似度（使用当前对象的向量）
     */
    public float cosineSimilarity(EmbeddingData other) {
        if (this.dimension != other.dimension) {
            throw new IllegalArgumentException("向量维度不匹配：" + this.dimension + " vs " + other.dimension);
        }
        float dotProduct = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (int i = 0; i < dimension; i++) {
            dotProduct += this.vector[i] * other.vector[i];
            norm1 += this.vector[i] * this.vector[i];
            norm2 += other.vector[i] * other.vector[i];
        }
        if (norm1 == 0 || norm2 == 0) return 0.0f;
        return (float) (dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2)));
    }

    /**
     * Arrow适配：返回当前对象的向量（原始/归一化）
     */
    public ListViewData toArrowFixedSizeList() {
        java.util.List<Object> vectorList = new java.util.ArrayList<>();
        for (float v : vector) vectorList.add(v);
        return new ListViewData(vectorList, ColumnType.FLOAT64, 0, dimension);
    }

    // ========== 核心修复：准确的状态判断 ==========
    public boolean isNormalized() {
        return isNormalizedFlag; // 直接返回标记，不再依赖缓存
    }

    // Getter（返回副本，避免外部修改）
    public float[] getVector() { return Arrays.copyOf(vector, vector.length); }
    public int getDimension() { return dimension; }
    public String getModelName() { return modelName; }

    /** Convert embedding to a 1-D float javacpp-pytorch {@link org.bytedeco.pytorch.Tensor}. */
    public org.bytedeco.pytorch.Tensor toTensor() {
        return TensorBridge.toTensor(this);
    }

    /** Build from a torch Tensor (flattened to 1-D float). */
    public static EmbeddingData fromTensor(org.bytedeco.pytorch.Tensor t, String modelName) {
        return TensorBridge.toEmbeddingData(t, modelName);
    }

    public static EmbeddingData fromTensor(org.bytedeco.pytorch.Tensor t) {
        return fromTensor(t, "tensor");
    }

    @Override
    public String toString() {
        String vectorPreview = dimension > 5 ?
                Arrays.toString(Arrays.copyOf(vector, 5)) + "..." : Arrays.toString(vector);
        return String.format("EmbeddingData[dim=%d, model=%s, normalized=%s, vector=%s]",
                dimension, modelName, isNormalizedFlag, vectorPreview);
    }

    @Override
    public String getShortDesc() {
        String vectorPreview = dimension > 5 ?
                Arrays.toString(Arrays.copyOf(vector, 5)) + "..." : Arrays.toString(vector);
        return String.format("dim=%d, model=%s, normalized=%s, vector=%s",
                dimension, modelName, isNormalizedFlag, vectorPreview);
    }

    // ========== 重写有效性校验 ==========
    @Override
    public boolean isValid() {
        return super.isValid() && vector != null && dimension > 0;
    }

    @Override
    public String getDataType() {
        return "EMBEDDING";
    }

    @Override
    public Object toArrowCompatible() {
        // 转换为Arrow FixedSizeList格式
        return toArrowFixedSizeList();
    }

    // ========== 实现StructuredData接口 ==========
    @Override
    public int getSize() {
        return dimension; // 向量维度即为大小
    }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("vector", vector);
        map.put("dimension", dimension);
        map.put("modelName", modelName);
        map.put("isNormalized", isNormalizedFlag);
        return map;
    }

    @Override
    public Number getNumericValue(){
        return null;
    }


}

//package lance.dtype;
//
// import org.bytedeco.pytorch.dataframe.enums.ColumnType;
//
//import java.io.Serializable;
//import java.util.Arrays;
//import java.util.Objects;
//
///**
// * 嵌入向量容器（适配 Arrow FixedSizeList/Float64 类型）
// * 修复：1. 保留原始向量，归一化返回新对象（不修改原向量）；2. toArrowFixedSizeList 返回原始向量
// */
//public class EmbeddingData implements Serializable {
//    private static final long serialVersionUID = 1L;
//
//    // 原始嵌入向量数据（浮点数组，主流为float32/float64）
//    private final float[] originalVector;
//    // 向量维度
//    private final int dimension;
//    // 模型名称（生成该嵌入的模型）
//    private final String modelName;
//    // 归一化后的向量（缓存，懒加载）
//    private float[] normalizedVector;
//
//    // 构造：原始向量 + 模型名
//    public EmbeddingData(float[] vector, String modelName) {
//        this.originalVector = Arrays.copyOf(Objects.requireNonNull(vector), vector.length);
//        this.dimension = vector.length;
//        this.modelName = modelName;
//        if (dimension == 0) {
//            throw new IllegalArgumentException("嵌入向量维度不能为0");
//        }
//    }
//
//    /**
//     * 归一化（L2归一化）：返回新对象，不修改原始向量
//     */
//    public EmbeddingData normalize() {
//        // 若已计算过归一化向量，直接返回新对象
//        if (normalizedVector != null) {
//            return new EmbeddingData(normalizedVector, modelName);
//        }
//        // 计算L2归一化
//        float sum = 0.0;
//        for (float v : originalVector) sum += v * v;
//        float norm = Math.sqrt(sum);
//        if (norm == 0) {
//            normalizedVector = Arrays.copyOf(originalVector, originalVector.length);
//            return new EmbeddingData(normalizedVector, modelName);
//        }
//        // 计算归一化向量（不修改原始向量）
//        normalizedVector = new float[dimension];
//        for (int i = 0; i < dimension; i++) {
//            normalizedVector[i] = originalVector[i] / norm;
//        }
//        return new EmbeddingData(normalizedVector, modelName);
//    }
//
//    /**
//     * 计算余弦相似度（和另一个嵌入向量）
//     * @param other 另一个嵌入向量
//     * @return 余弦相似度
//     */
//    public float cosineSimilarity(EmbeddingData other) {
//        if (this.dimension != other.dimension) {
//            throw new IllegalArgumentException("向量维度不匹配：" + this.dimension + " vs " + other.dimension);
//        }
//        // 使用原始向量计算相似度
//        float dotProduct = 0.0;
//        float norm1 = 0.0;
//        float norm2 = 0.0;
//        for (int i = 0; i < dimension; i++) {
//            dotProduct += this.originalVector[i] * other.originalVector[i];
//            norm1 += this.originalVector[i] * this.originalVector[i];
//            norm2 += other.originalVector[i] * other.originalVector[i];
//        }
//        if (norm1 == 0 || norm2 == 0) return 0.0;
//        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
//    }
//
//    /**
//     * Arrow适配：转换为FixedSizeList（返回原始向量，非归一化）
//     */
//    public ListViewData toArrowFixedSizeList() {
//        // 转换float[]为List<Object>（原始向量）
//        java.util.List<Object> vectorList = new java.util.ArrayList<>();
//        for (float v : originalVector) vectorList.add(v);
//        return new ListViewData(vectorList, ColumnType.FLOAT64, 0, dimension);
//    }
//
//    // Getter
//    public float[] getVector() { return Arrays.copyOf(originalVector, originalVector.length); }
//    public float[] getNormalizedVector() {
//        if (normalizedVector == null) {
//            normalize(); // 触发归一化计算
//        }
//        return Arrays.copyOf(normalizedVector, normalizedVector.length);
//    }
//    public int getDimension() { return dimension; }
//    public String getModelName() { return modelName; }
//    public boolean isNormalized() { return normalizedVector != null; }
//
//    @Override
//    public String toString() {
//        String vectorPreview = dimension > 5 ?
//                Arrays.toString(Arrays.copyOf(originalVector, 5)) + "..." : Arrays.toString(originalVector);
//        return String.format("EmbeddingData[dim=%d, model=%s, normalized=%s, vector=%s]",
//                dimension, modelName, isNormalized(), vectorPreview);
//    }
//}

//package lance.dtype;
//
// import org.bytedeco.pytorch.dataframe.enums.ColumnType;
//
//import java.io.Serializable;
//import java.util.Arrays;
//import java.util.Objects;
//
///**
// * 嵌入向量容器（适配 Arrow FixedSizeList/Float64 类型）
// * 用于AI模型的向量输出，支持维度、相似度计算、归一化
// */
//public class EmbeddingData implements Serializable {
//    private static final long serialVersionUID = 1L;
//
//    // 嵌入向量数据（浮点数组，主流为float32/float64）
//    private float[] vector;
//    // 向量维度
//    private int dimension;
//    // 模型名称（生成该嵌入的模型）
//    private String modelName;
//    // 归一化标志
//    private boolean normalized = false;
//
//    // 构造：原始向量 + 模型名
//    public EmbeddingData(float[] vector, String modelName) {
//        this.vector = Arrays.copyOf(Objects.requireNonNull(vector), vector.length);
//        this.dimension = vector.length;
//        this.modelName = modelName;
//        if (dimension == 0) {
//            throw new IllegalArgumentException("嵌入向量维度不能为0");
//        }
//    }
//
//    // 归一化（L2归一化，AI场景常用）
//    public EmbeddingData normalize() {
//        if (normalized) return this;
//        float sum = 0.0;
//        for (float v : vector) sum += v * v;
//        float norm = Math.sqrt(sum);
//        if (norm == 0) return this;
//        for (int i = 0; i < vector.length; i++) {
//            vector[i] /= norm;
//        }
//        this.normalized = true;
//        return this;
//    }
//
//    // 计算余弦相似度（和另一个嵌入向量）
//    public float cosineSimilarity(EmbeddingData other) {
//        if (this.dimension != other.dimension) {
//            throw new IllegalArgumentException("向量维度不匹配：" + this.dimension + " vs " + other.dimension);
//        }
//        float dotProduct = 0.0;
//        float norm1 = 0.0;
//        float norm2 = 0.0;
//        for (int i = 0; i < dimension; i++) {
//            dotProduct += this.vector[i] * other.vector[i];
//            norm1 += this.vector[i] * this.vector[i];
//            norm2 += other.vector[i] * other.vector[i];
//        }
//        if (norm1 == 0 || norm2 == 0) return 0.0;
//        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
//    }
//
//    // Arrow适配：转换为FixedSizeList（Arrow推荐的嵌入存储方式）
//    public ListViewData toArrowFixedSizeList() {
//        // 转换float[]为List<Object>
//        java.util.List<Object> vectorList = new java.util.ArrayList<>();
//        for (float v : vector) vectorList.add(v);
//        return new ListViewData(vectorList, ColumnType.FLOAT64, 0, dimension);
//    }
//
//    // Getter
//    public float[] getVector() { return Arrays.copyOf(vector, vector.length); }
//    public int getDimension() { return dimension; }
//    public String getModelName() { return modelName; }
//    public boolean isNormalized() { return normalized; }
//
//    @Override
//    public String toString() {
//        String vectorPreview = dimension > 5 ?
//                Arrays.toString(Arrays.copyOf(vector, 5)) + "..." : Arrays.toString(vector);
//        return String.format("EmbeddingData[dim=%d, model=%s, normalized=%s, vector=%s]",
//                dimension, modelName, normalized, vectorPreview);
//    }
//}