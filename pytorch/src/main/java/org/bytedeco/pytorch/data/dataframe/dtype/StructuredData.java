package org.bytedeco.pytorch.data.dataframe.dtype;

/**
 * 结构化数据（结构体/JSON/嵌入向量）通用接口
 */
public interface StructuredData extends DataValue {

    /**
     * 获取数据维度/字段数量
     * @return 维度/字段数
     */
    int getSize();

    /**
     * 转换为Map格式（便于序列化/解析）
     * @return Map格式数据
     */
    java.util.Map<String, Object> toMap();
}