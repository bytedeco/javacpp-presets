package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.ColumnType;

import java.util.*;

/**
 * 结构化数据容器（适配 Arrow StructType）
 * 最终修复：1. 区分类型定义和值设置；2. addField 校验类型一致性；3. 明确异常场景
 */
public class StructData  extends AbstractDataValue implements StructuredData {
    private static final long serialVersionUID = 1L;

    // 字段元信息：字段名 -> 字段类型（对齐ColumnType）
    private final Map<String, ColumnType> fieldTypes = new LinkedHashMap<>();
    // 字段值：字段名 -> 字段值（支持嵌套StructData/其他数据类型）
    private final Map<String, Object> fieldValues = new LinkedHashMap<>();
    // 结构名称（可选，用于Arrow schema标识）
    private String structName;

    // 空构造（Arrow反序列化用）
    public StructData() {}

    // 基础构造：指定结构名称 + 字段元信息（仅初始化类型，不初始化值）
    public StructData(String structName, Map<String, ColumnType> fieldTypes) {
        this.structName = Objects.requireNonNull(structName, "结构名称不能为空");
        this.fieldTypes.putAll(Objects.requireNonNull(fieldTypes, "字段类型映射不能为空"));
    }

    // 完整构造：指定字段名、类型、初始值
    public StructData(String structName, List<String> fieldNames, List<ColumnType> fieldTypes, List<Object> fieldValues) {
        this.structName = Objects.requireNonNull(structName);
        if (fieldNames.size() != fieldTypes.size() || fieldNames.size() != fieldValues.size()) {
            throw new IllegalArgumentException("字段名、类型、值的数量必须一致");
        }
        for (int i = 0; i < fieldNames.size(); i++) {
            String fieldName = fieldNames.get(i);
            ColumnType fieldType = fieldTypes.get(i);
            Object fieldValue = fieldValues.get(i);
            this.fieldTypes.put(fieldName, fieldType);
            this.fieldValues.put(fieldName, fieldValue);
        }
    }

    /**
     * 设置字段值（仅校验字段是否已定义类型，不校验值类型）
     * @param fieldName 字段名（必须已在fieldTypes中定义）
     * @param fieldValue 字段值
     */
    public void setFieldValue(String fieldName, Object fieldValue) {
        Objects.requireNonNull(fieldName, "字段名不能为空");
        // 校验字段是否已定义类型
        if (!fieldTypes.containsKey(fieldName)) {
            throw new IllegalArgumentException("字段 " + fieldName + " 未定义类型，无法设置值");
        }
        // 直接设置值（允许覆盖）
        fieldValues.put(fieldName, fieldValue);
    }

    /**
     * 添加字段（类型+值）：
     * 1. 字段未定义：添加类型 + 设置值；
     * 2. 字段已定义：校验类型一致性，不一致则抛异常，一致则覆盖值
     */
    public void addField(String fieldName, ColumnType fieldType, Object fieldValue) {
        Objects.requireNonNull(fieldName, "字段名不能为空");
        Objects.requireNonNull(fieldType, "字段类型不能为空");

        // 场景1：字段未定义类型 -> 添加类型 + 设置值
        if (!fieldTypes.containsKey(fieldName)) {
            fieldTypes.put(fieldName, fieldType);
            fieldValues.put(fieldName, fieldValue);
        }
        // 场景2：字段已定义类型 -> 校验类型一致性
        else {
            ColumnType existingType = fieldTypes.get(fieldName);
            if (!existingType.equals(fieldType)) {
                throw new IllegalArgumentException(
                        "字段 " + fieldName + " 类型不匹配：已定义 " + existingType + "，传入 " + fieldType);
            }
            // 类型一致 -> 覆盖值
            fieldValues.put(fieldName, fieldValue);
        }
    }

    // 获取字段值（类型安全）
    @SuppressWarnings("unchecked")
    public <T> T getFieldValue(String fieldName) {
        if (!fieldTypes.containsKey(fieldName)) {
            throw new IllegalArgumentException("字段 " + fieldName + " 不存在");
        }
        return (T) fieldValues.get(fieldName);
    }

    // Arrow类型适配：转换为Arrow StructType的字段列表描述
    public Map<String, String> toArrowFieldDesc() {
        Map<String, String> desc = new LinkedHashMap<>();
        fieldTypes.forEach((name, type) -> desc.put(name, type.name().toLowerCase()));
        return desc;
    }

    // Getter & Setter
    public String getStructName() { return structName; }
    public void setStructName(String structName) { this.structName = structName; }
    public Set<String> getFieldNames() { return new LinkedHashSet<>(fieldTypes.keySet()); }
    public Map<String, ColumnType> getFieldTypes() { return Collections.unmodifiableMap(fieldTypes); }
    public Map<String, Object> getFieldValues() { return Collections.unmodifiableMap(fieldValues); }
    public int getFieldCount() { return fieldTypes.size(); }

    @Override
    public String getDataType() {
        return "STRUCT";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回包含字段类型和值的Map
        Map<String, Object> arrowData = new LinkedHashMap<>();
        arrowData.put("structName", structName);
        arrowData.put("fieldDescriptions", toArrowFieldDesc());
        arrowData.put("fieldValues", fieldValues);
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        return String.format("name=%s, fields=%d", structName, getFieldCount());
    }

    @Override
    public boolean isValid() {
        // 基础校验 + Struct专属校验：字段类型和值的数量一致
        return super.isValid()
                && structName != null
                && fieldTypes.size() == fieldValues.size();
    }

    // ========== 实现 StructuredData 接口 ==========
    @Override
    public int getSize() {
        // Struct大小：字段数量
        return getFieldCount();
    }

    @Override
    public Number getNumericValue(){
        return null;
    }
    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("structName", structName);
        map.put("fieldTypes", fieldTypes);
        map.put("fieldValues", fieldValues);
        return map;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("StructData[name=").append(structName).append(", fields=[");
        fieldTypes.forEach((name, type) -> {
            Object value = fieldValues.get(name);
            sb.append(name).append(":").append(type).append("=").append(value).append(", ");
        });
        if (!fieldTypes.isEmpty()) {
            sb.delete(sb.length() - 2, sb.length());
        }
        sb.append("]]");
        return sb.toString();
    }
}

//
//package lance.dtype;
//
// import org.bytedeco.pytorch.dataframe.enums.ColumnType;
//import java.io.Serializable;
//import java.util.*;
//
///**
// * 结构化数据容器（适配 Arrow StructType）
// * 修复：构造函数仅初始化字段类型，addField 仅设置值（不再校验类型存在性）
// */
//public class StructData implements Serializable {
//    private static final long serialVersionUID = 1L;
//
//    // 字段元信息：字段名 -> 字段类型（对齐ColumnType）
//    private final Map<String, ColumnType> fieldTypes = new LinkedHashMap<>();
//    // 字段值：字段名 -> 字段值（支持嵌套StructData/其他数据类型）
//    private final Map<String, Object> fieldValues = new LinkedHashMap<>();
//    // 结构名称（可选，用于Arrow schema标识）
//    private String structName;
//
//    // 空构造（Arrow反序列化用）
//    public StructData() {}
//
//    // 基础构造：指定结构名称 + 字段元信息（仅初始化类型，不初始化值）
//    public StructData(String structName, Map<String, ColumnType> fieldTypes) {
//        this.structName = Objects.requireNonNull(structName, "结构名称不能为空");
//        this.fieldTypes.putAll(Objects.requireNonNull(fieldTypes, "字段类型映射不能为空"));
//    }
//
//    // 完整构造：指定字段名、类型、初始值
//    public StructData(String structName, List<String> fieldNames, List<ColumnType> fieldTypes, List<Object> fieldValues) {
//        this.structName = Objects.requireNonNull(structName);
//        if (fieldNames.size() != fieldTypes.size() || fieldNames.size() != fieldValues.size()) {
//            throw new IllegalArgumentException("字段名、类型、值的数量必须一致");
//        }
//        for (int i = 0; i < fieldNames.size(); i++) {
//            String fieldName = fieldNames.get(i);
//            ColumnType fieldType = fieldTypes.get(i);
//            Object fieldValue = fieldValues.get(i);
//            this.fieldTypes.put(fieldName, fieldType);
//            this.fieldValues.put(fieldName, fieldValue);
//        }
//    }
//
//    /**
//     * 修复核心：添加/设置字段值（不再校验类型是否存在，仅校验字段是否已定义类型）
//     * @param fieldName 字段名（必须已在fieldTypes中定义）
//     * @param fieldValue 字段值
//     */
//    public void setFieldValue(String fieldName, Object fieldValue) {
//        Objects.requireNonNull(fieldName, "字段名不能为空");
//        // 校验字段是否已定义类型
//        if (!fieldTypes.containsKey(fieldName)) {
//            throw new IllegalArgumentException("字段 " + fieldName + " 未定义类型，无法设置值");
//        }
//        // 直接设置值（允许覆盖）
//        fieldValues.put(fieldName, fieldValue);
//    }
//
//    // 兼容旧方法：保留addField，内部调用setFieldValue
//    @Deprecated
//    public void addField(String fieldName, ColumnType fieldType, Object fieldValue) {
//        // 若类型未定义，先添加类型（兼容旧逻辑）
//        if (!fieldTypes.containsKey(fieldName)) {
//            fieldTypes.put(fieldName, fieldType);
//        }
//        setFieldValue(fieldName, fieldValue);
//    }
//
//    // 获取字段值（类型安全）
//    @SuppressWarnings("unchecked")
//    public <T> T getFieldValue(String fieldName) {
//        if (!fieldTypes.containsKey(fieldName)) {
//            throw new IllegalArgumentException("字段 " + fieldName + " 不存在");
//        }
//        return (T) fieldValues.get(fieldName);
//    }
//
//    // Arrow类型适配：转换为Arrow StructType的字段列表描述
//    public Map<String, String> toArrowFieldDesc() {
//        Map<String, String> desc = new LinkedHashMap<>();
//        fieldTypes.forEach((name, type) -> desc.put(name, type.name().toLowerCase()));
//        return desc;
//    }
//
//    // Getter & Setter
//    public String getStructName() { return structName; }
//    public void setStructName(String structName) { this.structName = structName; }
//    public Set<String> getFieldNames() { return new LinkedHashSet<>(fieldTypes.keySet()); }
//    public Map<String, ColumnType> getFieldTypes() { return Collections.unmodifiableMap(fieldTypes); }
//    public Map<String, Object> getFieldValues() { return Collections.unmodifiableMap(fieldValues); }
//    public int getFieldCount() { return fieldTypes.size(); }
//
//    @Override
//    public String toString() {
//        StringBuilder sb = new StringBuilder();
//        sb.append("StructData[name=").append(structName).append(", fields=[");
//        fieldTypes.forEach((name, type) -> {
//            Object value = fieldValues.get(name);
//            sb.append(name).append(":").append(type).append("=").append(value).append(", ");
//        });
//        if (!fieldTypes.isEmpty()) {
//            sb.delete(sb.length() - 2, sb.length());
//        }
//        sb.append("]]");
//        return sb.toString();
//    }
//}
//package lance.dtype;
//
// import org.bytedeco.pytorch.dataframe.enums.ColumnType;
//import java.io.Serializable;
//import java.util.*;
//
///**
// * 结构化数据容器（适配 Arrow StructType）
// * 包含字段名-字段类型-字段值的映射，支持嵌套结构
// */
//public class StructData implements Serializable {
//    private static final long serialVersionUID = 1L;
//
//    // 字段元信息：字段名 -> 字段类型（对齐ColumnType）
//    private final Map<String, ColumnType> fieldTypes = new LinkedHashMap<>();
//    // 字段值：字段名 -> 字段值（支持嵌套StructData/其他数据类型）
//    private final Map<String, Object> fieldValues = new LinkedHashMap<>();
//    // 结构名称（可选，用于Arrow schema标识）
//    private String structName;
//
//    // 空构造（Arrow反序列化用）
//    public StructData() {}
//
//    // 基础构造：指定结构名称 + 字段元信息
//    public StructData(String structName, Map<String, ColumnType> fieldTypes) {
//        this.structName = Objects.requireNonNull(structName, "结构名称不能为空");
//        this.fieldTypes.putAll(Objects.requireNonNull(fieldTypes, "字段类型映射不能为空"));
//    }
//
//    // 完整构造：指定字段名、类型、初始值
//    public StructData(String structName, List<String> fieldNames, List<ColumnType> fieldTypes, List<Object> fieldValues) {
//        this.structName = Objects.requireNonNull(structName);
//        if (fieldNames.size() != fieldTypes.size() || fieldNames.size() != fieldValues.size()) {
//            throw new IllegalArgumentException("字段名、类型、值的数量必须一致");
//        }
//        for (int i = 0; i < fieldNames.size(); i++) {
//            String fieldName = fieldNames.get(i);
//            ColumnType fieldType = fieldTypes.get(i);
//            Object fieldValue = fieldValues.get(i);
//            this.fieldTypes.put(fieldName, fieldType);
//            this.fieldValues.put(fieldName, fieldValue);
//        }
//    }
//
//    // 添加字段（支持动态扩展）
//    public void addField(String fieldName, ColumnType fieldType, Object fieldValue) {
//        Objects.requireNonNull(fieldName, "字段名不能为空");
//        Objects.requireNonNull(fieldType, "字段类型不能为空");
//        if (fieldTypes.containsKey(fieldName)) {
//            throw new IllegalArgumentException("字段 " + fieldName + " 已存在");
//        }
//        fieldTypes.put(fieldName, fieldType);
//        fieldValues.put(fieldName, fieldValue);
//    }
//
//    // 获取字段值（类型安全）
//    @SuppressWarnings("unchecked")
//    public <T> T getFieldValue(String fieldName) {
//        if (!fieldTypes.containsKey(fieldName)) {
//            throw new IllegalArgumentException("字段 " + fieldName + " 不存在");
//        }
//        return (T) fieldValues.get(fieldName);
//    }
//
//    // Arrow类型适配：转换为Arrow StructType的字段列表描述
//    public Map<String, String> toArrowFieldDesc() {
//        Map<String, String> desc = new LinkedHashMap<>();
//        fieldTypes.forEach((name, type) -> desc.put(name, type.name().toLowerCase()));
//        return desc;
//    }
//
//    // Getter & Setter
//    public String getStructName() { return structName; }
//    public void setStructName(String structName) { this.structName = structName; }
//    public Set<String> getFieldNames() { return new LinkedHashSet<>(fieldTypes.keySet()); }
//    public Map<String, ColumnType> getFieldTypes() { return Collections.unmodifiableMap(fieldTypes); }
//    public Map<String, Object> getFieldValues() { return Collections.unmodifiableMap(fieldValues); }
//    public int getFieldCount() { return fieldTypes.size(); }
//
//    @Override
//    public String toString() {
//        StringBuilder sb = new StringBuilder();
//        sb.append("StructData[name=").append(structName).append(", fields=[");
//        fieldTypes.forEach((name, type) -> {
//            Object value = fieldValues.get(name);
//            sb.append(name).append(":").append(type).append("=").append(value).append(", ");
//        });
//        if (!fieldTypes.isEmpty()) {
//            sb.delete(sb.length() - 2, sb.length());
//        }
//        sb.append("]]");
//        return sb.toString();
//    }
//}