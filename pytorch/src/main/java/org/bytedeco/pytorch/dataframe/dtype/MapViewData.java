package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.ColumnType;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Map视图容器（适配 Arrow MapType）
 * 支持键类型、值类型约束，轻量级视图范围（过滤条件）
 */
public class MapViewData extends AbstractDataValue implements StructuredData{
    private static final long serialVersionUID = 1L;

    // 基础Map数据（底层存储）
    private Map<Object, Object> baseMap;
    // 键类型（统一类型，Arrow Map要求键类型一致）
    private ColumnType keyType;
    // 值类型（统一类型）
    private ColumnType valueType;
    // 视图过滤条件（可选，如键前缀、值范围）
    private String filterExpr;

    // 完整Map构造
    public MapViewData(Map<Object, Object> baseMap, ColumnType keyType, ColumnType valueType) {
        this.baseMap = new LinkedHashMap<>(Objects.requireNonNull(baseMap));
        this.keyType = Objects.requireNonNull(keyType);
        this.valueType = Objects.requireNonNull(valueType);
        // 校验键值类型一致性
        validateTypeConsistency();
    }

    // 带过滤条件的视图构造
    public MapViewData(Map<Object, Object> baseMap, ColumnType keyType, ColumnType valueType, String filterExpr) {
        this(baseMap, keyType, valueType);
        this.filterExpr = filterExpr;
    }

    // 校验键值类型一致性（Arrow Map要求）
    private void validateTypeConsistency() {
        for (Map.Entry<Object, Object> entry : baseMap.entrySet()) {
            if (!isTypeMatch(entry.getKey(), keyType)) {
                throw new IllegalArgumentException("键类型不匹配：期望" + keyType + "，实际" + entry.getKey().getClass());
            }
            if (!isTypeMatch(entry.getValue(), valueType)) {
                throw new IllegalArgumentException("值类型不匹配：期望" + valueType + "，实际" + entry.getValue().getClass());
            }
        }
    }

    // 类型匹配校验
    private boolean isTypeMatch(Object value, ColumnType type) {
        if (value == null) return true;
        return switch (type) {
            case INT32 -> value instanceof Integer;
            case INT64 -> value instanceof Long;
            case FLOAT32 -> value instanceof Float;
            case FLOAT64 -> value instanceof Double;
            case STRING -> value instanceof String;
            case BOOLEAN -> value instanceof Boolean;
            default -> true; // 复杂类型宽松校验
        };
    }

    // 获取视图内的Map（应用过滤条件，此处简化为返回全部，可扩展）
    public Map<Object, Object> getViewMap() {
        // 此处可扩展：根据filterExpr过滤baseMap
        return Collections.unmodifiableMap(baseMap);
    }

    // Arrow适配：转换为Arrow MapType描述
    public Map<String, String> toArrowMapDesc() {
        Map<String, String> desc = new HashMap<>();
        desc.put("key_type", keyType.name().toLowerCase());
        desc.put("value_type", valueType.name().toLowerCase());
        desc.put("filter_expr", filterExpr == null ? "none" : filterExpr);
        return desc;
    }

    // Getter & Setter
    public Map<Object, Object> getBaseMap() { return Collections.unmodifiableMap(baseMap); }
    public ColumnType getKeyType() { return keyType; }
    public ColumnType getValueType() { return valueType; }
    public String getFilterExpr() { return filterExpr; }
    public void setFilterExpr(String filterExpr) { this.filterExpr = filterExpr; }
    public int size() { return baseMap.size(); }

    @Override
    public String getDataType() {
        return "MAP_VIEW";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回MapView描述 + 视图数据
        Map<String, Object> arrowData = new HashMap<>();
        arrowData.put("desc", toArrowMapDesc());
        arrowData.put("view_map", getViewMap());
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        return String.format("keyType=%s, valueType=%s, size=%d, filter=%s",
                keyType, valueType, baseMap.size(), filterExpr == null ? "none" : filterExpr);
    }

    @Override
    public boolean isValid() {
        // 基础校验 + MapView专属校验：基础Map非空、键/值类型非空、类型一致性校验通过
        try {
            validateTypeConsistency();
            return super.isValid()
                    && baseMap != null && !baseMap.isEmpty()
                    && keyType != null && valueType != null;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    // ========== 实现 StructuredData 接口 ==========
    @Override
    public int getSize() {
        // MapView大小：基础Map的键值对数量
        return baseMap.size();
    }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("baseMap", baseMap);
        map.put("keyType", keyType);
        map.put("valueType", valueType);
        map.put("filterExpr", filterExpr);
        map.put("viewMap", getViewMap());
        map.put("size", size());
        return map;
    }

    @Override
    public String toString() {
        return String.format("MapViewData[keyType=%s, valueType=%s, size=%d, filter=%s, map=%s]",
                keyType, valueType, baseMap.size(), filterExpr, getViewMap());
    }

    @Override
    public Number getNumericValue(){
        return null;
    }

    /**
     * 构造函数
     * @param mapData 原始Map数据
     */
//    public MapViewData(Map<Object, Object> mapData) {
//        this.baseMap = Objects.requireNonNull(mapData, "Map数据不能为空");
//        // 自动推断键值类型（如果Map非空）
//        this.keyType = mapData.isEmpty() ? Object.class : mapData.keySet().iterator().next().getClass();
//        this.valueType = mapData.isEmpty() ? Object.class : mapData.values().iterator().next().getClass();
//    }

    /**
     * 获取Map中的所有键值对，转换为List<Object>
     * 每个元素为 "key=value" 格式的字符串，便于表格展示
     * @return 键值对列表
     */
    public List<Object> getEntries() {
        if (baseMap.isEmpty()) {
            return Collections.emptyList();
        }
        // 将Map的entrySet转换为List，元素格式：key=value
        return baseMap.entrySet().stream()
                .map(entry -> entry.getKey() + "=" + formatEntryValue(entry.getValue()))
                .collect(Collectors.toList());
    }

    /**
     * 格式化单个entry的值（处理复杂类型，避免显示内存地址）
     * @param value 原始值
     * @return 格式化后的字符串
     */
    private String formatEntryValue(Object value) {
        if (value == null) {
            return "null";
        }
        // 处理集合类型
        if (value instanceof Collection<?>) {
            Collection<?> coll = (Collection<?>) value;
            return coll.size() > 5 ? coll.stream().limit(5).toList() + " ... 共" + coll.size() + "个元素" : coll.toString();
        }
        // 处理数组类型
        if (value.getClass().isArray()) {
            if (value instanceof int[]) return Arrays.toString((int[]) value);
            if (value instanceof long[]) return Arrays.toString((long[]) value);
            if (value instanceof float[]) return Arrays.toString((float[]) value);
            if (value instanceof double[]) return Arrays.toString((double[]) value);
            if (value instanceof Object[]) return Arrays.toString((Object[]) value);
            return "【数组】";
        }
        // 处理Map类型（递归格式化）
        if (value instanceof Map<?, ?>) {
            Map<?, ?> subMap = (Map<?, ?>) value;
            return subMap.size() > 5 ? "Map[" + subMap.size() + "] （显示前5项）: " +
                    subMap.entrySet().stream().limit(5).toList() : subMap.toString();
        }
        // 普通类型直接返回字符串
        return value.toString().length() > 30 ? value.toString().substring(0, 30) + "..." : value.toString();
    }

    /**
     * 获取原始Map数据
     * @return 不可修改的Map（避免外部修改）
     */
    public Map<Object, Object> getMapData() {
        return Collections.unmodifiableMap(baseMap);
    }
}