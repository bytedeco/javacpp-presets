package org.bytedeco.pytorch.dataframe.dtype;

import org.bytedeco.pytorch.dataframe.enums.ColumnType;

import java.util.*;

/**
 * 列表视图容器（适配 Arrow ListViewType）
 * 轻量级列表引用，支持固定元素类型、视图范围（offset/length）
 */
public class ListViewData extends AbstractDataValue implements StructuredData {
    private static final long serialVersionUID = 1L;

    // 基础列表数据（底层存储）
    private List<Object> baseList;
    // 元素类型（统一类型，对齐Arrow List的元素类型）
    private ColumnType elementType;
    // 视图偏移量（起始索引）
    private int offset = 0;
    // 视图长度（-1表示到末尾）
    private int length = -1;

    // 完整列表构造（视图覆盖全部）
    public ListViewData(List<Object> baseList, ColumnType elementType) {
        this.baseList = new ArrayList<>(Objects.requireNonNull(baseList));
        this.elementType = Objects.requireNonNull(elementType);
        this.length = baseList.size();
    }

    // 视图构造（指定偏移和长度）
    public ListViewData(List<Object> baseList, ColumnType elementType, int offset, int length) {
        this.baseList = new ArrayList<>(Objects.requireNonNull(baseList));
        this.elementType = Objects.requireNonNull(elementType);
        if (offset < 0 || offset >= baseList.size()) {
            throw new IllegalArgumentException("偏移量超出列表范围：" + offset);
        }
        if (length < -1 || (length > 0 && offset + length > baseList.size())) {
            throw new IllegalArgumentException("视图长度超出列表范围：offset=" + offset + ", length=" + length);
        }
        this.offset = offset;
        this.length = length == -1 ? baseList.size() - offset : length;
    }

    // 获取视图内的元素（只读）
    public List<Object> getViewElements() {
        int endIdx = offset + length;
        return Collections.unmodifiableList(baseList.subList(offset, endIdx));
    }

    // Arrow适配：转换为Arrow ListView的元信息
    public Map<String, Object> toArrowListViewMeta() {
        Map<String, Object> meta = new HashMap<>();
        meta.put("element_type", elementType.name().toLowerCase());
        meta.put("offset", offset);
        meta.put("length", length);
        meta.put("base_list_size", baseList.size());
        return meta;
    }

    // Getter & Setter
    public List<Object> getBaseList() { return Collections.unmodifiableList(baseList); }
    public ColumnType getElementType() { return elementType; }
    public int getOffset() { return offset; }
    public int getLength() { return length; }

    @Override
    public String getDataType() {
        return "LIST_VIEW";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回ListView元信息 + 视图数据
        Map<String, Object> arrowData = new HashMap<>();
        arrowData.put("meta", toArrowListViewMeta());
        arrowData.put("view_elements", getViewElements());
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        return String.format("elementType=%s, offset=%d, length=%d, baseSize=%d",
                elementType, offset, length, baseList.size());
    }

    @Override
    public boolean isValid() {
        // 基础校验 + ListView专属校验：基础列表非空、元素类型非空、偏移/长度合法
        return super.isValid()
                && baseList != null && !baseList.isEmpty()
                && elementType != null
                && offset >= 0 && offset < baseList.size()
                && length >= -1
                && (length == -1 || (offset + length) <= baseList.size());
    }

    // ========== 实现 StructuredData 接口 ==========
    @Override
    public int getSize() {
        // ListView大小：视图内元素数量
        return length;
    }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("baseList", baseList);
        map.put("elementType", elementType);
        map.put("offset", offset);
        map.put("length", length);
        map.put("viewElements", getViewElements());
        map.put("baseSize", baseList.size());
        return map;
    }
    @Override
    public String toString() {
        return String.format("ListViewData[elementType=%s, offset=%d, length=%d, baseSize=%d, view=%s]",
                elementType, offset, length, baseList.size(), getViewElements());
    }

    @Override
    public Number getNumericValue(){
        return null;
    }


    public List<Object>  getItems() {
        int endIdx = offset + length;
        return Collections.unmodifiableList(baseList.subList(offset, endIdx));
    }
}