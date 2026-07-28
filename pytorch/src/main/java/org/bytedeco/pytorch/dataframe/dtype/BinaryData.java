package org.bytedeco.pytorch.dataframe.dtype;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

//import com.azure.core.util.BinaryData as AzureBinaryData; // 导入微软Azure的BinaryData（如果需要）

/**
 * 二进制数据容器（适配 Arrow BinaryType/FixedSizeBinaryType）
 * 继承 AbstractDataValue，兼容 Lance 数据类型体系，适配 Azure BinaryData 能力
 */
public class BinaryData extends AbstractDataValue implements Serializable {
    private static final long serialVersionUID = 1L;

    // 核心二进制数据存储
    private byte[] data;
    // 二进制数据名称（可选，用于schema标识）
    private String binaryName;
    // 是否为固定长度二进制（适配Arrow FixedSizeBinaryType）
    private boolean fixedSize;
    // 固定长度值（仅当fixedSize=true时有效）
    private Integer fixedLength;

    // 空构造（Arrow反序列化用）
    public BinaryData() {}

    /**
     * 基础构造：字节数组 + 二进制名称
     * @param binaryName 二进制数据名称
     * @param data 字节数组数据
     */
    public BinaryData(String binaryName, byte[] data) {
        this.binaryName = Objects.requireNonNull(binaryName, "二进制数据名称不能为空");
        this.data = Objects.requireNonNull(data, "二进制数据不能为空");
        this.fixedSize = false;
    }

    /**
     * 固定长度二进制构造
     * @param binaryName 二进制数据名称
     * @param data 字节数组数据
     * @param fixedLength 固定长度（数据长度必须等于该值）
     */
    public BinaryData(String binaryName, byte[] data, int fixedLength) {
        this.binaryName = Objects.requireNonNull(binaryName, "二进制数据名称不能为空");
        this.data = Objects.requireNonNull(data, "二进制数据不能为空");
        this.fixedSize = true;
        this.fixedLength = fixedLength;
        
        // 校验数据长度是否符合固定长度要求
        if (data.length != fixedLength) {
            throw new IllegalArgumentException(
                    String.format("固定长度二进制数据长度不匹配：期望 %d 字节，实际 %d 字节", fixedLength, data.length));
        }
    }

    /**
     * 从字符串创建二进制数据（UTF-8编码）
     * @param binaryName 二进制数据名称
     * @param str 字符串内容
     * @return BinaryData实例
     */
    public static BinaryData fromString(String binaryName, String str) {
        return new BinaryData(binaryName, str.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * 从输入流创建二进制数据
     * @param binaryName 二进制数据名称
     * @param inputStream 输入流
     * @return BinaryData实例
     * @throws IOException 流读取异常
     */
    public static BinaryData fromInputStream(String binaryName, InputStream inputStream) throws IOException {
        Objects.requireNonNull(inputStream, "输入流不能为空");
        byte[] data = inputStream.readAllBytes();
        return new BinaryData(binaryName, data);
    }

    /**
     * 从Azure BinaryData转换（兼容微软SDK）
     * @param binaryName 二进制数据名称
     * @param azureBinaryData Azure BinaryData实例
     * @return BinaryData实例
     */
//    public static BinaryData fromAzureBinaryData(String binaryName, AzureBinaryData azureBinaryData) {
//        Objects.requireNonNull(azureBinaryData, "Azure BinaryData不能为空");
//        return new BinaryData(binaryName, azureBinaryData.toBytes());
//    }

    /**
     * 设置二进制数据（非固定长度）
     * @param data 字节数组数据
     */
    public void setData(byte[] data) {
        if (fixedSize) {
            throw new UnsupportedOperationException("固定长度二进制数据不允许修改数据（长度会变化）");
        }
        this.data = Objects.requireNonNull(data, "二进制数据不能为空");
    }

    /**
     * 设置固定长度二进制数据
     * @param data 字节数组数据
     */
    public void setFixedSizeData(byte[] data) {
        if (!fixedSize || fixedLength == null) {
            throw new UnsupportedOperationException("当前不是固定长度二进制数据，请先初始化固定长度");
        }
        if (data.length != fixedLength) {
            throw new IllegalArgumentException(
                    String.format("固定长度二进制数据长度不匹配：期望 %d 字节，实际 %d 字节", fixedLength, data.length));
        }
        this.data = data;
    }

    /**
     * 转换为字符串（UTF-8编码）
     * @return UTF-8编码的字符串
     */
    public String toStringUtf8() {
        return new String(data, StandardCharsets.UTF_8);
    }

    /**
     * 转换为ByteBuffer
     * @return 只读ByteBuffer
     */
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(data).asReadOnlyBuffer();
    }

    /**
     * 转换为输入流
     * @return 字节数组输入流
     */
    public InputStream toInputStream() {
        return new ByteArrayInputStream(data);
    }

    /**
     * Arrow类型适配：转换为Arrow BinaryType/FixedSizeBinaryType的描述
     */
    public Map<String, String> toArrowFieldDesc() {
        return Collections.singletonMap(
                "type", fixedSize ? "fixedsizebinary[" + fixedLength + "]" : "binary"
        );
    }

    @Override
    public Number getNumericValue(){
        return null;
    }

    // ========== 实现 AbstractDataValue 抽象方法 ==========
    @Override
    public String getDataType() {
        return fixedSize ? "FIXED_SIZE_BINARY" : "BINARY";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回包含二进制数据核心信息的Map
        Map<String, Object> arrowData = new java.util.LinkedHashMap<>();
        arrowData.put("binaryName", binaryName);
        arrowData.put("dataLength", data.length);
        arrowData.put("fixedSize", fixedSize);
        if (fixedSize) {
            arrowData.put("fixedLength", fixedLength);
        }
        // 注意：Arrow传输时通常不直接存储完整二进制数据，这里返回长度和类型信息即可
        arrowData.put("data", data); // 按需返回，实际场景可根据性能考虑只返回流
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        return String.format("name=%s, type=%s, length=%d",
                binaryName, getDataType(), data.length);
    }

    @Override
    public boolean isValid() {
        // 基础校验 + Binary专属校验
        return super.isValid()
                && binaryName != null
                && data != null
                && (!fixedSize || (fixedLength != null && data.length == fixedLength));
    }

    // ========== Getter & Setter ==========
    public byte[] getData() {
        // 返回数据副本，避免外部修改内部数据
        return data.clone();
    }

    public String getBinaryName() {
        return binaryName;
    }

    public void setBinaryName(String binaryName) {
        this.binaryName = Objects.requireNonNull(binaryName, "二进制数据名称不能为空");
    }

    public boolean isFixedSize() {
        return fixedSize;
    }

    public Integer getFixedLength() {
        return fixedLength;
    }

    public int getDataLength() {
        return data.length;
    }

    // ========== 辅助方法 ==========
    @Override
    public String toString() {
        // 避免打印大量二进制数据，只显示关键信息
        StringBuilder sb = new StringBuilder();
        sb.append("BinaryData[name=").append(binaryName)
                .append(", type=").append(getDataType())
                .append(", length=").append(data.length);
        if (fixedSize) {
            sb.append(", fixedLength=").append(fixedLength);
        }
        // 显示前16个字节的十六进制（便于调试）
        sb.append(", preview=").append(bytesToHex(data, 16));
        sb.append("]");
        return sb.toString();
    }

    /**
     * 字节数组转十六进制字符串（用于预览）
     * @param bytes 字节数组
     * @param maxLength 最大显示长度
     * @return 十六进制字符串
     */
    private String bytesToHex(byte[] bytes, int maxLength) {
        if (bytes == null || bytes.length == 0) {
            return "";
        }
        int length = Math.min(bytes.length, maxLength);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            sb.append(String.format("%02X", bytes[i]));
            if (i < length - 1) {
                sb.append(" ");
            }
        }
        if (bytes.length > maxLength) {
            sb.append(" ...");
        }
        return sb.toString();
    }
}