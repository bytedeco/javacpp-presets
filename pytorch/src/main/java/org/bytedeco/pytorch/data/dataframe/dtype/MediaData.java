package org.bytedeco.pytorch.data.dataframe.dtype;

/**
 * 媒体数据（音频/图像/视频）通用接口
 */
public interface MediaData extends DataValue {

    /**
     * 获取媒体文件路径
     * @return 路径字符串（本地/远程）
     */
    String getPath();

    /**
     * 获取媒体数据的原始字节数组
     * @return 原始字节数组（null表示无）
     */
    byte[] getRawBytes();

    /**
     * 获取媒体数据的维度/形状（如图片：width,height,channels；音频：sampleRate,channels）
     * @return 维度数组
     */
    int[] getDimensions();
}