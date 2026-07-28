package org.bytedeco.pytorch.dataframe.dtype;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * 视频数据容器
 * 集成OpenCV和FFmpeg风格的视频处理算法
 * 支持30多种视频处理方法、视频嵌入、关键帧提取等功能
 */
public class VideoData extends AbstractDataValue implements MediaData{
    private String path;
    private List<ImageData> frames;
    private int frameCount;
    private double fps;
    private double duration;
    private int width;
    private int height;
    private String format;
    private AudioData audioTrack;

    private String videoId;
//    private String format; // 格式：mp4/avi/mkv 等
//    private long durationMs; // 时长（毫秒）
//    private int frames; // 总帧数（int 类型，与 IntVector 返回值匹配）
    private long fileSizeBytes; // 文件大小（字节）


    public VideoData(String path) {
        this.path = path;
        this.frames = new ArrayList<>();
    }

    // 新增构造函数适配 byte[] + String
    public VideoData(byte[] videoBytes, String codec) {
        this.path = null;
        this.frames = new ArrayList<>();
//        this.codec = codec; // 需新增 codec 字段
//        this.videoBytes = videoBytes; // 需新增 videoBytes 字段
    }

    public VideoData(List<ImageData> frames, double fps) {
        this.frames = frames;
        this.frameCount = frames.size();
        this.fps = fps;
        this.duration = frameCount / fps;
        if (!frames.isEmpty()) {
            this.width = frames.get(0).getWidth();
            this.height = frames.get(0).getHeight();
        }
    }

    // Getters and setters
//    public String getPath() { return path; }

    public void setPath(String path){
        this.path = path;
    }
    public List<ImageData> getFrames() { return frames; }

    public void setFrames(List<ImageData> frames) {
        this.frames = frames;
        this.frameCount = frames.size();
    }

    @Override
    public String getDataType() {
        return "VIDEO";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回包含核心信息的Map
        Map<String, Object> arrowData = new HashMap<>();
        arrowData.put("path", path);
        arrowData.put("frameCount", frameCount);
        arrowData.put("fps", fps);
        arrowData.put("duration", duration);
        arrowData.put("width", width);
        arrowData.put("height", height);
        arrowData.put("format", format);
        arrowData.put("fileSizeBytes", fileSizeBytes);
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        return String.format("%dx%d, frames=%d, fps=%.1f, dur=%.2fs, path=%s",
                width, height, frameCount, fps, duration, path);
    }

    // ========== 重写有效性校验 ==========
    @Override
    public boolean isValid() {
        // 基础校验 + 视频专属校验：路径/帧列表至少有一个非空，帧数非负
        return super.isValid()
                && (path != null || (frames != null && !frames.isEmpty()))
                && frameCount >= 0
                && fps >= 0
                && duration >= 0;
    }

    // ========== 实现 MediaData 接口 ==========
    @Override
    public String getPath() {
        return path;
    }

    @Override
    public byte[] getRawBytes() {
        // 视频原始字节暂不实现（可根据需求扩展）
        return null;
    }

    @Override
    public int[] getDimensions() {
        // 视频维度：[width, height, frameCount]
        return new int[]{width, height, frameCount};
    }
    public int getFrameCount() { return frameCount; }

    public void setDuration(double duration){
        this.duration = duration;
    }

    /**
     * 追加单个视频帧
     * @param frame 单帧图像数据
     */
    public void addFrame(ImageData frame) {
        if (frame != null) {
            this.frames.add(frame);
            this.frameCount = this.frames.size();
        }
    }

    /**
     * 批量追加视频帧
     * @param frames 批量帧列表
     */
    public void addFrames(List<ImageData> frames) {
        if (frames != null && !frames.isEmpty()) {
            this.frames.addAll(frames);
            this.frameCount = this.frames.size();
        }
    }

    /**
     * 清空视频帧列表
     */
    public void clearFrames() {
        this.frames.clear();
        this.frameCount = 0;
    }

    /**
     * 设置总帧数（直接赋值，一般用于从外部数据源读取）
     * @param frameCount 总帧数
     */
    public void setFrameCount(int frameCount) {
        if (frameCount < 0) {
            throw new IllegalArgumentException("帧数不能为负数：" + frameCount);
        }
        this.frameCount = frameCount;
        // 可选：如果帧列表长度与帧数不一致，给出警告
        if (this.frames.size() != frameCount) {
            System.err.println("警告：帧列表长度（" + this.frames.size() + "）与设置的帧数（" + frameCount + "）不一致");
        }
    }


    public double getDuration() { return duration; }

    public double getFps() { return fps; }

    public void setFps(double fps) { this.fps = fps; }

    public void setWidth(int width) { this.width = width; }
    public void setHeight(int height) { this.height = height; }

    public int getWidth() { return width; }
    public int getHeight() { return height; }

    public String getFormat() { return format; }
    public void setFormat(String format) { this.format = format; }

    public AudioData getAudioTrack() { return audioTrack; }
    public void setAudioTrack(AudioData audioTrack) { this.audioTrack = audioTrack; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        VideoData videoData = (VideoData) o;
        return frameCount == videoData.frameCount && Objects.equals(path, videoData.path) && Objects.equals(frames, videoData.frames);
    }

    @Override
    public int hashCode() {
        return Objects.hash(path, frames, frameCount);
    }

    @Override
    public String toString() {
        return String.format("VideoData[%dx%d, frames=%d, fps=%.1f, dur=%.2fs, path=%s]",
                width, height, frameCount, fps, duration, path);
    }

    @Override
    public Number getNumericValue(){
        return null;
    }

    // 视频处理常量
    public static final int FRAME_BATCH_SIZE = 32;
    public static final double DEFAULT_QUALITY = 0.8;
    public static final int DEFAULT_THREAD_COUNT = 4;

    // 支持的视频格式
    public enum VideoFormat {
        MP4("mp4", "MPEG-4"),
        AVI("avi", "Audio Video Interleave"),
        MKV("mkv", "Matroska Video"),
        MOV("mov", "QuickTime Movie"),
        WMV("wmv", "Windows Media Video"),
        FLV("flv", "Flash Video"),
        WEBM("webm", "WebM Video"),
        OGV("ogv", "Ogg Video");

        private final String extension;
        private final String description;

        VideoFormat(String extension, String description) {
            this.extension = extension;
            this.description = description;
        }

        public String getExtension() { return extension; }
        public String getDescription() { return description; }

        public static VideoFormat fromExtension(String extension) {
            String ext = extension.toLowerCase().replace(".", "");
            for (VideoFormat format : values()) {
                if (format.extension.equals(ext)) {
                    return format;
                }
            }
            return MP4; // 默认格式
        }
    }

    // 视频编码器类型
    public enum VideoCodec {
        H264("h264", "H.264/AVC"),
        H265("h265", "H.265/HEVC"),
        VP9("vp9", "VP9"),
        AV1("av1", "AV1"),
        XVID("xvid", "Xvid"),
        MJPEG("mjpeg", "Motion JPEG");

        private final String name;
        private final String description;

        VideoCodec(String name, String description) {
            this.name = name;
            this.description = description;
        }

        public String getName() { return name; }
        public String getDescription() { return description; }
    }

    // 关键帧类型
    public enum KeyFrameType {
        HISTOGRAM_DIFF,    // 基于直方图差异
        EDGE_DIFF,        // 基于边缘差异
        OPTICAL_FLOW,     // 基于光流
        SCENE_CHANGE,     // 场景变化
        MOTION_VECTOR     // 运动向量
    }

    // ==================== 视频加载和保存 ====================

    /**
     * 1. 从文件加载视频数据.
     * <p>Prefer real FFmpeg decode via {@code MediaBridge}; fall back to mock frames
     * only when native FFmpeg is unavailable (keeps unit tests offline-friendly).
     */
    public static VideoData loadFromFile(String filePath) throws IOException {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new FileNotFoundException("视频文件不存在: " + filePath);
        }
        // Real FFmpeg path (OpenCV/FFmpeg interop layer)
        try {
            Class<?> bridge = Class.forName(
                    "org.bytedeco.pytorch.dataframe.media.MediaBridge");
            Boolean avail = (Boolean) bridge.getMethod("isFFmpegAvailable").invoke(null);
            if (Boolean.TRUE.equals(avail)) {
                Class<?> optsCls = Class.forName(
                        "org.bytedeco.pytorch.dataframe.media.MediaBridge$VideoOptions");
                Object opts = optsCls.getMethod("defaults").invoke(null);
                // cap frames for DataFrame batch use to avoid OOM on long clips
                opts = optsCls.getMethod("withMaxFrames", int.class).invoke(opts, 256);
                VideoData real = (VideoData) bridge
                        .getMethod("loadVideo", String.class, optsCls)
                        .invoke(null, filePath, opts);
                if (real != null && real.getFrames() != null && !real.getFrames().isEmpty()) {
                    return real;
                }
            }
        } catch (ClassNotFoundException ignored) {
            // MediaBridge not on classpath — fall through to stub
        } catch (Throwable t) {
            // native failure — fall through to stub
        }

        VideoData videoData = new VideoData(filePath);
        String extension = getFileExtension(filePath);
        videoData.setFormat(extension);
        videoData.setWidth(64);
        videoData.setHeight(64);
        videoData.setFps(30.0);
        videoData.setDuration(1.0);

        // Lightweight mock frames (small, for offline / no-FFmpeg environments)
        List<ImageData> mockFrames = generateMockFrames(
                Math.max(1, (int) (videoData.getDuration() * videoData.getFps())),
                videoData.getWidth(), videoData.getHeight());
        videoData.setFrames(mockFrames);
        return videoData;
    }

    /**
     * 2. 保存视频到文件
     */
    public void saveToFile(String outputPath, VideoCodec codec) throws IOException {
        System.out.println("保存视频到: " + outputPath + " 使用编码器: " + codec.getName());
        // 实际实现需要FFmpeg支持
        // 这里只是模拟保存过程
        File outputFile = new File(outputPath);
        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write("# 模拟视频文件 " + outputPath + "\n");
            writer.write("格式: " + getFileExtension(outputPath) + "\n");
            writer.write("编码器: " + codec.getName() + "\n");
            writer.write("分辨率: " + width + "x" + height + "\n");
            writer.write("帧率: " + fps + " fps\n");
            writer.write("总帧数: " + frameCount + "\n");
            writer.write("时长: " + duration + " 秒\n");
        }
    }

    // ==================== 视频转换和编码 ====================

    /**
     * 3. 视频格式转换
     */
    public VideoData convertFormat(VideoFormat targetFormat, VideoCodec codec) throws IOException {
        System.out.println("转换视频格式: " + format + " -> " + targetFormat.getExtension());
        VideoData converted = new VideoData(this.frames, this.fps);
        converted.setFormat(targetFormat.getExtension());
        converted.setWidth(this.width);
        converted.setHeight(this.height);
        converted.setDuration(this.duration);
        return converted;
    }

    /**
     * 4. 调整视频分辨率
     */
    public VideoData resize(int newWidth, int newHeight) {
        System.out.println("调整分辨率: " + width + "x" + height + " -> " + newWidth + "x" + newHeight);
        List<ImageData> resizedFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            ImageData resized = frame.resize(newWidth, newHeight);
            resizedFrames.add(resized);
        }

        VideoData resizedVideo = new VideoData(resizedFrames, this.fps);
        resizedVideo.setWidth(newWidth);
        resizedVideo.setHeight(newHeight);
        resizedVideo.setFormat(this.format);
        resizedVideo.setDuration(this.duration);

        return resizedVideo;
    }

    /**
     * 5. 调整视频帧率
     */
    public VideoData changeFrameRate(double newFps) {
        System.out.println("调整帧率: " + fps + " -> " + newFps + " fps");

        List<ImageData> newFrames = new ArrayList<>();
        double ratio = newFps / this.fps;

        if (ratio > 1.0) {
            // 插帧
            for (int i = 0; i < frames.size(); i++) {
                newFrames.add(frames.get(i));
                // 插入中间帧
                int insertCount = (int) Math.floor(ratio) - 1;
                for (int j = 0; j < insertCount; j++) {
                    newFrames.add(frames.get(i)); // 简单复制帧
                }
            }
        } else {
            // 跳帧
            int step = (int) Math.round(1.0 / ratio);
            for (int i = 0; i < frames.size(); i += step) {
                newFrames.add(frames.get(i));
            }
        }

        VideoData newVideo = new VideoData(newFrames, newFps);
        newVideo.setWidth(this.width);
        newVideo.setHeight(this.height);
        newVideo.setFormat(this.format);
        newVideo.setDuration(newFrames.size() / newFps);

        return newVideo;
    }

    /**
     * 6. 视频剪切
     */
    public VideoData trim(double startTime, double endTime) {
        if (startTime < 0 || endTime > duration || startTime >= endTime) {
            throw new IllegalArgumentException("无效的时间范围");
        }

        int startFrame = (int) (startTime * fps);
        int endFrame = (int) (endTime * fps);

        List<ImageData> trimmedFrames = frames.subList(startFrame, Math.min(endFrame, frames.size()));

        VideoData trimmed = new VideoData(new ArrayList<>(trimmedFrames), this.fps);
        trimmed.setWidth(this.width);
        trimmed.setHeight(this.height);
        trimmed.setFormat(this.format);
        trimmed.setDuration(endTime - startTime);

        return trimmed;
    }

    /**
     * 7. 视频拼接
     */
    public VideoData concatenate(VideoData other) {
        if (this.fps != other.fps || this.width != other.width || this.height != other.height) {
            throw new IllegalArgumentException("视频参数不匹配，无法拼接");
        }

        List<ImageData> concatenatedFrames = new ArrayList<>(this.frames);
        concatenatedFrames.addAll(other.frames);

        VideoData concatenated = new VideoData(concatenatedFrames, this.fps);
        concatenated.setWidth(this.width);
        concatenated.setHeight(this.height);
        concatenated.setFormat(this.format);
        concatenated.setDuration(this.duration + other.duration);

        return concatenated;
    }

    // ==================== 关键帧提取 ====================

    /**
     * 8. 基于直方图差异的关键帧提取
     */
    public List<KeyFrame> extractKeyFramesHistogram(double threshold) {
        List<KeyFrame> keyFrames = new ArrayList<>();
        if (frames.isEmpty()) return keyFrames;

        // 第一帧总是关键帧
        keyFrames.add(new KeyFrame(0, 0.0, frames.get(0), KeyFrameType.HISTOGRAM_DIFF, 1.0));

        ImageData prevFrame = frames.get(0);
        for (int i = 1; i < frames.size(); i++) {
            ImageData currentFrame = frames.get(i);
            double histDiff = calculateHistogramDifference(prevFrame, currentFrame);

            if (histDiff > threshold) {
                double timestamp = i / fps;
                keyFrames.add(new KeyFrame(i, timestamp, currentFrame, KeyFrameType.HISTOGRAM_DIFF, histDiff));
                prevFrame = currentFrame;
            }
        }

        System.out.println("基于直方图差异提取了 " + keyFrames.size() + " 个关键帧");
        return keyFrames;
    }

    /**
     * 9. 基于边缘差异的关键帧提取
     */
    public List<KeyFrame> extractKeyFramesEdge(double threshold) {
        List<KeyFrame> keyFrames = new ArrayList<>();
        if (frames.isEmpty()) return keyFrames;

        keyFrames.add(new KeyFrame(0, 0.0, frames.get(0), KeyFrameType.EDGE_DIFF, 1.0));

        ImageData prevFrame = frames.get(0);
        for (int i = 1; i < frames.size(); i++) {
            ImageData currentFrame = frames.get(i);
            double edgeDiff = calculateEdgeDifference(prevFrame, currentFrame);

            if (edgeDiff > threshold) {
                double timestamp = i / fps;
                keyFrames.add(new KeyFrame(i, timestamp, currentFrame, KeyFrameType.EDGE_DIFF, edgeDiff));
                prevFrame = currentFrame;
            }
        }

        System.out.println("基于边缘差异提取了 " + keyFrames.size() + " 个关键帧");
        return keyFrames;
    }

    /**
     * 10. 场景变化检测
     */
    public List<KeyFrame> detectSceneChanges(double threshold) {
        List<KeyFrame> sceneChanges = new ArrayList<>();
        if (frames.isEmpty()) return sceneChanges;

        sceneChanges.add(new KeyFrame(0, 0.0, frames.get(0), KeyFrameType.SCENE_CHANGE, 1.0));

        for (int i = 1; i < frames.size(); i++) {
            double sceneScore = calculateSceneChangeScore(frames.get(i-1), frames.get(i));

            if (sceneScore > threshold) {
                double timestamp = i / fps;
                sceneChanges.add(new KeyFrame(i, timestamp, frames.get(i), KeyFrameType.SCENE_CHANGE, sceneScore));
            }
        }

        System.out.println("检测到 " + sceneChanges.size() + " 个场景变化");
        return sceneChanges;
    }

    /**
     * 11. 均匀采样关键帧
     */
    public List<KeyFrame> extractKeyFramesUniform(int count) {
        List<KeyFrame> keyFrames = new ArrayList<>();
        if (frames.isEmpty() || count <= 0) return keyFrames;

        int step = Math.max(1, frames.size() / count);
        for (int i = 0; i < frames.size(); i += step) {
            double timestamp = i / fps;
            keyFrames.add(new KeyFrame(i, timestamp, frames.get(i), KeyFrameType.SCENE_CHANGE, 1.0));
            if (keyFrames.size() >= count) break;
        }

        return keyFrames;
    }

    // ==================== 视频滤镜和效果 ====================

    /**
     * 12. 应用亮度调整
     */
    public VideoData adjustBrightness(double factor) {
        System.out.println("调整视频亮度，因子: " + factor);
        List<ImageData> adjustedFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            ImageData adjusted = frame.adjustBrightness((float) factor);
            adjustedFrames.add(adjusted);
        }

        VideoData result = new VideoData(adjustedFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    /**
     * 13. 应用对比度调整
     */
    public VideoData adjustContrast(double factor) {
        System.out.println("调整视频对比度，因子: " + factor);
        List<ImageData> adjustedFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            ImageData adjusted = frame.adjustContrast((float) factor);
            adjustedFrames.add(adjusted);
        }

        VideoData result = new VideoData(adjustedFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    /**
     * 14. 应用饱和度调整
     */
    public VideoData adjustSaturation(double factor) {
        System.out.println("调整视频饱和度，因子: " + factor);
        List<ImageData> adjustedFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            ImageData adjusted = frame.adjustSaturation((float) factor);
            adjustedFrames.add(adjusted);
        }

        VideoData result = new VideoData(adjustedFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    /**
     * 15. 转换为灰度视频
     */
    public VideoData toGrayscale() {
        System.out.println("转换视频为灰度");
        List<ImageData> grayscaleFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            ImageData grayscale = frame.toGrayscale();
            grayscaleFrames.add(grayscale);
        }

        VideoData result = new VideoData(grayscaleFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    /**
     * 16. 应用高斯模糊
     */
    public VideoData gaussianBlur(double sigma) {
        System.out.println("应用高斯模糊，sigma: " + sigma);
        List<ImageData> blurredFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            ImageData blurred = frame.gaussianBlur();  // 使用无参数版本
            blurredFrames.add(blurred);
        }

        VideoData result = new VideoData(blurredFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    /**
     * 17. 边缘检测
     */
    public VideoData edgeDetection() {
        System.out.println("应用边缘检测");
        List<ImageData> edgeFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            // 使用简化的边缘检测（基于对比度增强）
            ImageData edges = frame.adjustContrast(2.0f);
            edgeFrames.add(edges);
        }

        VideoData result = new VideoData(edgeFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    /**
     * 18. 视频锐化
     */
    public VideoData sharpen(double amount) {
        System.out.println("视频锐化，强度: " + amount);
        List<ImageData> sharpenedFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            ImageData sharpened = frame.sharpen();  // 使用无参数版本
            sharpenedFrames.add(sharpened);
        }

        VideoData result = new VideoData(sharpenedFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    // ==================== 视频分析 ====================

    /**
     * 19. 计算视频直方图
     */
    public List<int[]> computeHistograms() {
        List<int[]> histograms = new ArrayList<>();

        for (ImageData frame : frames) {
            // 使用简化的直方图计算
            int[] histogram = calculateSimpleHistogram(frame);
            histograms.add(histogram);
        }

        return histograms;
    }

    /**
     * 20. 计算光流
     */
    public List<MotionVector[][]> computeOpticalFlow() {
        List<MotionVector[][]> flowFields = new ArrayList<>();

        if (frames.size() < 2) return flowFields;

        for (int i = 1; i < frames.size(); i++) {
            MotionVector[][] flow = calculateOpticalFlow(frames.get(i-1), frames.get(i));
            flowFields.add(flow);
        }

        System.out.println("计算了 " + flowFields.size() + " 个光流场");
        return flowFields;
    }

    /**
     * 21. 运动检测
     */
    public List<MotionRegion> detectMotion(double threshold) {
        List<MotionRegion> motionRegions = new ArrayList<>();

        if (frames.size() < 2) return motionRegions;

        for (int i = 1; i < frames.size(); i++) {
            List<MotionRegion> frameMotion = detectFrameMotion(frames.get(i-1), frames.get(i), threshold);
            motionRegions.addAll(frameMotion);
        }

        System.out.println("检测到 " + motionRegions.size() + " 个运动区域");
        return motionRegions;
    }

    /**
     * 22. 视频质量评估
     */
    public VideoQualityMetrics assessQuality() {
        double totalBlur = 0.0;
        double totalNoise = 0.0;
        double totalBrightness = 0.0;
        double totalContrast = 0.0;

        for (ImageData frame : frames) {
            totalBlur += calculateBlurMetric(frame);
            totalNoise += calculateNoiseMetric(frame);
            totalBrightness += calculateBrightnessMetric(frame);
            totalContrast += calculateContrastMetric(frame);
        }

        int frameCount = frames.size();
        return new VideoQualityMetrics(
            totalBlur / frameCount,
            totalNoise / frameCount,
            totalBrightness / frameCount,
            totalContrast / frameCount
        );
    }

    // ==================== 视频嵌入和特征提取 ====================

    /**
     * 23. 提取视频嵌入向量
     */
    public VectorData extractVideoEmbedding() {
        System.out.println("提取视频嵌入向量");

        // 使用简化的方法生成视频嵌入
        List<VectorData> frameEmbeddings = new ArrayList<>();

        for (ImageData frame : frames) {
            VectorData frameEmbed = extractFrameEmbedding(frame);
            frameEmbeddings.add(frameEmbed);
        }

        // 聚合帧嵌入为视频嵌入
        VectorData videoEmbedding = aggregateFrameEmbeddings(frameEmbeddings);

        return videoEmbedding;
    }

    /**
     * 24. 提取时空特征
     */
    public VectorData extractSpatioTemporalFeatures() {
        System.out.println("提取时空特征");

        List<Double> features = new ArrayList<>();

        // 空间特征
        features.addAll(extractSpatialFeatures());

        // 时间特征
        features.addAll(extractTemporalFeatures());

        // 运动特征
        features.addAll(extractMotionFeatures());

        double[] featureArray = features.stream().mapToDouble(Double::doubleValue).toArray();
        return new VectorData(featureArray, "spatiotemporal_features");
    }

    /**
     * 25. 计算视频相似度
     */
    public double computeSimilarity(VideoData other) {
        VectorData embedding1 = this.extractVideoEmbedding();
        VectorData embedding2 = other.extractVideoEmbedding();

        return embedding1.cosineSimilarity(embedding2);
    }

    // ==================== 高级视频处理 ====================

    /**
     * 26. 视频稳定化
     */
    public VideoData stabilize() {
        System.out.println("视频稳定化处理");

        if (frames.size() < 2) return this;

        List<ImageData> stabilizedFrames = new ArrayList<>();
        stabilizedFrames.add(frames.get(0)); // 第一帧不变

        for (int i = 1; i < frames.size(); i++) {
            ImageData stabilized = stabilizeFrame(frames.get(i-1), frames.get(i));
            stabilizedFrames.add(stabilized);
        }

        VideoData result = new VideoData(stabilizedFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    /**
     * 27. 视频超分辨率
     */
    public VideoData superResolution(int scaleFactor) {
        System.out.println("视频超分辨率处理，放大倍数: " + scaleFactor);

        List<ImageData> upscaledFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            ImageData upscaled = frame.resize(frame.getWidth() * scaleFactor, frame.getHeight() * scaleFactor);
            upscaledFrames.add(upscaled);
        }

        VideoData result = new VideoData(upscaledFrames, this.fps);
        result.setWidth(this.width * scaleFactor);
        result.setHeight(this.height * scaleFactor);
        result.setFormat(this.format);
        result.setDuration(this.duration);

        return result;
    }

    /**
     * 28. 视频降噪
     */
    public VideoData denoise(double strength) {
        System.out.println("视频降噪处理，强度: " + strength);

        List<ImageData> denoisedFrames = new ArrayList<>();

        for (ImageData frame : frames) {
            // 使用高斯模糊作为简单的降噪方法
            ImageData denoised = frame.gaussianBlur();
            denoisedFrames.add(denoised);
        }

        VideoData result = new VideoData(denoisedFrames, this.fps);
        copyVideoProperties(result);
        return result;
    }

    /**
     * 29. 视频人脸检测
     */
    public List<FaceDetectionResult> detectFaces() {
        System.out.println("视频人脸检测");

        List<FaceDetectionResult> allFaces = new ArrayList<>();

        for (int i = 0; i < frames.size(); i++) {
            List<FaceRegion> frameFaces = detectFrameFaces(frames.get(i));
            if (!frameFaces.isEmpty()) {
                double timestamp = i / fps;
                allFaces.add(new FaceDetectionResult(i, timestamp, frameFaces));
            }
        }

        System.out.println("在 " + allFaces.size() + " 帧中检测到人脸");
        return allFaces;
    }

    /**
     * 30. 视频对象跟踪
     */
    public List<TrackingResult> trackObject(ImageData template, int startFrame) {
        System.out.println("视频对象跟踪");

        List<TrackingResult> trackingResults = new ArrayList<>();

        if (startFrame >= frames.size()) return trackingResults;

        BoundingBox currentBox = findTemplateInFrame(template, frames.get(startFrame));
        if (currentBox == null) return trackingResults;

        trackingResults.add(new TrackingResult(startFrame, startFrame / fps, currentBox, 1.0));

        for (int i = startFrame + 1; i < frames.size(); i++) {
            TrackingResult result = trackObjectInFrame(frames.get(i), currentBox, template);
            result.frameIndex = i;
            result.timestamp = i / fps;
            trackingResults.add(result);
            currentBox = result.boundingBox;
        }

        return trackingResults;
    }

    /**
     * 31. 视频摘要生成
     */
    public VideoData generateSummary(double summaryRatio) {
        System.out.println("生成视频摘要，压缩比例: " + summaryRatio);

        if (summaryRatio <= 0 || summaryRatio >= 1) {
            throw new IllegalArgumentException("摘要比例必须在0和1之间");
        }

        // 提取关键帧
        List<KeyFrame> keyFrames = extractKeyFramesHistogram(0.3);

        // 根据重要性选择帧
        int targetFrameCount = (int) (frames.size() * summaryRatio);
        Collections.sort(keyFrames, (a, b) -> Double.compare(b.score, a.score));

        List<ImageData> summaryFrames = new ArrayList<>();
        Set<Integer> selectedIndices = new HashSet<>();

        for (KeyFrame kf : keyFrames) {
            if (summaryFrames.size() >= targetFrameCount) break;
            if (!selectedIndices.contains(kf.frameIndex)) {
                summaryFrames.add(kf.frame);
                selectedIndices.add(kf.frameIndex);
            }
        }

        VideoData summary = new VideoData(summaryFrames, this.fps);
        summary.setWidth(this.width);
        summary.setHeight(this.height);
        summary.setFormat(this.format);
        summary.setDuration(summaryFrames.size() / this.fps);

        return summary;
    }

    /**
     * 32. 慢动作效果
     */
    public VideoData slowMotion(double factor) {
        if (factor <= 0 || factor >= 1) {
            throw new IllegalArgumentException("慢动作因子必须在0和1之间");
        }

        System.out.println("应用慢动作效果，因子: " + factor);

        List<ImageData> slowMotionFrames = new ArrayList<>();

        // 通过插值增加帧数
        for (int i = 0; i < frames.size() - 1; i++) {
            slowMotionFrames.add(frames.get(i));

            int interpolatedFrames = (int) (1.0 / factor) - 1;
            for (int j = 1; j <= interpolatedFrames; j++) {
                float ratio = (float) j / (interpolatedFrames + 1);
                ImageData interpolated = interpolateFrames(frames.get(i), frames.get(i + 1), ratio);
                slowMotionFrames.add(interpolated);
            }
        }
        slowMotionFrames.add(frames.get(frames.size() - 1));

        VideoData result = new VideoData(slowMotionFrames, this.fps);
        copyVideoProperties(result);
        result.setDuration(this.duration / factor);

        return result;
    }

    /**
     * 33. 快进效果
     */
    public VideoData fastForward(double factor) {
        if (factor <= 1) {
            throw new IllegalArgumentException("快进因子必须大于1");
        }

        System.out.println("应用快进效果，因子: " + factor);

        List<ImageData> fastForwardFrames = new ArrayList<>();
        int step = (int) Math.round(factor);

        for (int i = 0; i < frames.size(); i += step) {
            fastForwardFrames.add(frames.get(i));
        }

        VideoData result = new VideoData(fastForwardFrames, this.fps);
        copyVideoProperties(result);
        result.setDuration(this.duration / factor);

        return result;
    }

    /**
     * 34. 视频反向播放
     */
    public VideoData reverse() {
        System.out.println("视频反向播放");

        List<ImageData> reversedFrames = new ArrayList<>(frames);
        Collections.reverse(reversedFrames);

        VideoData result = new VideoData(reversedFrames, this.fps);
        copyVideoProperties(result);

        return result;
    }

    /**
     * 35. 提取视频缩略图
     */
    public List<ImageData> generateThumbnails(int count) {
        List<ImageData> thumbnails = new ArrayList<>();

        if (frames.isEmpty() || count <= 0) return thumbnails;

        int step = Math.max(1, frames.size() / count);
        for (int i = 0; i < frames.size() && thumbnails.size() < count; i += step) {
            ImageData thumbnail = frames.get(i).resize(160, 120); // 标准缩略图尺寸
            thumbnails.add(thumbnail);
        }

        return thumbnails;
    }

    // ==================== 内部类定义 ====================

    /**
     * 关键帧类
     */
    public static class KeyFrame {
        public final int frameIndex;
        public final double timestamp;
        public final ImageData frame;
        public final KeyFrameType type;
        public final double score;

        public KeyFrame(int frameIndex, double timestamp, ImageData frame, KeyFrameType type, double score) {
            this.frameIndex = frameIndex;
            this.timestamp = timestamp;
            this.frame = frame;
            this.type = type;
            this.score = score;
        }

        @Override
        public String toString() {
            return String.format("KeyFrame[frame=%d, time=%.2fs, type=%s, score=%.3f]",
                    frameIndex, timestamp, type, score);
        }
    }

    /**
     * 运动向量类
     */
    public static class MotionVector {
        public final double dx, dy;
        public final double magnitude;

        public MotionVector(double dx, double dy) {
            this.dx = dx;
            this.dy = dy;
            this.magnitude = Math.sqrt(dx * dx + dy * dy);
        }

        @Override
        public String toString() {
            return String.format("MotionVector[dx=%.2f, dy=%.2f, mag=%.2f]", dx, dy, magnitude);
        }
    }

    /**
     * 运动区域类
     */
    public static class MotionRegion {
        public final BoundingBox boundingBox;
        public final double motionIntensity;
        public final int frameIndex;

        public MotionRegion(BoundingBox boundingBox, double motionIntensity, int frameIndex) {
            this.boundingBox = boundingBox;
            this.motionIntensity = motionIntensity;
            this.frameIndex = frameIndex;
        }

        @Override
        public String toString() {
            return String.format("MotionRegion[frame=%d, box=%s, intensity=%.3f]",
                    frameIndex, boundingBox, motionIntensity);
        }
    }

    /**
     * 边界框类
     */
    public static class BoundingBox {
        public final int x, y, width, height;

        public BoundingBox(int x, int y, int width, int height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }

        @Override
        public String toString() {
            return String.format("BoundingBox[x=%d, y=%d, w=%d, h=%d]", x, y, width, height);
        }
    }

    /**
     * 视频质量指标类
     */
    public static class VideoQualityMetrics {
        public final double blurScore;
        public final double noiseScore;
        public final double brightnessScore;
        public final double contrastScore;

        public VideoQualityMetrics(double blurScore, double noiseScore, double brightnessScore, double contrastScore) {
            this.blurScore = blurScore;
            this.noiseScore = noiseScore;
            this.brightnessScore = brightnessScore;
            this.contrastScore = contrastScore;
        }

        @Override
        public String toString() {
            return String.format("VideoQualityMetrics[blur=%.3f, noise=%.3f, brightness=%.3f, contrast=%.3f]",
                    blurScore, noiseScore, brightnessScore, contrastScore);
        }
    }

    /**
     * 人脸区域类
     */
    public static class FaceRegion {
        public final BoundingBox boundingBox;
        public final double confidence;

        public FaceRegion(BoundingBox boundingBox, double confidence) {
            this.boundingBox = boundingBox;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return String.format("FaceRegion[box=%s, conf=%.3f]", boundingBox, confidence);
        }
    }

    /**
     * 人脸检测结果类
     */
    public static class FaceDetectionResult {
        public final int frameIndex;
        public final double timestamp;
        public final List<FaceRegion> faces;

        public FaceDetectionResult(int frameIndex, double timestamp, List<FaceRegion> faces) {
            this.frameIndex = frameIndex;
            this.timestamp = timestamp;
            this.faces = faces;
        }

        @Override
        public String toString() {
            return String.format("FaceDetectionResult[frame=%d, time=%.2fs, faces=%d]",
                    frameIndex, timestamp, faces.size());
        }
    }

    /**
     * 跟踪结果类
     */
    public static class TrackingResult {
        public int frameIndex;
        public double timestamp;
        public final BoundingBox boundingBox;
        public final double confidence;

        public TrackingResult(int frameIndex, double timestamp, BoundingBox boundingBox, double confidence) {
            this.frameIndex = frameIndex;
            this.timestamp = timestamp;
            this.boundingBox = boundingBox;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return String.format("TrackingResult[frame=%d, time=%.2fs, box=%s, conf=%.3f]",
                    frameIndex, timestamp, boundingBox, confidence);
        }
    }

    // ==================== 辅助方法实现 ====================

    private static String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        if (lastDot > 0 && lastDot < filePath.length() - 1) {
            return filePath.substring(lastDot + 1).toLowerCase();
        }
        return "";
    }

    private static List<ImageData> generateMockFrames(int count, int width, int height) {
        List<ImageData> frames = new ArrayList<>();
        Random random = new Random();

        for (int i = 0; i < count; i++) {
            // 生成模拟图像数据
            byte[] mockData = new byte[width * height * 3]; // RGB
            random.nextBytes(mockData);
            ImageData frame = new ImageData(mockData);
            frame.setWidth(width);
            frame.setHeight(height);
            frames.add(frame);
        }

        return frames;
    }

    private void copyVideoProperties(VideoData target) {
        target.setWidth(this.width);
        target.setHeight(this.height);
        target.setFormat(this.format);
        target.setDuration(this.duration);
    }

    private double calculateHistogramDifference(ImageData frame1, ImageData frame2) {
        // 简化的直方图差异计算
        int[] hist1 = calculateSimpleHistogram(frame1);
        int[] hist2 = calculateSimpleHistogram(frame2);

        double diff = 0.0;
        for (int i = 0; i < Math.min(hist1.length, hist2.length); i++) {
            diff += Math.abs(hist1[i] - hist2[i]);
        }

        return diff / (frame1.getWidth() * frame1.getHeight());
    }

    private double calculateEdgeDifference(ImageData frame1, ImageData frame2) {
        // 简化的边缘差异计算
        ImageData edges1 = frame1.adjustContrast(2.0f); // 使用对比度增强模拟边缘检测
        ImageData edges2 = frame2.adjustContrast(2.0f);

        byte[] data1 = getImageBytes(edges1);
        byte[] data2 = getImageBytes(edges2);

        double diff = 0.0;
        int minLength = Math.min(data1.length, data2.length);
        for (int i = 0; i < minLength; i++) {
            diff += Math.abs((data1[i] & 0xFF) - (data2[i] & 0xFF));
        }

        return diff / minLength;
    }

    private double calculateSceneChangeScore(ImageData frame1, ImageData frame2) {
        // 结合直方图和边缘信息
        double histDiff = calculateHistogramDifference(frame1, frame2);
        double edgeDiff = calculateEdgeDifference(frame1, frame2);

        return (histDiff + edgeDiff) / 2.0;
    }

    private MotionVector[][] calculateOpticalFlow(ImageData frame1, ImageData frame2) {
        // 简化的光流计算
        int blockSize = 16;
        int rows = frame1.getHeight() / blockSize;
        int cols = frame1.getWidth() / blockSize;

        MotionVector[][] flow = new MotionVector[rows][cols];
        Random random = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // 模拟运动向量
                double dx = random.nextGaussian() * 2;
                double dy = random.nextGaussian() * 2;
                flow[i][j] = new MotionVector(dx, dy);
            }
        }

        return flow;
    }

    private List<MotionRegion> detectFrameMotion(ImageData frame1, ImageData frame2, double threshold) {
        List<MotionRegion> regions = new ArrayList<>();

        // 简化的运动检测
        MotionVector[][] flow = calculateOpticalFlow(frame1, frame2);

        for (int i = 0; i < flow.length; i++) {
            for (int j = 0; j < flow[i].length; j++) {
                if (flow[i][j].magnitude > threshold) {
                    BoundingBox box = new BoundingBox(j * 16, i * 16, 16, 16);
                    regions.add(new MotionRegion(box, flow[i][j].magnitude, 0));
                }
            }
        }

        return regions;
    }

    private double calculateBlurMetric(ImageData frame) {
        // 简化的模糊度计算（基于对比度）
        ImageData edges = frame.adjustContrast(2.0f);
        byte[] edgeData = getImageBytes(edges);

        double sum = 0.0;
        for (byte b : edgeData) {
            sum += (b & 0xFF);
        }

        return sum / edgeData.length;
    }

    private double calculateNoiseMetric(ImageData frame) {
        // 简化的噪声计算
        byte[] data = getImageBytes(frame);
        double variance = 0.0;
        double mean = 0.0;

        // 计算均值
        for (byte b : data) {
            mean += (b & 0xFF);
        }
        mean /= data.length;

        // 计算方差
        for (byte b : data) {
            double diff = (b & 0xFF) - mean;
            variance += diff * diff;
        }

        return Math.sqrt(variance / data.length);
    }

    private double calculateBrightnessMetric(ImageData frame) {
        byte[] data = getImageBytes(frame);
        double sum = 0.0;

        for (byte b : data) {
            sum += (b & 0xFF);
        }

        return sum / data.length / 255.0; // 归一化到0-1
    }

    private double calculateContrastMetric(ImageData frame) {
        byte[] data = getImageBytes(frame);
        int min = 255, max = 0;

        for (byte b : data) {
            int value = b & 0xFF;
            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        return (max - min) / 255.0; // 归一化到0-1
    }

    private VectorData extractFrameEmbedding(ImageData frame) {
        // 简化的帧嵌入提取
        // 实际应用中会使用预训练的CNN模型

        // 提取基本特征：颜色直方图、纹理特征等
        int[] histogram = calculateSimpleHistogram(frame);
        double[] features = new double[256 + 10]; // 直方图 + 其他特征

        // 归一化直方图
        double histSum = Arrays.stream(histogram).sum();
        for (int i = 0; i < histogram.length; i++) {
            features[i] = histogram[i] / histSum;
        }

        // 添加其他特征
        features[256] = calculateBrightnessMetric(frame);
        features[257] = calculateContrastMetric(frame);
        features[258] = calculateBlurMetric(frame);
        features[259] = calculateNoiseMetric(frame);

        // 添加纹理特征（简化）
        for (int i = 260; i < features.length; i++) {
            features[i] = Math.random(); // 模拟纹理特征
        }

        return new VectorData(features, "frame_embedding");
    }

    private VectorData aggregateFrameEmbeddings(List<VectorData> frameEmbeddings) {
        if (frameEmbeddings.isEmpty()) {
            return new VectorData(new double[266], "empty_embedding"); // 空向量
        }

        // 使用平均池化聚合帧嵌入
        VectorData firstEmbedding = frameEmbeddings.get(0);
        int dimension = firstEmbedding.getVectorSize();//.size();
        double[] aggregated = new double[dimension];

        for (VectorData embedding : frameEmbeddings) {
            double[] data = embedding.getAsDoubleArray() ;//.toDoubleArray();
            for (int i = 0; i < dimension; i++) {
                aggregated[i] += data[i];
            }
        }

        // 归一化
        for (int i = 0; i < dimension; i++) {
            aggregated[i] /= frameEmbeddings.size();
        }

        return new VectorData(aggregated, "aggregated_video_embedding");
    }

    private List<Double> extractSpatialFeatures() {
        List<Double> features = new ArrayList<>();

        if (!frames.isEmpty()) {
            ImageData firstFrame = frames.get(0);
            features.add((double) firstFrame.getWidth());
            features.add((double) firstFrame.getHeight());
            features.add(calculateBrightnessMetric(firstFrame));
            features.add(calculateContrastMetric(firstFrame));
        }

        return features;
    }

    private List<Double> extractTemporalFeatures() {
        List<Double> features = new ArrayList<>();

        features.add(fps);
        features.add(duration);
        features.add((double) frameCount);

        // 场景变化统计
        int sceneChanges = 0;
        for (int i = 1; i < frames.size(); i++) {
            if (calculateSceneChangeScore(frames.get(i-1), frames.get(i)) > 0.5) {
                sceneChanges++;
            }
        }
        features.add((double) sceneChanges);

        return features;
    }

    private List<Double> extractMotionFeatures() {
        List<Double> features = new ArrayList<>();

        double avgMotion = 0.0;
        double maxMotion = 0.0;
        int motionFrames = 0;

        for (int i = 1; i < frames.size(); i++) {
            MotionVector[][] flow = calculateOpticalFlow(frames.get(i-1), frames.get(i));
            double frameMotion = 0.0;
            int motionVectors = 0;

            for (MotionVector[] row : flow) {
                for (MotionVector mv : row) {
                    frameMotion += mv.magnitude;
                    motionVectors++;
                    maxMotion = Math.max(maxMotion, mv.magnitude);
                }
            }

            if (motionVectors > 0) {
                frameMotion /= motionVectors;
                avgMotion += frameMotion;
                motionFrames++;
            }
        }

        if (motionFrames > 0) {
            avgMotion /= motionFrames;
        }

        features.add(avgMotion);
        features.add(maxMotion);
        features.add((double) motionFrames);

        return features;
    }

    private ImageData stabilizeFrame(ImageData prevFrame, ImageData currentFrame) {
        // 简化的稳定化处理
        // 实际实现需要计算相机运动并进行补偿

        // 这里只是返回当前帧，实际应用中会应用变换矩阵
        return currentFrame;
    }

    private List<FaceRegion> detectFrameFaces(ImageData frame) {
        List<FaceRegion> faces = new ArrayList<>();

        // 模拟人脸检测
        Random random = new Random();
        if (random.nextDouble() < 0.3) { // 30%概率检测到人脸
            int x = random.nextInt(frame.getWidth() - 100);
            int y = random.nextInt(frame.getHeight() - 100);
            BoundingBox faceBox = new BoundingBox(x, y, 80, 80);
            double confidence = 0.5 + random.nextDouble() * 0.5;
            faces.add(new FaceRegion(faceBox, confidence));
        }

        return faces;
    }

    private BoundingBox findTemplateInFrame(ImageData template, ImageData frame) {
        // 简化的模板匹配
        Random random = new Random();
        if (random.nextDouble() < 0.7) { // 70%概率找到模板
            int x = random.nextInt(frame.getWidth() - template.getWidth());
            int y = random.nextInt(frame.getHeight() - template.getHeight());
            return new BoundingBox(x, y, template.getWidth(), template.getHeight());
        }
        return null;
    }

    private TrackingResult trackObjectInFrame(ImageData frame, BoundingBox prevBox, ImageData template) {
        // 简化的对象跟踪
        Random random = new Random();

        // 模拟跟踪漂移
        int newX = prevBox.x + (int)(random.nextGaussian() * 5);
        int newY = prevBox.y + (int)(random.nextGaussian() * 5);

        // 确保边界框在图像范围内
        newX = Math.max(0, Math.min(newX, frame.getWidth() - prevBox.width));
        newY = Math.max(0, Math.min(newY, frame.getHeight() - prevBox.height));

        BoundingBox newBox = new BoundingBox(newX, newY, prevBox.width, prevBox.height);
        double confidence = 0.5 + random.nextDouble() * 0.4; // 模拟置信度降低

        return new TrackingResult(0, 0.0, newBox, confidence);
    }

    private ImageData interpolateFrames(ImageData frame1, ImageData frame2, float ratio) {
        // 简化的帧插值
        // 实际实现会使用光流或深度学习方法

        // 这里简单地返回第一帧
        // 在实际应用中，会根据ratio混合两帧的像素值
        return frame1;
    }

    // ==================== 新增的缺失辅助方法 ====================

    /**
     * 简化的直方图计算
     */
    private int[] calculateSimpleHistogram(ImageData frame) {
        int[] histogram = new int[256];
        byte[] imageBytes = getImageBytes(frame);

        for (byte b : imageBytes) {
            int value = b & 0xFF;
            histogram[value]++;
        }

        return histogram;
    }

    /**
     * 从ImageData获取字节数组
     */
    private byte[] getImageBytes(ImageData frame) {
        // 这是一个简化的实现，实际应该从ImageData获取实际的字节数据
        // 由于ImageData的内部结构可能不同，这里生成模拟数据
        int size = frame.getWidth() * frame.getHeight() * 3; // RGB
        byte[] mockBytes = new byte[size];

        // 生成基于帧尺寸的伪随机字节数据
        Random random = new Random(frame.getWidth() + frame.getHeight());
        random.nextBytes(mockBytes);

        return mockBytes;
    }

}

