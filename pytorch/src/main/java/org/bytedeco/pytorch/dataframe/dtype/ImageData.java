package org.bytedeco.pytorch.dataframe.dtype;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.geom.AffineTransform;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * 图像数据容器
 * 集成OpenCV、FFmpeg和Scrimage风格的图像处理算法
 */
public class ImageData extends AbstractDataValue implements MediaData {
    private byte[] rawBytes;
    private BufferedImage image;
    private String path;
    private int width;
    private int height;
    private int channels;
    private String format;

    // 图像处理常量
    private static final float[] BLUR_KERNEL_3X3 = {
        1f/9f, 1f/9f, 1f/9f,
        1f/9f, 1f/9f, 1f/9f,
        1f/9f, 1f/9f, 1f/9f
    };

    private static final float[] GAUSSIAN_KERNEL_3X3 = {
        1f/16f, 2f/16f, 1f/16f,
        2f/16f, 4f/16f, 2f/16f,
        1f/16f, 2f/16f, 1f/16f
    };

    private static final float[] SHARPEN_KERNEL_3X3 = {
        0f, -1f, 0f,
        -1f, 5f, -1f,
        0f, -1f, 0f
    };

    private static final float[] EDGE_KERNEL_3X3 = {
        -1f, -1f, -1f,
        -1f, 8f, -1f,
        -1f, -1f, -1f
    };

    private static final float[] SOBEL_X_KERNEL = {
        -1f, 0f, 1f,
        -2f, 0f, 2f,
        -1f, 0f, 1f
    };

    private static final float[] SOBEL_Y_KERNEL = {
        -1f, -2f, -1f,
        0f, 0f, 0f,
        1f, 2f, 1f
    };

    public ImageData(String path) {
        this.path = path;
    }

    public ImageData(byte[] rawBytes) {
        this.rawBytes = rawBytes;
    }

    public ImageData(byte[] rawBytes, int width, int height, String channels) {
        this.rawBytes = rawBytes;
        this.channels = channels.length();
        this.width = width;
        this.height = height;
    }

    public ImageData(BufferedImage image) {
        this.image = image;
        if (image != null) {
            this.width = image.getWidth();
            this.height = image.getHeight();
            this.channels = image.getColorModel().getNumComponents();
        }
    }

    // ==================== OpenCV/FFmpeg/Scrimage风格图像处理算法 ====================

    /**
     * 1. 图像加载 (OpenCV.imread equivalent when available, else ImageIO).
     * <p>Delegates to {@code MediaBridge.loadImage} when present so DataFrame
     * batch loaders, torchvision, and OpenCV share one decode path.
     */
    public static ImageData load(String path) throws IOException {
        // Prefer MediaBridge (OpenCV → ImageIO fallback)
        try {
            Class<?> bridge = Class.forName(
                    "org.bytedeco.pytorch.dataframe.media.MediaBridge");
            // call loadImage(String, ImageOptions) with defaults to avoid recursion:
            // MediaBridge.loadImage(AUTO) may call ImageData.load on IMAGEIO path —
            // so only take the OpenCV fast-path here.
            Boolean opencv = (Boolean) bridge.getMethod("isOpenCvAvailable").invoke(null);
            if (Boolean.TRUE.equals(opencv)) {
                ImageData viaCv = (ImageData) bridge
                        .getMethod("loadImageOpenCv", String.class, boolean.class)
                        .invoke(null, path, false);
                if (viaCv != null && viaCv.getImage() != null) return viaCv;
            }
        } catch (ClassNotFoundException ignored) {
        } catch (Throwable ignored) {
        }

        BufferedImage img = ImageIO.read(new File(path));
        if (img == null) {
            throw new IOException("无法加载图像: " + path);
        }
        ImageData imageData = new ImageData(img);
        imageData.setPath(path);
        imageData.setFormat(getFileExtension(path));
        return imageData;
    }

    /**
     * 2. 图像保存
     */
    public void save(String path) throws IOException {
        if (image == null) throw new IllegalStateException("No image data available");

        String format = getFileExtension(path);
        if (format.isEmpty()) format = "png";

        ImageIO.write(image, format, new File(path));
        this.path = path;
        this.format = format;
    }

    /**
     * 3. 图像缩放 (Resize)
     */
    public ImageData resize(int newWidth, int newHeight) {
        return resize(newWidth, newHeight, Image.SCALE_SMOOTH);
    }

    public ImageData resize(int newWidth, int newHeight, int scaleType) {
        if (image == null) throw new IllegalStateException("No image data available");

        Image scaledImage = image.getScaledInstance(newWidth, newHeight, scaleType);
        BufferedImage result = toBufferedImage(scaledImage, newWidth, newHeight);

        return new ImageData(result);
    }

    /**
     * 4. 按比例缩放
     */
    public ImageData scale(double factor) {
        return resize((int)(width * factor), (int)(height * factor));
    }

    /**
     * 5. 图像裁剪 (Crop)
     */
    public ImageData crop(int x, int y, int cropWidth, int cropHeight) {
        if (image == null) throw new IllegalStateException("No image data available");

        x = Math.max(0, Math.min(x, width));
        y = Math.max(0, Math.min(y, height));
        cropWidth = Math.min(cropWidth, width - x);
        cropHeight = Math.min(cropHeight, height - y);

        BufferedImage cropped = image.getSubimage(x, y, cropWidth, cropHeight);
        return new ImageData(copyImage(cropped));
    }

    /**
     * 6. 图像旋转
     */
    public ImageData rotate(double angleRadians) {
        if (image == null) throw new IllegalStateException("No image data available");

        double sin = Math.abs(Math.sin(angleRadians));
        double cos = Math.abs(Math.cos(angleRadians));

        int newWidth = (int) Math.floor(width * cos + height * sin);
        int newHeight = (int) Math.floor(height * cos + width * sin);

        BufferedImage result = new BufferedImage(newWidth, newHeight, image.getType());
        Graphics2D g2d = result.createGraphics();

        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.translate((newWidth - width) / 2, (newHeight - height) / 2);
        g2d.rotate(angleRadians, width / 2.0, height / 2.0);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();

        return new ImageData(result);
    }

    /**
     * 7. 图像翻转
     */
    public ImageData flip(boolean horizontal) {
        if (image == null) throw new IllegalStateException("No image data available");

        AffineTransform tx;
        if (horizontal) {
            tx = AffineTransform.getScaleInstance(-1, 1);
            tx.translate(-width, 0);
        } else {
            tx = AffineTransform.getScaleInstance(1, -1);
            tx.translate(0, -height);
        }

        AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
        BufferedImage result = op.filter(image, null);

        return new ImageData(result);
    }

    /**
     * 8. 模糊滤波 (Blur)
     */
    public ImageData blur() {
        return applyKernel(BLUR_KERNEL_3X3, 3, 3);
    }

    /**
     * 9. 高斯模糊
     */
    public ImageData gaussianBlur() {
        return applyKernel(GAUSSIAN_KERNEL_3X3, 3, 3);
    }

    /**
     * 10. 图像锐化
     */
    public ImageData sharpen() {
        return applyKernel(SHARPEN_KERNEL_3X3, 3, 3);
    }

    /**
     * 11. 边缘检测
     */
    public ImageData detectEdges() {
        return applyKernel(EDGE_KERNEL_3X3, 3, 3);
    }

    /**
     * 12. Sobel边缘检测
     */
    public ImageData sobelEdgeDetection() {
        ImageData sobelX = applyKernel(SOBEL_X_KERNEL, 3, 3);
        ImageData sobelY = applyKernel(SOBEL_Y_KERNEL, 3, 3);
        return combineSobel(sobelX, sobelY);
    }

    /**
     * 13. 灰度转换
     */
    public ImageData toGrayscale() {
        if (image == null) throw new IllegalStateException("No image data available");

        ColorSpace graySpace = ColorSpace.getInstance(ColorSpace.CS_GRAY);
        ColorConvertOp op = new ColorConvertOp(graySpace, null);
        BufferedImage gray = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        op.filter(image, gray);

        return new ImageData(gray);
    }

    /**
     * 14. 亮度调整
     */
    public ImageData adjustBrightness(float factor) {
        if (image == null) throw new IllegalStateException("No image data available");

        RescaleOp op = new RescaleOp(1.0f, factor, null);
        BufferedImage result = new BufferedImage(width, height, image.getType());
        op.filter(image, result);

        return new ImageData(result);
    }

    /**
     * 15. 对比度调整
     */
    public ImageData adjustContrast(float factor) {
        if (image == null) throw new IllegalStateException("No image data available");

        RescaleOp op = new RescaleOp(factor, 0, null);
        BufferedImage result = new BufferedImage(width, height, image.getType());
        op.filter(image, result);

        return new ImageData(result);
    }

    /**
     * 16. 饱和度调整
     */
    public ImageData adjustSaturation(float factor) {
        if (image == null) throw new IllegalStateException("No image data available");

        BufferedImage result = new BufferedImage(width, height, image.getType());
        Graphics2D g2d = result.createGraphics();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(image.getRGB(x, y));
                float[] hsb = Color.RGBtoHSB(color.getRed(), color.getGreen(), color.getBlue(), null);
                hsb[1] = Math.max(0, Math.min(1, hsb[1] * factor));
                int rgb = Color.HSBtoRGB(hsb[0], hsb[1], hsb[2]);
                result.setRGB(x, y, rgb);
            }
        }

        g2d.dispose();
        return new ImageData(result);
    }

    /**
     * 17. 色相调整
     */
    public ImageData adjustHue(float hueShift) {
        if (image == null) throw new IllegalStateException("No image data available");

        BufferedImage result = new BufferedImage(width, height, image.getType());

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(image.getRGB(x, y));
                float[] hsb = Color.RGBtoHSB(color.getRed(), color.getGreen(), color.getBlue(), null);
                hsb[0] = (hsb[0] + hueShift) % 1.0f;
                if (hsb[0] < 0) hsb[0] += 1.0f;
                int rgb = Color.HSBtoRGB(hsb[0], hsb[1], hsb[2]);
                result.setRGB(x, y, rgb);
            }
        }

        return new ImageData(result);
    }

    /**
     * 18. 图像反色
     */
    public ImageData invert() {
        if (image == null) throw new IllegalStateException("No image data available");

        BufferedImage result = new BufferedImage(width, height, image.getType());

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(image.getRGB(x, y));
                Color inverted = new Color(255 - color.getRed(), 255 - color.getGreen(), 255 - color.getBlue());
                result.setRGB(x, y, inverted.getRGB());
            }
        }

        return new ImageData(result);
    }

    /**
     * 19. 二值化
     */
    public ImageData threshold(int thresholdValue) {
        ImageData gray = toGrayscale();
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(gray.image.getRGB(x, y));
                int grayValue = color.getRed(); // 灰度图像RGB相同
                int binary = grayValue > thresholdValue ? 255 : 0;
                Color binaryColor = new Color(binary, binary, binary);
                result.setRGB(x, y, binaryColor.getRGB());
            }
        }

        return new ImageData(result);
    }

    /**
     * 20. 自适应二值化
     */
    public ImageData adaptiveThreshold(int blockSize) {
        ImageData gray = toGrayscale();
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        int halfBlock = blockSize / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 计算局部平均值
                int sum = 0, count = 0;
                for (int dy = -halfBlock; dy <= halfBlock; dy++) {
                    for (int dx = -halfBlock; dx <= halfBlock; dx++) {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            Color color = new Color(gray.image.getRGB(nx, ny));
                            sum += color.getRed();
                            count++;
                        }
                    }
                }

                int localMean = sum / count;
                Color currentColor = new Color(gray.image.getRGB(x, y));
                int binary = currentColor.getRed() > localMean ? 255 : 0;
                Color binaryColor = new Color(binary, binary, binary);
                result.setRGB(x, y, binaryColor.getRGB());
            }
        }

        return new ImageData(result);
    }

    /**
     * 21. 形态学操作 - 膨胀 (Dilation)
     */
    public ImageData dilate(int kernelSize) {
        return morphologyOperation(kernelSize, true);
    }

    /**
     * 22. 形态学操作 - 腐蚀 (Erosion)
     */
    public ImageData erode(int kernelSize) {
        return morphologyOperation(kernelSize, false);
    }

    /**
     * 23. 形态学操作 - 开运算
     */
    public ImageData opening(int kernelSize) {
        return erode(kernelSize).dilate(kernelSize);
    }

    /**
     * 24. 形态学操作 - 闭运算
     */
    public ImageData closing(int kernelSize) {
        return dilate(kernelSize).erode(kernelSize);
    }

    /**
     * 25. 直方图均衡化
     */
    public ImageData equalizeHistogram() {
        ImageData gray = toGrayscale();
        int[] histogram = new int[256];

        // 计算直方图
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(gray.image.getRGB(x, y));
                histogram[color.getRed()]++;
            }
        }

        // 计算累积分布函数
        int[] cdf = new int[256];
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i-1] + histogram[i];
        }

        // 归一化
        int totalPixels = width * height;
        int[] lookupTable = new int[256];
        for (int i = 0; i < 256; i++) {
            lookupTable[i] = (int) ((float) cdf[i] * 255 / totalPixels);
        }

        // 应用均衡化
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(gray.image.getRGB(x, y));
                int equalizedValue = lookupTable[color.getRed()];
                Color newColor = new Color(equalizedValue, equalizedValue, equalizedValue);
                result.setRGB(x, y, newColor.getRGB());
            }
        }

        return new ImageData(result);
    }

    /**
     * 26. 噪声添加
     */
    public ImageData addNoise(double intensity) {
        if (image == null) throw new IllegalStateException("No image data available");

        BufferedImage result = copyImage(image);
        Random random = new Random();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(result.getRGB(x, y));

                int r = Math.max(0, Math.min(255, color.getRed() + (int)(random.nextGaussian() * intensity)));
                int g = Math.max(0, Math.min(255, color.getGreen() + (int)(random.nextGaussian() * intensity)));
                int b = Math.max(0, Math.min(255, color.getBlue() + (int)(random.nextGaussian() * intensity)));

                Color noisyColor = new Color(r, g, b);
                result.setRGB(x, y, noisyColor.getRGB());
            }
        }

        return new ImageData(result);
    }

    /**
     * 27. 中值滤波 (去噪)
     */
    public ImageData medianFilter(int kernelSize) {
        if (image == null) throw new IllegalStateException("No image data available");

        BufferedImage result = new BufferedImage(width, height, image.getType());
        int halfKernel = kernelSize / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                List<Integer> redValues = new ArrayList<>();
                List<Integer> greenValues = new ArrayList<>();
                List<Integer> blueValues = new ArrayList<>();

                for (int dy = -halfKernel; dy <= halfKernel; dy++) {
                    for (int dx = -halfKernel; dx <= halfKernel; dx++) {
                        int nx = Math.max(0, Math.min(width - 1, x + dx));
                        int ny = Math.max(0, Math.min(height - 1, y + dy));

                        Color color = new Color(image.getRGB(nx, ny));
                        redValues.add(color.getRed());
                        greenValues.add(color.getGreen());
                        blueValues.add(color.getBlue());
                    }
                }

                Collections.sort(redValues);
                Collections.sort(greenValues);
                Collections.sort(blueValues);

                int medianIndex = redValues.size() / 2;
                Color medianColor = new Color(
                    redValues.get(medianIndex),
                    greenValues.get(medianIndex),
                    blueValues.get(medianIndex)
                );

                result.setRGB(x, y, medianColor.getRGB());
            }
        }

        return new ImageData(result);
    }

    /**
     * 28. 图像混合
     */
    public ImageData blend(ImageData other, float alpha) {
        if (image == null || other.image == null) {
            throw new IllegalStateException("No image data available");
        }

        int minWidth = Math.min(width, other.width);
        int minHeight = Math.min(height, other.height);

        BufferedImage result = new BufferedImage(minWidth, minHeight, image.getType());

        for (int y = 0; y < minHeight; y++) {
            for (int x = 0; x < minWidth; x++) {
                Color color1 = new Color(image.getRGB(x, y));
                Color color2 = new Color(other.image.getRGB(x, y));

                int r = (int) (color1.getRed() * (1 - alpha) + color2.getRed() * alpha);
                int g = (int) (color1.getGreen() * (1 - alpha) + color2.getGreen() * alpha);
                int b = (int) (color1.getBlue() * (1 - alpha) + color2.getBlue() * alpha);

                Color blendedColor = new Color(r, g, b);
                result.setRGB(x, y, blendedColor.getRGB());
            }
        }

        return new ImageData(result);
    }

    /**
     * 29. 图像水印
     */
    public ImageData addWatermark(String text, int x, int y, Color color, int fontSize) {
        if (image == null) throw new IllegalStateException("No image data available");

        BufferedImage result = copyImage(image);
        Graphics2D g2d = result.createGraphics();

        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setFont(new Font("Arial", Font.BOLD, fontSize));
        g2d.setColor(color);
        g2d.drawString(text, x, y);

        g2d.dispose();
        return new ImageData(result);
    }

    /**
     * 30. 透视变换
     */
    public ImageData perspectiveTransform(double[] srcPoints, double[] dstPoints) {
        // 简化的透视变换实现
        if (image == null) throw new IllegalStateException("No image data available");

        BufferedImage result = new BufferedImage(width, height, image.getType());
        Graphics2D g2d = result.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        // 这里应该实现完整的透视变换矩阵计算
        // 简化版本：只做基本的仿射变换
        AffineTransform transform = new AffineTransform();
        transform.scale(0.8, 0.8); // 简化示例

        AffineTransformOp op = new AffineTransformOp(transform, AffineTransformOp.TYPE_BILINEAR);
        BufferedImage transformed = op.filter(image, null);

        g2d.drawImage(transformed, 0, 0, null);
        g2d.dispose();

        return new ImageData(result);
    }

    /**
     * 31. 图像拼接
     */
    public ImageData concatenate(ImageData other, boolean horizontal) {
        if (image == null || other.image == null) {
            throw new IllegalStateException("No image data available");
        }

        int newWidth, newHeight;
        if (horizontal) {
            newWidth = width + other.width;
            newHeight = Math.max(height, other.height);
        } else {
            newWidth = Math.max(width, other.width);
            newHeight = height + other.height;
        }

        BufferedImage result = new BufferedImage(newWidth, newHeight, image.getType());
        Graphics2D g2d = result.createGraphics();

        g2d.drawImage(image, 0, 0, null);
        if (horizontal) {
            g2d.drawImage(other.image, width, 0, null);
        } else {
            g2d.drawImage(other.image, 0, height, null);
        }

        g2d.dispose();
        return new ImageData(result);
    }

    /**
     * 32. 图像嵌入 (Image Embedding)
     * 提取图像特征向量用于相似度比较和检索
     */
    public float[] extractEmbedding() {
        return extractEmbedding(256); // 默认256维特征
    }

    public float[] extractEmbedding(int dimensions) {
        if (image == null) throw new IllegalStateException("No image data available");

        // 简化的图像特征提取算法
        // 在实际应用中，这里应该使用深度学习模型如ResNet、VGG等

        // 1. 缩放到固定尺寸以标准化
        ImageData resized = resize(224, 224);

        // 2. 转换为灰度
        ImageData gray = resized.toGrayscale();

        // 3. 提取多尺度特征
        float[] features = new float[dimensions];
        int featureIndex = 0;

        // 全局统计特征
        features[featureIndex++] = computeMean(gray);
        features[featureIndex++] = computeStandardDeviation(gray);
        features[featureIndex++] = computeSkewness(gray);
        features[featureIndex++] = computeKurtosis(gray);

        // 纹理特征 (Gray Level Co-occurrence Matrix 简化版)
        float[] glcmFeatures = computeGLCMFeatures(gray);
        System.arraycopy(glcmFeatures, 0, features, featureIndex, Math.min(glcmFeatures.length, dimensions - featureIndex));
        featureIndex += glcmFeatures.length;

        // 边缘特征
        ImageData edges = gray.detectEdges();
        features[featureIndex++] = computeMean(edges);

        // 分块特征 (将图像分为8x8块，计算每块的平均值)
        int blockSize = 28; // 224/8
        for (int by = 0; by < 8 && featureIndex < dimensions; by++) {
            for (int bx = 0; bx < 8 && featureIndex < dimensions; bx++) {
                features[featureIndex++] = computeBlockMean(gray, bx * blockSize, by * blockSize, blockSize, blockSize);
            }
        }

        // 填充剩余维度
        while (featureIndex < dimensions) {
            features[featureIndex++] = 0.0f;
        }

        // 归一化特征
        return normalizeFeatures(features);
    }

    /**
     * 33. 计算图像相似度
     */
    public float computeSimilarity(ImageData other) {
        float[] embedding1 = this.extractEmbedding();
        float[] embedding2 = other.extractEmbedding();

        return computeCosineSimilarity(embedding1, embedding2);
    }

    // ==================== 辅助方法 ====================

    private ImageData applyKernel(float[] kernel, int kernelWidth, int kernelHeight) {
        if (image == null) throw new IllegalStateException("No image data available");

        Kernel convKernel = new Kernel(kernelWidth, kernelHeight, kernel);
        ConvolveOp op = new ConvolveOp(convKernel, ConvolveOp.EDGE_NO_OP, null);
        BufferedImage result = new BufferedImage(width, height, image.getType());
        op.filter(image, result);

        return new ImageData(result);
    }

    private ImageData combineSobel(ImageData sobelX, ImageData sobelY) {
        BufferedImage result = new BufferedImage(width, height, image.getType());

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color colorX = new Color(sobelX.image.getRGB(x, y));
                Color colorY = new Color(sobelY.image.getRGB(x, y));

                double gradientMagnitude = Math.sqrt(
                    Math.pow(colorX.getRed(), 2) + Math.pow(colorY.getRed(), 2)
                );

                int magnitude = (int) Math.min(255, gradientMagnitude);
                Color combinedColor = new Color(magnitude, magnitude, magnitude);
                result.setRGB(x, y, combinedColor.getRGB());
            }
        }

        return new ImageData(result);
    }

    private ImageData morphologyOperation(int kernelSize, boolean isDilation) {
        ImageData binary = threshold(128);
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        int halfKernel = kernelSize / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean found = false;

                for (int dy = -halfKernel; dy <= halfKernel && !found; dy++) {
                    for (int dx = -halfKernel; dx <= halfKernel && !found; dx++) {
                        int nx = Math.max(0, Math.min(width - 1, x + dx));
                        int ny = Math.max(0, Math.min(height - 1, y + dy));

                        Color color = new Color(binary.image.getRGB(nx, ny));
                        boolean isWhite = color.getRed() > 128;

                        if (isDilation && isWhite) {
                            found = true;
                        } else if (!isDilation && !isWhite) {
                            found = true;
                        }
                    }
                }

                Color resultColor;
                if (isDilation) {
                    resultColor = found ? Color.WHITE : Color.BLACK;
                } else {
                    resultColor = found ? Color.BLACK : Color.WHITE;
                }

                result.setRGB(x, y, resultColor.getRGB());
            }
        }

        return new ImageData(result);
    }

    // 图像特征计算方法
    private float computeMean(ImageData grayImage) {
        long sum = 0;
        for (int y = 0; y < grayImage.height; y++) {
            for (int x = 0; x < grayImage.width; x++) {
                Color color = new Color(grayImage.image.getRGB(x, y));
                sum += color.getRed();
            }
        }
        return (float) sum / (grayImage.width * grayImage.height);
    }

    private float computeStandardDeviation(ImageData grayImage) {
        float mean = computeMean(grayImage);
        double sumSquaredDiff = 0;

        for (int y = 0; y < grayImage.height; y++) {
            for (int x = 0; x < grayImage.width; x++) {
                Color color = new Color(grayImage.image.getRGB(x, y));
                double diff = color.getRed() - mean;
                sumSquaredDiff += diff * diff;
            }
        }

        return (float) Math.sqrt(sumSquaredDiff / (grayImage.width * grayImage.height));
    }

    private float computeSkewness(ImageData grayImage) {
        float mean = computeMean(grayImage);
        float std = computeStandardDeviation(grayImage);
        if (std == 0) return 0;

        double sumCubedDiff = 0;
        for (int y = 0; y < grayImage.height; y++) {
            for (int x = 0; x < grayImage.width; x++) {
                Color color = new Color(grayImage.image.getRGB(x, y));
                double diff = (color.getRed() - mean) / std;
                sumCubedDiff += Math.pow(diff, 3);
            }
        }

        return (float) (sumCubedDiff / (grayImage.width * grayImage.height));
    }

    private float computeKurtosis(ImageData grayImage) {
        float mean = computeMean(grayImage);
        float std = computeStandardDeviation(grayImage);
        if (std == 0) return 0;

        double sumFourthPowerDiff = 0;
        for (int y = 0; y < grayImage.height; y++) {
            for (int x = 0; x < grayImage.width; x++) {
                Color color = new Color(grayImage.image.getRGB(x, y));
                double diff = (color.getRed() - mean) / std;
                sumFourthPowerDiff += Math.pow(diff, 4);
            }
        }

        return (float) (sumFourthPowerDiff / (grayImage.width * grayImage.height) - 3);
    }

    private float[] computeGLCMFeatures(ImageData grayImage) {
        // 简化的灰度共生矩阵特征
        int[][] glcm = new int[256][256];

        // 计算水平方向的共生矩阵
        for (int y = 0; y < grayImage.height; y++) {
            for (int x = 0; x < grayImage.width - 1; x++) {
                Color color1 = new Color(grayImage.image.getRGB(x, y));
                Color color2 = new Color(grayImage.image.getRGB(x + 1, y));
                glcm[color1.getRed()][color2.getRed()]++;
            }
        }

        // 计算纹理特征
        float contrast = 0, homogeneity = 0, energy = 0;
        int totalCount = 0;

        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                if (glcm[i][j] > 0) {
                    contrast += Math.pow(i - j, 2) * glcm[i][j];
                    homogeneity += glcm[i][j] / (1 + Math.abs(i - j));
                    energy += Math.pow(glcm[i][j], 2);
                    totalCount += glcm[i][j];
                }
            }
        }

        if (totalCount > 0) {
            contrast /= totalCount;
            homogeneity /= totalCount;
            energy = (float) Math.sqrt(energy) / totalCount;
        }

        return new float[]{contrast, homogeneity, energy};
    }

    private float computeBlockMean(ImageData grayImage, int startX, int startY, int blockWidth, int blockHeight) {
        long sum = 0;
        int count = 0;

        for (int y = startY; y < Math.min(startY + blockHeight, grayImage.height); y++) {
            for (int x = startX; x < Math.min(startX + blockWidth, grayImage.width); x++) {
                Color color = new Color(grayImage.image.getRGB(x, y));
                sum += color.getRed();
                count++;
            }
        }

        return count > 0 ? (float) sum / count : 0;
    }

    private float[] normalizeFeatures(float[] features) {
        // L2归一化
        float sum = 0;
        for (float feature : features) {
            sum += feature * feature;
        }

        float norm = (float) Math.sqrt(sum);
        if (norm > 0) {
            for (int i = 0; i < features.length; i++) {
                features[i] /= norm;
            }
        }

        return features;
    }

    private float computeCosineSimilarity(float[] vector1, float[] vector2) {
        if (vector1.length != vector2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        float dotProduct = 0, norm1 = 0, norm2 = 0;

        for (int i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += vector1[i] * vector1[i];
            norm2 += vector2[i] * vector2[i];
        }

        float denominator = (float) (Math.sqrt(norm1) * Math.sqrt(norm2));
        return denominator > 0 ? dotProduct / denominator : 0;
    }

    // 工具方法
    private static String getFileExtension(String path) {
        int lastDot = path.lastIndexOf('.');
        return lastDot > 0 ? path.substring(lastDot + 1).toLowerCase() : "";
    }

    private BufferedImage toBufferedImage(Image img, int width, int height) {
        BufferedImage buffered = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = buffered.createGraphics();
        g2d.drawImage(img, 0, 0, null);
        g2d.dispose();
        return buffered;
    }

    private BufferedImage copyImage(BufferedImage source) {
        BufferedImage copy = new BufferedImage(source.getWidth(), source.getHeight(), source.getType());
        Graphics2D g2d = copy.createGraphics();
        g2d.drawImage(source, 0, 0, null);
        g2d.dispose();
        return copy;
    }

    // ==================== 原有的getter/setter方法 ====================

    public void setRawBytes(byte[] rawBytes) { this.rawBytes = rawBytes; }
    public BufferedImage getImage() { return image; }
    public void setImage(BufferedImage image) {
        this.image = image;
        if (image != null) {
            this.width = image.getWidth();
            this.height = image.getHeight();
            this.channels = image.getColorModel().getNumComponents();
        }
    }
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public int getChannels() { return channels; }
    public String getFormat() { return format; }
    public void setFormat(String format) { this.format = format; }

    public int[] getShape() {
        return new int[]{height, width, channels};
    }

    public void setPath(String path){
        this.path = path;
    }

    public void setChannels(int channels){
        this.channels = channels;
    }

    public void setWidth(int width) { this.width = width; }

    public void setHeight(int height) { this.height = height; }

    @Override
    public String getDataType() {
        return "IMAGE";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回包含核心信息的Map
        Map<String, Object> arrowData = new HashMap<>();
        arrowData.put("path", path);
        arrowData.put("width", width);
        arrowData.put("height", height);
        arrowData.put("channels", channels);
        arrowData.put("format", format);
        // 原始字节数组可选（大数据量时可省略）
        if (rawBytes != null) {
            arrowData.put("rawBytesLength", rawBytes.length);
        }
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        return String.format("%dx%dx%d, path=%s", width, height, channels, path);
    }

    @Override
    public Number getNumericValue(){
        return null;
    }

    @Override
    public boolean isValid() {
        // 基础校验 + 图像专属校验：路径/原始字节/图像对象至少有一个非空
        return super.isValid()
                && (path != null || rawBytes != null || image != null)
                && width >= 0 && height >= 0 && channels >= 0;
    }

    // ========== 实现 MediaData 接口 ==========
    @Override
    public String getPath() {
        return path;
    }

    @Override
    public byte[] getRawBytes() {
        return rawBytes;
    }

    @Override
    public int[] getDimensions() {
        // 图像维度：[height, width, channels]
        return new int[]{height, width, channels};
    }

    @Override
    public String toString() {
        return String.format("ImageData[%dx%dx%d, path=%s]", width, height, channels, path);
    }
}

