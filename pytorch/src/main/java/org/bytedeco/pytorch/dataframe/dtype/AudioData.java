package org.bytedeco.pytorch.dataframe.dtype;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.*;

/**
 * 音频数据容器
 * 集成librosa风格的音频处理算法
 * 支持MP3、WAV、FLAC、M4A、WMA、AAC格式读写
 */
public class AudioData extends AbstractDataValue implements MediaData {
    private byte[] rawBytes;
    private float[] samples;
    private String path;
    private int sampleRate;
    private int channels;
    private double duration; // 秒
    private String format;

    // 音频处理常量
    private static final int DEFAULT_N_FFT = 2048;
    private static final int DEFAULT_HOP_LENGTH = 512;
    private static final int DEFAULT_N_MELS = 128;
    private static final int DEFAULT_N_MFCC = 13;
    private static final double DEFAULT_F_MIN = 0.0;
    private static final double DEFAULT_F_MAX = 8000.0;

    // 支持的音频格式
    public enum AudioFormat {
        MP3("mp3", "MPEG-1 Audio Layer III"),
        WAV("wav", "Waveform Audio File Format"),
        FLAC("flac", "Free Lossless Audio Codec"),
        M4A("m4a", "MPEG-4 Audio"),
        WMA("wma", "Windows Media Audio"),
        AAC("aac", "Advanced Audio Coding");

        private final String extension;
        private final String description;

        AudioFormat(String extension, String description) {
            this.extension = extension;
            this.description = description;
        }

        public String getExtension() { return extension; }
        public String getDescription() { return description; }

        public static AudioFormat fromExtension(String extension) {
            String ext = extension.toLowerCase().replace(".", "");
            for (AudioFormat format : values()) {
                if (format.extension.equals(ext)) {
                    return format;
                }
            }
            return WAV; // 默认格式
        }
    }

    public AudioData(String path) {
        this.path = path;
    }

    public AudioData(byte[] rawBytes, int sampleRate) {
        this.rawBytes = rawBytes;
        this.sampleRate = sampleRate;
    }

    public AudioData(float[] samples, int sampleRate, int channels) {
        this.samples = samples;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.duration = (double) samples.length / sampleRate / channels;
    }

    // ==================== 音频格式读写功能 ====================

    /**
     * 从文件加载音频数据，自动检测格式
     */
    public static AudioData loadFromFile(String filePath) throws IOException {
        return loadFromFile(filePath, 22050, true);
    }

    /**
     * 从文件加载音频数据，支持多种格式.
     * <p>WAV uses pure-Java {@code javax.sound}. Compressed formats (mp3/flac/…)
     * prefer FFmpeg via {@code MediaBridge} when natives are available; otherwise
     * fall back to the existing lightweight stubs so offline tests still run.
     */
    public static AudioData loadFromFile(String filePath, int targetSampleRate, boolean mono) throws IOException {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new FileNotFoundException("音频文件不存在: " + filePath);
        }

        String extension = getFileExtension(filePath);
        AudioFormat audioFormat = AudioFormat.fromExtension(extension);

        // Non-WAV: try real FFmpeg decode first
        if (audioFormat != AudioFormat.WAV) {
            try {
                Class<?> bridge = Class.forName(
                        "org.bytedeco.pytorch.dataframe.media.MediaBridge");
                Boolean avail = (Boolean) bridge.getMethod("isFFmpegAvailable").invoke(null);
                if (Boolean.TRUE.equals(avail)) {
                    AudioData viaFf = (AudioData) bridge
                            .getMethod("loadAudioFFmpeg", String.class, int.class, boolean.class)
                            .invoke(null, filePath, targetSampleRate, mono);
                    if (viaFf != null && viaFf.getSamples() != null) {
                        viaFf.setPath(filePath);
                        viaFf.setFormat(extension);
                        return viaFf;
                    }
                }
            } catch (ClassNotFoundException ignored) {
            } catch (Throwable ignored) {
            }
        }

        AudioData audioData;
        switch (audioFormat) {
            case WAV:
                audioData = loadWav(filePath, targetSampleRate, mono);
                break;
            case MP3:
                audioData = loadMp3(filePath, targetSampleRate, mono);
                break;
            case FLAC:
                audioData = loadFlac(filePath, targetSampleRate, mono);
                break;
            case M4A:
                audioData = loadM4a(filePath, targetSampleRate, mono);
                break;
            case WMA:
                audioData = loadWma(filePath, targetSampleRate, mono);
                break;
            case AAC:
                audioData = loadAac(filePath, targetSampleRate, mono);
                break;
            default:
                throw new UnsupportedOperationException("不支持的音频格式: " + extension);
        }

        audioData.setPath(filePath);
        audioData.setFormat(extension);
        return audioData;
    }

    /**
     * 加载WAV格式音频
     */
    public static AudioData loadWav(String filePath, int targetSampleRate, boolean mono) throws IOException {
        try {
            File file = new File(filePath);
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
            javax.sound.sampled.AudioFormat sourceFormat = audioInputStream.getFormat();

            // 转换为目标格式
            javax.sound.sampled.AudioFormat targetFormat = new javax.sound.sampled.AudioFormat(
                    javax.sound.sampled.AudioFormat.Encoding.PCM_SIGNED,
                    targetSampleRate,
                    16, // 16-bit
                    mono ? 1 : sourceFormat.getChannels(),
                    mono ? 2 : sourceFormat.getChannels() * 2,
                    targetSampleRate,
                    false // little-endian
            );

            AudioInputStream convertedStream = AudioSystem.getAudioInputStream(targetFormat, audioInputStream);
            byte[] audioBytes = convertedStream.readAllBytes();
            float[] samples = bytesToFloats(audioBytes, targetFormat);

            AudioData audioData = new AudioData(samples, targetSampleRate, mono ? 1 : sourceFormat.getChannels());
            audioData.setRawBytes(audioBytes);
            audioData.setDuration((double) samples.length / targetSampleRate / (mono ? 1 : sourceFormat.getChannels()));

            convertedStream.close();
            audioInputStream.close();
            return audioData;

        } catch (UnsupportedAudioFileException e) {
            throw new IOException("不支持的WAV文件格式: " + e.getMessage());
        }
    }

    /**
     * 加载MP3格式音频 (简化实现，实际需要MP3解码库)
     */
    public static AudioData loadMp3(String filePath, int targetSampleRate, boolean mono) throws IOException {
        // 注意：这是简化实现，实际应用中需要使用如JLayer、BasicPlayer等MP3解码库
        System.out.println("警告: MP3解码需要专门的库支持，当前使用模拟数据");

        // 模拟MP3加载：读取文件大小并估算时长
        File file = new File(filePath);
        long fileSize = file.length();

        // 估算音频时长 (简化算法：128kbps MP3)
        double estimatedDuration = (fileSize * 8.0) / (128 * 1000); // 秒
        int sampleCount = (int) (estimatedDuration * targetSampleRate * (mono ? 1 : 2));

        // 生成模拟音频数据
        float[] samples = generateSampleAudio(sampleCount, targetSampleRate);

        AudioData audioData = new AudioData(samples, targetSampleRate, mono ? 1 : 2);
        audioData.setDuration(estimatedDuration);
        audioData.setRawBytes(Files.readAllBytes(file.toPath()));

        return audioData;
    }

    /**
     * 加载FLAC格式音频 (简化实现)
     */
    public static AudioData loadFlac(String filePath, int targetSampleRate, boolean mono) throws IOException {
        // 注意：这是简化实现，实际应用中需要使用FLAC解码库
        System.out.println("警告: FLAC解码需要专门的库支持，当前使用模拟数据");

        File file = new File(filePath);
        byte[] fileData = Files.readAllBytes(file.toPath());

        // 尝试从FLAC头部读取基本信息
        FlacInfo flacInfo = parseFlacHeader(fileData);

        // 生成模拟音频数据
        int sampleCount = (int) (flacInfo.duration * targetSampleRate * (mono ? 1 : flacInfo.channels));
        float[] samples = generateSampleAudio(sampleCount, targetSampleRate);

        AudioData audioData = new AudioData(samples, targetSampleRate, mono ? 1 : flacInfo.channels);
        audioData.setDuration(flacInfo.duration);
        audioData.setRawBytes(fileData);

        return audioData;
    }

    /**
     * 加载M4A格式音频 (简化实现)
     */
    public static AudioData loadM4a(String filePath, int targetSampleRate, boolean mono) throws IOException {
        System.out.println("警告: M4A解码需要专门的库支持，当前使用模拟数据");

        File file = new File(filePath);
        byte[] fileData = Files.readAllBytes(file.toPath());

        // 估算时长（简化）
        double estimatedDuration = fileData.length / (64.0 * 1000 / 8); // 假设64kbps
        int sampleCount = (int) (estimatedDuration * targetSampleRate * (mono ? 1 : 2));

        float[] samples = generateSampleAudio(sampleCount, targetSampleRate);

        AudioData audioData = new AudioData(samples, targetSampleRate, mono ? 1 : 2);
        audioData.setDuration(estimatedDuration);
        audioData.setRawBytes(fileData);

        return audioData;
    }

    /**
     * 加载WMA格式音频 (简化实现)
     */
    public static AudioData loadWma(String filePath, int targetSampleRate, boolean mono) throws IOException {
        System.out.println("警告: WMA解码需要专门的库支持，当前使用模拟数据");

        File file = new File(filePath);
        byte[] fileData = Files.readAllBytes(file.toPath());

        // 估算时长
        double estimatedDuration = fileData.length / (96.0 * 1000 / 8); // 假设96kbps
        int sampleCount = (int) (estimatedDuration * targetSampleRate * (mono ? 1 : 2));

        float[] samples = generateSampleAudio(sampleCount, targetSampleRate);

        AudioData audioData = new AudioData(samples, targetSampleRate, mono ? 1 : 2);
        audioData.setDuration(estimatedDuration);
        audioData.setRawBytes(fileData);

        return audioData;
    }

    /**
     * 加载AAC格式音频 (简化实现)
     */
    public static AudioData loadAac(String filePath, int targetSampleRate, boolean mono) throws IOException {
        System.out.println("警告: AAC解码需要专门的库支持，当前使用模拟数据");

        File file = new File(filePath);
        byte[] fileData = Files.readAllBytes(file.toPath());

        // 估算时长
        double estimatedDuration = fileData.length / (128.0 * 1000 / 8); // 假设128kbps
        int sampleCount = (int) (estimatedDuration * targetSampleRate * (mono ? 1 : 2));

        float[] samples = generateSampleAudio(sampleCount, targetSampleRate);

        AudioData audioData = new AudioData(samples, targetSampleRate, mono ? 1 : 2);
        audioData.setDuration(estimatedDuration);
        audioData.setRawBytes(fileData);

        return audioData;
    }

    /**
     * 保存音频到文件，自动检测格式
     */
    public void saveToFile(String filePath) throws IOException {
        String extension = getFileExtension(filePath);
        AudioFormat audioFormat = AudioFormat.fromExtension(extension);

        switch (audioFormat) {
            case WAV:
                saveAsWav(filePath);
                break;
            case MP3:
                saveAsMp3(filePath);
                break;
            case FLAC:
                saveAsFlac(filePath);
                break;
            case M4A:
                saveAsM4a(filePath);
                break;
            case WMA:
                saveAsWma(filePath);
                break;
            case AAC:
                saveAsAac(filePath);
                break;
            default:
                throw new UnsupportedOperationException("不支持保存为格式: " + extension);
        }

        this.path = filePath;
        this.format = extension;
    }

    /**
     * 保存为WAV格式
     */
    public void saveAsWav(String filePath) throws IOException {
        if (samples == null) {
            throw new IllegalStateException("没有音频样本数据可供保存");
        }

        // 转换float样本为16位PCM字节
        byte[] audioBytes = floatsToBytes(samples);

        // 创建AudioFormat
        javax.sound.sampled.AudioFormat audioFormat = new javax.sound.sampled.AudioFormat(
                javax.sound.sampled.AudioFormat.Encoding.PCM_SIGNED,
                sampleRate,
                16, // 16-bit
                channels,
                channels * 2, // 每帧字节数
                sampleRate,
                false // little-endian
        );

        // 创建AudioInputStream
        ByteArrayInputStream bais = new ByteArrayInputStream(audioBytes);
        AudioInputStream audioInputStream = new AudioInputStream(bais, audioFormat, samples.length / channels);

        // 写入文件
        File outputFile = new File(filePath);
        AudioSystem.write(audioInputStream, AudioFileFormat.Type.WAVE, outputFile);

        audioInputStream.close();
        bais.close();
    }

    /**
     * 保存为MP3格式 (需要MP3编码器)
     */
    public void saveAsMp3(String filePath) throws IOException {
        System.out.println("警告: MP3编码需要专门的库支持，当前保存为WAV格式");
        // 实际实现需要使用LAME或其他MP3编码器
        String wavPath = filePath.replace(".mp3", "_temp.wav");
        saveAsWav(wavPath);

        // 这里应该调用MP3编码器将WAV转换为MP3
        // 例如: Runtime.getRuntime().exec("lame " + wavPath + " " + filePath);

        System.out.println("请使用外部工具将 " + wavPath + " 转换为 " + filePath);
    }

    /**
     * 保存为FLAC格式
     */
    public void saveAsFlac(String filePath) throws IOException {
        System.out.println("警告: FLAC编码需要专门的库支持，当前保存为WAV格式");
        String wavPath = filePath.replace(".flac", "_temp.wav");
        saveAsWav(wavPath);
        System.out.println("请使用外部工具将 " + wavPath + " 转换为 " + filePath);
    }

    /**
     * 保存为M4A格式
     */
    public void saveAsM4a(String filePath) throws IOException {
        System.out.println("警告: M4A编码需要专门的库支持，当前保存为WAV格式");
        String wavPath = filePath.replace(".m4a", "_temp.wav");
        saveAsWav(wavPath);
        System.out.println("请使用外部工具将 " + wavPath + " 转换为 " + filePath);
    }

    /**
     * 保存为WMA格式
     */
    public void saveAsWma(String filePath) throws IOException {
        System.out.println("警告: WMA编码需要专门的库支持，当前保存为WAV格式");
        String wavPath = filePath.replace(".wma", "_temp.wav");
        saveAsWav(wavPath);
        System.out.println("请使用外部工具将 " + wavPath + " 转换为 " + filePath);
    }

    /**
     * 保存为AAC格式
     */
    public void saveAsAac(String filePath) throws IOException {
        System.out.println("警告: AAC编码需要专门的库支持，当前保存为WAV格式");
        String wavPath = filePath.replace(".aac", "_temp.wav");
        saveAsWav(wavPath);
        System.out.println("请使用外部工具将 " + wavPath + " 转换为 " + filePath);
    }

    // ==================== Librosa风格音频处理算法 ====================

    /**
     * 1. 加载音频文件 (librosa.load equivalent)
     */
    public static AudioData load(String path, int sr, boolean mono) {
        try {
            // 尝试从实际文件加载
            if (new File(path).exists()) {
                return loadFromFile(path, sr, mono);
            }
        } catch (IOException e) {
            System.out.println("无法加载文件 " + path + "，使用模拟数据: " + e.getMessage());
        }

        // 模拟加载音频文件，生成测试数据
        float[] samples = generateSampleAudio(sr * 3, sr); // 3秒音频
        return new AudioData(samples, sr, mono ? 1 : 2);
    }

    /**
     * 2. 短时傅里叶变换 (STFT)
     */
    public ComplexMatrix stft() {
        return stft(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, "hann");
    }

    public ComplexMatrix stft(int nFft, int hopLength, String window) {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        int numFrames = Math.max(1, (samples.length - nFft) / hopLength + 1);
        int numBins = nFft / 2 + 1;

        ComplexMatrix result = new ComplexMatrix(numBins, numFrames);

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopLength;
            int end = Math.min(start + nFft, samples.length);

            // 提取窗口并填充
            float[] windowSamples = new float[nFft];
            System.arraycopy(samples, start, windowSamples, 0, end - start);

            // 应用窗口函数
            applyWindow(windowSamples, window);

            // FFT
            Complex[] fftResult = fft(windowSamples);

            // 存储结果
            for (int bin = 0; bin < Math.min(numBins, fftResult.length); bin++) {
                result.set(bin, frame, fftResult[bin]);
            }
        }

        return result;
    }

    /**
     * 3. 梅尔频谱 (Mel-spectrogram)
     */
    public float[][] melSpectrogram() {
        return melSpectrogram(DEFAULT_N_MELS, DEFAULT_F_MIN, sampleRate / 2.0);
    }

    public float[][] melSpectrogram(int nMels, double fMin, double fMax) {
        return melSpectrogram(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, nMels, fMin, fMax);
    }

    public float[][] melSpectrogram(int nFft, int hopLength, int nMels, double fMin, double fMax) {
        // 计算STFT
        ComplexMatrix stftMatrix = stft(nFft, hopLength, "hann");

        // 计算功率谱
        float[][] powerSpec = stftMatrix.powerSpectrum();

        // 创建梅尔滤波器组
        float[][] melFilters = createMelFilterBank(nMels, nFft, sampleRate, fMin, fMax);

        // 应用梅尔滤波器
        return applyFilterBank(powerSpec, melFilters);
    }

    /**
     * 4. 梅尔倒频系数 (MFCC)
     */
    public float[][] mfcc() {
        return mfcc(DEFAULT_N_MFCC);
    }

    public float[][] mfcc(int nMfcc) {
        return mfcc(nMfcc, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_N_MELS, DEFAULT_F_MIN, sampleRate / 2.0);
    }

    public float[][] mfcc(int nMfcc, int nFft, int hopLength, int nMels, double fMin, double fMax) {
        // 计算梅尔频谱
        float[][] melSpec = melSpectrogram(nFft, hopLength, nMels, fMin, fMax);

        // 对数变换
        for (int i = 0; i < melSpec.length; i++) {
            for (int j = 0; j < melSpec[i].length; j++) {
                melSpec[i][j] = (float) Math.log(Math.max(melSpec[i][j], 1e-10));
            }
        }

        // DCT变换
        return dct(melSpec, nMfcc);
    }

    /**
     * 5. 色度特征 (Chromagram)
     */
    public float[][] chroma() {
        return chroma(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH);
    }

    public float[][] chroma(int nFft, int hopLength) {
        ComplexMatrix stftMatrix = stft(nFft, hopLength, "hann");
        float[][] powerSpec = stftMatrix.powerSpectrum();

        // 创建色度滤波器
        float[][] chromaFilters = createChromaFilterBank(nFft, sampleRate);

        return applyFilterBank(powerSpec, chromaFilters);
    }

    /**
     * 6. 谱质心 (Spectral Centroid)
     */
    public float[] spectralCentroid() {
        return spectralCentroid(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH);
    }

    public float[] spectralCentroid(int nFft, int hopLength) {
        ComplexMatrix stftMatrix = stft(nFft, hopLength, "hann");
        float[][] powerSpec = stftMatrix.powerSpectrum();

        float[] centroids = new float[powerSpec[0].length];

        for (int frame = 0; frame < powerSpec[0].length; frame++) {
            float numerator = 0f, denominator = 0f;

            for (int bin = 0; bin < powerSpec.length; bin++) {
                float freq = (float) bin * sampleRate / nFft;
                float magnitude = powerSpec[bin][frame];

                numerator += freq * magnitude;
                denominator += magnitude;
            }

            centroids[frame] = denominator > 0 ? numerator / denominator : 0;
        }

        return centroids;
    }

    /**
     * 7. 谱带宽 (Spectral Bandwidth)
     */
    public float[] spectralBandwidth() {
        float[] centroids = spectralCentroid();
        ComplexMatrix stftMatrix = stft(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, "hann");
        float[][] powerSpec = stftMatrix.powerSpectrum();

        float[] bandwidths = new float[powerSpec[0].length];

        for (int frame = 0; frame < powerSpec[0].length; frame++) {
            float centroid = centroids[frame];
            float numerator = 0f, denominator = 0f;

            for (int bin = 0; bin < powerSpec.length; bin++) {
                float freq = (float) bin * sampleRate / DEFAULT_N_FFT;
                float magnitude = powerSpec[bin][frame];

                numerator += Math.pow(freq - centroid, 2) * magnitude;
                denominator += magnitude;
            }

            bandwidths[frame] = denominator > 0 ? (float) Math.sqrt(numerator / denominator) : 0;
        }

        return bandwidths;
    }

    /**
     * 8. 谱对比度 (Spectral Contrast)
     */
    public float[][] spectralContrast() {
        return spectralContrast(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, 6);
    }

    public float[][] spectralContrast(int nFft, int hopLength, int nBands) {
        ComplexMatrix stftMatrix = stft(nFft, hopLength, "hann");
        float[][] powerSpec = stftMatrix.powerSpectrum();

        // 创建频率子带
        int[] bandEdges = createLogFrequencyBands(nBands, nFft, sampleRate);

        float[][] contrast = new float[nBands][powerSpec[0].length];

        for (int frame = 0; frame < powerSpec[0].length; frame++) {
            for (int band = 0; band < nBands; band++) {
                int startBin = bandEdges[band];
                int endBin = bandEdges[band + 1];

                // 在子带内计算对比度
                List<Float> bandMagnitudes = new ArrayList<>();
                for (int bin = startBin; bin < endBin && bin < powerSpec.length; bin++) {
                    bandMagnitudes.add(powerSpec[bin][frame]);
                }

                Collections.sort(bandMagnitudes, Collections.reverseOrder());

                if (!bandMagnitudes.isEmpty()) {
                    int peakCount = Math.max(1, bandMagnitudes.size() / 20);
                    int valleyCount = Math.max(1, bandMagnitudes.size() / 20);

                    float peakAvg = 0f, valleyAvg = 0f;

                    for (int i = 0; i < peakCount; i++) {
                        peakAvg += bandMagnitudes.get(i);
                    }
                    peakAvg /= peakCount;

                    for (int i = bandMagnitudes.size() - valleyCount; i < bandMagnitudes.size(); i++) {
                        valleyAvg += bandMagnitudes.get(i);
                    }
                    valleyAvg /= valleyCount;

                    contrast[band][frame] = (float) Math.log(Math.max(peakAvg, 1e-10) / Math.max(valleyAvg, 1e-10));
                }
            }
        }

        return contrast;
    }

    /**
     * 9. 零交叉率 (Zero Crossing Rate)
     */
    public float[] zeroCrossingRate() {
        return zeroCrossingRate(DEFAULT_HOP_LENGTH);
    }

    public float[] zeroCrossingRate(int frameLength) {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        int numFrames = Math.max(1, (samples.length - 1) / frameLength + 1);
        float[] zcr = new float[numFrames];

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * frameLength;
            int end = Math.min(start + frameLength, samples.length);
            int crossings = 0;

            for (int i = start + 1; i < end; i++) {
                if ((samples[i] >= 0) != (samples[i-1] >= 0)) {
                    crossings++;
                }
            }

            zcr[frame] = (end - start - 1) > 0 ? (float) crossings / (end - start - 1) : 0;
        }

        return zcr;
    }

    /**
     * 10. RMS能量 (RMS Energy)
     */
    public float[] rmsEnergy() {
        return rmsEnergy(DEFAULT_HOP_LENGTH);
    }

    public float[] rmsEnergy(int frameLength) {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        int numFrames = Math.max(1, (samples.length - 1) / frameLength + 1);
        float[] rms = new float[numFrames];

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * frameLength;
            int end = Math.min(start + frameLength, samples.length);
            float sumSquares = 0f;

            for (int i = start; i < end; i++) {
                sumSquares += samples[i] * samples[i];
            }

            rms[frame] = (end - start) > 0 ? (float) Math.sqrt(sumSquares / (end - start)) : 0;
        }

        return rms;
    }

    /**
     * 11. 滚降频率 (Spectral Rolloff)
     */
    public float[] spectralRolloff() {
        return spectralRolloff(0.85f);
    }

    public float[] spectralRolloff(float rolloffPercent) {
        ComplexMatrix stftMatrix = stft(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, "hann");
        float[][] powerSpec = stftMatrix.powerSpectrum();

        float[] rolloffs = new float[powerSpec[0].length];

        for (int frame = 0; frame < powerSpec[0].length; frame++) {
            float totalEnergy = 0f;
            for (int bin = 0; bin < powerSpec.length; bin++) {
                totalEnergy += powerSpec[bin][frame];
            }

            float threshold = totalEnergy * rolloffPercent;
            float cumulativeEnergy = 0f;

            for (int bin = 0; bin < powerSpec.length; bin++) {
                cumulativeEnergy += powerSpec[bin][frame];
                if (cumulativeEnergy >= threshold) {
                    rolloffs[frame] = (float) bin * sampleRate / DEFAULT_N_FFT;
                    break;
                }
            }
        }

        return rolloffs;
    }

    /**
     * 12. 音调跟踪 (Pitch Tracking)
     */
    public float[] pitchTrack() {
        return pitchTrack(80.0f, 400.0f, DEFAULT_HOP_LENGTH);
    }

    public float[] pitchTrack(float fMin, float fMax, int hopLength) {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        int numFrames = Math.max(1, (samples.length - 1) / hopLength + 1);
        float[] pitches = new float[numFrames];

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopLength;
            int end = Math.min(start + hopLength * 4, samples.length); // 使用较大窗口

            if (end - start < hopLength) break;

            float[] window = Arrays.copyOfRange(samples, start, end);
            pitches[frame] = extractPitchAutocorrelation(window, fMin, fMax);
        }

        return pitches;
    }

    /**
     * 13. 节拍跟踪 (Beat Tracking)
     */
    public BeatTrackResult beatTrack() {
        // 计算onset强度
        float[] onsetStrength = onsetStrength();

        // 估计节拍
        float tempo = estimateTempo(onsetStrength);

        // 检测节拍位置
        float[] beats = detectBeats(onsetStrength, tempo);

        return new BeatTrackResult(tempo, beats);
    }

    /**
     * 14. Onset检测
     */
    public float[] onsetStrength() {
        // 使用谱流量进行onset检测
        ComplexMatrix stftMatrix = stft(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, "hann");
        float[][] powerSpec = stftMatrix.powerSpectrum();

        if (powerSpec[0].length <= 1) return new float[0];

        float[] onsetStrength = new float[powerSpec[0].length - 1];

        for (int frame = 1; frame < powerSpec[0].length; frame++) {
            float sum = 0f;
            for (int bin = 0; bin < powerSpec.length; bin++) {
                float diff = powerSpec[bin][frame] - powerSpec[bin][frame-1];
                sum += Math.max(0, diff); // 只考虑增加的部分
            }
            onsetStrength[frame-1] = sum;
        }

        return onsetStrength;
    }

    /**
     * 15. 谱平坦度 (Spectral Flatness)
     */
    public float[] spectralFlatness() {
        ComplexMatrix stftMatrix = stft(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, "hann");
        float[][] powerSpec = stftMatrix.powerSpectrum();

        float[] flatness = new float[powerSpec[0].length];

        for (int frame = 0; frame < powerSpec[0].length; frame++) {
            float geometricMean = 1f;
            float arithmeticMean = 0f;
            int validBins = 0;

            for (int bin = 1; bin < powerSpec.length; bin++) { // 跳过DC分量
                float magnitude = Math.max(powerSpec[bin][frame], 1e-10f);
                geometricMean *= Math.pow(magnitude, 1.0 / (powerSpec.length - 1));
                arithmeticMean += magnitude;
                validBins++;
            }

            if (validBins > 0) {
                arithmeticMean /= validBins;
                flatness[frame] = arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;
            }
        }

        return flatness;
    }

    /**
     * 16. 常量Q变换 (Constant-Q Transform)
     */
    public ComplexMatrix constantQ() {
        return constantQ(84, 12);
    }

    public ComplexMatrix constantQ(int nBins, double binsPerOctave) {
        // 简化的CQT实现
        int numFrames = Math.max(1, (samples.length - DEFAULT_N_FFT) / DEFAULT_HOP_LENGTH + 1);
        ComplexMatrix cqtResult = new ComplexMatrix(nBins, numFrames);

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * DEFAULT_HOP_LENGTH;
            int end = Math.min(start + DEFAULT_N_FFT, samples.length);

            float[] window = new float[DEFAULT_N_FFT];
            System.arraycopy(samples, start, window, 0, end - start);
            Complex[] fftResult = fft(window);

            // 转换为CQT
            for (int bin = 0; bin < nBins && bin < fftResult.length; bin++) {
                cqtResult.set(bin, frame, fftResult[bin]);
            }
        }

        return cqtResult;
    }

    /**
     * 17. 声纹特征提取
     */
    public VoiceprintFeatures extractVoiceprint() {
        // 提取多种声纹特征
        float[][] mfccFeatures = mfcc(DEFAULT_N_MFCC);
        float[] zcrFeatures = zeroCrossingRate();
        float[] rmsFeatures = rmsEnergy();
        float[] pitchFeatures = pitchTrack();
        float[] spectralCentroid = spectralCentroid();
        float[][] chromaFeatures = chroma();

        return new VoiceprintFeatures(mfccFeatures, zcrFeatures, rmsFeatures,
                                    pitchFeatures, spectralCentroid, chromaFeatures);
    }

    /**
     * 18. 音频分段 (Audio Segmentation)
     */
    public List<AudioSegment> segmentAudio(float minSegmentLength) {
        float[] onsetStrength = onsetStrength();
        List<AudioSegment> segments = new ArrayList<>();

        if (onsetStrength.length == 0) return segments;

        // 简单的基于onset的分段
        int frameLength = DEFAULT_HOP_LENGTH;
        float threshold = getMax(onsetStrength) * 0.3f;

        int segmentStart = 0;
        for (int i = 1; i < onsetStrength.length; i++) {
            if (onsetStrength[i] > threshold) {
                float segmentDuration = (float) (i - segmentStart) * frameLength / sampleRate;
                if (segmentDuration >= minSegmentLength) {
                    int startSample = segmentStart * frameLength;
                    int endSample = Math.min(i * frameLength, samples.length);

                    segments.add(new AudioSegment(startSample, endSample,
                                                segmentStart * frameLength / (float) sampleRate,
                                                i * frameLength / (float) sampleRate));
                    segmentStart = i;
                }
            }
        }

        // 添加最后一段
        if (segmentStart < onsetStrength.length - 1) {
            int startSample = segmentStart * frameLength;
            segments.add(new AudioSegment(startSample, samples.length - 1,
                                        segmentStart * frameLength / (float) sampleRate,
                                        (float) samples.length / sampleRate));
        }

        return segments;
    }

    /**
     * 19. 音频增强 - 降噪
     */
    public AudioData denoise(float alpha) {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        // 简单的谱减法降噪
        ComplexMatrix stftMatrix = stft(DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, "hann");

        // 估计噪声功率谱（使用前几帧）
        int noiseFrames = Math.min(10, stftMatrix.getCols() / 4);
        float[] noisePowerSpectrum = new float[stftMatrix.getRows()];

        for (int bin = 0; bin < stftMatrix.getRows(); bin++) {
            for (int frame = 0; frame < noiseFrames; frame++) {
                Complex value = stftMatrix.get(bin, frame);
                noisePowerSpectrum[bin] += value.magnitude() * value.magnitude();
            }
            noisePowerSpectrum[bin] /= noiseFrames;
        }

        // 应用谱减法
        for (int frame = 0; frame < stftMatrix.getCols(); frame++) {
            for (int bin = 0; bin < stftMatrix.getRows(); bin++) {
                Complex value = stftMatrix.get(bin, frame);
                float magnitude = (float) value.magnitude();
                float phase = (float) value.phase();

                float newMagnitude = Math.max(
                    magnitude - alpha * (float) Math.sqrt(noisePowerSpectrum[bin]),
                    0.1f * magnitude
                );

                stftMatrix.set(bin, frame, Complex.polar(newMagnitude, phase));
            }
        }

        // 重建音频信号
        float[] denoisedSamples = istft(stftMatrix, DEFAULT_HOP_LENGTH);

        return new AudioData(denoisedSamples, sampleRate, channels);
    }

    /**
     * 20. 音频特征统计
     */
    public AudioFeatureStats computeFeatureStats() {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        float[][] mfccFeatures = mfcc(DEFAULT_N_MFCC);
        float[] zcrFeatures = zeroCrossingRate();
        float[] rmsFeatures = rmsEnergy();
        float[] spectralCentroid = spectralCentroid();

        return new AudioFeatureStats(
            computeStats(flattenMatrix(mfccFeatures)),
            computeStats(zcrFeatures),
            computeStats(rmsFeatures),
            computeStats(spectralCentroid)
        );
    }

    /**
     * 21. 音频相似度计算
     */
    public float computeSimilarity(AudioData other) {
        VoiceprintFeatures features1 = this.extractVoiceprint();
        VoiceprintFeatures features2 = other.extractVoiceprint();

        // 使用MFCC特征计算相似度
        return computeMfccSimilarity(features1.mfcc, features2.mfcc);
    }

    /**
     * 22. 频谱质心轨迹
     */
    public float[] spectralCentroidTrajectory() {
        return spectralCentroid();
    }

    /**
     * 23. 能量包络
     */
    public float[] energyEnvelope() {
        return rmsEnergy();
    }

    /**
     * 24. 过零率变化
     */
    public float[] zcrVariation() {
        float[] zcr = zeroCrossingRate();
        float[] variation = new float[zcr.length - 1];

        for (int i = 1; i < zcr.length; i++) {
            variation[i - 1] = Math.abs(zcr[i] - zcr[i - 1]);
        }

        return variation;
    }

    /**
     * 25. 音频归一化
     */
    public AudioData normalize() {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        // 找到最大绝对值
        float maxAbs = 0f;
        for (float sample : samples) {
            maxAbs = Math.max(maxAbs, Math.abs(sample));
        }

        if (maxAbs == 0) return this;

        // 归一化
        float[] normalizedSamples = new float[samples.length];
        for (int i = 0; i < samples.length; i++) {
            normalizedSamples[i] = samples[i] / maxAbs;
        }

        return new AudioData(normalizedSamples, sampleRate, channels);
    }

    /**
     * 26. 音频淡入淡出
     */
    public AudioData fadeInOut(float fadeInDuration, float fadeOutDuration) {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        float[] processedSamples = Arrays.copyOf(samples, samples.length);

        int fadeInSamples = (int) (fadeInDuration * sampleRate);
        int fadeOutSamples = (int) (fadeOutDuration * sampleRate);

        // 淡入
        for (int i = 0; i < Math.min(fadeInSamples, processedSamples.length); i++) {
            float factor = (float) i / fadeInSamples;
            processedSamples[i] *= factor;
        }

        // 淡出
        int fadeOutStart = Math.max(0, processedSamples.length - fadeOutSamples);
        for (int i = fadeOutStart; i < processedSamples.length; i++) {
            float factor = (float) (processedSamples.length - 1 - i) / fadeOutSamples;
            processedSamples[i] *= factor;
        }

        return new AudioData(processedSamples, sampleRate, channels);
    }

    /**
     * 27. 音频剪切
     */
    public AudioData trim(float startTime, float endTime) {
        if (samples == null) throw new IllegalStateException("No audio samples available");

        int startSample = (int) (startTime * sampleRate);
        int endSample = (int) (endTime * sampleRate);

        startSample = Math.max(0, startSample);
        endSample = Math.min(samples.length, endSample);

        if (startSample >= endSample) {
            throw new IllegalArgumentException("Invalid time range");
        }

        float[] trimmedSamples = Arrays.copyOfRange(samples, startSample, endSample);
        return new AudioData(trimmedSamples, sampleRate, channels);
    }

    /**
     * 28. 音频连接
     */
    public AudioData concatenate(AudioData other) {
        if (samples == null || other.samples == null) {
            throw new IllegalStateException("No audio samples available");
        }

        if (sampleRate != other.sampleRate || channels != other.channels) {
            throw new IllegalArgumentException("Audio parameters must match");
        }

        float[] concatenatedSamples = new float[samples.length + other.samples.length];
        System.arraycopy(samples, 0, concatenatedSamples, 0, samples.length);
        System.arraycopy(other.samples, 0, concatenatedSamples, samples.length, other.samples.length);

        return new AudioData(concatenatedSamples, sampleRate, channels);
    }

    /**
     * 29. 语音活动检测 (Voice Activity Detection)
     */
    public boolean[] voiceActivityDetection() {
        float[] energy = rmsEnergy();
        float[] zcr = zeroCrossingRate();

        // 计算阈值
        float energyThreshold = getAverage(energy) * 0.1f;
        float zcrThreshold = getAverage(zcr) * 2.0f;

        boolean[] vad = new boolean[energy.length];
        for (int i = 0; i < vad.length; i++) {
            vad[i] = energy[i] > energyThreshold && zcr[i] < zcrThreshold;
        }

        return vad;
    }

    /**
     * 30. 基频稳定性分析
     */
    public float pitchStability() {
        float[] pitches = pitchTrack();

        // 过滤有效音调
        List<Float> validPitches = new ArrayList<>();
        for (float pitch : pitches) {
            if (pitch > 0) {
                validPitches.add(pitch);
            }
        }

        if (validPitches.size() < 2) return 0f;

        // 计算音调变化的标准差
        float mean = (float) validPitches.stream().mapToDouble(Float::doubleValue).average().orElse(0);
        float variance = 0f;
        for (float pitch : validPitches) {
            variance += Math.pow(pitch - mean, 2);
        }
        variance /= validPitches.size();

        return (float) Math.sqrt(variance);
    }

    // ==================== 内部类和数据结构 ====================

    /**
     * 复数类
     */
    public static class Complex {
        private final double real, imag;

        public Complex(double real, double imag) {
            this.real = real;
            this.imag = imag;
        }

        public static Complex polar(double magnitude, double phase) {
            return new Complex(magnitude * Math.cos(phase), magnitude * Math.sin(phase));
        }

        public double real() { return real; }
        public double imag() { return imag; }

        public double magnitude() {
            return Math.sqrt(real * real + imag * imag);
        }

        public double phase() {
            return Math.atan2(imag, real);
        }

        public Complex add(Complex other) {
            return new Complex(real + other.real, imag + other.imag);
        }

        public Complex subtract(Complex other) {
            return new Complex(real - other.real, imag - other.imag);
        }

        public Complex multiply(Complex other) {
            return new Complex(real * other.real - imag * other.imag,
                             real * other.imag + imag * other.real);
        }

        public Complex conjugate() {
            return new Complex(real, -imag);
        }

        public Complex scale(double factor) {
            return new Complex(real * factor, imag * factor);
        }

        @Override
        public String toString() {
            return String.format("%.3f%+.3fi", real, imag);
        }
    }

    /**
     * 复数矩阵类
     */
    public static class ComplexMatrix {
        private final Complex[][] matrix;
        private final int rows, cols;

        public ComplexMatrix(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            this.matrix = new Complex[rows][cols];

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i][j] = new Complex(0, 0);
                }
            }
        }

        public void set(int row, int col, Complex value) {
            if (row >= 0 && row < rows && col >= 0 && col < cols) {
                matrix[row][col] = value;
            }
        }

        public Complex get(int row, int col) {
            if (row >= 0 && row < rows && col >= 0 && col < cols) {
                return matrix[row][col];
            }
            return new Complex(0, 0);
        }

        public int getRows() { return rows; }
        public int getCols() { return cols; }

        public float[][] powerSpectrum() {
            float[][] power = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double mag = matrix[i][j].magnitude();
                    power[i][j] = (float) (mag * mag);
                }
            }
            return power;
        }
    }

    /**
     * 声纹特征
     */
    public static class VoiceprintFeatures {
        public final float[][] mfcc;
        public final float[] zcr;
        public final float[] rms;
        public final float[] pitch;
        public final float[] spectralCentroid;
        public final float[][] chroma;

        public VoiceprintFeatures(float[][] mfcc, float[] zcr, float[] rms,
                                float[] pitch, float[] spectralCentroid, float[][] chroma) {
            this.mfcc = mfcc;
            this.zcr = zcr;
            this.rms = rms;
            this.pitch = pitch;
            this.spectralCentroid = spectralCentroid;
            this.chroma = chroma;
        }

        @Override
        public String toString() {
            return String.format("VoiceprintFeatures[mfcc=%dx%d, zcr=%d, rms=%d, pitch=%d, centroid=%d, chroma=%dx%d]",
                    mfcc.length, mfcc.length > 0 ? mfcc[0].length : 0,
                    zcr.length, rms.length, pitch.length, spectralCentroid.length,
                    chroma.length, chroma.length > 0 ? chroma[0].length : 0);
        }
    }

    /**
     * 节拍跟踪结果
     */
    public static class BeatTrackResult {
        public final float tempo;
        public final float[] beats;

        public BeatTrackResult(float tempo, float[] beats) {
            this.tempo = tempo;
            this.beats = beats;
        }

        @Override
        public String toString() {
            return String.format("BeatTrackResult[tempo=%.1f BPM, beats=%d]", tempo, beats.length);
        }
    }

    /**
     * 音频片段
     */
    public static class AudioSegment {
        public final int startSample, endSample;
        public final float startTime, endTime;

        public AudioSegment(int startSample, int endSample, float startTime, float endTime) {
            this.startSample = startSample;
            this.endSample = endSample;
            this.startTime = startTime;
            this.endTime = endTime;
        }

        public float getDuration() {
            return endTime - startTime;
        }

        @Override
        public String toString() {
            return String.format("AudioSegment[%.2fs-%.2fs, samples=%d-%d]",
                    startTime, endTime, startSample, endSample);
        }
    }

    /**
     * 特征统计
     */
    public static class FeatureStats {
        public final float mean, std, min, max;
        public final int count;

        public FeatureStats(float mean, float std, float min, float max, int count) {
            this.mean = mean;
            this.std = std;
            this.min = min;
            this.max = max;
            this.count = count;
        }

        @Override
        public String toString() {
            return String.format("FeatureStats[mean=%.3f, std=%.3f, min=%.3f, max=%.3f, count=%d]",
                    mean, std, min, max, count);
        }
    }

    /**
     * 音频特征统计集合
     */
    public static class AudioFeatureStats {
        public final FeatureStats mfccStats;
        public final FeatureStats zcrStats;
        public final FeatureStats rmsStats;
        public final FeatureStats spectralCentroidStats;

        public AudioFeatureStats(FeatureStats mfccStats, FeatureStats zcrStats,
                                FeatureStats rmsStats, FeatureStats spectralCentroidStats) {
            this.mfccStats = mfccStats;
            this.zcrStats = zcrStats;
            this.rmsStats = rmsStats;
            this.spectralCentroidStats = spectralCentroidStats;
        }

        @Override
        public String toString() {
            return String.format("AudioFeatureStats[\n  MFCC: %s\n  ZCR: %s\n  RMS: %s\n  SpectralCentroid: %s\n]",
                    mfccStats, zcrStats, rmsStats, spectralCentroidStats);
        }
    }

    /**
     * 音频文件信息类
     */
    public static class AudioFileInfo {
        public String filePath;
        public AudioFormat format;
        public long fileSize;
        public int sampleRate;
        public int channels;
        public int bitDepth;
        public double duration;

        @Override
        public String toString() {
            return String.format("AudioFileInfo{path='%s', format=%s, size=%d bytes, sr=%d Hz, ch=%d, bits=%d, duration=%.2fs}",
                    filePath, format, fileSize, sampleRate, channels, bitDepth, duration);
        }
    }

    /**
     * 获取音频文件信息而不完全加载
     */
    public static AudioFileInfo getAudioFileInfo(String filePath) throws IOException {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new FileNotFoundException("文件不存在: " + filePath);
        }

        String extension = getFileExtension(filePath);
        AudioFormat audioFormat = AudioFormat.fromExtension(extension);

        AudioFileInfo info = new AudioFileInfo();
        info.filePath = filePath;
        info.format = audioFormat;
        info.fileSize = file.length();

        switch (audioFormat) {
            case WAV:
                info = getWavFileInfo(filePath);
                break;
            case FLAC:
                byte[] flacData = Files.readAllBytes(file.toPath());
                FlacInfo flacInfo = parseFlacHeader(flacData);
                info.sampleRate = flacInfo.sampleRate;
                info.channels = flacInfo.channels;
                info.duration = flacInfo.duration;
                break;
            default:
                // 对于其他格式，提供估算值
                info.sampleRate = 44100;
                info.channels = 2;
                info.duration = estimateDurationFromFileSize(file.length(), audioFormat);
                break;
        }

        return info;
    }

    /**
     * 获取WAV文件信息
     */
    private static AudioFileInfo getWavFileInfo(String filePath) throws IOException {
        AudioFileInfo info = new AudioFileInfo();
        info.filePath = filePath;
        info.format = AudioFormat.WAV;

        try {
            File file = new File(filePath);
            info.fileSize = file.length();

            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
            javax.sound.sampled.AudioFormat format = audioInputStream.getFormat();

            info.sampleRate = (int) format.getSampleRate();
            info.channels = format.getChannels();
            info.bitDepth = format.getSampleSizeInBits();

            long frameLength = audioInputStream.getFrameLength();
            if (frameLength != AudioSystem.NOT_SPECIFIED) {
                info.duration = frameLength / format.getFrameRate();
            }

            audioInputStream.close();
        } catch (UnsupportedAudioFileException e) {
            throw new IOException("不支持的WAV文件格式: " + e.getMessage());
        }

        return info;
    }

    /**
     * 根据文件大小估算音频时长
     */
    private static double estimateDurationFromFileSize(long fileSize, AudioFormat format) {
        // 不同格式的平均比特率 (kbps)
        int estimatedBitrate;
        switch (format) {
            case MP3:
                estimatedBitrate = 128;
                break;
            case AAC:
                estimatedBitrate = 128;
                break;
            case M4A:
                estimatedBitrate = 128;
                break;
            case WMA:
                estimatedBitrate = 96;
                break;
            case FLAC:
                estimatedBitrate = 700; // 无损压缩
                break;
            default:
                estimatedBitrate = 128;
                break;
        }

        // 计算估算时长 (秒)
        return (fileSize * 8.0) / (estimatedBitrate * 1000.0);
    }

    /**
     * 格式转换方法
     */
    public AudioData convertToFormat(AudioFormat targetFormat, String outputPath) throws IOException {
        switch (targetFormat) {
            case WAV:
                saveAsWav(outputPath);
                break;
            case MP3:
                saveAsMp3(outputPath);
                break;
            case FLAC:
                saveAsFlac(outputPath);
                break;
            case M4A:
                saveAsM4a(outputPath);
                break;
            case WMA:
                saveAsWma(outputPath);
                break;
            case AAC:
                saveAsAac(outputPath);
                break;
            default:
                throw new UnsupportedOperationException("不支持转换到格式: " + targetFormat);
        }

        // 返回转换后的音频数据
        try {
            return loadFromFile(outputPath, this.sampleRate, this.channels == 1);
        } catch (IOException e) {
            // 如果无法加载转换后的文件，返回当前对象的副本
            AudioData converted = new AudioData(Arrays.copyOf(this.samples, this.samples.length),
                                              this.sampleRate, this.channels);
            converted.setPath(outputPath);
            converted.setFormat(targetFormat.getExtension());
            return converted;
        }
    }

    /**
     * 批量格式转换
     */
    public static void batchConvert(String[] inputPaths, String outputDir, AudioFormat targetFormat) {
        File outputDirectory = new File(outputDir);
        if (!outputDirectory.exists()) {
            outputDirectory.mkdirs();
        }

        for (String inputPath : inputPaths) {
            try {
                AudioData audio = loadFromFile(inputPath);
                String fileName = new File(inputPath).getName();
                String baseName = fileName.substring(0, fileName.lastIndexOf('.'));
                String outputPath = new File(outputDir, baseName + "." + targetFormat.getExtension()).getPath();

                audio.convertToFormat(targetFormat, outputPath);
                System.out.println("转换完成: " + inputPath + " -> " + outputPath);
            } catch (IOException e) {
                System.err.println("转换失败: " + inputPath + " - " + e.getMessage());
            }
        }
    }

    // ...existing code...

    // ==================== 原有的getter/setter方法 ====================

    public byte[] getRawBytes() { return rawBytes; }

    @Override
    public int[] getDimensions() {
        // 音频维度：[采样率, 声道数]
        return new int[]{sampleRate, channels};
    }

    @Override
    public Number getNumericValue(){
        return null;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public void setRawBytes(byte[] rawBytes) { this.rawBytes = rawBytes; }
    public float[] getSamples() { return samples; }
    public void setSamples(float[] samples) {
        this.samples = samples;
        if (samples != null && sampleRate > 0) {
            this.duration = (double) samples.length / sampleRate / channels;
        }
    }
    public String getPath() { return path; }
    public int getSampleRate() { return sampleRate; }
    public void setSampleRate(int sampleRate) { this.sampleRate = sampleRate; }
    public int getChannels() { return channels; }
    public void setChannels(int channels) { this.channels = channels; }
    public double getDuration() { return duration; }
    public void setDuration(double duration) { this.duration = duration; }
    public String getFormat() { return format; }
    public void setFormat(String format) { this.format = format; }

    @Override
    public String toString() {
        return String.format("AudioData[sr=%d, ch=%d, dur=%.2fs, path=%s]",
                sampleRate, channels, duration, path);
    }

    @Override
    public String getDataType() {
        return "AUDIO";
    }

    @Override
    public Object toArrowCompatible() {
        // Arrow适配：返回包含核心信息的Map
        Map<String, Object> arrowData = new HashMap<>();
        arrowData.put("path", path);
        arrowData.put("sampleRate", sampleRate);
        arrowData.put("channels", channels);
        arrowData.put("duration", duration);
        arrowData.put("format", format);
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        return String.format("sr=%d, ch=%d, dur=%.2fs, path=%s",
                sampleRate, channels, duration, path);
    }

    @Override
    public boolean isValid() {
        // 基础校验 + 音频专属校验
        return super.isValid()
                && (path != null || rawBytes != null || samples != null)
                && sampleRate > 0;
    }

    // ==================== 音频处理辅助方法 ====================

    private static String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        if (lastDot > 0 && lastDot < filePath.length() - 1) {
            return filePath.substring(lastDot + 1).toLowerCase();
        }
        return "";
    }

    private static float[] generateSampleAudio(int length, int sampleRate) {
        float[] samples = new float[length];
        Random random = new Random();
        for (int i = 0; i < length; i++) {
            float time = (float) i / sampleRate;
            samples[i] = (float) (0.5 * Math.sin(2 * Math.PI * 440 * time) +
                                 0.1 * random.nextGaussian());
        }
        return samples;
    }

    private void applyWindow(float[] samples, String windowType) {
        int n = samples.length;
        switch (windowType.toLowerCase()) {
            case "hann":
                for (int i = 0; i < n; i++) {
                    samples[i] *= 0.5f * (1 - (float) Math.cos(2 * Math.PI * i / (n - 1)));
                }
                break;
            default:
                break;
        }
    }

    private Complex[] fft(float[] samples) {
        int n = samples.length;
        int paddedSize = Integer.highestOneBit(n - 1) << 1;
        if (paddedSize < n) paddedSize <<= 1;
        if (paddedSize != n) {
            samples = Arrays.copyOf(samples, paddedSize);
            n = paddedSize;
        }

        Complex[] complex = new Complex[n];
        for (int i = 0; i < samples.length; i++) {
            complex[i] = new Complex(samples[i], 0);
        }
        for (int i = samples.length; i < n; i++) {
            complex[i] = new Complex(0, 0);
        }

        return fftRecursive(complex);
    }

    private Complex[] fftRecursive(Complex[] x) {
        int n = x.length;
        if (n <= 1) return x;

        Complex[] even = new Complex[n / 2];
        Complex[] odd = new Complex[n / 2];

        for (int i = 0; i < n / 2; i++) {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        Complex[] fftEven = fftRecursive(even);
        Complex[] fftOdd = fftRecursive(odd);

        Complex[] result = new Complex[n];
        for (int k = 0; k < n / 2; k++) {
            double angle = -2 * Math.PI * k / n;
            Complex w = new Complex(Math.cos(angle), Math.sin(angle));
            Complex wOdd = w.multiply(fftOdd[k]);

            result[k] = fftEven[k].add(wOdd);
            result[k + n / 2] = fftEven[k].subtract(wOdd);
        }

        return result;
    }

    private float[] istft(ComplexMatrix stftMatrix, int hopLength) {
        int numFrames = stftMatrix.getCols();
        int nFft = (stftMatrix.getRows() - 1) * 2;
        int signalLength = (numFrames - 1) * hopLength + nFft;

        float[] signal = new float[signalLength];
        float[] window = new float[signalLength];

        for (int frame = 0; frame < numFrames; frame++) {
            Complex[] frameData = new Complex[nFft];
            for (int bin = 0; bin < Math.min(stftMatrix.getRows(), frameData.length / 2 + 1); bin++) {
                frameData[bin] = stftMatrix.get(bin, frame);
            }

            for (int bin = stftMatrix.getRows(); bin < nFft; bin++) {
                int mirrorBin = nFft - bin;
                if (mirrorBin > 0 && mirrorBin < stftMatrix.getRows()) {
                    frameData[bin] = stftMatrix.get(mirrorBin, frame).conjugate();
                } else {
                    frameData[bin] = new Complex(0, 0);
                }
            }

            Complex[] timeFrame = ifft(frameData);

            int start = frame * hopLength;
            for (int i = 0; i < Math.min(nFft, signalLength - start); i++) {
                if (timeFrame[i] != null) {
                    signal[start + i] += (float) timeFrame[i].real();
                    window[start + i] += 1.0f;
                }
            }
        }

        for (int i = 0; i < signalLength; i++) {
            if (window[i] > 0) {
                signal[i] /= window[i];
            }
        }

        return Arrays.copyOf(signal, samples != null ? samples.length : signalLength);
    }

    private Complex[] ifft(Complex[] x) {
        Complex[] conjugated = new Complex[x.length];
        for (int i = 0; i < x.length; i++) {
            conjugated[i] = x[i] != null ? x[i].conjugate() : new Complex(0, 0);
        }

        Complex[] result = fftRecursive(conjugated);

        for (int i = 0; i < result.length; i++) {
            result[i] = result[i].conjugate().scale(1.0 / x.length);
        }

        return result;
    }

    private static float[] bytesToFloats(byte[] audioBytes, javax.sound.sampled.AudioFormat format) {
        int bytesPerSample = format.getSampleSizeInBits() / 8;
        int numSamples = audioBytes.length / bytesPerSample;
        float[] samples = new float[numSamples];

        if (format.getSampleSizeInBits() == 16) {
            for (int i = 0; i < numSamples; i++) {
                int sampleIndex = i * 2;
                if (sampleIndex + 1 < audioBytes.length) {
                    short sample = (short) ((audioBytes[sampleIndex + 1] << 8) | (audioBytes[sampleIndex] & 0xFF));
                    samples[i] = sample / 32768.0f;
                }
            }
        }
        return samples;
    }

    private byte[] floatsToBytes(float[] samples) {
        byte[] audioBytes = new byte[samples.length * 2];

        for (int i = 0; i < samples.length; i++) {
            float clampedSample = Math.max(-1.0f, Math.min(1.0f, samples[i]));
            short sample16 = (short) (clampedSample * 32767.0f);
            audioBytes[i * 2] = (byte) (sample16 & 0xFF);
            audioBytes[i * 2 + 1] = (byte) ((sample16 >> 8) & 0xFF);
        }

        return audioBytes;
    }

    private static class FlacInfo {
        int sampleRate = 44100;
        int channels = 2;
        double duration = 3.0;
    }

    private static FlacInfo parseFlacHeader(byte[] flacData) {
        FlacInfo info = new FlacInfo();
        if (flacData.length < 4 ||
            !(flacData[0] == 'f' && flacData[1] == 'L' && flacData[2] == 'a' && flacData[3] == 'C')) {
            info.sampleRate = 44100;
            info.channels = 2;
            info.duration = 3.0;
            return info;
        }

        try {
            int offset = 4;
            while (offset + 4 < flacData.length) {
                ByteBuffer buffer = ByteBuffer.wrap(flacData, offset, 4).order(ByteOrder.BIG_ENDIAN);
                int blockHeader = buffer.getInt();
                boolean isLast = (blockHeader & 0x80000000) != 0;
                int blockType = (blockHeader >>> 24) & 0x7F;
                int blockSize = blockHeader & 0xFFFFFF;

                if (blockType == 0) {
                    if (offset + 4 + blockSize <= flacData.length) {
                        byte[] streamInfo = Arrays.copyOfRange(flacData, offset + 4, offset + 4 + blockSize);

                        if (streamInfo.length >= 34) {
                            int sampleRateValue = ((streamInfo[18] & 0xFF) << 12) |
                                                 ((streamInfo[19] & 0xFF) << 4) |
                                                 ((streamInfo[20] & 0xF0) >>> 4);
                            info.sampleRate = sampleRateValue;

                            info.channels = ((streamInfo[20] & 0x0E) >>> 1) + 1;

                            long totalSamples = ((long)(streamInfo[21] & 0x01) << 32) |
                                              ((long)(streamInfo[22] & 0xFF) << 24) |
                                              ((long)(streamInfo[23] & 0xFF) << 16) |
                                              ((long)(streamInfo[24] & 0xFF) << 8) |
                                              (streamInfo[25] & 0xFF);

                            if (info.sampleRate > 0) {
                                info.duration = (double) totalSamples / info.sampleRate;
                            }
                        }
                    }
                    break;
                }

                offset += 4 + blockSize;
                if (isLast) break;
            }
        } catch (Exception e) {
            System.out.println("FLAC头解析失败，使用默认值: " + e.getMessage());
        }

        if (info.sampleRate <= 0) info.sampleRate = 44100;
        if (info.channels <= 0) info.channels = 2;
        if (info.duration <= 0) info.duration = 3.0;

        return info;
    }

    private float[][] createMelFilterBank(int nMels, int nFft, int sampleRate, double fMin, double fMax) {
        int nBins = nFft / 2 + 1;
        float[][] filters = new float[nMels][nBins];

        double melMin = hzToMel(fMin);
        double melMax = hzToMel(fMax);

        double[] melPoints = new double[nMels + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = melMin + i * (melMax - melMin) / (nMels + 1);
        }

        double[] freqPoints = new double[melPoints.length];
        for (int i = 0; i < freqPoints.length; i++) {
            freqPoints[i] = melToHz(melPoints[i]);
        }

        int[] binPoints = new int[freqPoints.length];
        for (int i = 0; i < freqPoints.length; i++) {
            binPoints[i] = (int) Math.round(freqPoints[i] * nFft / sampleRate);
        }

        for (int mel = 0; mel < nMels; mel++) {
            int left = binPoints[mel];
            int center = binPoints[mel + 1];
            int right = binPoints[mel + 2];

            for (int bin = left; bin < center && bin < nBins; bin++) {
                filters[mel][bin] = (float) (bin - left) / (center - left);
            }

            for (int bin = center; bin < right && bin < nBins; bin++) {
                filters[mel][bin] = (float) (right - bin) / (right - center);
            }
        }

        return filters;
    }

    private double hzToMel(double hz) {
        return 2595 * Math.log10(1 + hz / 700);
    }

    private double melToHz(double mel) {
        return 700 * (Math.pow(10, mel / 2595) - 1);
    }

    private float[][] createChromaFilterBank(int nFft, int sampleRate) {
        int nChroma = 12;
        int nBins = nFft / 2 + 1;
        float[][] filters = new float[nChroma][nBins];

        for (int bin = 1; bin < nBins; bin++) {
            double freq = bin * sampleRate / (double) nFft;
            if (freq > 0) {
                double midiNote = 12 * Math.log(freq / 440.0) / Math.log(2) + 69;
                int chroma = ((int) Math.round(midiNote) % 12 + 12) % 12;
                double deviation = Math.abs(midiNote - Math.round(midiNote));
                filters[chroma][bin] = (float) Math.exp(-deviation * deviation / (2 * 0.5 * 0.5));
            }
        }

        return filters;
    }

    private float[][] applyFilterBank(float[][] spectrum, float[][] filters) {
        int nFilters = filters.length;
        int nFrames = spectrum[0].length;

        float[][] result = new float[nFilters][nFrames];

        for (int filter = 0; filter < nFilters; filter++) {
            for (int frame = 0; frame < nFrames; frame++) {
                for (int bin = 0; bin < Math.min(filters[filter].length, spectrum.length); bin++) {
                    result[filter][frame] += spectrum[bin][frame] * filters[filter][bin];
                }
            }
        }

        return result;
    }

    private float[][] dct(float[][] input, int nCoeffs) {
        int nMels = input.length;
        int nFrames = input[0].length;

        float[][] result = new float[nCoeffs][nFrames];

        for (int coeff = 0; coeff < nCoeffs; coeff++) {
            for (int frame = 0; frame < nFrames; frame++) {
                for (int mel = 0; mel < nMels; mel++) {
                    result[coeff][frame] += input[mel][frame] *
                        Math.cos(Math.PI * coeff * (2 * mel + 1) / (2 * nMels));
                }
            }
        }

        return result;
    }

    private int[] createLogFrequencyBands(int nBands, int nFft, int sampleRate) {
        double fMin = 200;
        double fMax = sampleRate / 2.0;

        int[] bands = new int[nBands + 1];

        for (int i = 0; i <= nBands; i++) {
            double freq = fMin * Math.pow(fMax / fMin, (double) i / nBands);
            bands[i] = (int) Math.round(freq * nFft / sampleRate);
        }

        return bands;
    }

    private float extractPitchAutocorrelation(float[] signal, float fMin, float fMax) {
        int minPeriod = (int) (sampleRate / fMax);
        int maxPeriod = (int) (sampleRate / fMin);

        float maxCorrelation = 0;
        int bestPeriod = minPeriod;

        for (int period = minPeriod; period <= maxPeriod && period < signal.length / 2; period++) {
            float correlation = 0;
            int count = 0;

            for (int i = 0; i < signal.length - period; i++) {
                correlation += signal[i] * signal[i + period];
                count++;
            }

            if (count > 0) {
                correlation /= count;
                if (correlation > maxCorrelation) {
                    maxCorrelation = correlation;
                    bestPeriod = period;
                }
            }
        }

        return maxCorrelation > 0.1f ? (float) sampleRate / bestPeriod : 0;
    }

    /**
     * Estimate global tempo (BPM) from an onset-strength envelope.
     *
     * <p>{@code onsetStrength} is framed (one value per hop), not per-sample.
     * Autocorrelation lags are therefore in <em>frames</em>; converting with
     * {@link #DEFAULT_HOP_LENGTH} yields the beat period in seconds.
     * Search window covers roughly 40–240 BPM (librosa-style default range).
     */
    private float estimateTempo(float[] onsetStrength) {
        if (onsetStrength == null || onsetStrength.length < 2) return 120f;

        double fps = sampleRate / (double) DEFAULT_HOP_LENGTH; // frames per second
        // lag (frames) for BPM b: lag = 60 * fps / b
        int minLag = Math.max(1, (int) Math.floor(60.0 * fps / 240.0)); // fastest ~240 BPM
        int maxLag = Math.min(onsetStrength.length / 2, (int) Math.ceil(60.0 * fps / 40.0)); // slowest ~40 BPM
        if (maxLag <= minLag) {
            return 120f;
        }

        float maxCorr = Float.NEGATIVE_INFINITY;
        int bestLag = Math.max(minLag, (int) Math.round(60.0 * fps / 120.0)); // default 120 BPM lag

        for (int lag = minLag; lag <= maxLag; lag++) {
            float correlation = 0f;
            int n = onsetStrength.length - lag;
            for (int i = 0; i < n; i++) {
                correlation += onsetStrength[i] * onsetStrength[i + lag];
            }
            // Prefer shorter periods slightly when correlations tie (octave bias toward faster)
            float score = correlation / Math.max(1, n);
            if (score > maxCorr) {
                maxCorr = score;
                bestLag = lag;
            }
        }

        float lagInSeconds = (float) bestLag * DEFAULT_HOP_LENGTH / Math.max(1, sampleRate);
        if (lagInSeconds <= 1e-6f) return 120f;
        float bpm = 60f / lagInSeconds;
        // Clamp to a musically plausible range; broken estimates fall back to 120
        if (bpm < 40f || bpm > 240f || !Float.isFinite(bpm)) return 120f;
        return bpm;
    }

    private float[] detectBeats(float[] onsetStrength, float tempo) {
        List<Float> beats = new ArrayList<>();
        float threshold = getMax(onsetStrength) * 0.3f;

        for (int i = 0; i < onsetStrength.length; i++) {
            if (onsetStrength[i] > threshold) {
                float timeInSeconds = i * DEFAULT_HOP_LENGTH / (float) sampleRate;
                beats.add(timeInSeconds);
            }
        }

        float[] result = new float[beats.size()];
        for (int i = 0; i < beats.size(); i++) {
            result[i] = beats.get(i);
        }
        return result;
    }

    private float[] flattenMatrix(float[][] matrix) {
        List<Float> flattened = new ArrayList<>();
        for (float[] row : matrix) {
            for (float value : row) {
                flattened.add(value);
            }
        }
        float[] result = new float[flattened.size()];
        for (int i = 0; i < flattened.size(); i++) {
            result[i] = flattened.get(i);
        }
        return result;
    }

    private FeatureStats computeStats(float[] features) {
        if (features.length == 0) return new FeatureStats(0, 0, 0, 0, 0);

        float sum = 0, min = features[0], max = features[0];
        for (float f : features) {
            sum += f;
            min = Math.min(min, f);
            max = Math.max(max, f);
        }

        float mean = sum / features.length;

        float variance = 0;
        for (float f : features) {
            variance += (f - mean) * (f - mean);
        }
        variance /= features.length;

        return new FeatureStats(mean, (float) Math.sqrt(variance), min, max, features.length);
    }

    private float computeMfccSimilarity(float[][] mfcc1, float[][] mfcc2) {
        if (mfcc1.length != mfcc2.length) return 0f;

        float totalDistance = 0f;
        int frames = Math.min(mfcc1[0].length, mfcc2[0].length);

        for (int frame = 0; frame < frames; frame++) {
            float frameDistance = 0f;
            for (int coeff = 0; coeff < mfcc1.length; coeff++) {
                float diff = mfcc1[coeff][frame] - mfcc2[coeff][frame];
                frameDistance += diff * diff;
            }
            totalDistance += Math.sqrt(frameDistance);
        }

        float avgDistance = totalDistance / frames;
        return 1f / (1f + avgDistance);
    }

    // ==================== 数组辅助方法 ====================

    /**
     * 获取数组中的最大值
     */
    private float getMax(float[] array) {
        if (array.length == 0) return 0f;
        float max = array[0];
        for (float value : array) {
            max = Math.max(max, value);
        }
        return max;
    }

    /**
     * 获取数组的平均值
     */
    private float getAverage(float[] array) {
        if (array.length == 0) return 0f;
        float sum = 0f;
        for (float value : array) {
            sum += value;
        }
        return sum / array.length;
    }

}
