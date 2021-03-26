package org.bytedeco.nvcodec.samples.encoder;

import org.bytedeco.nvcodec.samples.dispose.Disposable;
import org.bytedeco.nvcodec.samples.exceptions.NvCodecException;
import org.bytedeco.nvcodec.samples.util.VectorEx;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.nvcodec.nvencodeapi.*;

import java.util.Vector;

import static org.bytedeco.nvcodec.global.nvencodeapi.*;
import static org.bytedeco.nvcodec.samples.util.NvCodecUtil.*;
import static org.bytedeco.javacpp.Pointer.*;

public abstract class NvEncoder implements Disposable {
    private int width;
    private int height;
    private int bufferFormat;
    private Pointer device;
    private int deviceType;
    private NV_ENC_INITIALIZE_PARAMS initializeParams;
    private NV_ENC_CONFIG encodeConfig;
    private boolean encoderInitialized;
    private int extraOutputDelay;  // To ensure encode and graphics can work in parallel, m_nExtraOutputDelay should be set to at least 1
    private VectorEx<NV_ENC_OUTPUT_PTR> bitstreamOutputBuffer;
    private VectorEx<NV_ENC_OUTPUT_PTR> mvDataOutputBuffer;
    private int maxEncodeWidth;
    private int maxEncodedHeight;

    protected int toSend;
    protected int got;
    protected int encoderBuffer;
    protected int outputDelay;

    protected boolean motionEstimationOnly;
    protected boolean outputInVideoMemory;
    protected Pointer encoder;
    protected NV_ENCODE_API_FUNCTION_LIST nvEncodeApiFunctionList;
    protected VectorEx<NvEncoderInputFrame> inputFrames;
    protected VectorEx<NV_ENC_REGISTERED_PTR> registeredResources;
    protected VectorEx<NvEncoderInputFrame> referenceFrames;
    protected VectorEx<NV_ENC_REGISTERED_PTR> registeredResourcesForReference;
    protected VectorEx<NV_ENC_INPUT_PTR> mappedInputBuffers;
    protected VectorEx<NV_ENC_INPUT_PTR> mappedRefBuffers;
    protected VectorEx<Pointer> completionEventPointers;

    /**
     * @brief This a static function to get chroma offsets for YUV planar formats.
     */
    public static void getChromaSubPlaneOffsets(final int bufferFormat, final int pitch, final int height, Vector<Integer> chromaOffsets) {
        chromaOffsets.clear();
        switch (bufferFormat) {
            case NV_ENC_BUFFER_FORMAT_NV12:
            case NV_ENC_BUFFER_FORMAT_YUV420_10BIT: {
                chromaOffsets.add(pitch * height);
                break;
            }
            case NV_ENC_BUFFER_FORMAT_YV12:
            case NV_ENC_BUFFER_FORMAT_IYUV: {
                chromaOffsets.add(pitch * height);
                chromaOffsets.add(chromaOffsets.get(0) + (getChromaPitch(bufferFormat, pitch) * getChromaHeight(bufferFormat, height)));
                break;
            }
            case NV_ENC_BUFFER_FORMAT_YUV444:
            case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: {
                chromaOffsets.add(pitch * height);
                chromaOffsets.add(chromaOffsets.get(0) + (pitch * height));
                break;
            }
            case NV_ENC_BUFFER_FORMAT_ARGB:
            case NV_ENC_BUFFER_FORMAT_ARGB10:
            case NV_ENC_BUFFER_FORMAT_AYUV:
            case NV_ENC_BUFFER_FORMAT_ABGR:
            case NV_ENC_BUFFER_FORMAT_ABGR10: {
                break;
            }
            default: {
                System.err.println("Invalid Buffer format");
                break;
            }
        }

    }

    /**
     * @brief This a static function to get the chroma plane pitch for YUV planar formats.
     */
    public static int getChromaPitch(final int bufferFormat, final int lumaPitch) {
        switch (bufferFormat) {
            case NV_ENC_BUFFER_FORMAT_NV12:
            case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
            case NV_ENC_BUFFER_FORMAT_YUV444:
            case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: {
                return lumaPitch;
            }
            case NV_ENC_BUFFER_FORMAT_YV12:
            case NV_ENC_BUFFER_FORMAT_IYUV: {
                return (lumaPitch + 1) / 2;
            }
            case NV_ENC_BUFFER_FORMAT_ARGB:
            case NV_ENC_BUFFER_FORMAT_ARGB10:
            case NV_ENC_BUFFER_FORMAT_AYUV:
            case NV_ENC_BUFFER_FORMAT_ABGR:
            case NV_ENC_BUFFER_FORMAT_ABGR10: {
                return 0;
            }
            default: {
                System.err.println("Invalid Buffer format");
                return -1;
            }
        }
    }

    /**
     * @brief This a static function to get the number of chroma planes for YUV planar formats.
     */
    public static int getNumChromaPlanes(final int bufferFormat) {
        switch (bufferFormat) {
            case NV_ENC_BUFFER_FORMAT_NV12:
            case NV_ENC_BUFFER_FORMAT_YUV420_10BIT: {
                return 1;
            }
            case NV_ENC_BUFFER_FORMAT_YV12:
            case NV_ENC_BUFFER_FORMAT_IYUV:
            case NV_ENC_BUFFER_FORMAT_YUV444:
            case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: {
                return 2;
            }
            case NV_ENC_BUFFER_FORMAT_ARGB:
            case NV_ENC_BUFFER_FORMAT_ARGB10:
            case NV_ENC_BUFFER_FORMAT_AYUV:
            case NV_ENC_BUFFER_FORMAT_ABGR:
            case NV_ENC_BUFFER_FORMAT_ABGR10: {
                return 0;
            }
            default: {
                System.err.println("Invalid Buffer format");
                return -1;
            }
        }
    }

    /**
     * @brief This a static function to get the chroma plane width in bytes for YUV planar formats.
     */
    public static int getChromaWidthInBytes(final int bufferFormat, final int lumaWidth) {
        switch (bufferFormat) {
            case NV_ENC_BUFFER_FORMAT_YV12:
            case NV_ENC_BUFFER_FORMAT_IYUV: {
                return (lumaWidth + 1) / 2;
            }
            case NV_ENC_BUFFER_FORMAT_NV12:
            case NV_ENC_BUFFER_FORMAT_YUV444: {
                return lumaWidth;
            }
            case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
            case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: {
                return lumaWidth * 2;
            }
            case NV_ENC_BUFFER_FORMAT_ARGB:
            case NV_ENC_BUFFER_FORMAT_ARGB10:
            case NV_ENC_BUFFER_FORMAT_AYUV:
            case NV_ENC_BUFFER_FORMAT_ABGR:
            case NV_ENC_BUFFER_FORMAT_ABGR10: {
                return 0;
            }
            default: {
                System.err.println("Invalid Buffer format");
                return 0;
            }
        }
    }

    /**
     * @brief This a static function to get the chroma planes height in bytes for YUV planar formats.
     */
    public static int getChromaHeight(final int bufferFormat, final int lumaHeight) {
        switch (bufferFormat) {
            case NV_ENC_BUFFER_FORMAT_YV12:
            case NV_ENC_BUFFER_FORMAT_IYUV:
            case NV_ENC_BUFFER_FORMAT_NV12:
            case NV_ENC_BUFFER_FORMAT_YUV420_10BIT: {
                return (lumaHeight + 1) / 2;
            }
            case NV_ENC_BUFFER_FORMAT_YUV444:
            case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: {
                return lumaHeight;
            }
            case NV_ENC_BUFFER_FORMAT_ARGB:
            case NV_ENC_BUFFER_FORMAT_ARGB10:
            case NV_ENC_BUFFER_FORMAT_AYUV:
            case NV_ENC_BUFFER_FORMAT_ABGR:
            case NV_ENC_BUFFER_FORMAT_ABGR10: {
                return 0;
            }
            default: {
                System.err.println("Invalid Buffer format");
                return 0;
            }
        }
    }

    /**
     * @brief This a static function to get the width in bytes for the frame.
     * For YUV planar format this is the width in bytes of the luma plane.
     */
    public static int getWidthInBytes(final int bufferFormat, final int width) {
        switch (bufferFormat) {
            case NV_ENC_BUFFER_FORMAT_NV12:
            case NV_ENC_BUFFER_FORMAT_YV12:
            case NV_ENC_BUFFER_FORMAT_IYUV:
            case NV_ENC_BUFFER_FORMAT_YUV444: {
                return width;
            }
            case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
            case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: {
                return width * 2;
            }
            case NV_ENC_BUFFER_FORMAT_ARGB:
            case NV_ENC_BUFFER_FORMAT_ARGB10:
            case NV_ENC_BUFFER_FORMAT_AYUV:
            case NV_ENC_BUFFER_FORMAT_ABGR:
            case NV_ENC_BUFFER_FORMAT_ABGR10: {
                return width * 4;
            }
            default: {
                System.err.println("Invalid Buffer format");
                return 0;
            }
        }
    }

    /**
     * @brief This function is used to get the current device on which encoder is running.
     */
    public Pointer getDevice() {
        return this.device;
    }

    /**
     * @brief This function is used to get the current device type which encoder is running.
     */
    public int getDeviceType() {
        return deviceType;
    }

    /**
     * @brief This function is used to get the current encode width.
     * The encode width can be modified by Reconfigure() function.
     */
    public int getEncodeWidth() {
        return this.width;
    }

    /**
     * @brief This function is used to get the current encode height.
     * The encode height can be modified by Reconfigure() function.
     */
    public int getEncodeHeight() {
        return this.height;
    }

    /**
     * @brief This is a private function which is used to check if there is any
     * buffering done by encoder.
     * The encoder generally buffers data to encode B frames or for lookahead
     * or pipelining.
     */
    private boolean isZeroDelay() {
        return this.outputDelay == 0;
    }

    /**
     * @brief This function is used to check if hardware encoder is properly initialized.
     */
    protected boolean isHWEncoderInitialized() {
        return (this.encoder != null && !this.encoder.isNull()) && this.encoderInitialized;
    }

    /**
     * @brief This function returns the number of allocated buffers.
     */
    public int getEncoderBufferCount() {
        return this.encoderBuffer;
    }

    /**
     * @brief This function returns maximum width used to open the encoder session.
     * All encode input buffers are allocated using maximum dimensions.
     */
    protected int getMaxEncodeWidth() {
        return this.maxEncodeWidth;
    }

    /**
     * @brief This function returns maximum height used to open the encoder session.
     * All encode input buffers are allocated using maximum dimensions.
     */
    protected int getMaxEncodedHeight() {
        return maxEncodedHeight;
    }

    /**
     * @brief This function returns the completion event.
     */
    protected Pointer getCompletionEventPointer(int eventIndex) {
        return (this.completionEventPointers.size() == this.encoderBuffer) ? this.completionEventPointers.get(eventIndex) : null;
    }

    /**
     * @brief This function returns the current pixel format.
     */
    protected int getPixelFormat() {
        return this.bufferFormat;
    }

    public NV_ENCODE_API_FUNCTION_LIST getNvEncodeApiFunctionList() {
        return nvEncodeApiFunctionList;
    }

    public NvEncoder(int deviceType, Pointer devicePointer, int width, int height, int bufferFormat, int outputDelay, boolean motionEstimationOnly) {
        this(deviceType, devicePointer, width, height, bufferFormat, outputDelay, motionEstimationOnly, false);
    }

    public NvEncoder(int deviceType, Pointer devicePointer, int width, int height, int bufferFormat, int outputDelay, boolean motionEstimationOnly, boolean outputInVideoMemory) {
        this.device = devicePointer;
        this.deviceType = deviceType;
        this.width = width;
        this.height = height;
        this.maxEncodeWidth = width;
        this.maxEncodedHeight = height;
        this.bufferFormat = bufferFormat;
        this.motionEstimationOnly = motionEstimationOnly;
        this.outputInVideoMemory = outputInVideoMemory;
        this.extraOutputDelay = outputDelay;
        this.encoder = null;
        this.encodeConfig = new NV_ENC_CONFIG();
        this.initializeParams = new NV_ENC_INITIALIZE_PARAMS();

        this.referenceFrames = new VectorEx<>();
        this.inputFrames = new VectorEx<>();
        this.registeredResources = new VectorEx<>();
        this.registeredResourcesForReference = new VectorEx<>();
        this.mappedInputBuffers = new VectorEx<>();
        this.mappedRefBuffers = new VectorEx<>();
        this.completionEventPointers = new VectorEx<>();

        this.bitstreamOutputBuffer = new VectorEx<>();
        this.mvDataOutputBuffer = new VectorEx<>();
        this.loadNvEncAPI();

        if (this.nvEncodeApiFunctionList.nvEncOpenEncodeSessionEx() == null || this.nvEncodeApiFunctionList.nvEncOpenEncodeSessionEx().isNull()) {
            this.encoderBuffer = 0;
            System.err.println("EncodeAPI not found");
        }

        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = new NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS();
        encodeSessionExParams.version(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER);
        encodeSessionExParams.device(this.device);
        encodeSessionExParams.deviceType(this.deviceType);
        encodeSessionExParams.apiVersion(NVENCAPI_VERSION);


        PointerPointer<Pointer> encoderPointer = new PointerPointer<Pointer>(1);

        try {
            checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncOpenEncodeSessionEx().call(encodeSessionExParams, encoderPointer));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        this.encoder = encoderPointer.get();
    }

    /**
     * @brief This is a private function which is used to load the encode api shared library.
     */
    private void loadNvEncAPI() {
        IntPointer version = new IntPointer(1);
        int currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;

        try {
            checkNvCodecApiCall(NvEncodeAPIGetMaxSupportedVersion(version));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        if (currentVersion > version.get()) {
            System.err.println("Current Driver Version does not support this NvEncodeAPI version, please upgrade driver");
        }

        this.nvEncodeApiFunctionList = new NV_ENCODE_API_FUNCTION_LIST();
        this.nvEncodeApiFunctionList.version(NV_ENCODE_API_FUNCTION_LIST_VER);

        try {
            checkNvCodecApiCall(NvEncodeAPICreateInstance(this.nvEncodeApiFunctionList));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }
    }


    public void createDefaultEncoderParams(NV_ENC_INITIALIZE_PARAMS initializeParams, GUID codecGuid, GUID presetGuid) {
        this.createDefaultEncoderParams(initializeParams, codecGuid, presetGuid, NV_ENC_TUNING_INFO_UNDEFINED);
    }

    /**
     * @brief This function is used to initialize config parameters based on
     * given codec and preset guids.
     * The application can call this function to get the default configuration
     * for a certain preset. The application can either use these parameters
     * directly or override them with application-specific settings before
     * using them in CreateEncoder() function.
     */
    public void createDefaultEncoderParams(NV_ENC_INITIALIZE_PARAMS initializeParams, GUID codecGuid, GUID presetGuid, int tuningInfo) {
        if (this.encoder == null || this.encoder.isNull()) {
            System.err.println("Encoder Initialization failed");
            return;
        }
        NV_ENC_CONFIG configPointer = initializeParams.encodeConfig();
        if ((initializeParams == null || initializeParams.isNull()) || (configPointer == null || configPointer.isNull())) {
            System.err.println("initializeParams and initializeParams->encodeConfig can't be NULL");
        }

        memset(configPointer, 0, Pointer.sizeof(NV_ENC_CONFIG.class));
        memset(initializeParams, 0, Pointer.sizeof(NV_ENC_INITIALIZE_PARAMS.class));

        configPointer.version(NV_ENC_CONFIG_VER);

        initializeParams.encodeConfig(configPointer);
        initializeParams.version(NV_ENC_INITIALIZE_PARAMS_VER);

        initializeParams.encodeGUID(codecGuid);
        initializeParams.presetGUID(presetGuid);
        initializeParams.encodeWidth(this.width);
        initializeParams.encodeHeight(this.height);
        initializeParams.darWidth(this.width);
        initializeParams.darHeight(this.height);
        initializeParams.frameRateNum(30);
        initializeParams.frameRateDen(1);
        initializeParams.enablePTD(1);
        initializeParams.reportSliceOffsets(0);
        initializeParams.enableSubFrameWrite(0);
        initializeParams.maxEncodeWidth(this.width);
        initializeParams.maxEncodeHeight(this.height);
        initializeParams.enableMEOnlyMode(this.motionEstimationOnly ? 1 : 0);
        initializeParams.enableOutputInVidmem(this.outputInVideoMemory ? 1 : 0);

        NV_ENC_PRESET_CONFIG presetConfig = new NV_ENC_PRESET_CONFIG();
        presetConfig.version(NV_ENC_PRESET_CONFIG_VER);
        presetConfig.presetCfg(new NV_ENC_CONFIG() {{
            version(NV_ENC_CONFIG_VER);
        }});

        this.nvEncodeApiFunctionList.nvEncGetEncodePresetConfig().call(this.encoder, codecGuid, presetGuid, presetConfig);

        memcpy(configPointer, presetConfig.presetCfg(), Pointer.sizeof(NV_ENC_CONFIG.class));

        initializeParams.encodeConfig().frameIntervalP(1);
        initializeParams.encodeConfig().gopLength(NVENC_INFINITE_GOPLENGTH);

        initializeParams.encodeConfig().rcParams().rateControlMode(NV_ENC_PARAMS_RC_CONSTQP);

        if (!this.motionEstimationOnly) {
            initializeParams.tuningInfo(tuningInfo);

            NV_ENC_PRESET_CONFIG presetConfig2 = new NV_ENC_PRESET_CONFIG();
            presetConfig2.version(NV_ENC_PRESET_CONFIG_VER);
            presetConfig2.presetCfg(new NV_ENC_CONFIG() {{
                version(NV_ENC_CONFIG_VER);
            }});

            this.nvEncodeApiFunctionList.nvEncGetEncodePresetConfigEx().call(this.encoder, codecGuid, presetGuid, tuningInfo, presetConfig2);

            memcpy(configPointer, presetConfig2.presetCfg(), Pointer.sizeof(NV_ENC_CONFIG.class));
        } else {
            this.encodeConfig.version(NV_ENC_CONFIG_VER);
            this.encodeConfig.rcParams().rateControlMode(NV_ENC_PARAMS_RC_CONSTQP);

            NV_ENC_QP constQP = new NV_ENC_QP();
            constQP.qpInterP(28);
            constQP.qpInterB(31);
            constQP.qpIntra(25);

            this.encodeConfig.rcParams().constQP(constQP);
        }

        if (initializeParams.encodeGUID().Data1() == NV_ENC_CODEC_H264_GUID().Data1()) {
            if (this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
                initializeParams.encodeConfig().encodeCodecConfig().h264Config().chromaFormatIDC(3);
            }
            initializeParams.encodeConfig().encodeCodecConfig().h264Config().idrPeriod(initializeParams.encodeConfig().gopLength());
        } else if (initializeParams.encodeGUID().Data1() == NV_ENC_CODEC_HEVC_GUID().Data1()) {
            initializeParams.encodeConfig().encodeCodecConfig().hevcConfig().pixelBitDepthMinus8((this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 2 : 0);

            if (bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
                initializeParams.encodeConfig().encodeCodecConfig().hevcConfig().chromaFormatIDC(3);
            }

            initializeParams.encodeConfig().encodeCodecConfig().hevcConfig().idrPeriod(initializeParams.encodeConfig().gopLength());
        }
    }

    /**
     * @brief This function is used to initialize the encoder session.
     * Application must call this function to initialize the encoder, before
     * starting to encode any frames.
     */
    public void createEncoder(final NV_ENC_INITIALIZE_PARAMS encoderParams) {
        if (this.encoder == null || this.encoder.isNull()) {
            System.err.println("Encoder Initialization failed");
        }

        if (encoderParams == null || encoderParams.isNull()) {
            System.err.println("Invalid NV_ENC_INITIALIZE_PARAMS ptr");
        }

        if (encoderParams.encodeWidth() == 0 || encoderParams.encodeHeight() == 0) {
            System.err.println("Invalid encoder width and height");
        }

        if (!compareGUID(encoderParams.encodeGUID(), NV_ENC_CODEC_H264_GUID()) && !compareGUID(encoderParams.encodeGUID(), NV_ENC_CODEC_HEVC_GUID())) {
            System.err.println("Invalid codec guid");
        }

        // set other necessary params if not set yet
        if (compareGUID(encoderParams.encodeGUID(), NV_ENC_CODEC_H264_GUID())) {
            if (this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
                System.err.println("10-bit format isn't supported by H264 encoder");
            }
        }

        if (compareGUID(encoderParams.encodeGUID(), NV_ENC_CODEC_HEVC_GUID())) {
            boolean yuv10BitFormat = (this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT);
            if (yuv10BitFormat && encoderParams.encodeConfig().encodeCodecConfig().hevcConfig().pixelBitDepthMinus8() != 2) {
                System.err.println("Invalid PixelBitdepth");
            }

            if ((this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || this.bufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) && (encoderParams.encodeConfig().encodeCodecConfig().hevcConfig().chromaFormatIDC() != 3)) {
                System.err.println("Invalid ChromaFormatIDC");
            }
        }

        memcpy(this.initializeParams, encoderParams, this.initializeParams.sizeof());


        this.initializeParams.version(NV_ENC_INITIALIZE_PARAMS_VER);

        if (encoderParams.encodeConfig() != null && !encoderParams.encodeConfig().isNull()) {
            memcpy(this.encodeConfig, encoderParams.encodeConfig(), this.encodeConfig.sizeof());
            this.encodeConfig.version(NV_ENC_CONFIG_VER);
        } else {
            NV_ENC_PRESET_CONFIG presetConfig = new NV_ENC_PRESET_CONFIG();
            presetConfig.version(NV_ENC_PRESET_CONFIG_VER);
            presetConfig.presetCfg(new NV_ENC_CONFIG() {
                {
                    version(NV_ENC_CONFIG_VER);
                }
            });

            if (!this.motionEstimationOnly) {
                this.nvEncodeApiFunctionList.nvEncGetEncodePresetConfigEx().call(this.encoder, encoderParams.encodeGUID(), encoderParams.presetGUID(), encoderParams.tuningInfo(), presetConfig);
                memcpy(this.encodeConfig, presetConfig.presetCfg(), this.encodeConfig.sizeof());
            } else {
                this.encodeConfig.version(NV_ENC_CONFIG_VER);
                this.encodeConfig.rcParams().rateControlMode(NV_ENC_PARAMS_RC_CONSTQP);
                this.encodeConfig.rcParams().constQP(new NV_ENC_QP() {
                    {
                        qpInterP(28);
                        qpInterB(31);
                        qpIntra(25);
                    }
                });
            }
        }

        this.initializeParams.encodeConfig(this.encodeConfig);

        try {
            checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncInitializeEncoder().call(this.encoder, this.initializeParams));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        this.encoderInitialized = true;
        this.width = this.initializeParams.encodeWidth();
        this.height = this.initializeParams.encodeHeight();
        this.maxEncodeWidth = this.initializeParams.maxEncodeWidth();
        this.maxEncodedHeight = this.initializeParams.maxEncodeHeight();

        this.encoderBuffer = this.encodeConfig.frameIntervalP() + this.encodeConfig.rcParams().lookaheadDepth() + this.extraOutputDelay;
        this.outputDelay = this.encoderBuffer - 1;

        this.mappedInputBuffers.resize(this.encoderBuffer, null);

        if (!this.outputInVideoMemory) {
            this.completionEventPointers.resize(this.encoderBuffer, null);
        }

        if (this.motionEstimationOnly) {
            this.mappedRefBuffers.resize(this.encoderBuffer, null);

            if (!this.outputInVideoMemory) {
                this.initializeMVOutputBuffer();
            }
        } else {
            if (!this.outputInVideoMemory) {
                this.bitstreamOutputBuffer.resize(this.encoderBuffer, null);
                this.initializeBitstreamBuffer();
            }
        }

        this.allocateInputBuffers(this.encoderBuffer);
    }


    /**
     * @brief This function is used to destroy the encoder session.
     * Application must call this function to destroy the encoder session and
     * clean up any allocated resources. The application must call EndEncode()
     * function to get any queued encoded frames before calling DestroyEncoder().
     */
    public void destroyEncoder() {
        if (this.encoder != null && !this.encoder.isNull()) {
            this.releaseInputBuffers();
            this.destroyHWEncoder();
        }
    }

    /**
     * @brief This is a private function which is used to destroy HW encoder.
     */
    private void destroyHWEncoder() {
        if (this.encoder != null && !this.encoder.isNull()) {
            if (this.motionEstimationOnly) {
                this.destroyMVOutputBuffer();
            } else {
                this.destroyBitstreamBuffer();
            }
            this.nvEncodeApiFunctionList.nvEncDestroyEncoder().call(this.encoder);

            this.encoder = null;
            this.encoderInitialized = false;
        }
    }

    /**
     * @brief This function is used to get the next available input buffer.
     * Applications must call this function to obtain a pointer to the next
     * input buffer. The application must copy the uncompressed data to the
     * input buffer and then call EncodeFrame() function to encode it.
     */
    public NvEncoderInputFrame getNextInputFrame() {
        int index = this.toSend % this.encoderBuffer;
        return this.inputFrames.get(index);
    }

    /**
     * @brief This function is used to get an available reference frame.
     * Application must call this function to get a pointer to reference buffer,
     * to be used in the subsequent RunMotionEstimation() function.
     */
    public NvEncoderInputFrame getNextReferenceFrame() {
        int index = this.toSend % this.encoderBuffer;
        return this.referenceFrames.get(index);
    }

    /**
     * @brief This function is used to map the input buffers to NvEncodeAPI.
     */
    protected void mapResources(int index) {
        NV_ENC_MAP_INPUT_RESOURCE mapInputResource = new NV_ENC_MAP_INPUT_RESOURCE();
        mapInputResource.version(NV_ENC_MAP_INPUT_RESOURCE_VER);

        mapInputResource.registeredResource(this.registeredResources.get(index));

        try {
            checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncMapInputResource().call(this.encoder, mapInputResource));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        this.mappedInputBuffers.set(index, mapInputResource.mappedResource());

        if (this.motionEstimationOnly) {
            mapInputResource.registeredResource(this.registeredResourcesForReference.get(index));
            try {
                checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncMapInputResource().call(this.encoder, mapInputResource));
            } catch (NvCodecException e) {
                e.printStackTrace();
            }

            this.mappedRefBuffers.set(index, mapInputResource.mappedResource());
        }
    }

    public void encodeFrame(Vector<Vector<Byte>> packet) {
        this.encodeFrame(packet, null);
    }

    /**
     * @brief This function is used to encode a frame.
     * Applications must call EncodeFrame() function to encode the uncompressed
     * data, which has been copied to an input buffer obtained from the
     * GetNextInputFrame() function.
     */
    public void encodeFrame(Vector<Vector<Byte>> packet, NV_ENC_PIC_PARAMS picParams) {
        packet.clear();
        if (!this.isHWEncoderInitialized()) {
            System.err.println("Encoder device not found");
        }
        int index = this.toSend % this.encoderBuffer;

        this.mapResources(index);

        int nvStatus = this.doEncode(this.mappedInputBuffers.get(index), this.bitstreamOutputBuffer.get(index), picParams);
        if (nvStatus == NV_ENC_SUCCESS || nvStatus == NV_ENC_ERR_NEED_MORE_INPUT) {
            this.toSend++;
            this.getEncodedPacket(this.bitstreamOutputBuffer, packet, true);
        } else {
            System.err.println("nvEncEncodePicture API failed");
        }
    }

    /**
     * @brief This function is used to run motion estimation
     * This is used to run motion estimation on a a pair of frames. The
     * application must copy the reference frame data to the buffer obtained
     * by calling GetNextReferenceFrame(), and copy the input frame data to
     * the buffer obtained by calling GetNextInputFrame() before calling the
     * RunMotionEstimation() function.
     */
    public void runMotionEstimation(Vector<Byte> mvData) {
        if (this.encoder != null && !this.encoder.isNull()) {
            final int index = this.toSend % this.encoderBuffer;

            this.mapResources(index);

            int nvStatus = this.doMotionEstimation(this.mappedInputBuffers.get(index), this.mappedRefBuffers.get(index), this.mvDataOutputBuffer.get(index));

            if (nvStatus == NV_ENC_SUCCESS) {
                this.toSend++;

                Vector<Vector<Byte>> packet = new Vector<>();

                this.getEncodedPacket(this.mvDataOutputBuffer, packet, true);

                if (packet.size() != 1) {
                    System.err.println("GetEncodedPacket() doesn't return one (and only one) MVData");
                }

                mvData = packet.get(0);
            } else {
                System.err.println("nvEncEncodePicture API failed");
            }
        } else {
            System.err.println("Encoder Initialization failed");
        }
    }

    /**
     * @brief This function is used to get sequence and picture parameter headers.
     * Application can call this function after encoder is initialized to get SPS and PPS
     * nalus for the current encoder instance. The sequence header data might change when
     * application calls Reconfigure() function.
     */
    public void getSequenceParams(Vector<Byte> seqParams) {
        byte[] data = new byte[1024]; // Assume maximum spspps data is 1KB or less
        BytePointer spsppsData = new BytePointer(data);

        memset(spsppsData, 0, data.length);

        NV_ENC_SEQUENCE_PARAM_PAYLOAD payload = new NV_ENC_SEQUENCE_PARAM_PAYLOAD();
        payload.version(NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER);
        IntPointer spsppsSize = new IntPointer();

        payload.spsppsBuffer(spsppsData);
        payload.inBufferSize(spsppsData.sizeof());
        payload.outSPSPPSPayloadSize(spsppsSize);
        try {
            checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncGetSequenceParams().call(this.encoder, payload));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }
        spsppsData.asBuffer().get(data);
        seqParams.clear();

        for (int index = 0; index < data.length; index++) {
            seqParams.add(data[index]);
        }
    }

    /**
     * @brief This function is used to submit the encode commands to the
     * NVENC hardware.
     */
    protected int doEncode(NV_ENC_INPUT_PTR inputBuffer, NV_ENC_OUTPUT_PTR outputBuffer, NV_ENC_PIC_PARAMS picParams) {
        if (picParams == null || picParams.isNull()) {
            picParams = new NV_ENC_PIC_PARAMS();
        }

        picParams.version(NV_ENC_PIC_PARAMS_VER);
        picParams.pictureStruct(NV_ENC_PIC_STRUCT_FRAME);
        picParams.inputBuffer(inputBuffer);
        picParams.bufferFmt(this.getPixelFormat());
        picParams.inputWidth(this.getEncodeWidth());
        picParams.inputHeight(this.getEncodeHeight());
        picParams.outputBitstream(outputBuffer);
        picParams.completionEvent(this.getCompletionEventPointer(this.toSend % this.encoderBuffer));

        return this.nvEncodeApiFunctionList.nvEncEncodePicture().call(this.encoder, picParams);
    }

    /**
     * @brief This function is used to send EOS to HW encoder.
     */
    protected void sendEOS() {
        NV_ENC_PIC_PARAMS picParams = new NV_ENC_PIC_PARAMS();
        picParams.version(NV_ENC_PIC_PARAMS_VER);
        picParams.encodePicFlags(NV_ENC_PIC_FLAG_EOS);
        picParams.completionEvent(this.getCompletionEventPointer(this.toSend % this.encoderBuffer));
        try {
            checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncEncodePicture().call(this.encoder, picParams));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }
    }

    /**
     * @brief This function to flush the encoder queue.
     * The encoder might be queuing frames for B picture encoding or lookahead;
     * the application must call EndEncode() to get all the queued encoded frames
     * from the encoder. The application must call this function before destroying
     * an encoder session.
     */
    public void endEncode(Vector<Vector<Byte>> packet) {
        packet.clear();

        if (!this.isHWEncoderInitialized()) {
            System.err.println("Encoder device not initialized");
        }

        this.sendEOS();
        this.getEncodedPacket(this.bitstreamOutputBuffer, packet, false);
    }

    /**
     * @brief This is a private function which is used to get the output packets
     * from the encoder HW.
     * This is called by DoEncode() function. If there is buffering enabled,
     * this may return without any output data.
     */
    private void getEncodedPacket(Vector<NV_ENC_OUTPUT_PTR> outputBuffer, Vector<Vector<Byte>> packet, boolean outputDelay) {
        long index = 0;

        int end = outputDelay ? this.toSend - this.outputDelay : this.toSend;
        for (; this.got < end; this.got++) {
            this.waitForCompletionEvent(this.got % this.encoderBuffer);
            NV_ENC_LOCK_BITSTREAM lockBitstreamData = new NV_ENC_LOCK_BITSTREAM();
            lockBitstreamData.version(NV_ENC_LOCK_BITSTREAM_VER);
            lockBitstreamData.outputBitstream(outputBuffer.get(this.got % this.encoderBuffer));
            lockBitstreamData.doNotWait(0);

            try {
                checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncLockBitstream().call(this.encoder, lockBitstreamData));
            } catch (NvCodecException e) {
                e.printStackTrace();
            }

            Pointer dataPointer = lockBitstreamData.bitstreamBufferPtr();
            if (packet.size() < index + 1) {
                packet.add(new Vector<Byte>());
            }

            Vector<Byte> packetData = packet.get((int) index);
            packetData.clear();

            dataPointer.limit(lockBitstreamData.bitstreamSizeInBytes());
            byte[] data = new byte[lockBitstreamData.bitstreamSizeInBytes()];

            dataPointer.asByteBuffer().get(data);

            for (byte value : data) {
                packetData.add(value);
            }
            index++;

            try {
                checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncUnlockBitstream().call(this.encoder, new NV_ENC_OUTPUT_PTR(lockBitstreamData.outputBitstream())));
            } catch (NvCodecException e) {
                e.printStackTrace();
            }


            NV_ENC_INPUT_PTR inputPointer = this.mappedInputBuffers.get(this.got % this.encoderBuffer);
            if (inputPointer != null && !inputPointer.isNull()) {
                try {
                    checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncUnmapInputResource().call(this.encoder, inputPointer));
                } catch (NvCodecException e) {
                    e.printStackTrace();
                }

                this.mappedInputBuffers.set(this.got % this.encoderBuffer, null);
            }
            if (this.motionEstimationOnly) {
                NV_ENC_INPUT_PTR refPointer = this.mappedRefBuffers.get(this.got % this.encoderBuffer);
                if (refPointer != null && !refPointer.isNull()) {
                    try {
                        checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncUnmapInputResource().call(this.encoder, refPointer));
                    } catch (NvCodecException e) {
                        e.printStackTrace();
                    }
                    this.mappedRefBuffers.set(this.got % this.encoderBuffer, null);
                }
            }
        }
    }

    /**
     * @brief This function is used to reconfigure an existing encoder session.
     * Application can use this function to dynamically change the bitrate,
     * resolution and other QOS parameters. If the application changes the
     * resolution, it must set NV_ENC_RECONFIGURE_PARAMS::forceIDR.
     */
    public boolean reconfigure(final NV_ENC_RECONFIGURE_PARAMS reconfigureParams) {
        try {
            checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncReconfigureEncoder().call(this.encoder, reconfigureParams));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        memcpy(this.initializeParams, reconfigureParams.reInitEncodeParams(), this.initializeParams.sizeof());


        NV_ENC_CONFIG config = reconfigureParams.reInitEncodeParams().encodeConfig();
        if (config != null && !config.isNull()) {
            memcpy(this.encodeConfig, config, this.encodeConfig.sizeof());
        }

        this.width = this.initializeParams.encodeWidth();
        this.height = this.initializeParams.encodeHeight();
        this.maxEncodeWidth = this.initializeParams.maxEncodeWidth();
        this.maxEncodedHeight = this.initializeParams.maxEncodeHeight();

        return true;
    }

    protected NV_ENC_REGISTERED_PTR registerResource(Pointer bufferPointer, int resourceType, int width, int height, int pitch, int bufferFormat) {
        return this.registerResource(bufferPointer, resourceType, width, height, pitch, bufferFormat, NV_ENC_INPUT_IMAGE);
    }

    /**
     * @brief This function is used to register CUDA, D3D or OpenGL input or output buffers with NvEncodeAPI.
     */
    protected NV_ENC_REGISTERED_PTR registerResource(Pointer bufferPointer, int resourceType, int width, int height, int pitch, int bufferFormat, int bufferUsage) {
        NV_ENC_REGISTER_RESOURCE registerResource = new NV_ENC_REGISTER_RESOURCE();
        registerResource.version(NV_ENC_REGISTER_RESOURCE_VER);
        registerResource.resourceType(resourceType);
        registerResource.resourceToRegister(bufferPointer);
        registerResource.width(width);
        registerResource.height(height);
        registerResource.pitch(pitch);
        registerResource.bufferFormat(bufferFormat);
        registerResource.bufferUsage(bufferUsage);

        try {
            checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncRegisterResource().call(this.encoder, registerResource));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        return registerResource.registeredResource();
    }

    /**
     * @brief This function is used to register CUDA, D3D or OpenGL input buffers with NvEncodeAPI.
     * This is non public function and is called by derived class for allocating
     * and registering input buffers.
     */
    protected void registerInputResources(Vector<Pointer> inputFrames, int resourceType, int width, int height, int pitch, int bufferFormat, boolean referenceFrame) {
        for (int index = 0; index < inputFrames.size(); ++index) {
            NV_ENC_REGISTERED_PTR registeredPtr = this.registerResource(inputFrames.get(index), resourceType, width, height, pitch, bufferFormat, NV_ENC_INPUT_IMAGE);

            Vector<Integer> chromaOffsets = new Vector<>();
            getChromaSubPlaneOffsets(bufferFormat, pitch, height, chromaOffsets);

            NvEncoderInputFrame inputFrame = new NvEncoderInputFrame();
            inputFrame.setInputPointer(inputFrames.get(index));
            inputFrame.setChromaOffsets(new int[]{0, 0});

            for (int ch = 0; ch < chromaOffsets.size(); ch++) {
                inputFrame.getChromaOffsets()[ch] = chromaOffsets.get(ch);
            }
            inputFrame.setNumChromaPlanes(getNumChromaPlanes(bufferFormat));
            inputFrame.setPitch(pitch);
            inputFrame.setChromaPitch(getChromaPitch(bufferFormat, pitch));
            inputFrame.setBufferFormat(bufferFormat);
            inputFrame.setResourceType(resourceType);

            if (referenceFrame) {
                this.registeredResourcesForReference.add(registeredPtr);
                this.referenceFrames.add(inputFrame);
            } else {
                this.registeredResources.add(registeredPtr);
                this.inputFrames.add(inputFrame);
            }
        }
    }

    /**
     * @brief This function is used to flush the encoder queue.
     */
    private void flushEncoder() {
        if (!this.motionEstimationOnly && !this.outputInVideoMemory) {
            // Incase of error it is possible for buffers still mapped to encoder.
            // flush the encoder queue and then unmapped it if any surface is still mapped
            try {
                Vector<Vector<Byte>> packet = new Vector<>();
                this.endEncode(packet);
            } catch (Exception e) {

            }
        }
    }

    /**
     * @brief This function is used to unregister resources which had been previously registered for encoding
     * using RegisterInputResources() function.
     */
    protected void unregisterInputResources() {
        this.flushEncoder();

        if (this.motionEstimationOnly) {
            for (NV_ENC_INPUT_PTR refPointer : this.mappedRefBuffers) {
                if (refPointer != null && !refPointer.isNull()) {
                    this.nvEncodeApiFunctionList.nvEncUnmapInputResource().call(this.encoder, refPointer);
                }
            }
        }
        this.mappedRefBuffers.clear();

        for (NV_ENC_INPUT_PTR inputPointer : this.mappedInputBuffers) {
            if (inputPointer != null && !inputPointer.isNull()) {
                this.nvEncodeApiFunctionList.nvEncUnmapInputResource().call(this.encoder, inputPointer);
            }
        }
        this.mappedInputBuffers.clear();

        for (NV_ENC_REGISTERED_PTR registeredPointer : this.registeredResources) {
            if (registeredPointer != null && !registeredPointer.isNull()) {
                this.nvEncodeApiFunctionList.nvEncUnregisterResource().call(this.encoder, registeredPointer);
            }
        }
        this.registeredResources.clear();

        for (NV_ENC_REGISTERED_PTR registeredPointer : this.registeredResourcesForReference) {
            if (registeredPointer != null && !registeredPointer.isNull()) {
                this.nvEncodeApiFunctionList.nvEncUnregisterResource().call(this.encoder, registeredPointer);
            }
        }
        this.registeredResourcesForReference.clear();
    }

    /**
     * @brief This function is used to wait for completion of encode command.
     */
    protected void waitForCompletionEvent(int event) {
        //not support win32api
    }

    /**
     * @brief This function is used to query hardware encoder capabilities.
     * Applications can call this function to query capabilities like maximum encode
     * dimensions, support for lookahead or the ME-only mode etc.
     */
    public int getCapabilityValue(GUID guidCodec, int capsToQuery) {
        if (this.encoder != null && !this.encoder.isNull()) {
            NV_ENC_CAPS_PARAM capsParam = new NV_ENC_CAPS_PARAM();
            capsParam.version(NV_ENC_CAPS_PARAM_VER);
            capsParam.capsToQuery(capsToQuery);

            IntPointer value = new IntPointer();
            this.nvEncodeApiFunctionList.nvEncGetEncodeCaps().call(this.encoder, guidCodec, capsParam, value);

            return value.get();
        }
        return 0;
    }

    /**
     * @brief This function is used to get the current frame size based on pixel format.
     */
    public int getFrameSize() {
        switch (this.getPixelFormat()) {
            case NV_ENC_BUFFER_FORMAT_YV12:
            case NV_ENC_BUFFER_FORMAT_IYUV:
            case NV_ENC_BUFFER_FORMAT_NV12: {
                return this.getEncodeWidth() * (this.getEncodeHeight() + (this.getEncodeHeight() + 1) / 2);
            }
            case NV_ENC_BUFFER_FORMAT_YUV420_10BIT: {
                return 2 * this.getEncodeWidth() * (this.getEncodeHeight() + (this.getEncodeHeight() + 1) / 2);
            }
            case NV_ENC_BUFFER_FORMAT_YUV444: {
                return this.getEncodeWidth() * this.getEncodeHeight() * 3;
            }
            case NV_ENC_BUFFER_FORMAT_YUV444_10BIT: {
                return 2 * this.getEncodeWidth() * getEncodeHeight() * 3;
            }
            case NV_ENC_BUFFER_FORMAT_ARGB:
            case NV_ENC_BUFFER_FORMAT_ARGB10:
            case NV_ENC_BUFFER_FORMAT_AYUV:
            case NV_ENC_BUFFER_FORMAT_ABGR:
            case NV_ENC_BUFFER_FORMAT_ABGR10: {
                return 4 * this.getEncodeWidth() * this.getEncodeHeight();
            }
            default: {
                System.err.println("Invalid Buffer format");
                return 0;
            }
        }
    }

    /**
     * @brief This function is used to get the current initialization parameters,
     * which had been used to configure the encoder session.
     * The initialization parameters are modified if the application calls
     * Reconfigure() function.
     */
    public void getInitializeParams(NV_ENC_INITIALIZE_PARAMS initializeParams) {
        NV_ENC_CONFIG encodeConfig = initializeParams.encodeConfig();
        if ((initializeParams == null || initializeParams.isNull()) || (encodeConfig == null || encodeConfig.isNull())) {
            System.err.println("Both pInitializeParams and pInitializeParams->encodeConfig can't be NULL");
        }

        initializeParams = this.initializeParams;
        initializeParams.encodeConfig(this.encodeConfig);
    }

    /**
     * @brief This is a private function which is used to initialize the bitstream buffers.
     * This is only used in the encoding mode.
     */
    private void initializeBitstreamBuffer() {
        for (int index = 0; index < this.encoderBuffer; index++) {
            NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = new NV_ENC_CREATE_BITSTREAM_BUFFER();
            createBitstreamBuffer.version(NV_ENC_CREATE_BITSTREAM_BUFFER_VER);

            try {
                checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncCreateBitstreamBuffer().call(this.encoder, createBitstreamBuffer));
            } catch (NvCodecException e) {
                e.printStackTrace();
            }
            this.bitstreamOutputBuffer.set(index, createBitstreamBuffer.bitstreamBuffer());
        }
    }

    /**
     * @brief This is a private function which is used to destroy the bitstream buffers.
     * This is only used in the encoding mode.
     */
    private void destroyBitstreamBuffer() {
        for (int index = 0; index < this.bitstreamOutputBuffer.size(); index++) {
            NV_ENC_OUTPUT_PTR pointer = this.bitstreamOutputBuffer.get(index);
            if (pointer != null && !pointer.isNull()) {
                this.nvEncodeApiFunctionList.nvEncDestroyBitstreamBuffer().call(this.encoder, pointer);
            }
        }
        this.bitstreamOutputBuffer.clear();
    }

    /**
     * @brief This is a private function which is used to initialize MV output buffers.
     * This is only used in ME-only Mode.
     */
    private void initializeMVOutputBuffer() {
        for (int index = 0; index < this.encoderBuffer; index++) {
            NV_ENC_CREATE_MV_BUFFER createMVBuffer = new NV_ENC_CREATE_MV_BUFFER();
            createMVBuffer.version(NV_ENC_CREATE_BITSTREAM_BUFFER_VER);
            try {
                checkNvCodecApiCall(this.nvEncodeApiFunctionList.nvEncCreateMVBuffer().call(this.encoder, createMVBuffer));
            } catch (NvCodecException e) {
                e.printStackTrace();
            }

            this.mvDataOutputBuffer.add(createMVBuffer.mvBuffer());
        }
    }

    /**
     * @brief This is a private function which is used to destroy MV output buffers.
     * This is only used in ME-only Mode.
     */
    private void destroyMVOutputBuffer() {
        for (int index = 0; index < this.mvDataOutputBuffer.size(); index++) {
            NV_ENC_OUTPUT_PTR pointer = this.mvDataOutputBuffer.get(index);
            if (pointer != null && !pointer.isNull()) {
                this.nvEncodeApiFunctionList.nvEncDestroyMVBuffer().call(this.encoder, pointer);
            }
        }
        this.mvDataOutputBuffer.clear();
    }

    /**
     * @brief This function is used to submit the encode commands to the
     * NVENC hardware for ME only mode.
     */
    protected int doMotionEstimation(NV_ENC_INPUT_PTR inputBuffer, NV_ENC_INPUT_PTR inputBufferForReference, NV_ENC_OUTPUT_PTR outputBuffer) {
        NV_ENC_MEONLY_PARAMS meParams = new NV_ENC_MEONLY_PARAMS();
        meParams.version(NV_ENC_MEONLY_PARAMS_VER);
        meParams.inputBuffer(inputBuffer);
        meParams.referenceFrame(inputBufferForReference);
        meParams.inputWidth(this.getEncodeWidth());
        meParams.inputHeight(this.getEncodeHeight());
        meParams.mvBuffer(outputBuffer);
        meParams.completionEvent(this.getCompletionEventPointer(this.toSend % this.encoderBuffer));

        return this.nvEncodeApiFunctionList.nvEncRunMotionEstimationOnly().call(this.encoder, meParams);
    }

    /**
     * @brief This is a pure virtual function which is used to allocate input buffers.
     * The derived classes must implement this function.
     */
    protected abstract void allocateInputBuffers(int numInputBuffers);

    /**
     * @brief This is a pure virtual function which is used to destroy input buffers.
     * The derived classes must implement this function.
     */
    protected abstract void releaseInputBuffers();

    @Override
    public void dispose() {
        this.destroyHWEncoder();
    }
}
