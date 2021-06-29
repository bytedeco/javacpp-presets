package org.bytedeco.nvcodec.samples.decoder;


import org.bytedeco.nvcodec.samples.callback.PfnVideoDecodeCallback;
import org.bytedeco.nvcodec.samples.callback.PfnVideoDisplayCallback;
import org.bytedeco.nvcodec.samples.callback.PfnVideoSequenceCallback;
import org.bytedeco.nvcodec.samples.dispose.Disposable;
import org.bytedeco.nvcodec.samples.exceptions.CudaException;
import org.bytedeco.nvcodec.samples.exceptions.NvCodecException;
import org.bytedeco.nvcodec.samples.util.Dimension;
import org.bytedeco.nvcodec.samples.util.Rectangle;
import org.bytedeco.cuda.cudart.CUDA_MEMCPY2D_v2;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.javacpp.*;
import org.bytedeco.nvcodec.global.nvcuvid;
import org.bytedeco.nvcodec.global.nvcuvid.*;
import org.bytedeco.nvcodec.nvcuvid.*;

import java.util.*;

import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.nvcodec.global.nvcuvid.*;
import static org.bytedeco.nvcodec.samples.util.CudaUtil.*;
import static org.bytedeco.nvcodec.samples.util.NvCodecUtil.*;

/**
 * @brief Base class for decoder interface.
 */
public class NvDecoder extends Pointer implements Disposable {
    private CUctx_st cuContext;

    private _CUcontextlock_st ctxLock;
    private CUvideoparser parser;
    private CUvideodecoder decoder;
    private boolean useDeviceFrame;

    //dimension of the output
    private int width;
    private int lumaHeight;
    private int chromaHeight;
    private int numChromaPlanes;

    //height of the mapped surface
    private int surfaceWidth;
    private int surfaceHeight;

    private int codec;
    private int chromaFormat;
    private int outputFormat;

    private int bitDepthMinus8;
    private int bpp;

    private CUVIDEOFORMAT videoFormat;
    private Rectangle displayRectangle;

    // stock of frames
    private Vector<BytePointer> frames;
    // timestamps of decoded frames
    private Vector<Long> timestamps;

    private int decodedFrame;
    private int decodedFrameReturned;

    private int decodePicCount;
    private int[] picNumInDecodeOrder;
    private boolean endDecodeDone;

    private final Object vpFrameLock;

    private int frameAlloc;

    private CUstream_st cuvidStream;

    private boolean deviceFramePitched;
    private SizeTPointer deviceFramePitch;

    private Rectangle cropRect;
    private Dimension resizeDimension;

    private StringBuilder videoInfo;

    private int maxWidth;
    private int maxHeight;

    private boolean reconfigExternal;
    private boolean reconfigExtPPChanged;

    private static Map<Integer, String> codecNames = new HashMap<Integer, String>() {
        {
            put(cudaVideoCodec_MPEG1, "MPEG-1");
            put(cudaVideoCodec_MPEG2, "MPEG-2");
            put(cudaVideoCodec_MPEG4, "MPEG-4 (ASP)");
            put(cudaVideoCodec_VC1, "VC-1/WMV");
            put(cudaVideoCodec_H264, "AVC/H.264");
            put(cudaVideoCodec_JPEG, "M-JPEG");
            put(cudaVideoCodec_H264_SVC, "H.264/SVC");
            put(cudaVideoCodec_H264_MVC, "H.264/MVC");
            put(cudaVideoCodec_HEVC, "H.265/HEVC");
            put(cudaVideoCodec_VP8, "VP8");
            put(cudaVideoCodec_VP9, "VP9");
            put(cudaVideoCodec_NumCodecs, "Invalid");
            put(cudaVideoCodec_YUV420, "YUV  4:2:0");
            put(cudaVideoCodec_YV12, "YV12 4:2:0");
            put(cudaVideoCodec_NV12, "NV12 4:2:0");
            put(cudaVideoCodec_YUYV, "YUYV 4:2:2");
            put(cudaVideoCodec_UYVY, "UYVY 4:2:2");
        }
    };

    private static Map<Integer, String> chromaFormatNames = new HashMap<Integer, String>() {
        {
            put(cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)");
            put(cudaVideoChromaFormat_420, "YUV 420");
            put(cudaVideoChromaFormat_422, "YUV 422");
            put(cudaVideoChromaFormat_444, "YUV 444");
        }
    };

    public static String getVideoCodecString(int codec) {
        return codecNames.get(codec);
    }

    public static String getVideoChromaFormatString(int chromaFormat) {
        return chromaFormatNames.get(chromaFormat);
    }

    public static float getChromaHeightFactor(int surfaceFormat) {
        float factor = 0.5f;

        switch (surfaceFormat) {
            case cudaVideoSurfaceFormat_NV12:
            case cudaVideoSurfaceFormat_P016: {
                factor = 0.5f;
                break;
            }
            case cudaVideoSurfaceFormat_YUV444:
            case cudaVideoSurfaceFormat_YUV444_16Bit: {
                factor = 1.0f;
                break;
            }
        }

        return factor;
    }

    public static int getChromaPlaneCount(int surfaceFormat) {
        int numPlane = 1;

        switch (surfaceFormat) {
            case cudaVideoSurfaceFormat_NV12:
            case cudaVideoSurfaceFormat_P016: {
                numPlane = 1;
                break;
            }
            case cudaVideoSurfaceFormat_YUV444:
            case cudaVideoSurfaceFormat_YUV444_16Bit: {
                numPlane = 2;
                break;
            }
        }

        return numPlane;
    }

    /**
     * @brief This function is used to get the current CUDA context.
     */
    public CUctx_st getContext() {
        return cuContext;
    }

    /**
     * @brief This function is used to get the current decode width.
     */
    public int getWidth() {
        return width;
    }

    /**
     * @brief This function is used to get the current decode height (Luma height).
     */
    public int getHeight() {
        return this.lumaHeight;
    }

    /**
     * @brief This function is used to get the current chroma height.
     */
    public int getChromaHeight() {
        return chromaHeight;
    }

    /**
     * @brief This function is used to get the number of chroma planes.
     */
    public int getNumChromaPlanes() {
        return numChromaPlanes;
    }

    /**
     * @brief This function is used to get the current frame size based on pixel format.
     */
    public int getFrameSize() {
        return this.width * (this.lumaHeight + this.chromaHeight * this.numChromaPlanes) * this.bpp;
    }

    /**
     * @brief This function is used to get the pitch of the device buffer holding the decoded frame.
     */
    public int getDeviceFramePitch() {
        return this.deviceFramePitch != null ? (int) this.deviceFramePitch.get() : this.width * this.bpp;
    }

    /**
     * @brief This function is used to get the bit depth associated with the pixel format.
     */
    public int getBitDepth() {
        return this.bitDepthMinus8 + 8;
    }

    /**
     * @brief This function is used to get the bytes used per pixel.
     */
    public int getBPP() {
        return this.bpp;
    }

    /**
     * @brief This function is used to get the YUV chroma format
     */
    public int getOutputFormat() {
        return outputFormat;
    }

    /**
     * @brief This function is used to get information about the video stream (codec, display parameters etc)
     */
    public CUVIDEOFORMAT getVideoFormat() {
        return videoFormat;
    }

    /**
     * @brief This function is used to get codec string from codec id
     */
    public String getCodecString(int codec) {
        return getVideoCodecString(codec);
    }

    /**
     * @brief This function is used to print information about the video stream
     */
    public StringBuilder getVideoInfo() {
        return videoInfo;
    }

    /**
     * @brief This function is used to initialize the decoder session.
     * Application must call this function to initialize the decoder, before
     * starting to decode any frames.
     */
    public NvDecoder(CUctx_st cuContext, boolean useDeviceFrame, int codec, boolean lowLatency, boolean deviceFramePitched, Rectangle cropRect, Dimension resizeDimension, int maxWidth, int maxHeight, int clkRate) {
        this.vpFrameLock = new Object();

        this.ctxLock = new _CUcontextlock_st();
        this.parser = new CUvideoparser();
        this.decoder = new CUvideodecoder();
        this.cuvidStream = new CUstream_st();

        this.frameAlloc = 0;
        this.bpp = 1;
        this.codec = cudaVideoCodec_NumCodecs;

        this.videoFormat = new CUVIDEOFORMAT();

        this.displayRectangle = new Rectangle();

        this.cropRect = cropRect;
        this.resizeDimension = resizeDimension;

        this.picNumInDecodeOrder = new int[32];

        this.cuContext = cuContext;
        this.useDeviceFrame = useDeviceFrame;

        this.codec = codec;

        this.deviceFramePitch = new SizeTPointer(1) {{
            put(0);
        }};

        this.deviceFramePitched = deviceFramePitched;

        this.maxWidth = maxWidth;
        this.maxHeight = maxHeight;

        this.frames = new Vector<BytePointer>();
        this.timestamps = new Vector<Long>();

        this.videoInfo = new StringBuilder();

        if (cropRect != null) {
            this.cropRect = cropRect;
        }

        if (resizeDimension != null) {
            this.resizeDimension = resizeDimension;
        }

        try {
            checkNvCodecApiCall(cuvidCtxLockCreate(this.ctxLock, this.cuContext));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        CUVIDPARSERPARAMS parseParameter = new CUVIDPARSERPARAMS();

        parseParameter.CodecType(this.codec);
        parseParameter.ulMaxNumDecodeSurfaces(1);
        parseParameter.ulClockRate(clkRate);
        parseParameter.pUserData(this);
        parseParameter.ulMaxDisplayDelay(lowLatency ? 0 : 1);

        parseParameter.pfnSequenceCallback(PfnVideoSequenceCallback.getInstance());

        parseParameter.pfnDecodePicture(PfnVideoDecodeCallback.getInstance());

        parseParameter.pfnDisplayPicture(PfnVideoDisplayCallback.getInstance());

        PfnVideoSequenceCallback.getInstance().setDecoder(this);
        PfnVideoDecodeCallback.getInstance().setDecoder(this);
        PfnVideoDisplayCallback.getInstance().setDecoder(this);
        try {
            checkNvCodecApiCall(cuvidCreateVideoParser(this.parser, parseParameter));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }
    }

    /**
     * @brief This function gets called when a sequence is ready to be decoded. The function also gets called
     * when there is format change
     */
    public int handleVideoSequence(CUVIDEOFORMAT videoFormat) {
        long startTime = System.currentTimeMillis();

        this.videoInfo.setLength(0);

        this.videoInfo.append("Video Input Information \n")
                .append("\t Codec : ")
                .append(getVideoCodecString(videoFormat.codec()))
                .append("\n")
                .append("\t Frame rate : ")
                .append(String.format("%d / %d = %f fps \n", videoFormat.frame_rate_numerator(), videoFormat.frame_rate_denominator(), 1.0f * videoFormat.frame_rate_numerator() / videoFormat.frame_rate_denominator()))
                .append("\t Sequence : ")
                .append(videoFormat.progressive_sequence() == 1 ? "Progressive" : "Interlaced")
                .append("\n")
                .append("\t Coded size : ")
                .append(String.format("[ %d, %d ] \n", videoFormat.coded_width(), videoFormat.coded_height()))
                .append("\t Display area : ")
                .append(String.format("[%d, %d, %d, %d] \n", videoFormat.display_area_left(), videoFormat.display_area_top(), videoFormat.display_area_right(), videoFormat.display_area_bottom()))
                .append("\t Chroma : ")
                .append(getVideoChromaFormatString(videoFormat.chroma_format()))
                .append("\n")
                .append("\t Bit depth : ")
                .append(videoFormat.bit_depth_chroma_minus8() + 8)
                .append("\n");

        int decodeSurface = videoFormat.min_num_decode_surfaces();
        CUVIDDECODECAPS decodecaps = new CUVIDDECODECAPS();
        cudaMemset(decodecaps, 0, Pointer.sizeof(CUVIDDECODECAPS.class));

        decodecaps.eCodecType(videoFormat.codec());
        decodecaps.eChromaFormat(videoFormat.chroma_format());
        decodecaps.nBitDepthMinus8(videoFormat.bit_depth_luma_minus8());

        try {
            checkCudaApiCall(cuCtxPushCurrent(this.cuContext));
            checkNvCodecApiCall(cuvidGetDecoderCaps(decodecaps));
            checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException | NvCodecException e) {
            e.printStackTrace();
        }

        if (decodecaps.bIsSupported() == 0) {
            System.err.println("Codec not supported on this GPU");

            return decodeSurface;
        }

        if ((videoFormat.coded_width() > decodecaps.nMaxWidth()) || (videoFormat.coded_height() > decodecaps.nMaxHeight())) {
            System.err.printf("Resolution : %d x %d \n Max Supported (wxh) : %d x %d \n Resolution not supported on this GPU%n", videoFormat.coded_width(), videoFormat.coded_height(), decodecaps.nMaxWidth(), decodecaps.nMaxHeight());

            return decodeSurface;
        }

        if ((videoFormat.coded_width() >> 4) * (videoFormat.coded_height() >> 4) > decodecaps.nMaxMBCount()) {
            System.err.printf("MBCount : %d \n Max Supported mbcnt : %d \n MBCount not supported on this GPU%n", (videoFormat.coded_width() >> 4) * (videoFormat.coded_height() >> 4), decodecaps.nMaxMBCount());
        }

        if (this.width != 0 && this.lumaHeight != 0 && this.chromaHeight != 0) {
            // cuvidCreateDecoder() has been called before, and now there's possible config change
            return reconfigureDecoder(videoFormat);
        }

        this.codec = videoFormat.codec();
        this.chromaFormat = videoFormat.chroma_format();
        this.bitDepthMinus8 = videoFormat.bit_depth_luma_minus8();
        this.bpp = this.bitDepthMinus8 > 0 ? 2 : 1;

        switch (this.chromaFormat) {
            case cudaVideoChromaFormat_420:
                this.outputFormat = videoFormat.bit_depth_luma_minus8() == 1 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
                break;
            case cudaVideoChromaFormat_422:
                this.outputFormat = cudaVideoSurfaceFormat_NV12;
                break;
            case cudaVideoChromaFormat_444:
                this.outputFormat = videoFormat.bit_depth_luma_minus8() == 1 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
                break;
        }

        if ((decodecaps.nOutputFormatMask() & (1 << this.outputFormat)) == 0) {
            if ((decodecaps.nOutputFormatMask() & (1 << cudaVideoSurfaceFormat_NV12)) == 1) {
                this.outputFormat = cudaVideoSurfaceFormat_NV12;
            } else if ((decodecaps.nOutputFormatMask() & (1 << cudaVideoSurfaceFormat_P016)) == 1) {
                this.outputFormat = cudaVideoSurfaceFormat_P016;
            } else if ((decodecaps.nOutputFormatMask() & (1 << cudaVideoSurfaceFormat_YUV444)) == 1) {
                this.outputFormat = cudaVideoSurfaceFormat_YUV444;
            } else if ((decodecaps.nOutputFormatMask() & (1 << cudaVideoSurfaceFormat_YUV444_16Bit)) == 1) {
                this.outputFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
            } else {
                System.err.println("No supported output format found");
            }
        }

        this.videoFormat = videoFormat;

        CUVIDDECODECREATEINFO videoDecodeCreateInfo = new CUVIDDECODECREATEINFO();
        videoDecodeCreateInfo.CodecType(videoFormat.codec());
        videoDecodeCreateInfo.ChromaFormat(videoFormat.chroma_format());
        videoDecodeCreateInfo.OutputFormat(this.outputFormat);
        videoDecodeCreateInfo.bitDepthMinus8(videoFormat.bit_depth_luma_minus8());

        if (videoFormat.progressive_sequence() == 1) {
            videoDecodeCreateInfo.DeinterlaceMode(cudaVideoDeinterlaceMode_Weave);
        } else {
            videoDecodeCreateInfo.DeinterlaceMode(cudaVideoDeinterlaceMode_Adaptive);
        }

        videoDecodeCreateInfo.ulNumOutputSurfaces(2);

        //With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware

        videoDecodeCreateInfo.ulCreationFlags(cudaVideoCreate_PreferCUVID);
        videoDecodeCreateInfo.ulNumDecodeSurfaces(decodeSurface);
        videoDecodeCreateInfo.vidLock(this.ctxLock);
        videoDecodeCreateInfo.ulWidth(videoFormat.coded_width());
        videoDecodeCreateInfo.ulHeight(videoFormat.coded_height());

        if (this.maxWidth < videoFormat.coded_width()) {
            this.maxWidth = videoFormat.coded_width();
        }

        if (this.maxHeight < videoFormat.coded_height()) {
            this.maxHeight = videoFormat.coded_height();
        }

        videoDecodeCreateInfo.ulMaxWidth(this.maxWidth);
        videoDecodeCreateInfo.ulMaxHeight(this.maxHeight);

        if ((this.cropRect.getRight() == 0 || this.cropRect.getBottom() == 0) && (this.resizeDimension.getWidth() == 0 || this.resizeDimension.getHeight() == 0)) {
            this.width = videoFormat.display_area_right() - this.videoFormat.display_area_left();
            this.lumaHeight = videoFormat.display_area_bottom() - this.videoFormat.display_area_top();

            videoDecodeCreateInfo.ulTargetWidth(videoFormat.coded_width());
            videoDecodeCreateInfo.ulTargetHeight(videoFormat.coded_height());
        } else {
            if (this.resizeDimension.getWidth() != 0 && this.resizeDimension.getHeight() != 0) {
                videoDecodeCreateInfo.display_area_left((short) videoFormat.display_area_left());
                videoDecodeCreateInfo.display_area_top((short) videoFormat.display_area_top());
                videoDecodeCreateInfo.display_area_right((short) videoFormat.display_area_right());
                videoDecodeCreateInfo.display_area_bottom((short) videoFormat.display_area_bottom());
                this.width = this.resizeDimension.getWidth();
                this.lumaHeight = this.resizeDimension.getHeight();
            }

            if (this.cropRect.getRight() != 0 && this.cropRect.getBottom() != 0) {
                videoDecodeCreateInfo.display_area_left((short) this.cropRect.getLeft());
                videoDecodeCreateInfo.display_area_top((short) this.cropRect.getTop());
                videoDecodeCreateInfo.display_area_right((short) this.cropRect.getRight());
                videoDecodeCreateInfo.display_area_bottom((short) this.cropRect.getBottom());

                this.width = this.cropRect.getRight() - this.cropRect.getLeft();
                this.lumaHeight = this.cropRect.getBottom() - this.cropRect.getTop();
            }

            videoDecodeCreateInfo.ulTargetWidth(this.width);
            videoDecodeCreateInfo.ulTargetHeight(this.lumaHeight);
        }

        this.chromaHeight = (int) (this.lumaHeight * getChromaHeightFactor(this.outputFormat));
        this.numChromaPlanes = getChromaPlaneCount(this.outputFormat);
        this.surfaceHeight = (int) videoDecodeCreateInfo.ulTargetHeight();
        this.surfaceWidth = (int) videoDecodeCreateInfo.ulTargetWidth();
        this.displayRectangle.setBottom(videoDecodeCreateInfo.display_area_bottom());
        this.displayRectangle.setTop(videoDecodeCreateInfo.display_area_top());
        this.displayRectangle.setLeft(videoDecodeCreateInfo.display_area_left());
        this.displayRectangle.setRight(videoDecodeCreateInfo.display_area_right());

        String[] deinterlaceModes = {"Weave", "Bob", "Adaptive"};

        this.videoInfo.append("Video Decoding Params : \n")
                .append(String.format("\t Num Surfaces : %d \n", videoDecodeCreateInfo.ulNumDecodeSurfaces()))
                .append(String.format("\t Crop : [%d, %d, %d, %d] \n", videoDecodeCreateInfo.display_area_left(), videoDecodeCreateInfo.display_area_top(), videoDecodeCreateInfo.display_area_right(), videoDecodeCreateInfo.display_area_bottom()))
                .append(String.format("\t Resize : %d x %d \n", videoDecodeCreateInfo.ulTargetWidth(), videoDecodeCreateInfo.ulTargetHeight()))
                .append(String.format("\t Deinterlace : %s \n", deinterlaceModes[videoDecodeCreateInfo.DeinterlaceMode()]));

        try {
            checkCudaApiCall(cuCtxPushCurrent(this.cuContext));
            checkNvCodecApiCall(cuvidCreateDecoder(this.decoder, videoDecodeCreateInfo));
            checkCudaApiCall(cuCtxPopCurrent(null));
        } catch (CudaException | NvCodecException e) {
            e.printStackTrace();
        }

        System.out.printf("Session Initialization Time: %d ms %n", (System.currentTimeMillis() - startTime));

        return decodeSurface;
    }

    /**
     * @brief This function reconfigure decoder if there is a change in sequence params.
     */
    public int reconfigureDecoder(CUVIDEOFORMAT videoFormat) {
        if (videoFormat.bit_depth_luma_minus8() != this.videoFormat.bit_depth_luma_minus8() || videoFormat.bit_depth_chroma_minus8() != this.videoFormat.bit_depth_chroma_minus8()) {
            System.err.println("Reconfigure Not supported for bit depth change");
        }

        if (videoFormat.chroma_format() != this.videoFormat.chroma_format()) {
            System.err.println("Reconfigure Not supported for chroma format change");
        }

        boolean decodeResChange = !(videoFormat.coded_width() == this.videoFormat.coded_width() && videoFormat.coded_height() == this.videoFormat.coded_height());
        boolean displayRectChange = !(videoFormat.display_area_bottom() == this.videoFormat.display_area_bottom() && videoFormat.display_area_top() == this.videoFormat.display_area_top()
                && videoFormat.display_area_left() == this.videoFormat.display_area_left() && videoFormat.display_area_right() == this.videoFormat.display_area_right());
        int decodeSurface = videoFormat.min_num_decode_surfaces();

        if ((videoFormat.coded_width() > this.maxWidth) || (videoFormat.coded_height() > this.maxHeight)) {
            // For VP9, let driver  handle the change if new width/height > maxwidth/maxheight
            if ((this.codec != cudaVideoCodec_VP9) || this.reconfigExternal) {
                System.err.println("Reconfigure Not supported when width/height > maxwidth/maxheight");
            }
        }

        if (!decodeResChange && !this.reconfigExtPPChanged) {
            // if the coded_width/coded_height hasn't changed but display resolution has changed, then need to update width/height for
            // correct output without cropping. Example : 1920x1080 vs 1920x1088
            if (displayRectChange) {
                this.width = videoFormat.display_area_right() - videoFormat.display_area_left();
                this.lumaHeight = videoFormat.display_area_bottom() - videoFormat.display_area_top();
                this.chromaHeight = (int) (this.lumaHeight * getChromaHeightFactor(this.outputFormat));
                this.numChromaPlanes = getChromaPlaneCount(this.outputFormat);
            }

            // no need for reconfigureDecoder(). Just return
            return 1;
        }

        CUVIDRECONFIGUREDECODERINFO reconfigParams = new CUVIDRECONFIGUREDECODERINFO(0);

        this.videoFormat.coded_width(videoFormat.coded_width());
        reconfigParams.ulWidth(this.videoFormat.coded_width());

        this.videoFormat.coded_height(videoFormat.coded_height());
        reconfigParams.ulHeight(this.videoFormat.coded_height());


        // Dont change display rect and get scaled output from decoder. This will help display app to present apps smoothly
        reconfigParams.display_area_bottom((short) this.displayRectangle.getBottom());
        reconfigParams.display_area_top((short) this.displayRectangle.getTop());
        reconfigParams.display_area_left((short) this.displayRectangle.getLeft());
        reconfigParams.display_area_right((short) this.displayRectangle.getRight());
        reconfigParams.ulTargetWidth(this.surfaceWidth);
        reconfigParams.ulTargetHeight(this.surfaceHeight);


        // If external reconfigure is called along with resolution change even if post processing params is not changed,
        // do full reconfigure params update
        if ((this.reconfigExternal && decodeResChange) || this.reconfigExtPPChanged) {
            // update display rect and target resolution if requested explicitely
            this.reconfigExternal = false;
            this.reconfigExtPPChanged = false;
            this.videoFormat = videoFormat;

            if (!(this.cropRect.getRight() != 0 && this.cropRect.getBottom() != 0) && !(this.resizeDimension.getWidth() != 0 && this.resizeDimension.getHeight() != 0)) {
                this.width = videoFormat.display_area_right() - videoFormat.display_area_left();
                this.lumaHeight = videoFormat.display_area_bottom() - videoFormat.display_area_top();
                reconfigParams.ulTargetWidth(videoFormat.coded_width());
                reconfigParams.ulTargetHeight(videoFormat.coded_height());
            } else {
                if (this.resizeDimension.getWidth() != 0 && this.resizeDimension.getHeight() != 0) {
                    reconfigParams.display_area_left((short) videoFormat.display_area_left());
                    reconfigParams.display_area_top((short) videoFormat.display_area_top());
                    reconfigParams.display_area_right((short) videoFormat.display_area_right());
                    reconfigParams.display_area_bottom((short) videoFormat.display_area_bottom());
                    this.width = this.resizeDimension.getWidth();
                    this.lumaHeight = this.resizeDimension.getHeight();
                }

                if (this.cropRect.getRight() != 0 && this.cropRect.getBottom() != 0) {
                    reconfigParams.display_area_left((short) this.cropRect.getLeft());
                    reconfigParams.display_area_top((short) this.cropRect.getTop());
                    reconfigParams.display_area_right((short) this.cropRect.getRight());
                    reconfigParams.display_area_bottom((short) this.cropRect.getBottom());
                    this.width = this.cropRect.getRight() - this.cropRect.getLeft();
                    this.lumaHeight = this.cropRect.getBottom() - this.cropRect.getTop();
                }

                reconfigParams.ulTargetWidth(this.width);
                reconfigParams.ulTargetHeight(this.lumaHeight);
            }

            this.chromaHeight = (int) (this.lumaHeight * getChromaHeightFactor(this.outputFormat));
            this.numChromaPlanes = getChromaPlaneCount(this.outputFormat);
            this.surfaceHeight = reconfigParams.ulTargetHeight();
            this.surfaceWidth = reconfigParams.ulTargetWidth();
            this.displayRectangle.setBottom(reconfigParams.display_area_bottom());
            this.displayRectangle.setTop(reconfigParams.display_area_top());
            this.displayRectangle.setLeft(reconfigParams.display_area_left());
            this.displayRectangle.setRight(reconfigParams.display_area_right());
        }

        reconfigParams.ulNumDecodeSurfaces(decodeSurface);

        try {
            long startTime = System.currentTimeMillis();
            checkCudaApiCall(cuCtxPushCurrent(this.cuContext));
            checkNvCodecApiCall(cuvidReconfigureDecoder(this.decoder, reconfigParams));
            checkCudaApiCall(cuCtxPopCurrent(null));
            System.out.printf("Session Reconfigure Time: %d ms", (System.currentTimeMillis() - startTime));
        } catch (CudaException | NvCodecException e) {
            e.printStackTrace();
        }

        return decodeSurface;
    }

    /**
     * @param cropRect        - cropping rectangle coordinates
     * @param resizeDimension - width and height of resized output
     * @brief This function allow app to set decoder reconfig params
     */
    public int setReconfigParams(Rectangle cropRect, Dimension resizeDimension) {
        this.reconfigExternal = true;
        this.reconfigExtPPChanged = false;
        if (cropRect != null) {
            if (!((cropRect.getTop() == this.cropRect.getTop()) && (cropRect.getLeft() == this.cropRect.getLeft())
                    && (cropRect.getBottom() == this.cropRect.getBottom()) && (cropRect.getRight() == this.cropRect.getRight()))) {
                this.reconfigExtPPChanged = true;
                this.cropRect = cropRect;
            }
        }

        if (resizeDimension != null) {
            if (!((resizeDimension.getWidth() == this.resizeDimension.getWidth()) && (resizeDimension.getHeight() == this.resizeDimension.getHeight()))) {
                this.reconfigExtPPChanged = true;
                this.resizeDimension = resizeDimension;
            }
        }


        // Clear existing output buffers of different size


        BytePointer framePointer = null;

        while (!this.frames.isEmpty()) {
            framePointer = this.frames.lastElement();
            this.frames.remove(this.frames.size() - 1);

            if (this.useDeviceFrame) {
                try {
                    checkCudaApiCall(cuCtxPushCurrent(this.cuContext));
                    checkCudaApiCall(cudaFree(framePointer));
                    checkCudaApiCall(cuCtxPopCurrent(null));
                } catch (CudaException e) {
                    e.printStackTrace();
                }
            }
        }

        return 1;
    }


    /**
     * @brief This function gets called when a picture is ready to be decoded. cuvidDecodePicture is called from this function
     * to decode the picture
     */
    public int handlePictureDecode(CUVIDPICPARAMS picParams) {
        if (this.decoder == null || this.decoder.isNull()) {
            System.err.println("Decoder not initialized.");
            return 0;
        }

        this.picNumInDecodeOrder[picParams.CurrPicIdx()] = this.decodePicCount++;

        try {
            checkNvCodecApiCall(cuvidDecodePicture(this.decoder, picParams));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        return 1;
    }

    /**
     * @brief This function gets called after a picture is decoded and available for display. Frames are fetched and stored in
     * internal buffer
     */
    public int handlePictureDisplay(CUVIDPARSERDISPINFO displayInfo) {
        try {
            CUVIDPROCPARAMS videoProcessingParameters = new CUVIDPROCPARAMS();
            videoProcessingParameters.progressive_frame(displayInfo.progressive_frame());
            videoProcessingParameters.second_field(displayInfo.repeat_first_field() + 1);
            videoProcessingParameters.top_field_first(displayInfo.top_field_first());
            videoProcessingParameters.unpaired_field(displayInfo.repeat_first_field() < 0 ? 1 : 0);
            videoProcessingParameters.output_stream(this.cuvidStream);

            LongPointer srcFrame = new LongPointer(1);
            IntPointer srcPitch = new IntPointer(1);

            try {
                checkNvCodecApiCall(cuvidMapVideoFrame64(this.decoder, displayInfo.picture_index(), srcFrame, srcPitch, videoProcessingParameters));
            } catch (NvCodecException e) {
                e.printStackTrace();
            }

            CUVIDGETDECODESTATUS decodeStatus = new CUVIDGETDECODESTATUS();
            cudaMemset(decodeStatus, 0, Pointer.sizeof(CUVIDGETDECODESTATUS.class));

            int result = cuvidGetDecodeStatus(this.decoder, displayInfo.picture_index(), decodeStatus);

            if (result == CUDA_SUCCESS && (decodeStatus.decodeStatus() == cuvidDecodeStatus_Error || decodeStatus.decodeStatus() == cuvidDecodeStatus_Error_Concealed)) {
                System.err.printf("Decode Error occurred for picture %d%n", this.picNumInDecodeOrder[displayInfo.picture_index()]);
            }

            BytePointer decodedFrame;

            synchronized (this.vpFrameLock) {
                if (++this.decodedFrame > this.frames.size()) {
                    // Not enough frames in stock
                    this.frameAlloc++;

                    BytePointer frame = new BytePointer();

                    if (this.useDeviceFrame) {
                        checkCudaApiCall(cuCtxPushCurrent(this.cuContext));
                        if (this.deviceFramePitched) {
                            checkCudaApiCall(cuMemAllocPitch(frame.getPointer(LongPointer.class), this.deviceFramePitch, (long) this.width * this.bpp, this.lumaHeight + ((long) this.chromaHeight * this.numChromaPlanes), 16));
                        } else {
                            checkCudaApiCall(cuMemAlloc(frame.getPointer(LongPointer.class), this.getFrameSize()));
                        }
                        checkCudaApiCall(cuCtxPopCurrent(null));
                    } else {
                        frame = new BytePointer(getFrameSize());
                    }
                    this.frames.add(frame);
                }
                decodedFrame = this.frames.get(this.decodedFrame - 1);
            }

            checkCudaApiCall(cuCtxPushCurrent(this.cuContext));

            CUDA_MEMCPY2D_v2 memcpy2D = new CUDA_MEMCPY2D_v2();
            memcpy2D.srcXInBytes(0);
            memcpy2D.srcMemoryType(CU_MEMORYTYPE_DEVICE);
            memcpy2D.srcDevice(srcFrame.get());
            memcpy2D.srcPitch(srcPitch.get());
            memcpy2D.dstMemoryType(this.useDeviceFrame ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST);
            memcpy2D.dstHost(decodedFrame);
            memcpy2D.dstDevice(decodedFrame.address());
            memcpy2D.dstPitch(this.deviceFramePitch.get() != 0 ? this.deviceFramePitch.get() : (long) this.width * this.bpp);
            memcpy2D.WidthInBytes((long) this.width * this.bpp);
            memcpy2D.Height(this.lumaHeight);
            checkCudaApiCall(cuMemcpy2DAsync(memcpy2D, this.cuvidStream));

            memcpy2D.srcDevice(srcFrame.get() + memcpy2D.srcPitch() * this.surfaceHeight);
            memcpy2D.dstHost(decodedFrame.getPointer(memcpy2D.dstPitch() * this.lumaHeight));
            memcpy2D.dstDevice(memcpy2D.dstHost().address());
            memcpy2D.Height(this.chromaHeight);
            checkCudaApiCall(cuMemcpy2DAsync(memcpy2D, this.cuvidStream));

            if (this.numChromaPlanes == 2) {
                memcpy2D.srcDevice(srcFrame.get() + memcpy2D.srcPitch() * this.surfaceHeight * 2);
                memcpy2D.dstHost(decodedFrame.getPointer(memcpy2D.dstPitch() * this.lumaHeight * 2));
                memcpy2D.dstDevice(memcpy2D.dstHost().address());
                memcpy2D.Height(this.chromaHeight);

                checkCudaApiCall(cuMemcpy2DAsync(memcpy2D, this.cuvidStream));
            }

            checkCudaApiCall(cuStreamSynchronize(this.cuvidStream));
            checkCudaApiCall(cuCtxPopCurrent(null));

            if (this.timestamps.size() < this.decodedFrame) {
                for (int index = this.timestamps.size(); index < this.frames.size(); index++) {
                    this.timestamps.add(-1L);
                }
            }

            this.timestamps.setElementAt(displayInfo.timestamp(), this.decodedFrame - 1);

            checkNvCodecApiCall(cuvidUnmapVideoFrame64(this.decoder, srcFrame.get()));
        } catch (CudaException | NvCodecException e) {
            e.printStackTrace();
        }

        return 1;
    }

    /**
     * @param data      - pointer to the data buffer that is to be decoded
     * @param size      - size of the data buffer in bytes
     * @param flags     - CUvideopacketflags for setting decode options
     * @param timestamp - presentation timestamp
     * @brief This function decodes a frame and returns the number of frames that are available for
     * display. All frames that are available for display should be read before making a subsequent decode call.
     */
    public int decode(BytePointer data, int size, int flags, long timestamp) {
        this.decodedFrame = 0;
        this.decodedFrameReturned = 0;

        CUVIDSOURCEDATAPACKET packet = new CUVIDSOURCEDATAPACKET();
        packet.payload(data);
        packet.payload_size(size);
        packet.flags(flags | CUVID_PKT_TIMESTAMP);
        packet.timestamp(timestamp);

        if ((data == null || data.isNull()) || size == 0) {
            packet.flags(packet.flags() | CUVID_PKT_ENDOFSTREAM);
        }
        try {
            checkNvCodecApiCall(nvcuvid.cuvidParseVideoData(this.parser, packet));
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        return this.decodedFrame;
    }

    /**
     * @brief This function returns a decoded frame and timestamp. This function should be called in a loop for
     * fetching all the frames that are available for display.
     */
    public BytePointer getFrame(LongPointer timestamp) {
        if (this.decodedFrame > 0) {
            synchronized (this.vpFrameLock) {
                this.decodedFrame--;

                if (timestamp != null && !timestamp.isNull()) {
                    timestamp.put(this.timestamps.get(this.decodedFrameReturned));
                }

                return this.frames.get(this.decodedFrameReturned++);
            }
        }

        return null;
    }

    /**
     * @brief This function decodes a frame and returns the locked frame buffers
     * This makes the buffers available for use by the application without the buffers
     * getting overwritten, even if subsequent decode calls are made. The frame buffers
     * remain locked, until UnlockFrame() is called
     */
    public BytePointer getLockedFrame(LongPointer timestamp) {
        long time;
        BytePointer frame;

        if (this.decodedFrame > 0) {
            synchronized (this.vpFrameLock) {
                this.decodedFrame--;
                frame = this.frames.get(0);

                this.frames.remove(0);
                this.frames.remove(1);

                time = this.timestamps.get(0);
                this.timestamps.remove(0);
                this.timestamps.remove(1);

                if (timestamp != null && !timestamp.isNull()) {
                    timestamp.put(time);
                }

                return frame;
            }
        }

        return null;
    }

    @Override
    public void dispose() {
        try {
            long startTime = System.currentTimeMillis();
            checkCudaApiCall(cuCtxPushCurrent(this.cuContext));
            checkCudaApiCall(cuCtxPopCurrent(null));

            if (this.parser != null && !this.parser.isNull()) {
                cuvidDestroyVideoParser(this.parser);
            }

            if (this.decoder != null && !this.decoder.isNull()) {
                cuvidDestroyDecoder(this.decoder);
            }

            synchronized (this.vpFrameLock) {
                for (BytePointer framePointer : this.frames) {
                    if (this.useDeviceFrame) {
                        checkCudaApiCall(cuCtxPushCurrent(this.cuContext));
                        cuMemFree(framePointer.address());
                        checkCudaApiCall(cuCtxPopCurrent(null));
                    } else {
                        cuMemFreeHost(framePointer);
                    }
                }
            }

            cuvidCtxLockDestroy(this.ctxLock);

            System.out.println("Session Deinitialization Time: " + (System.currentTimeMillis() - startTime) + "ms");
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }
}
