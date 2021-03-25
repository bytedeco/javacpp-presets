package org.bytedeco.nvcodec.samples.encoder;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.nvcodec.nvencodeapi.*;
import org.bytedeco.nvcodec.samples.exceptions.NvCodecException;
import org.bytedeco.nvcodec.samples.util.NvCodecUtil;

import static org.bytedeco.nvcodec.global.nvencodeapi.*;

public class NvEncoderInitParam {
    public static class InitFunction extends FunctionPointer {
        public void call(NV_ENC_INITIALIZE_PARAMS params) {

        }
    }

    private int tuningInfo;

    private String param;
    private String[] tokens;

    private boolean lowLatency;

    private GUID guidCodec;
    private GUID guidPreset;

    private InitFunction initFunction;

    private final String codecNames = "h264 hevc";
    private GUID[] codecs = new GUID[]{
            NV_ENC_CODEC_H264_GUID(),
            NV_ENC_CODEC_HEVC_GUID()
    };

    private final String chromaNames = "yuv420 yuv444";
    private Integer[] chromaTypes = new Integer[]{
            1, 3
    };

    private final String presetNames = "p1 p2 p3 p4 p5 p6 p7";
    private GUID[] presets = new GUID[]{
            NV_ENC_PRESET_P1_GUID(),
            NV_ENC_PRESET_P2_GUID(),
            NV_ENC_PRESET_P3_GUID(),
            NV_ENC_PRESET_P4_GUID(),
            NV_ENC_PRESET_P5_GUID(),
            NV_ENC_PRESET_P6_GUID(),
            NV_ENC_PRESET_P7_GUID()
    };

    private final String h264ProfileNames = "baseline main high high444";
    private GUID[] h264Profiles = new GUID[]{
            NV_ENC_H264_PROFILE_BASELINE_GUID(),
            NV_ENC_H264_PROFILE_MAIN_GUID(),
            NV_ENC_H264_PROFILE_HIGH_GUID(),
            NV_ENC_H264_PROFILE_HIGH_444_GUID()
    };

    private final String hevcProfileNames = "main main10 frext";
    private GUID[] hevcProfiles = new GUID[]{
            NV_ENC_HEVC_PROFILE_MAIN_GUID(),
            NV_ENC_HEVC_PROFILE_MAIN10_GUID(),
            NV_ENC_HEVC_PROFILE_FREXT_GUID()
    };

    private final String profileNames = "(default) auto baseline(h264) main(h264) high(h264) high444(h264) stereo(h264) svc_temporal_scalability(h264) progressiv_high(h264) constrained_high(h264) main(hevc) main10(hevc) frext(hevc)";
    private GUID[] profiles = new GUID[]{
            new GUID(),
            NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID(),
            NV_ENC_H264_PROFILE_BASELINE_GUID(),
            NV_ENC_H264_PROFILE_MAIN_GUID(),
            NV_ENC_H264_PROFILE_HIGH_GUID(),
            NV_ENC_H264_PROFILE_HIGH_444_GUID(),
            NV_ENC_H264_PROFILE_STEREO_GUID(),
            NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID(),
            NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID(),
            NV_ENC_HEVC_PROFILE_MAIN_GUID(),
            NV_ENC_HEVC_PROFILE_MAIN10_GUID(),
            NV_ENC_HEVC_PROFILE_FREXT_GUID()
    };

    private final String lowLatencyTuningInfoNames = "lowlatency ultralowlatency";
    private final String tuningInfoNames = "hq lowlatency ultralowlatency lossless";
    private Integer[] tuningInfos = new Integer[]{
            NV_ENC_TUNING_INFO_HIGH_QUALITY,
            NV_ENC_TUNING_INFO_LOW_LATENCY,
            NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
            NV_ENC_TUNING_INFO_LOSSLESS
    };

    private final String rcModeNames = "constqp vbr cbr";
    private Integer[] rcModes = new Integer[]{
            NV_ENC_PARAMS_RC_CONSTQP,
            NV_ENC_PARAMS_RC_VBR,
            NV_ENC_PARAMS_RC_CBR
    };

    private final String multiPassNames = "disabled qres fullres";
    private Integer[] multiPassValues = new Integer[]{
            NV_ENC_MULTI_PASS_DISABLED,
            NV_ENC_TWO_PASS_QUARTER_RESOLUTION,
            NV_ENC_TWO_PASS_FULL_RESOLUTION
    };

    private final String qpMapModeNames = "disabled emphasis_level_map delta_qp_map qp_map";
    private Integer[] qpMapModes = new Integer[]{
            NV_ENC_QP_MAP_DISABLED,
            NV_ENC_QP_MAP_EMPHASIS,
            NV_ENC_QP_MAP_DELTA,
            NV_ENC_QP_MAP
    };

    private <T> T parseString(String optionValue, T[] values, String valueNames) {
        String[] optionValueNames = valueNames.split(" ");

        for (int index = 0; index < optionValueNames.length; index++) {
            if (optionValueNames[index].equals(optionValue)) {
                return values[index];
            }
        }

        System.err.println("Invalid value. Can't parse option");

        return null;
    }

    private String convertGUIDToString(GUID[] values, String valueNames, GUID value) {
        for (int index = 0; index < values.length; index++) {
            if (NvCodecUtil.compareGUID(value, values[index])) {
                return valueNames.split(" ")[index];
            }
        }

        System.err.println("Invalid value. Can't convert to one of " + valueNames);

        return null;
    }

    private <T> String convertValueToString(T[] values, String valueNames, T value) {
        for (int index = 0; index < values.length; index++) {
            if (value.equals(values[index])) {
                return valueNames.split(" ")[index];
            }
        }

        System.err.println("Invalid value. Can't convert to one of " + valueNames);

        return null;
    }

    private int parseInt(String optionName, String optionValue) {
        try {
            return Integer.parseInt(optionValue);
        } catch (Exception e) {
            System.err.println(optionName + " need a value of positive number");
        }

        return -1;
    }

    private int parseBitRate(String optionName, String optionValue) {
        try {
            Double.parseDouble(optionValue);
            for (int index = 0; index < optionValue.length(); index++) {
                char ch = optionValue.charAt(index);

                if ((ch < 48 || 57 < ch) && ch != 46) {
                    double r = Double.parseDouble(optionValue.substring(0, index));

                    if (ch != 0 && ch != 'k' && ch != 'm') {
                        System.err.println(optionName + " units: 1, K, M (lower case also allowed)");
                    }

                    return (int) ((ch == 'm' ? 1000000 : (ch == 'k' ? 1000 : 1)) * r);
                }
            }
        } catch (Exception e) {
            System.err.println(optionName + " units: 1, K, M (lower case also allowed)");
        }

        return -1;
    }

    private NV_ENC_QP parseQp(String optionName, String optionValue) {
        String[] qps = optionValue.split(",");
        NV_ENC_QP qp = null;
        try {
            if (qps.length == 1) {
                int qpValue = Integer.parseInt(qps[0]);
                qp = new NV_ENC_QP();
                qp.qpInterP(qpValue);
                qp.qpInterB(qpValue);
                qp.qpIntra(qpValue);
            } else if (qps.length == 3) {
                qp = new NV_ENC_QP();
                qp.qpInterP(Integer.parseInt(qps[0]));
                qp.qpInterB(Integer.parseInt(qps[1]));
                qp.qpIntra(Integer.parseInt(qps[2]));
            } else {
                System.err.println(optionName + " qp_for_P_B_I or qp_P,qp_B,qp_I (no space is allowed)");
            }
        } catch (Exception e) {
            qp = null;
        }

        return qp;
    }

    public GUID getEncodeGUID() {
        return this.guidCodec;
    }

    public GUID getPresetGUID() {
        return this.guidPreset;
    }

    public int getTuningInfo() {
        return this.tuningInfo;
    }

    public boolean isCodecH264() {
        return NvCodecUtil.compareGUID(this.getEncodeGUID(), NV_ENC_CODEC_H264_GUID());
    }

    public boolean isCodecHEVC() {
        return NvCodecUtil.compareGUID(this.getEncodeGUID(), NV_ENC_CODEC_HEVC_GUID());
    }

    public NvEncoderInitParam(String param) {
        this(param, null, false);
    }

    public NvEncoderInitParam(String param, InitFunction initFunction, boolean lowLatency) {
        this.param = param.toLowerCase();
        this.lowLatency = lowLatency;
        this.initFunction = new InitFunction();

        this.guidCodec = NV_ENC_CODEC_H264_GUID();
        this.guidPreset = NV_ENC_PRESET_P3_GUID();
        this.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;

        if (initFunction != null) {
            this.initFunction = initFunction;
        }

        this.tokens = this.param.split(" ");

        for (int index = 0; index < this.tokens.length; index++) {
            String token = this.tokens[index];
            if (index + 1 < this.tokens.length) {
                switch (token) {
                    case "-codec":
                        this.guidCodec = this.parseString(this.tokens[++index], this.codecs, this.codecNames);
                        break;
                    case "-preset":
                        this.guidPreset = this.parseString(this.tokens[++index], this.presets, this.presetNames);
                        break;
                    case "-tuninginfo":
                        this.tuningInfo = this.parseString(this.tokens[++index], this.tuningInfos, this.tuningInfoNames);
                        break;
                }
            }
        }
    }

    public String getHelpMessage() {
        return this.getHelpMessage(false, false, false, false);
    }

    public String getHelpMessage(boolean meOnly, boolean unbuffered, boolean hide444, boolean outputInVidMem) {
        StringBuilder sb = new StringBuilder();

        if (outputInVidMem && meOnly) {
            sb.append("-codec       Codec: ").append("h264\n");
        } else {
            sb.append("-codec       Codec: ").append(this.codecNames).append("\n");
        }

        sb.append("-preset      Preset: ").append(this.presetNames).append("\n")
                .append("-profile     H264: ").append(this.h264ProfileNames);

        if (outputInVidMem && meOnly) {
            sb.append("\n");
        } else {
            sb.append("; HEVC: ").append(this.hevcProfileNames).append("\n");
        }

        if (!meOnly) {
            if (!this.lowLatency) {
                sb.append("-tuninginfo  TuningInfo: ").append(this.tuningInfoNames).append("\n");
            } else {
                sb.append("-tuninginfo  TuningInfo: ").append(this.lowLatencyTuningInfoNames).append("\n");
            }

            sb.append("-multipass   Multipass: ").append(this.multiPassNames).append("\n");
        }

        if (!hide444 && !this.lowLatency) {
            sb.append("-444         (Only for RGB input) YUV444 encode\n");
        }

        if (!meOnly) {
            sb.append("-rc          Rate control mode: ").append(this.rcModeNames).append("\n")
                    .append("-fps         Frame rate\n")
                    .append("-gop         Length of GOP (Group of Pictures)\n");

            if (!unbuffered && !this.lowLatency) {
                sb.append("-bf          Number of consecutive B-frames\n");
            }

            sb.append("-bitrate     Average bit rate, can be in unit of 1, K, M\n")
                    .append("-maxbitrate  Max bit rate, can be in unit of 1, K, M\n")
                    .append("-vbvbufsize  VBV buffer size in bits, can be in unit of 1, K, M\n")
                    .append("-vbvinit     VBV initial delay in bits, can be in unit of 1, K, M\n");

            if (!this.lowLatency) {
                sb.append("-aq          Enable spatial AQ and set its stength (range 1-15, 0-auto)\n")
                        .append("-temporalaq  (No value) Enable temporal AQ\n");
            }

            if (!unbuffered && !this.lowLatency) {
                sb.append("-lookahead   Maximum depth of lookahead (range 0-(31 - number of B frames))\n");
            }

            sb.append("-cq          Target constant quality level for VBR mode (range 1-51, 0-auto)\n")
                    .append("-qmin        Min QP value\n")
                    .append("-qmax        Max QP value\n")
                    .append("-initqp      Initial QP value\n");

            if (!this.lowLatency) {
                sb.append("-constqp     QP value for constqp rate control mode\n")
                        .append("Note: QP value can be in the form of qp_of_P_B_I or qp_P,qp_B,qp_I (no space)\n");
            }

            if (unbuffered && !this.lowLatency) {
                sb.append("Note: Options -bf and -lookahead are unavailable for this app\n");
            }
        }

        return sb.toString();
    }

    public String mainParamToString(final NV_ENC_INITIALIZE_PARAMS pParams) {
        StringBuilder sb = new StringBuilder();

        sb.append("Encoding Parameters:")
                .append("\n").append("\tcodec        : ").append(this.convertGUIDToString(this.codecs, this.codecNames, pParams.encodeGUID()))
                .append("\n").append("\tpreset       : ").append(this.convertGUIDToString(this.presets, this.presetNames, pParams.presetGUID()));

        if (pParams.tuningInfo() != 0) {
            sb.append("\n").append("\ttuningInfo   : ").append(this.convertValueToString(this.tuningInfos, this.tuningInfoNames, pParams.tuningInfo()));
        }

        sb.append("\n").append("\tprofile      : ").append(this.convertGUIDToString(this.profiles, this.profileNames, pParams.encodeConfig().profileGUID()))
                .append("\n").append("\tchroma       : ").append(this.convertValueToString(chromaTypes, chromaNames, (NvCodecUtil.compareGUID(pParams.encodeGUID(), NV_ENC_CODEC_H264_GUID())) ? pParams.encodeConfig().encodeCodecConfig().h264Config().chromaFormatIDC() : pParams.encodeConfig().encodeCodecConfig().hevcConfig().chromaFormatIDC()))
                .append("\n").append("\tbitdepth     : ").append((NvCodecUtil.compareGUID(pParams.encodeGUID(), NV_ENC_CODEC_H264_GUID()) ? 0 : pParams.encodeConfig().encodeCodecConfig().hevcConfig().pixelBitDepthMinus8()) + 8)
                .append("\n").append("\trc           : ").append(this.convertValueToString(this.rcModes, this.rcModeNames, pParams.encodeConfig().rcParams().rateControlMode()));

        if (pParams.encodeConfig().rcParams().rateControlMode() == NV_ENC_PARAMS_RC_CONSTQP) {
            sb.append(" (P,B,I=").append(pParams.encodeConfig().rcParams().constQP().qpInterP()).append(",").append(pParams.encodeConfig().rcParams().constQP().qpInterB()).append(",").append(pParams.encodeConfig().rcParams().constQP().qpIntra()).append(")");
        }

        sb.append("\n").append("\tfps          : ").append(pParams.frameRateNum()).append("/").append(pParams.frameRateDen())
                .append("\n").append("\tgop          : ").append((pParams.encodeConfig().gopLength() == NVENC_INFINITE_GOPLENGTH ? "INF" : pParams.encodeConfig().gopLength()))
                .append("\n").append("\tbf           : ").append(pParams.encodeConfig().frameIntervalP() - 1)
                .append("\n").append("\tmultipass    : ").append(pParams.encodeConfig().rcParams().multiPass())
                .append("\n").append("\tsize         : ").append(pParams.encodeWidth()).append("x").append(pParams.encodeHeight())
                .append("\n").append("\tbitrate      : ").append(pParams.encodeConfig().rcParams().averageBitRate())
                .append("\n").append("\tmaxbitrate   : ").append(pParams.encodeConfig().rcParams().maxBitRate())
                .append("\n").append("\tvbvbufsize   : ").append(pParams.encodeConfig().rcParams().vbvBufferSize())
                .append("\n").append("\tvbvinit      : ").append(pParams.encodeConfig().rcParams().vbvInitialDelay())
                .append("\n").append("\taq           : ").append((pParams.encodeConfig().rcParams().enableAQ() != 0 ? (pParams.encodeConfig().rcParams().aqStrength() != 0 ? pParams.encodeConfig().rcParams().aqStrength() : "auto") : "disabled"))
                .append("\n").append("\ttemporalaq   : ").append((pParams.encodeConfig().rcParams().enableTemporalAQ() != 0 ? "enabled" : "disabled"))
                .append("\n").append("\tlookahead    : ").append((pParams.encodeConfig().rcParams().enableLookahead() != 0 ? pParams.encodeConfig().rcParams().lookaheadDepth() : "disabled"))
                .append("\n").append("\tcq           : ").append(pParams.encodeConfig().rcParams().targetQuality())
                .append("\n").append("\tqmin         : P,B,I=").append(pParams.encodeConfig().rcParams().minQP().qpInterP()).append(",").append(pParams.encodeConfig().rcParams().minQP().qpInterB()).append(",").append(pParams.encodeConfig().rcParams().minQP().qpIntra())
                .append("\n").append("\tqmax         : P,B,I=").append(pParams.encodeConfig().rcParams().maxQP().qpInterP()).append(",").append(pParams.encodeConfig().rcParams().maxQP().qpInterB()).append(",").append(pParams.encodeConfig().rcParams().maxQP().qpIntra())
                .append("\n").append("\tinitqp       : P,B,I=").append(pParams.encodeConfig().rcParams().initialRCQP().qpInterP()).append(",").append(pParams.encodeConfig().rcParams().initialRCQP().qpInterB()).append(",").append(pParams.encodeConfig().rcParams().initialRCQP().qpIntra());

        return sb.toString();
    }

    public void setInitParams(NV_ENC_INITIALIZE_PARAMS pParams, int eBufferFormat) throws NvCodecException {
        NV_ENC_CONFIG config = pParams.encodeConfig();
        for (int i = 0; i < this.tokens.length; i++) {
            {
                if (this.tokens[i].equals("-codec") && ++i != 0) {
                    continue;
                }
                if (this.tokens[i].equals("-preset") && ++i != 0) {
                    continue;

                }
                if (this.tokens[i].equals("-multipass") && ++i != tokens.length) {
                    Integer multiPass = this.parseString(tokens[i], multiPassValues, multiPassNames);

                    if (multiPass != null) {
                        config.rcParams().multiPass(multiPass);

                        continue;
                    }
                }
                if (tokens[i].equals("-profile") && ++i != tokens.length) {
                    GUID profileGUID = null;

                    if (this.isCodecH264()) {
                        profileGUID = this.parseString(tokens[i], h264Profiles, h264ProfileNames);
                    } else {
                        profileGUID = this.parseString(tokens[i], hevcProfiles, hevcProfileNames);
                    }

                    if (profileGUID != null) {
                        config.profileGUID(profileGUID);
                        continue;
                    }
                }

                if (tokens[i].equals("-rc") && ++i != tokens.length) {
                    int rateControlMode = this.parseString(tokens[i], rcModes, rcModeNames);
                    if (rateControlMode != -1) {
                        config.rcParams().rateControlMode(rateControlMode);
                        continue;
                    }
                }
                if (tokens[i].equals("-fps") && ++i != tokens.length) {
                    int frameRate = this.parseInt("-fps", tokens[i]);

                    if (frameRate != -1) {
                        pParams.frameRateNum(frameRate);
                        continue;
                    }
                }
                if (tokens[i].equals("-bf") && ++i != tokens.length) {
                    int frameIntervalP = this.parseInt("-bf", tokens[i]);

                    if (frameIntervalP != -1) {
                        config.frameIntervalP(++frameIntervalP);

                        continue;
                    }
                }
                if (tokens[i].equals("-bitrate") && ++i != tokens.length) {
                    int bitRate = this.parseBitRate("-bitrate", tokens[i]);

                    if (bitRate != -1) {
                        config.rcParams().averageBitRate(bitRate);
                        continue;
                    }
                }
                if (tokens[i].equals("-maxbitrate") && ++i != tokens.length) {
                    int bitRate = this.parseBitRate("-maxbitrate", tokens[i]);

                    if (bitRate != -1) {
                        config.rcParams().maxBitRate(bitRate);
                        continue;
                    }
                }
                if (tokens[i].equals("-vbvbufsize") && ++i != tokens.length) {
                    int bitRate = this.parseBitRate("-vbvbufsize", tokens[i]);

                    if (bitRate != -1) {
                        config.rcParams().vbvBufferSize(bitRate);
                        continue;
                    }
                }
                if (tokens[i].equals("-vbvinit") && ++i != tokens.length) {
                    int bitRate = this.parseBitRate("-vbvinit", tokens[i]);

                    if (bitRate != -1) {
                        config.rcParams().vbvInitialDelay(bitRate);
                        continue;
                    }
                }
                if (tokens[i].equals("-cq") && ++i != tokens.length) {
                    int frameIntervalP = this.parseInt("-cq", tokens[i]);

                    if (frameIntervalP != -1) {
                        config.rcParams().targetQuality((byte) frameIntervalP);

                        continue;
                    }
                }
                if (tokens[i].equals("-initqp") && ++i != tokens.length) {
                    NV_ENC_QP qp = this.parseQp("-initqp", tokens[i]);
                    if (qp != null) {
                        config.rcParams().initialRCQP(qp);
                        config.rcParams().enableInitialRCQP(1);

                        continue;
                    }
                }
                if (tokens[i].equals("-qmin") && ++i != tokens.length) {
                    NV_ENC_QP qp = this.parseQp("-qmin", tokens[i]);
                    if (qp != null) {
                        config.rcParams().minQP(qp);
                        config.rcParams().enableMinQP(1);

                        continue;
                    }
                }

                if (tokens[i].equals("-qmax") && ++i != tokens.length) {
                    NV_ENC_QP qp = this.parseQp("-qmax", tokens[i]);
                    if (qp != null) {
                        config.rcParams().maxQP(qp);
                        config.rcParams().enableMaxQP(1);

                        continue;
                    }
                }

                if (tokens[i].equals("-constqp") && ++i != tokens.length) {
                    NV_ENC_QP qp = this.parseQp("-constqp", tokens[i]);
                    if (qp != null) {
                        config.rcParams().constQP(qp);

                        continue;
                    }
                }

                if (tokens[i].equals("-temporalaq") && ++i != tokens.length) {
                    config.rcParams().enableTemporalAQ(1);
                    continue;
                }
            }


            if (tokens[i].equals("-lookahead") && ++i != tokens.length) {
                int lookaheadDepth = this.parseInt("-lookahead", tokens[i]);

                if (lookaheadDepth != -1) {
                    config.rcParams().lookaheadDepth((short) lookaheadDepth);
                    config.rcParams().enableLookahead(config.rcParams().lookaheadDepth() > 0 ? 1 : 0);
                    continue;
                }
            }

            if (tokens[i].equals("-aq") && ++i != tokens.length) {
                int aqStrength = this.parseInt("-aq", tokens[i]);

                if (aqStrength != -1) {
                    config.rcParams().enableAQ(1);
                    config.rcParams().aqStrength(aqStrength);
                    continue;
                }
            }

            if (tokens[i].equals("-gop") && ++i != tokens.length) {
                int gopLength = this.parseInt("-gop", tokens[i]);

                if (gopLength != -1) {
                    config.gopLength(gopLength);

                    if (this.isCodecH264()) {
                        config.encodeCodecConfig().h264Config().idrPeriod(gopLength);
                    } else {
                        config.encodeCodecConfig().hevcConfig().idrPeriod(gopLength);
                    }

                    continue;
                }
            }

            if (tokens[i].equals("-444")) {
                if (this.isCodecH264()) {
                    config.encodeCodecConfig().h264Config().chromaFormatIDC(3);
                } else {
                    config.encodeCodecConfig().hevcConfig().chromaFormatIDC(3);
                }

                continue;
            }

            System.err.println("Incorrect parameter: " + tokens[i] + "\nRe-run the application with the -h option to get a list of the supported options.");

            throw new NvCodecException("Incorrect parameter: " + tokens[i] + "\nRe-run the application with the -h option to get a list of the supported options.", 1);
        }

        if (this.isCodecHEVC()) {
            if (eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
                config.encodeCodecConfig().hevcConfig().pixelBitDepthMinus8(2);
            }
        }

        this.initFunction.call(pParams);
        System.out.println(new NvEncoderInitParam("").mainParamToString(pParams));
        System.out.println(new NvEncoderInitParam("").fullParamToString(pParams));
    }

    public String fullParamToString(final NV_ENC_INITIALIZE_PARAMS pInitializeParams) {
        StringBuilder sb = new StringBuilder();

        sb.append("NV_ENC_INITIALIZE_PARAMS:").append("\n")
                .append("encodeGUID: ").append(this.convertGUIDToString(this.codecs, this.codecNames, pInitializeParams.encodeGUID())).append("\n")
                .append("presetGUID: ").append(this.convertGUIDToString(this.presets, this.presetNames, pInitializeParams.presetGUID())).append("\n");

        if (pInitializeParams.tuningInfo() != 0) {
            sb.append("tuningInfo: ").append(this.convertValueToString(this.tuningInfos, this.tuningInfoNames, pInitializeParams.tuningInfo())).append("\n");
        }

        sb.append("encodeWidth: ").append(pInitializeParams.encodeWidth()).append("\n")
                .append("encodeHeight: ").append(pInitializeParams.encodeHeight()).append("\n")
                .append("darWidth: ").append(pInitializeParams.darWidth()).append("\n")
                .append("darHeight: ").append(pInitializeParams.darHeight()).append("\n")
                .append("frameRateNum: ").append(pInitializeParams.frameRateNum()).append("\n")
                .append("frameRateDen: ").append(pInitializeParams.frameRateDen()).append("\n")
                .append("enableEncodeAsync: ").append(pInitializeParams.enableEncodeAsync()).append("\n")
                .append("reportSliceOffsets: ").append(pInitializeParams.reportSliceOffsets()).append("\n")
                .append("enableSubFrameWrite: ").append(pInitializeParams.enableSubFrameWrite()).append("\n")
                .append("enableExternalMEHints: ").append(pInitializeParams.enableExternalMEHints()).append("\n")
                .append("enableMEOnlyMode: ").append(pInitializeParams.enableMEOnlyMode()).append("\n")
                .append("enableWeightedPrediction: ").append(pInitializeParams.enableWeightedPrediction()).append("\n")
                .append("maxEncodeWidth: ").append(pInitializeParams.maxEncodeWidth()).append("\n")
                .append("maxEncodeHeight: ").append(pInitializeParams.maxEncodeHeight()).append("\n")
                .append("maxMEHintCountsPerBlock: ").append(pInitializeParams.maxMEHintCountsPerBlock()).append("\n");

        NV_ENC_CONFIG pConfig = pInitializeParams.encodeConfig();

        sb.append("NV_ENC_CONFIG:").append("\n")
                .append("profile: ").append(this.convertGUIDToString(this.profiles, this.profileNames, pConfig.profileGUID())).append("\n")
                .append("gopLength: ").append(pConfig.gopLength()).append("\n")
                .append("frameIntervalP: ").append(pConfig.frameIntervalP()).append("\n")
                .append("monoChromeEncoding: ").append(pConfig.monoChromeEncoding()).append("\n")
                .append("frameFieldMode: ").append(pConfig.frameFieldMode()).append("\n")
                .append("mvPrecision: ").append(pConfig.mvPrecision()).append("\n")
                .append("NV_ENC_RC_PARAMS:").append("\n")
                .append("    rateControlMode: 0x").append(Integer.toHexString(pConfig.rcParams().rateControlMode())).append("\n")
                .append("    constQP: ").append(pConfig.rcParams().constQP().qpInterP()).append(", ").append(pConfig.rcParams().constQP().qpInterB()).append(", ").append(pConfig.rcParams().constQP().qpIntra()).append("\n")
                .append("    averageBitRate:  ").append(pConfig.rcParams().averageBitRate()).append("\n")
                .append("    maxBitRate:      ").append(pConfig.rcParams().maxBitRate()).append("\n")
                .append("    vbvBufferSize:   ").append(pConfig.rcParams().vbvBufferSize()).append("\n")
                .append("    vbvInitialDelay: ").append(pConfig.rcParams().vbvInitialDelay()).append("\n")
                .append("    enableMinQP: ").append(pConfig.rcParams().enableMinQP()).append("\n")
                .append("    enableMaxQP: ").append(pConfig.rcParams().enableMaxQP()).append("\n")
                .append("    enableInitialRCQP: ").append(pConfig.rcParams().enableInitialRCQP()).append("\n")
                .append("    enableAQ: ").append(pConfig.rcParams().enableAQ()).append("\n")
                .append("    qpMapMode: ").append(this.convertValueToString(this.qpMapModes, this.qpMapModeNames, pConfig.rcParams().qpMapMode())).append("\n")
                .append("    multipass: ").append(this.convertValueToString(this.multiPassValues, this.multiPassNames, pConfig.rcParams().multiPass())).append("\n")
                .append("    enableLookahead: ").append(pConfig.rcParams().enableLookahead()).append("\n")
                .append("    disableIadapt: ").append(pConfig.rcParams().disableIadapt()).append("\n")
                .append("    disableBadapt: ").append(pConfig.rcParams().disableBadapt()).append("\n")
                .append("    enableTemporalAQ: ").append(pConfig.rcParams().enableTemporalAQ()).append("\n")
                .append("    zeroReorderDelay: ").append(pConfig.rcParams().zeroReorderDelay()).append("\n")
                .append("    enableNonRefP: ").append(pConfig.rcParams().enableNonRefP()).append("\n")
                .append("    strictGOPTarget: ").append(pConfig.rcParams().strictGOPTarget()).append("\n")
                .append("    aqStrength: ").append(pConfig.rcParams().aqStrength()).append("\n")
                .append("    minQP: ").append(pConfig.rcParams().minQP().qpInterP()).append(", ").append(pConfig.rcParams().minQP().qpInterB()).append(", ").append(pConfig.rcParams().minQP().qpIntra()).append("\n")
                .append("    maxQP: ").append(pConfig.rcParams().maxQP().qpInterP()).append(", ").append(pConfig.rcParams().maxQP().qpInterB()).append(", ").append(pConfig.rcParams().maxQP().qpIntra()).append("\n")
                .append("    initialRCQP: ").append(pConfig.rcParams().initialRCQP().qpInterP()).append(", ").append(pConfig.rcParams().initialRCQP().qpInterB()).append(", ").append(pConfig.rcParams().initialRCQP().qpIntra()).append("\n")
                .append("    temporallayerIdxMask: ").append(pConfig.rcParams().temporallayerIdxMask()).append("\n")
                .append("    temporalLayerQP: ").append(pConfig.rcParams().temporalLayerQP(0)).append(", ").append(pConfig.rcParams().temporalLayerQP(1)).append(", ").append(pConfig.rcParams().temporalLayerQP(2)).append(", ").append(pConfig.rcParams().temporalLayerQP(3)).append(", ").append(pConfig.rcParams().temporalLayerQP(4)).append(", ").append(pConfig.rcParams().temporalLayerQP(5)).append(", ").append(pConfig.rcParams().temporalLayerQP(6)).append(", ").append(pConfig.rcParams().temporalLayerQP(7)).append("\n")
                .append("    targetQuality: ").append(pConfig.rcParams().targetQuality()).append("\n")
                .append("    lookaheadDepth: ").append(pConfig.rcParams().lookaheadDepth()).append("\n");
        if (NvCodecUtil.compareGUID(pInitializeParams.encodeGUID(), NV_ENC_CODEC_H264_GUID())) {
            sb.append("NV_ENC_CODEC_CONFIG (H264):").append("\n")
                    .append("    enableStereoMVC: ").append(pConfig.encodeCodecConfig().h264Config().enableStereoMVC()).append("\n")
                    .append("    hierarchicalPFrames: ").append(pConfig.encodeCodecConfig().h264Config().hierarchicalPFrames()).append("\n")
                    .append("    hierarchicalBFrames: ").append(pConfig.encodeCodecConfig().h264Config().hierarchicalBFrames()).append("\n")
                    .append("    outputBufferingPeriodSEI: ").append(pConfig.encodeCodecConfig().h264Config().outputBufferingPeriodSEI()).append("\n")
                    .append("    outputPictureTimingSEI: ").append(pConfig.encodeCodecConfig().h264Config().outputPictureTimingSEI()).append("\n")
                    .append("    outputAUD: ").append(pConfig.encodeCodecConfig().h264Config().outputAUD()).append("\n")
                    .append("    disableSPSPPS: ").append(pConfig.encodeCodecConfig().h264Config().disableSPSPPS()).append("\n")
                    .append("    outputFramePackingSEI: ").append(pConfig.encodeCodecConfig().h264Config().outputFramePackingSEI()).append("\n")
                    .append("    outputRecoveryPointSEI: ").append(pConfig.encodeCodecConfig().h264Config().outputRecoveryPointSEI()).append("\n")
                    .append("    enableIntraRefresh: ").append(pConfig.encodeCodecConfig().h264Config().enableIntraRefresh()).append("\n")
                    .append("    enableConstrainedEncoding: ").append(pConfig.encodeCodecConfig().h264Config().enableConstrainedEncoding()).append("\n")
                    .append("    repeatSPSPPS: ").append(pConfig.encodeCodecConfig().h264Config().repeatSPSPPS()).append("\n")
                    .append("    enableVFR: ").append(pConfig.encodeCodecConfig().h264Config().enableVFR()).append("\n")
                    .append("    enableLTR: ").append(pConfig.encodeCodecConfig().h264Config().enableLTR()).append("\n")
                    .append("    qpPrimeYZeroTransformBypassFlag: ").append(pConfig.encodeCodecConfig().h264Config().qpPrimeYZeroTransformBypassFlag()).append("\n")
                    .append("    useConstrainedIntraPred: ").append(pConfig.encodeCodecConfig().h264Config().useConstrainedIntraPred()).append("\n")
                    .append("    level: ").append(pConfig.encodeCodecConfig().h264Config().level()).append("\n")
                    .append("    idrPeriod: ").append(pConfig.encodeCodecConfig().h264Config().idrPeriod()).append("\n")
                    .append("    separateColourPlaneFlag: ").append(pConfig.encodeCodecConfig().h264Config().separateColourPlaneFlag()).append("\n")
                    .append("    disableDeblockingFilterIDC: ").append(pConfig.encodeCodecConfig().h264Config().disableDeblockingFilterIDC()).append("\n")
                    .append("    numTemporalLayers: ").append(pConfig.encodeCodecConfig().h264Config().numTemporalLayers()).append("\n")
                    .append("    spsId: ").append(pConfig.encodeCodecConfig().h264Config().spsId()).append("\n")
                    .append("    ppsId: ").append(pConfig.encodeCodecConfig().h264Config().ppsId()).append("\n")
                    .append("    adaptiveTransformMode: ").append(pConfig.encodeCodecConfig().h264Config().adaptiveTransformMode()).append("\n")
                    .append("    fmoMode: ").append(pConfig.encodeCodecConfig().h264Config().fmoMode()).append("\n")
                    .append("    bdirectMode: ").append(pConfig.encodeCodecConfig().h264Config().bdirectMode()).append("\n")
                    .append("    entropyCodingMode: ").append(pConfig.encodeCodecConfig().h264Config().entropyCodingMode()).append("\n")
                    .append("    stereoMode: ").append(pConfig.encodeCodecConfig().h264Config().stereoMode()).append("\n")
                    .append("    intraRefreshPeriod: ").append(pConfig.encodeCodecConfig().h264Config().intraRefreshPeriod()).append("\n")
                    .append("    intraRefreshCnt: ").append(pConfig.encodeCodecConfig().h264Config().intraRefreshCnt()).append("\n")
                    .append("    maxNumRefFrames: ").append(pConfig.encodeCodecConfig().h264Config().maxNumRefFrames()).append("\n")
                    .append("    sliceMode: ").append(pConfig.encodeCodecConfig().h264Config().sliceMode()).append("\n")
                    .append("    sliceModeData: ").append(pConfig.encodeCodecConfig().h264Config().sliceModeData()).append("\n")
                    .append("    NV_ENC_CONFIG_H264_VUI_PARAMETERS:").append("\n")
                    .append("        overscanInfoPresentFlag: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().overscanInfoPresentFlag()).append("\n")
                    .append("        overscanInfo: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().overscanInfo()).append("\n")
                    .append("        videoSignalTypePresentFlag: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().videoSignalTypePresentFlag()).append("\n")
                    .append("        videoFormat: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().videoFormat()).append("\n")
                    .append("        videoFullRangeFlag: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().videoFullRangeFlag()).append("\n")
                    .append("        colourDescriptionPresentFlag: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().colourDescriptionPresentFlag()).append("\n")
                    .append("        colourPrimaries: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().colourPrimaries()).append("\n")
                    .append("        transferCharacteristics: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().transferCharacteristics()).append("\n")
                    .append("        colourMatrix: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().colourMatrix()).append("\n")
                    .append("        chromaSampleLocationFlag: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().chromaSampleLocationFlag()).append("\n")
                    .append("        chromaSampleLocationTop: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().chromaSampleLocationTop()).append("\n")
                    .append("        chromaSampleLocationBot: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().chromaSampleLocationBot()).append("\n")
                    .append("        bitstreamRestrictionFlag: ").append(pConfig.encodeCodecConfig().h264Config().h264VUIParameters().bitstreamRestrictionFlag()).append("\n")
                    .append("    ltrNumFrames: ").append(pConfig.encodeCodecConfig().h264Config().ltrNumFrames()).append("\n")
                    .append("    ltrTrustMode: ").append(pConfig.encodeCodecConfig().h264Config().ltrTrustMode()).append("\n")
                    .append("    chromaFormatIDC: ").append(pConfig.encodeCodecConfig().h264Config().chromaFormatIDC()).append("\n")
                    .append("    maxTemporalLayers: ").append(pConfig.encodeCodecConfig().h264Config().maxTemporalLayers()).append("\n");
        } else if (NvCodecUtil.compareGUID(pInitializeParams.encodeGUID(), NV_ENC_CODEC_HEVC_GUID())) {
            sb.append("NV_ENC_CODEC_CONFIG (HEVC):").append("\n")
                    .append("    level: ").append(pConfig.encodeCodecConfig().hevcConfig().level()).append("\n")
                    .append("    tier: ").append(pConfig.encodeCodecConfig().hevcConfig().tier()).append("\n")
                    .append("    minCUSize: ").append(pConfig.encodeCodecConfig().hevcConfig().minCUSize()).append("\n")
                    .append("    maxCUSize: ").append(pConfig.encodeCodecConfig().hevcConfig().maxCUSize()).append("\n")
                    .append("    useConstrainedIntraPred: ").append(pConfig.encodeCodecConfig().hevcConfig().useConstrainedIntraPred()).append("\n")
                    .append("    disableDeblockAcrossSliceBoundary: ").append(pConfig.encodeCodecConfig().hevcConfig().disableDeblockAcrossSliceBoundary()).append("\n")
                    .append("    outputBufferingPeriodSEI: ").append(pConfig.encodeCodecConfig().hevcConfig().outputBufferingPeriodSEI()).append("\n")
                    .append("    outputPictureTimingSEI: ").append(pConfig.encodeCodecConfig().hevcConfig().outputPictureTimingSEI()).append("\n")
                    .append("    outputAUD: ").append(pConfig.encodeCodecConfig().hevcConfig().outputAUD()).append("\n")
                    .append("    enableLTR: ").append(pConfig.encodeCodecConfig().hevcConfig().enableLTR()).append("\n")
                    .append("    disableSPSPPS: ").append(pConfig.encodeCodecConfig().hevcConfig().disableSPSPPS()).append("\n")
                    .append("    repeatSPSPPS: ").append(pConfig.encodeCodecConfig().hevcConfig().repeatSPSPPS()).append("\n")
                    .append("    enableIntraRefresh: ").append(pConfig.encodeCodecConfig().hevcConfig().enableIntraRefresh()).append("\n")
                    .append("    chromaFormatIDC: ").append(pConfig.encodeCodecConfig().hevcConfig().chromaFormatIDC()).append("\n")
                    .append("    pixelBitDepthMinus8: ").append(pConfig.encodeCodecConfig().hevcConfig().pixelBitDepthMinus8()).append("\n")
                    .append("    idrPeriod: ").append(pConfig.encodeCodecConfig().hevcConfig().idrPeriod()).append("\n")
                    .append("    intraRefreshPeriod: ").append(pConfig.encodeCodecConfig().hevcConfig().intraRefreshPeriod()).append("\n")
                    .append("    intraRefreshCnt: ").append(pConfig.encodeCodecConfig().hevcConfig().intraRefreshCnt()).append("\n")
                    .append("    maxNumRefFramesInDPB: ").append(pConfig.encodeCodecConfig().hevcConfig().maxNumRefFramesInDPB()).append("\n")
                    .append("    ltrNumFrames: ").append(pConfig.encodeCodecConfig().hevcConfig().ltrNumFrames()).append("\n")
                    .append("    vpsId: ").append(pConfig.encodeCodecConfig().hevcConfig().vpsId()).append("\n")
                    .append("    spsId: ").append(pConfig.encodeCodecConfig().hevcConfig().spsId()).append("\n")
                    .append("    ppsId: ").append(pConfig.encodeCodecConfig().hevcConfig().ppsId()).append("\n")
                    .append("    sliceMode: ").append(pConfig.encodeCodecConfig().hevcConfig().sliceMode()).append("\n")
                    .append("    sliceModeData: ").append(pConfig.encodeCodecConfig().hevcConfig().sliceModeData()).append("\n")
                    .append("    maxTemporalLayersMinus1: ").append(pConfig.encodeCodecConfig().hevcConfig().maxTemporalLayersMinus1()).append("\n")
                    .append("    NV_ENC_CONFIG_HEVC_VUI_PARAMETERS:").append("\n")
                    .append("        overscanInfoPresentFlag: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().overscanInfoPresentFlag()).append("\n")
                    .append("        overscanInfo: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().overscanInfo()).append("\n")
                    .append("        videoSignalTypePresentFlag: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().videoSignalTypePresentFlag()).append("\n")
                    .append("        videoFormat: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().videoFormat()).append("\n")
                    .append("        videoFullRangeFlag: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().videoFullRangeFlag()).append("\n")
                    .append("        colourDescriptionPresentFlag: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().colourDescriptionPresentFlag()).append("\n")
                    .append("        colourPrimaries: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().colourPrimaries()).append("\n")
                    .append("        transferCharacteristics: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().transferCharacteristics()).append("\n")
                    .append("        colourMatrix: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().colourMatrix()).append("\n")
                    .append("        chromaSampleLocationFlag: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().chromaSampleLocationFlag()).append("\n")
                    .append("        chromaSampleLocationTop: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().chromaSampleLocationTop()).append("\n")
                    .append("        chromaSampleLocationBot: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().chromaSampleLocationBot()).append("\n")
                    .append("        bitstreamRestrictionFlag: ").append(pConfig.encodeCodecConfig().hevcConfig().hevcVUIParameters().bitstreamRestrictionFlag()).append("\n")
                    .append("    ltrTrustMode: ").append(pConfig.encodeCodecConfig().hevcConfig().ltrTrustMode()).append("\n");
        }
        return sb.toString();
    }
}