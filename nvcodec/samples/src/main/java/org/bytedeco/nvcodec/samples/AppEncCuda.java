package org.bytedeco.nvcodec.samples;

import org.bytedeco.nvcodec.samples.cuda.CRC;
import org.bytedeco.nvcodec.samples.encoder.*;
import org.bytedeco.nvcodec.samples.exceptions.InvalidArgument;
import org.bytedeco.nvcodec.samples.exceptions.NvCodecException;
import org.bytedeco.nvcodec.samples.util.NvCodecUtil;
import org.bytedeco.nvcodec.samples.exceptions.CudaException;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.nvcodec.global.nvencodeapi.*;
import org.bytedeco.nvcodec.nvencodeapi.NV_ENC_CONFIG;
import org.bytedeco.nvcodec.nvencodeapi.NV_ENC_INITIALIZE_PARAMS;
import org.bytedeco.nvcodec.nvencodeapi.NV_ENC_OUTPUT_PTR;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Vector;

import static org.bytedeco.nvcodec.samples.util.CudaUtil.checkCudaApiCall;
import static java.lang.System.exit;
import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.cuda.global.cudart.cuDeviceGetName;
import static org.bytedeco.nvcodec.global.nvencodeapi.*;

public class AppEncCuda {
    public static int iGpu = 0;
    public static int width = 0;
    public static int height = 0;
    public static int cuStreamType = -1;

    public static boolean bOutputInVidMem;

    public static String szInputFilePath = "";
    public static String szOutputFilePath = "";

    public static NvEncoderInitParam initParam;
    public static int eFormat = NV_ENC_BUFFER_FORMAT_IYUV;

    public static void showEncoderCapability() {
        StringBuilder sb = new StringBuilder();
        try {
            checkCudaApiCall(cuInit(0));
            IntPointer gpuNum = new IntPointer();
            checkCudaApiCall(cuDeviceGetCount(gpuNum));

            sb.append("Encoder Capability \n\n");
            for (int iGpu = 0; iGpu < gpuNum.get(); iGpu++) {
                IntPointer cuDevice = new IntPointer();
                checkCudaApiCall(cuDeviceGet(cuDevice, iGpu));
                BytePointer szDeviceName = new BytePointer(80);
                checkCudaApiCall(cuDeviceGetName(szDeviceName, szDeviceName.sizeof(), cuDevice.get()));
                CUctx_st cuContext = new CUctx_st();
                checkCudaApiCall(cuCtxCreate(cuContext, 0, cuDevice.get()));
                NvEncoderCuda enc = new NvEncoderCuda(cuContext, 1280, 720, NV_ENC_BUFFER_FORMAT_NV12);

                sb.append("GPU ").append(iGpu).append(" - ").append(szDeviceName.getString()).append("\n");
                sb.append("\tH264:\t\t  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_H264_GUID(), NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) != 0 ? "yes" : "no").append("\n")
                        .append("\tH264_444:\t  ")
                        .append(enc.getCapabilityValue(NV_ENC_CODEC_H264_GUID(), NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) != 0 ? "yes" : "no").append("\n").append(
                        "\tH264_ME:\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_H264_GUID(),
                                NV_ENC_CAPS_SUPPORT_MEONLY_MODE) != 0 ? "yes" : "no").append("\n").append(
                        "\tH264_WxH:\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_H264_GUID(),
                                NV_ENC_CAPS_WIDTH_MAX)).append("*").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_H264_GUID(), NV_ENC_CAPS_HEIGHT_MAX)).append("\n").append(
                        "\tHEVC:\t\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_HEVC_GUID(),
                                NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) != 0 ? "yes" : "no").append("\n").append(
                        "\tHEVC_Main10:\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_HEVC_GUID(),
                                NV_ENC_CAPS_SUPPORT_10BIT_ENCODE) != 0 ? "yes" : "no").append("\n").append(
                        "\tHEVC_Lossless:\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_HEVC_GUID(),
                                NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE) != 0 ? "yes" : "no").append("\n").append(
                        "\tHEVC_SAO:\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_HEVC_GUID(),
                                NV_ENC_CAPS_SUPPORT_SAO) != 0 ? "yes" : "no").append("\n").append(
                        "\tHEVC_444:\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_HEVC_GUID(),
                                NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) != 0 ? "yes" : "no").append("\n").append(
                        "\tHEVC_ME:\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_HEVC_GUID(),
                                NV_ENC_CAPS_SUPPORT_MEONLY_MODE) != 0 ? "yes" : "no").append("\n").append(
                        "\tHEVC_WxH:\t").append("  ").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_HEVC_GUID(),
                                NV_ENC_CAPS_WIDTH_MAX)).append("*").append(
                        enc.getCapabilityValue(NV_ENC_CODEC_HEVC_GUID(), NV_ENC_CAPS_HEIGHT_MAX)).append("\n\n");

                System.out.println(sb.toString());

                enc.destroyEncoder();
                checkCudaApiCall(cuCtxDestroy(cuContext));
            }
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }

    public static void showHelpAndExit() throws InvalidArgument {
        showHelpAndExit(null);
    }

    public static void showHelpAndExit(String badOption) throws InvalidArgument {
        boolean throwError = false;
        StringBuilder sb = new StringBuilder();
        if (badOption != null) {
            throwError = true;

            sb.append("Error parsing \"").append(badOption).append("\"").append("\n");
        }
        sb.append("Options:").append("\n")
                .append("-i               Input file path").append("\n")
                .append("-o               Output file path").append("\n")
                .append("-s               Input resolution in this form: WxH").append("\n")
                .append("-if              Input format: iyuv nv12 yuv444 p010 yuv444p16 bgra bgra10 ayuv abgr abgr10").append("\n")
                .append("-gpu             Ordinal of GPU to use").append("\n")
                .append("-outputInVidMem  Set this to 1 to enable output in Video Memory").append("\n")
                .append("-cuStreamType    Use CU stream for pre and post processing when outputInVidMem is set to 1").append("\n")
                .append("                 CRC of encoded frames will be computed and dumped to file with suffix '_crc.txt' added").append("\n")
                .append("                 to file specified by -o option ").append("\n")
                .append("                 0 : both pre and post processing are on NULL CUDA stream").append("\n")
                .append("                 1 : both pre and post processing are on SAME CUDA stream").append("\n")
                .append("                 2 : both pre and post processing are on DIFFERENT CUDA stream").append("\n");

        sb.append(new NvEncoderInitParam("").getHelpMessage()).append("\n");

        if (throwError) {
            throw new InvalidArgument(sb.toString());
        } else {
            System.out.println(sb.toString());
            showEncoderCapability();

            exit(0);
        }
    }

    public static void parseCommandLine(int argc, String[] argv) throws InvalidArgument {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < argc; i++) {
            if (argv[i].equals("-h")) {
                showHelpAndExit();
            }
            if (argv[i].equals("-i")) {
                if (++i == argc) {
                    showHelpAndExit("-i");
                }
                szInputFilePath = argv[i];
                continue;
            }
            if (argv[i].equals("-o")) {
                if (++i == argc) {
                    showHelpAndExit("-o");
                }
                szOutputFilePath = argv[i];
                continue;
            }
            if (argv[i].equals("-s")) {
                if (++i == argc) {
                    showHelpAndExit("-s");
                }
                String[] values = argv[i].split("x");

                if (values.length != 2) {
                    showHelpAndExit("-s");
                }

                width = Integer.parseInt(values[0]);
                height = Integer.parseInt(values[1]);
                continue;
            }
            String[] vszFileFormatName = {
                    "iyuv", "nv12", "yv12", "yuv444", "p010", "yuv444p16", "bgra", "bgra10", "ayuv", "abgr", "abgr10"
            };

            int[] aFormat =
                    {
                            NV_ENC_BUFFER_FORMAT_IYUV,
                            NV_ENC_BUFFER_FORMAT_NV12,
                            NV_ENC_BUFFER_FORMAT_YV12,
                            NV_ENC_BUFFER_FORMAT_YUV444,
                            NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
                            NV_ENC_BUFFER_FORMAT_YUV444_10BIT,
                            NV_ENC_BUFFER_FORMAT_ARGB,
                            NV_ENC_BUFFER_FORMAT_ARGB10,
                            NV_ENC_BUFFER_FORMAT_AYUV,
                            NV_ENC_BUFFER_FORMAT_ABGR,
                            NV_ENC_BUFFER_FORMAT_ABGR10,
                    };

            if (argv[i].equals("-if")) {
                if (++i == argc) {
                    showHelpAndExit("-if");
                }
                String value = argv[i];

                for (int index = 0; index < vszFileFormatName.length; index++) {
                    String fileFormat = vszFileFormatName[index];
                    if (fileFormat.equals(value)) {
                        eFormat = aFormat[index];
                        break;
                    }
                }

                continue;
            }
            if (argv[i].equals("-gpu")) {
                if (++i == argc) {
                    showHelpAndExit("-gpu");
                }
                iGpu = Integer.parseInt(argv[i]);
                continue;
            }
            if (argv[i].equals("-outputInVidMem")) {
                if (++i == argc) {
                    showHelpAndExit("-outputInVidMem");
                }
                bOutputInVidMem = !argv[i].equals("0");
                continue;
            }
            if (argv[i].equals("-cuStreamType")) {
                if (++i == argc) {
                    showHelpAndExit("-cuStreamType");
                }
                cuStreamType = Integer.parseInt(argv[i]);
                continue;
            }

            // Regard as encoder parameter
            if (argv[i].charAt(0) != '-') {
                showHelpAndExit(argv[i]);
            }

            sb.append(argv[i]).append(" ");
            while (i + 1 < argc && argv[i + 1].charAt(0) != '-') {
                sb.append(argv[++i]).append(" ");
            }
        }

        initParam = new NvEncoderInitParam(sb.toString());
    }

    public static void initializeEncoder(NvEncoder pEnc, NvEncoderInitParam encodeCLIOptions, int eFormat) {
        NV_ENC_INITIALIZE_PARAMS initializeParams = new NV_ENC_INITIALIZE_PARAMS();
        initializeParams.version(NV_ENC_INITIALIZE_PARAMS_VER);

        NV_ENC_CONFIG encodeConfig = new NV_ENC_CONFIG();
        encodeConfig.version(NV_ENC_CONFIG_VER);

        initializeParams.encodeConfig(encodeConfig);
        pEnc.createDefaultEncoderParams(initializeParams, encodeCLIOptions.getEncodeGUID(), encodeCLIOptions.getPresetGUID(), encodeCLIOptions.getTuningInfo());

        try {
            encodeCLIOptions.setInitParams(initializeParams, eFormat);
        } catch (NvCodecException e) {
            e.printStackTrace();
        }

        pEnc.createEncoder(initializeParams);
    }

    public static void encodeCuda(int nWidth, int nHeight, int eFormat, NvEncoderInitParam encodeCLIOptions, CUctx_st cuContext, FileInputStream input, FileOutputStream output) throws IOException {
        NvEncoderCuda encoder = new NvEncoderCuda(cuContext, nWidth, nHeight, eFormat);

        initializeEncoder(encoder, encodeCLIOptions, eFormat);

        int nFrameSize = encoder.getFrameSize();

        byte[] pHostFrame = new byte[nFrameSize];

        int nFrame = 0;

        while (true) {
            // Load the next frame from disk
            int nRead = input.read(pHostFrame);
            // For receiving encoded packets
            Vector<Vector<Byte>> vPacket = new Vector<>();
            if (nRead == nFrameSize) {
                final NvEncoderInputFrame encoderInputFrame = encoder.getNextInputFrame();
                NvEncoderCuda.copyToDeviceFrame(cuContext, new BytePointer(pHostFrame), 0, encoderInputFrame.getInputPointer().getPointer(LongPointer.class),
                        encoderInputFrame.getPitch(),
                        encoder.getEncodeWidth(),
                        encoder.getEncodeHeight(),
                        CU_MEMORYTYPE_HOST,
                        encoderInputFrame.getBufferFormat(),
                        encoderInputFrame.getChromaOffsets(),
                        encoderInputFrame.getNumChromaPlanes());

                encoder.encodeFrame(vPacket);
            } else {
                encoder.endEncode(vPacket);
            }
            nFrame += vPacket.size();

            for (Vector<Byte> packet : vPacket) {
                // For each encoded packet
                byte[] bytes = new byte[packet.size()];

                for (int index = 0; index < packet.size(); index++) {
                    bytes[index] = packet.get(index);
                }

                output.write(bytes);
            }

            if (nRead != nFrameSize)
                break;
        }

        encoder.dispose();

        encoder.destroyEncoder();

        System.out.println("Total frames encoded: " + nFrame);
    }

    public static void encodeCudaOpInVidMem(int nWidth, int nHeight, int eFormat, NvEncoderInitParam encodeCLIOptions, CUctx_st cuContext, FileInputStream input, FileOutputStream output, int cuStreamType) throws IOException {
        NvEncoderOutputInVidMemCuda encoder = new NvEncoderOutputInVidMemCuda(cuContext, nWidth, nHeight, eFormat);

        initializeEncoder(encoder, encodeCLIOptions, eFormat);

        int nFrameSize = encoder.getFrameSize();
        boolean useCUStream = cuStreamType != -1;

        CRC crc = null;
        NvCUStream cuStream = null;

        if (useCUStream) {
            // Allocate CUDA streams
            cuStream = new NvCUStream((CUctx_st) encoder.getDevice(), cuStreamType, encoder);

            // When CUDA streams are used, the encoded frame's CRC is computed using cuda kernel
            crc = new CRC((CUctx_st) encoder.getDevice(), encoder.getOutputBufferSize());
        }

        // For dumping output - encoded frame and CRC, to a file
        DumpVidMemOutput dumpVidMemOutput = new DumpVidMemOutput((CUctx_st) encoder.getDevice(), encoder.getOutputBufferSize(), szOutputFilePath, useCUStream);

        byte[] pHostFrame = new byte[nFrameSize];
        int nFrame = 0;

        // Encoding loop
        while (true) {
            // Load the next frame from disk
            int nRead = input.read(pHostFrame);
            // For receiving encoded packets
            Vector<NV_ENC_OUTPUT_PTR> pVideoMemBfr = new Vector<>();

            if (nRead == nFrameSize) {
                final NvEncoderInputFrame encoderInputFrame = encoder.getNextInputFrame();
                NvEncoderCuda.copyToDeviceFrame(cuContext, new BytePointer(pHostFrame), 0, encoderInputFrame.getInputPointer().getPointer(LongPointer.class),
                        encoderInputFrame.getPitch(),
                        encoder.getEncodeWidth(),
                        encoder.getEncodeHeight(),
                        CU_MEMORYTYPE_HOST,
                        encoderInputFrame.getBufferFormat(),
                        encoderInputFrame.getChromaOffsets(),
                        encoderInputFrame.getNumChromaPlanes(),
                        false,
                        useCUStream ? cuStream.getInputStream() : null);
                encoder.encodeFrame(pVideoMemBfr);
            } else {
                encoder.endEncode(pVideoMemBfr);
            }

            for (int i = 0; i < pVideoMemBfr.size(); ++i) {
                if (useCUStream) {
                    // Compute CRC of encoded stream
                    crc.getCRC(pVideoMemBfr.get(i), cuStream.getOutputStream());
                }

                dumpVidMemOutput.dumpOutputToFile(pVideoMemBfr.get(i).getPointer(LongPointer.class), useCUStream ? crc.getCrcVidMem() : new LongPointer(1) {{
                    put(0);
                }}, output, nFrame);

                nFrame++;
            }

            if (nRead != nFrameSize) break;
        }

        dumpVidMemOutput.dispose();

        if (useCUStream) {
            crc.dispose();
            cuStream.dispose();
        }

        encoder.dispose();

        encoder.destroyEncoder();

        System.out.println("Total frames encoded: " + nFrame);
    }

    public static void main(String[] args) {
        try {
            parseCommandLine(args.length, args);

            NvCodecUtil.checkInputFile(szInputFilePath);
            NvCodecUtil.validateResolution(width, height);

            if (szOutputFilePath == null) {
                szOutputFilePath = initParam.isCodecH264() ? "out.h264" : "out.hevc";
            }

            try {
                checkCudaApiCall(cuInit(0));

                IntPointer nGpu = new IntPointer(1);
                checkCudaApiCall(cuDeviceGetCount(nGpu));

                if (iGpu < 0 || iGpu >= nGpu.get()) {
                    System.out.println("GPU ordinal out of range. Should be within [0 ," + (nGpu.get() - 1) + "]");
                    return;
                }

                IntPointer cuDevice = new IntPointer(1);
                checkCudaApiCall(cuDeviceGet(cuDevice, iGpu));

                BytePointer szDeviceName = new BytePointer(80);
                checkCudaApiCall(cuDeviceGetName(szDeviceName, (int) szDeviceName.limit(), cuDevice.get()));
                System.out.println("GPU in use: " + szDeviceName.getString());

                CUctx_st cuContext = new CUctx_st();
                checkCudaApiCall(cuCtxCreate(cuContext, 0, cuDevice.get()));

                // Open input file
                FileInputStream input = new FileInputStream(szInputFilePath);
                // Open output file
                FileOutputStream output = new FileOutputStream(szOutputFilePath);

                // Encode
                if (bOutputInVidMem) {
                    encodeCudaOpInVidMem(width, height, eFormat, initParam, cuContext, input, output, cuStreamType);
                } else {
                    encodeCuda(width, height, eFormat, initParam, cuContext, input, output);
                }

                output.close();
                input.close();

                System.out.println("Bitstream saved in file " + szOutputFilePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (CudaException | InvalidArgument e) {
            e.printStackTrace();
        }
    }
}