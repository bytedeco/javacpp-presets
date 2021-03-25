package org.bytedeco.nvcodec.samples.util;

import org.bytedeco.nvcodec.samples.exceptions.InvalidArgument;
import org.bytedeco.nvcodec.samples.exceptions.CudaException;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.nvcodec.global.nvcuvid.*;
import org.bytedeco.nvcodec.nvcuvid.CUVIDDECODECAPS;

import static org.bytedeco.nvcodec.global.nvcuvid.*;
import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.nvcodec.samples.util.CudaUtil.checkCudaApiCall;
import static java.lang.System.exit;

public class AppDecUtils {
    public static void showHelpAndExit(final String badOption, String outputFilePath, boolean verbose, int d3d) throws InvalidArgument {
        StringBuilder sb = new StringBuilder();
        boolean bThrowError = false;

        if (badOption != null) {
            bThrowError = true;
            sb.append("Error parsing \"").append(badOption).append("\"").append("\n");
        }

        System.out.println("Options: \n"
                + "-i           Input file path\n"
                + (outputFilePath != null ? "-o           Output file path\n" : "")
                + "-gpu         Ordinal of GPU to use\n"
                + (verbose ? "-v           Verbose message\n" : "")
                + (d3d != 0 ? "-d3d         9 (default): display with D3D9; 11: display with D3D11\n" : ""));

        if (bThrowError) {
            throw new InvalidArgument(sb.toString());
        } else {
            System.out.println(sb.toString());
            exit(0);
        }
    }

    public static void parseCommandLine(int argc, String[] argv, String inputFilePath, String outputFilePath, int gpu, boolean verbose, int d3d) throws InvalidArgument {
        int i;
        for (i = 1; i < argc; i++) {
            if (argv[i].equals("-h")) {
                showHelpAndExit(null, outputFilePath, verbose, d3d);
            }
            if (argv[i].equals("-i")) {
                if (++i == argc) {
                    showHelpAndExit("-i", outputFilePath, verbose, d3d);
                }
                inputFilePath = argv[i];
                continue;
            }
            if (argv[i].equals("-o")) {
                if (++i == argc || outputFilePath == null) {
                    showHelpAndExit("-o", outputFilePath, verbose, d3d);
                }
                outputFilePath = argv[i];
                continue;
            }
            if (argv[i].equals("-gpu")) {
                if (++i == argc) {
                    showHelpAndExit("-gpu", outputFilePath, verbose, d3d);
                }
                gpu = Integer.parseInt(argv[i]);
                continue;
            }
            if (argv[i].equals("-v")) {
                if (!verbose) {
                    showHelpAndExit("-v", outputFilePath, verbose, d3d);
                }

                verbose = true;
                continue;
            }
            if (argv[i].equals("-d3d")) {
                if (++i == argc || d3d == 0) {
                    showHelpAndExit("-d3d", outputFilePath, verbose, d3d);
                }

                d3d = Integer.parseInt(argv[i]);
                continue;
            }

            showHelpAndExit(argv[i], outputFilePath, verbose, d3d);
        }
    }

    /**
     * @param outputFormatMask - Bit mask to represent supported cudaVideoSurfaceFormat in decoder
     * @param outputFormats    - Variable into which output string is written
     * @brief Function to generate space-separated list of supported video surface formats
     */
    public static String getOutputFormatNames(short outputFormatMask, String outputFormats) {
        if (outputFormatMask == 0) {
            outputFormats = "N/A";
        } else {
            if ((outputFormatMask & (1 << cudaVideoSurfaceFormat_NV12)) != 0) {
                outputFormats += "NV12 ";
            }

            if ((outputFormatMask & (1 << cudaVideoSurfaceFormat_P016)) != 0) {
                outputFormats += "P016 ";
            }

            if ((outputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444)) != 0) {
                outputFormats += "YUV444 ";
            }

            if ((outputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444_16Bit)) != 0) {
                outputFormats += "YUV444P16 ";
            }
        }
        return outputFormats;
    }

    public static void createCudaContext(CUctx_st cuContext, int gpu, int flags) {
        IntPointer cuDevice = new IntPointer(1);
        try {
            checkCudaApiCall(cuDeviceGet(cuDevice, gpu));

            byte[] szDeviceName = new byte[80];

            checkCudaApiCall(cuDeviceGetName(szDeviceName, szDeviceName.length, cuDevice.get()));

            System.out.println("GPU in use: " + new String(szDeviceName));

            checkCudaApiCall(cuCtxCreate(cuContext, flags, cuDevice.get()));
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }

    /**
     * @brief Print decoder capabilities on std::cout
     */
    public static void showDecoderCapability() {
        try {
            checkCudaApiCall(cuInit(0));

            IntPointer gpuCount = new IntPointer(1);

            checkCudaApiCall(cuDeviceGetCount(gpuCount));
            System.out.println("Decoder Capability\n\n");

            final String[] aszCodecName = {
                    "JPEG", "MPEG1", "MPEG2", "MPEG4", "H264", "HEVC", "HEVC", "HEVC", "HEVC", "HEVC", "HEVC", "VC1", "VP8", "VP9", "VP9", "VP9"
            };
            final String[] aszChromaFormat = {
                    "4:0:0", "4:2:0", "4:2:2", "4:4:4"
            };

            char[] strOutputFormats = new char[64];

            int aeCodec[] = {cudaVideoCodec_JPEG, cudaVideoCodec_MPEG1, cudaVideoCodec_MPEG2, cudaVideoCodec_MPEG4, cudaVideoCodec_H264, cudaVideoCodec_HEVC,
                    cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_VC1, cudaVideoCodec_VP8,
                    cudaVideoCodec_VP9, cudaVideoCodec_VP9, cudaVideoCodec_VP9};
            int anBitDepthMinus8[] = {0, 0, 0, 0, 0, 0, 2, 4, 0, 2, 4, 0, 0, 0, 2, 4};

            int aeChromaFormat[] = {cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420,
                    cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_444, cudaVideoChromaFormat_444,
                    cudaVideoChromaFormat_444, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420};

            for (int gpu = 0; gpu < gpuCount.get(); gpu++) {

                CUctx_st cuContext = new CUctx_st();
                createCudaContext(cuContext, gpu, 0);

                for (int i = 0; i < aeCodec.length; i++) {
                    CUVIDDECODECAPS decodeCaps = new CUVIDDECODECAPS();
                    decodeCaps.eCodecType(aeCodec[i]);
                    decodeCaps.eChromaFormat(aeChromaFormat[i]);
                    decodeCaps.nBitDepthMinus8(anBitDepthMinus8[i]);

                    cuvidGetDecoderCaps(decodeCaps);

                    strOutputFormats[0] = '\0';

                    String outputFormats = getOutputFormatNames(decodeCaps.nOutputFormatMask(), new String(strOutputFormats));

                    // setw() width = maximum_width_of_string + 2 spaces
                    System.out.printf("Codec  %-7s BitDepth  %-4d ChromaFormat  %-7s Supported  %-3d MaxWidth  %-7d MaxHeight  %-7d MaxMBCount  %-10d MinWidth  %-5d MinHeight  %-5d SurfaceFormat  %-11s",
                            aszCodecName[i], decodeCaps.nBitDepthMinus8() + 8, aszChromaFormat[decodeCaps.eChromaFormat()], decodeCaps.bIsSupported(), decodeCaps.nMaxWidth(), decodeCaps.nMaxHeight(),
                            decodeCaps.nMaxMBCount(), decodeCaps.nMinWidth(), decodeCaps.nMinHeight(), outputFormats);
                }

                System.out.println();

                checkCudaApiCall(cuCtxDestroy(cuContext));
            }
        } catch (CudaException e) {
            e.printStackTrace();
        }
    }
}
