package org.bytedeco.nvcodec.samples;

import org.bytedeco.nvcodec.samples.decoder.NvDecoder;
import org.bytedeco.nvcodec.samples.encoder.YuvConverter;
import org.bytedeco.nvcodec.samples.exceptions.InvalidArgument;
import org.bytedeco.nvcodec.samples.exceptions.NvCodecException;
import org.bytedeco.nvcodec.samples.util.AppDecUtils;
import org.bytedeco.nvcodec.samples.util.Dimension;
import org.bytedeco.nvcodec.samples.util.Rectangle;
import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.nvcodec.global.nvcuvid.*;

import org.jcodec.common.DemuxerTrack;
import org.jcodec.common.io.NIOUtils;
import org.jcodec.common.model.Packet;
import org.jcodec.containers.mp4.demuxer.MP4Demuxer;

import java.io.*;

import static org.bytedeco.nvcodec.global.nvcuvid.*;
import static org.bytedeco.nvcodec.global.nvencodeapi.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.nvcodec.samples.util.CudaUtil.checkCudaApiCall;
import static org.bytedeco.nvcodec.samples.util.NvCodecUtil.checkInputFile;
import static org.bytedeco.nvcodec.samples.util.NvCodecUtil.convertToNvCodec;
import static java.lang.System.exit;

public class AppDec {
    private static int gpu;
    private static boolean outPlanar;

    private static String inputFilePath;
    private static String outputFilePath;

    private static Rectangle cropRectangle;
    private static Dimension resizeDimension;

    public static void convertSemiPlanarToPlanar(BytePointer hostFramePointer, int width, int height, int bitDepth) {
        if (bitDepth == 8) {
            // nv12->iyuv
            YuvConverter converter8 = new YuvConverter(width, height, 8);
            converter8.UVInterleavedToPlanar(hostFramePointer);
        } else {
            // p016->yuv420p16
            YuvConverter converter16 = new YuvConverter(width, height, 16);
            converter16.UVInterleavedToPlanar(hostFramePointer);
        }
    }

    /**
     * @param cuContext       - Handle to CUDA context
     * @param inFilePath      - Path to file to be decoded
     * @param outFilePath     - Path to output file into which raw frames are stored
     * @param outPlanar       - Flag to indicate whether output needs to be converted to planar format
     * @param cropRectangle   - Cropping rectangle coordinates
     * @param resizeDimension - Resizing dimensions for output
     * @brief Function to decode media file and write raw frames into an output file.
     */
    public static void decodeMediaFile(CUctx_st cuContext, String inFilePath, String outFilePath, boolean outPlanar, Rectangle cropRectangle, Dimension resizeDimension) {
        try (FileOutputStream outputStream = new FileOutputStream(outFilePath)) {
            try (MP4Demuxer demuxer = MP4Demuxer.createMP4Demuxer(NIOUtils.readableChannel(new File(inFilePath)))) {
                DemuxerTrack videoTrack = demuxer.getVideoTrack();
                int codec = convertToNvCodec(videoTrack.getMeta().getCodec());

                NvDecoder nvDecoder = new NvDecoder(cuContext, false, codec, false, false, cropRectangle, resizeDimension, 0, 0, 1000);

                int frame = 0;
                int frameReturned;

                BytePointer framePointer;
                boolean decodedOutSemiPlanar;


                Packet packet;

                do {
                    packet = videoTrack.nextFrame();

                    byte[] packetDataArray;

                    if (packet == null) {
                        packetDataArray = new byte[0];
                    } else {
                        packetDataArray = packet.getData().array();
                    }

                    BytePointer bytePointer = new BytePointer(packetDataArray);

                    frameReturned = nvDecoder.decode(bytePointer, packetDataArray.length, 0, 0);

                    if (frame == 0 && frameReturned != 0) {
                        System.out.println(nvDecoder.getVideoInfo());
                    }

                    decodedOutSemiPlanar = (nvDecoder.getOutputFormat() == cudaVideoSurfaceFormat_NV12) || (nvDecoder.getOutputFormat() == cudaVideoSurfaceFormat_P016);


                    for (int index = 0; index < frameReturned; index++) {
                        framePointer = nvDecoder.getFrame(null);

                        if (outPlanar && decodedOutSemiPlanar) {
                            convertSemiPlanarToPlanar(framePointer, nvDecoder.getWidth(), nvDecoder.getHeight(), nvDecoder.getBitDepth());
                        }

                        byte[] frameData = new byte[nvDecoder.getFrameSize()];
                        framePointer.get(frameData);

                        outputStream.write(frameData);
                    }

                    frame += frameReturned;
                } while (packet != null);

                String[] aszDecodeOutFormat = new String[]{
                        "NV12", "P016", "YUV444", "YUV444P16"
                };

                if (outPlanar) {
                    aszDecodeOutFormat[0] = "iyuv";
                    aszDecodeOutFormat[1] = "yuv420p16";
                }

                System.out.println("Total frame decoded: " + frame);
                System.out.println("Saved in file " + outFilePath + " in " + aszDecodeOutFormat[nvDecoder.getOutputFormat()] + " format");

                nvDecoder.dispose();
            }
        } catch (NvCodecException | IOException e) {
            e.printStackTrace();
        }
    }

    public static void showHelpAndExit() throws InvalidArgument {
        showHelpAndExit(null);
    }

    public static void showHelpAndExit(final String badOption) throws InvalidArgument {
        boolean bThrowError = false;

        StringBuilder sb = new StringBuilder();
        if (badOption != null) {
            bThrowError = true;
            sb.append("Error parsing \"").append(badOption).append("\"").append("\n");
        }

        sb.append("Options:").append("\n")
                .append("-i             Input file path").append("\n")
                .append("-o             Output file path").append("\n")
                .append("-outplanar     Convert output to planar format").append("\n")
                .append("-gpu           Ordinal of GPU to use").append("\n")
                .append("-crop l,t,r,b  Crop rectangle in left,top,right,bottom (ignored for case 0)").append("\n")
                .append("-resize WxH    Resize to dimension W times H (ignored for case 0)").append("\n");
        sb.append("\n");

        if (bThrowError) {
            throw new InvalidArgument(sb.toString());
        } else {
            System.out.println(sb.toString());
            AppDecUtils.showDecoderCapability();
            exit(0);
        }
    }

    public static void parseCommandLine(int argc, String[] argv) throws InvalidArgument {
        for (int i = 0; i < argc; i++) {
            if (argv[i].equals("-h")) {
                showHelpAndExit();
            }
            if (argv[i].equals("-i")) {
                if (++i == argc) {
                    showHelpAndExit("-i");
                }
                inputFilePath = argv[i];
                continue;
            }
            if (argv[i].equals("-o")) {
                if (++i == argc) {
                    showHelpAndExit("-o");
                }
                outputFilePath = argv[i];
                continue;
            }
            if (argv[i].equals("-outplanar")) {
                outPlanar = true;
                continue;
            }
            if (argv[i].equals("-gpu")) {
                if (++i == argc) {
                    showHelpAndExit("-gpu");
                }

                gpu = Integer.parseInt(argv[i]);

                continue;
            }
            if (argv[i].equals("-crop")) {
                if (++i == argc) {
                    showHelpAndExit("-crop");
                }

                String[] values = argv[i].split(",");

                if (values.length != 4) {
                    showHelpAndExit("-crop");
                }

                cropRectangle.setLeft(Integer.parseInt(values[0]));
                cropRectangle.setTop(Integer.parseInt(values[1]));
                cropRectangle.setRight(Integer.parseInt(values[2]));
                cropRectangle.setBottom(Integer.parseInt(values[3]));

                if ((cropRectangle.getRight() - cropRectangle.getLeft()) % 2 == 1 || (cropRectangle.getBottom() - cropRectangle.getTop()) % 2 == 1) {
                    System.err.println("Cropping rect must have width and height of even numbers");
                    exit(1);
                }

                continue;
            }
            if (argv[i].equals("-resize")) {
                if (++i == argc) {
                    showHelpAndExit("-resize");
                }
                String[] values = argv[i].split("x");

                if (values.length != 2) {
                    showHelpAndExit("-resize");
                }

                resizeDimension.setWidth(Integer.parseInt(values[0]));
                resizeDimension.setHeight(Integer.parseInt(values[1]));

                if (resizeDimension.getWidth() % 2 == 1 || resizeDimension.getHeight() % 2 == 1) {
                    System.err.println("Resizing rect must have width and height of even numbers");
                    exit(1);
                }

                continue;
            }

            showHelpAndExit(argv[i]);
        }
    }

    public static void main(String[] args) {
        inputFilePath = "";
        outputFilePath = "";
        gpu = 0;
        cropRectangle = new Rectangle();
        resizeDimension = new Dimension();
        outPlanar = false;

        try {
            parseCommandLine(args.length, args);
            checkInputFile(inputFilePath);

            if (outputFilePath == null || outputFilePath.isEmpty()) {
                outputFilePath = outPlanar ? "out.planar" : "out.native";
            }

            checkCudaApiCall(cuInit(0));
            IntPointer nGpu = new IntPointer(1);
            checkCudaApiCall(cuDeviceGetCount(nGpu));

            if (gpu < 0 || gpu > nGpu.get()) {
                System.err.printf("GPU ordinal out of range. Should be within [%d,%d]\n", 0, nGpu.get() - 1);
                return;
            }

            CUctx_st cuContext = new CUctx_st();
            cuCtxCreate(cuContext, 0, gpu);

            System.out.println("Decode with demuxing.");
            decodeMediaFile(cuContext, inputFilePath, outputFilePath, outPlanar, cropRectangle, resizeDimension);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
