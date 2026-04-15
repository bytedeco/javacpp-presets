import org.bytedeco.hailort._hailo_activated_network_group;
import org.bytedeco.hailort._hailo_configured_network_group;
import org.bytedeco.hailort._hailo_device;
import org.bytedeco.hailort._hailo_hef;
import org.bytedeco.hailort.hailo_3d_image_shape_t;
import org.bytedeco.hailort.hailo_activate_network_group_params_t;
import org.bytedeco.hailort.hailo_configure_params_t;
import org.bytedeco.hailort.hailo_device_id_t;
import org.bytedeco.hailort.hailo_format_t;
import org.bytedeco.hailort.hailo_input_vstream_params_by_name_t;
import org.bytedeco.hailort.hailo_network_group_info_t;
import org.bytedeco.hailort.hailo_nms_shape_t;
import org.bytedeco.hailort.hailo_output_vstream_params_by_name_t;
import org.bytedeco.hailort.hailo_stream_raw_buffer_by_name_t;
import org.bytedeco.hailort.hailo_stream_raw_buffer_t;
import org.bytedeco.hailort.hailo_version_t;
import org.bytedeco.hailort.hailo_vstream_info_t;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.bytedeco.opencv.opencv_videoio.VideoWriter;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_TYPE_FLOAT32;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_TYPE_UINT8;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_NETWORK_GROUPS;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_NETWORK_GROUP_NAME_SIZE;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_STREAM_NAME_SIZE;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_STREAMS_COUNT;
import static org.bytedeco.hailort.global.hailort.HAILO_SUCCESS;
import static org.bytedeco.hailort.global.hailort.hailo_activate_network_group;
import static org.bytedeco.hailort.global.hailort.hailo_configure_device;
import static org.bytedeco.hailort.global.hailort.hailo_create_device_by_id;
import static org.bytedeco.hailort.global.hailort.hailo_create_hef_file;
import static org.bytedeco.hailort.global.hailort.hailo_deactivate_network_group;
import static org.bytedeco.hailort.global.hailort.hailo_get_library_version;
import static org.bytedeco.hailort.global.hailort.hailo_get_network_groups_infos;
import static org.bytedeco.hailort.global.hailort.hailo_get_vstream_frame_size;
import static org.bytedeco.hailort.global.hailort.hailo_hef_get_all_vstream_infos;
import static org.bytedeco.hailort.global.hailort.hailo_infer;
import static org.bytedeco.hailort.global.hailort.hailo_init_configure_params_by_device;
import static org.bytedeco.hailort.global.hailort.hailo_make_input_vstream_params;
import static org.bytedeco.hailort.global.hailort.hailo_make_output_vstream_params;
import static org.bytedeco.hailort.global.hailort.hailo_release_device;
import static org.bytedeco.hailort.global.hailort.hailo_release_hef;
import static org.bytedeco.hailort.global.hailort.hailo_scan_devices;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_GRAY2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_SIMPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FPS;

/**
 * Sample video inference loop for HailoRT.
 *
 * <p>This sample expects ByteDeco OpenCV on the classpath, for example
 * {@code org.bytedeco:opencv-platform}, because it uses OpenCV's {@link VideoCapture}
 * to decode frames from a video file.</p>
 *
 * <p>Usage:
 * <pre>
 * java HailVideoInferenceSample
 * java HailVideoInferenceSample path/to/model.hef path/to/video.mp4
 * java HailVideoInferenceSample path/to/model.hef path/to/video.mp4 0000:55:00.0
 * java HailVideoInferenceSample path/to/model.hef path/to/video.mp4 0000:55:00.0 path/to/output.mp4
 * </pre>
 * </p>
 */
public class HailVideoInferenceSample {
    private static final String DEFAULT_HEF = "C:\\Users\\BarryPitman\\work\\javacpp-presets\\hailort\\models\\yolov8s_hailo8l_v2.11.0.hef";
    private static final String DEFAULT_VIDEO = "C:\\Users\\BarryPitman\\work\\javacpp-presets\\hailort\\samples\\bottle-detection.mp4";
    private static final String DEFAULT_OUTPUT_VIDEO = "C:\\Users\\BarryPitman\\work\\javacpp-presets\\hailort\\samples\\bottle-detection-annotated.mp4";
    private static final int MAX_DEVICES = 16;
    private static final int MAX_DETECTIONS_TO_PRINT = 10;

    public static void main(String[] args) throws Exception {
        Loader.load(org.bytedeco.hailort.global.hailort.class);
        Loader.load(org.bytedeco.opencv.global.opencv_videoio.class);
        Loader.load(org.bytedeco.opencv.global.opencv_imgproc.class);

        String hefPath = args.length > 0 ? args[0] : DEFAULT_HEF;
        String videoPath = args.length > 1 ? args[1] : DEFAULT_VIDEO;
        String requestedDeviceId = args.length > 2 ? args[2] : null;
        String outputVideoPath = args.length > 3 ? args[3] : DEFAULT_OUTPUT_VIDEO;

        File hefFile = new File(hefPath);
        if (!hefFile.isFile()) {
            throw new IllegalArgumentException("HEF file was not found: " + hefFile.getAbsolutePath());
        }
        File videoFile = new File(videoPath);
        if (!videoFile.isFile()) {
            throw new IllegalArgumentException("Video file was not found: " + videoFile.getAbsolutePath());
        }

        hailo_version_t version = new hailo_version_t();
        checkStatus(hailo_get_library_version(version), "load HailoRT version");
        System.out.printf("HailoRT %d.%d.%d%n", version.major(), version.minor(), version.revision());
        System.out.println("Using HEF: " + hefFile.getAbsolutePath());
        System.out.println("Using video: " + videoFile.getAbsolutePath());
        System.out.println("Annotated output: " + new File(outputVideoPath).getAbsolutePath());

        hailo_device_id_t scannedIds = new hailo_device_id_t(MAX_DEVICES);
        SizeTPointer deviceCountPtr = new SizeTPointer(1).put(MAX_DEVICES);
        checkStatus(hailo_scan_devices(null, scannedIds, deviceCountPtr), "scan devices");
        long deviceCount = deviceCountPtr.get();
        if (deviceCount == 0) {
            throw new IllegalStateException("No Hailo devices were found.");
        }

        hailo_device_id_t selectedId = selectDevice(scannedIds, deviceCount, requestedDeviceId);
        System.out.println("Using device: " + selectedId.id().getString());

        _hailo_device device = new _hailo_device();
        _hailo_hef hef = new _hailo_hef();
        checkStatus(hailo_create_device_by_id(selectedId, device), "open device");
        checkStatus(hailo_create_hef_file(hef, hefFile.getAbsolutePath()), "open HEF");

        // Initialization phase:
        // - open the device
        // - load the HEF
        // - configure and activate the network group
        // - allocate reusable host buffers once
        //
        // Everything in this block can be reused across many frames of one video,
        // which is exactly what you want for real-time or batch video processing.
        try {
            String networkGroupName = getFirstNetworkGroupName(hef);
            hailo_vstream_info_t allVstreams = new hailo_vstream_info_t(HAILO_MAX_STREAMS_COUNT);
            SizeTPointer vstreamCountPtr = new SizeTPointer(1).put(HAILO_MAX_STREAMS_COUNT);
            checkStatus(hailo_hef_get_all_vstream_infos(hef, networkGroupName, allVstreams, vstreamCountPtr), "enumerate vstreams");

            hailo_configure_params_t configureParams = new hailo_configure_params_t();
            checkStatus(hailo_init_configure_params_by_device(hef, device, configureParams), "init configure params");

            _hailo_configured_network_group networkGroup = new _hailo_configured_network_group();
            SizeTPointer networkGroupCountPtr = new SizeTPointer(1).put(1);
            checkStatus(hailo_configure_device(device, hef, configureParams, networkGroup, networkGroupCountPtr), "configure device");
            if (networkGroupCountPtr.get() != 1) {
                throw new IllegalStateException("Expected exactly one configured network group but got " + networkGroupCountPtr.get());
            }

            hailo_activate_network_group_params_t activationParams = new hailo_activate_network_group_params_t();
            _hailo_activated_network_group activatedNetworkGroup = new _hailo_activated_network_group();
            checkStatus(hailo_activate_network_group(networkGroup, activationParams, activatedNetworkGroup), "activate network group");

            try {
                hailo_input_vstream_params_by_name_t inputParams = new hailo_input_vstream_params_by_name_t(HAILO_MAX_STREAMS_COUNT);
                SizeTPointer inputCountPtr = new SizeTPointer(1).put(HAILO_MAX_STREAMS_COUNT);
                checkStatus(hailo_make_input_vstream_params(networkGroup, false, HAILO_FORMAT_TYPE_UINT8, inputParams, inputCountPtr),
                        "build input vstream params");
                long inputCount = inputCountPtr.get();
                if (inputCount != 1) {
                    throw new IllegalStateException("This sample expects exactly one input vstream, got " + inputCount);
                }

                hailo_output_vstream_params_by_name_t outputParams = new hailo_output_vstream_params_by_name_t(HAILO_MAX_STREAMS_COUNT);
                SizeTPointer outputCountPtr = new SizeTPointer(1).put(HAILO_MAX_STREAMS_COUNT);
                checkStatus(hailo_make_output_vstream_params(networkGroup, false, HAILO_FORMAT_TYPE_FLOAT32, outputParams, outputCountPtr),
                        "build output vstream params");
                long outputCount = outputCountPtr.get();
                if (outputCount < 1) {
                    throw new IllegalStateException("No output vstreams were produced for the configured network group.");
                }

                hailo_input_vstream_params_by_name_t inputParam = inputParams.getPointer(0);
                String inputName = fixedLengthString(inputParam.name(), HAILO_MAX_STREAM_NAME_SIZE);
                hailo_vstream_info_t inputInfo = findVStreamInfo(allVstreams, vstreamCountPtr.get(), inputName);
                long inputFrameSize = getFrameSize(inputInfo, inputParam.params().user_buffer_format());
                hailo_3d_image_shape_t inputShape = inputInfo.shape();
                int inputWidth = inputShape.width();
                int inputHeight = inputShape.height();
                int inputChannels = inputShape.features();
                if (inputChannels != 3) {
                    throw new IllegalStateException("This sample currently expects a 3-channel input, got " + inputChannels);
                }

                BytePointer inputBufferPointer = new BytePointer(inputFrameSize);
                byte[] reusableRgbBytes = new byte[(int) inputFrameSize];
                hailo_stream_raw_buffer_by_name_t inputBuffers = new hailo_stream_raw_buffer_by_name_t(1);
                populateBufferByName(inputBuffers.getPointer(0), inputName, inputBufferPointer, inputFrameSize);

                List<Pointer> ownedOutputPointers = new ArrayList<Pointer>();
                List<Long> outputSizes = new ArrayList<Long>();
                List<String> outputNames = new ArrayList<String>();
                List<hailo_vstream_info_t> outputInfos = new ArrayList<hailo_vstream_info_t>();
                hailo_stream_raw_buffer_by_name_t outputBuffers = new hailo_stream_raw_buffer_by_name_t(outputCount);
                for (int i = 0; i < outputCount; i++) {
                    hailo_output_vstream_params_by_name_t outputParam = outputParams.getPointer(i);
                    String outputName = fixedLengthString(outputParam.name(), HAILO_MAX_STREAM_NAME_SIZE);
                    hailo_vstream_info_t outputInfo = findVStreamInfo(allVstreams, vstreamCountPtr.get(), outputName);
                    long outputFrameSize = getFrameSize(outputInfo, outputParam.params().user_buffer_format());

                    Pointer outputPointer;
                    if (outputParam.params().user_buffer_format().type() == HAILO_FORMAT_TYPE_FLOAT32) {
                        outputPointer = new FloatPointer(outputFrameSize / Float.BYTES);
                    } else {
                        outputPointer = new BytePointer(outputFrameSize);
                    }

                    zeroBuffer(outputPointer, outputFrameSize);
                    ownedOutputPointers.add(outputPointer);
                    outputSizes.add(outputFrameSize);
                    outputNames.add(outputName);
                    outputInfos.add(outputInfo);
                    populateBufferByName(outputBuffers.getPointer(i), outputName, outputPointer, outputFrameSize);
                }

                VideoCapture capture = new VideoCapture(videoFile.getAbsolutePath());
                Mat frame = new Mat();
                Mat resizedFrame = new Mat();
                Mat rgbFrame = new Mat();
                VideoWriter writer = new VideoWriter();
                Size inputSize = new Size(inputWidth, inputHeight);

                // Per-frame loop:
                // - reuse the same device, HEF, network group, params, and host buffers
                // - only replace the input pixels and clear the output buffers
                // - run hailo_infer() once for each decoded frame
                // - draw detections and append the annotated frame to the output video
                //
                // This is the main pattern to carry into a real video service.
                int framesProcessed = 0;
                try {
                    if (!capture.isOpened()) {
                        throw new IllegalStateException("Could not open video: " + videoFile.getAbsolutePath());
                    }

                    double fps = capture.get(CAP_PROP_FPS);
                    if (!(fps > 0.0)) {
                        fps = 30.0;
                    }

                    while (capture.read(frame)) {
                        if (frame.empty()) {
                            break;
                        }

                        long frameStartNanos = System.nanoTime();

                        if (!writer.isOpened()) {
                            openWriter(writer, outputVideoPath, frame.cols(), frame.rows(), fps);
                        }

                        prepareInputFrame(frame, resizedFrame, rgbFrame, inputSize, reusableRgbBytes, inputBufferPointer);
                        zeroOutputBuffers(ownedOutputPointers, outputSizes);
                        checkStatus(hailo_infer(networkGroup, inputParams, inputBuffers, inputCount, outputParams, outputBuffers, outputCount, 1),
                                "run inference for frame " + framesProcessed);
                        List<Detection> detections = collectFrameDetections(framesProcessed, outputNames, outputInfos, ownedOutputPointers, outputSizes);
                        double processingFps = calculateProcessingFps(frameStartNanos, System.nanoTime());
                        annotateFrame(frame, detections, fps, processingFps, framesProcessed);
                        writer.write(frame);
                        framesProcessed++;
                    }

                    System.out.println("Finished processing " + framesProcessed + " frames.");
                } finally {
                    // Cleanup phase for video resources:
                    // - release the decoder handle
                    // - release reusable OpenCV matrices
                    //
                    // Keep this separate from Hailo cleanup so it is easy to see what belongs
                    // to video I/O versus what belongs to the accelerator/runtime.
                    writer.release();
                    capture.release();
                    frame.release();
                    resizedFrame.release();
                    rgbFrame.release();
                }
            } finally {
                // Cleanup phase for Hailo inference state:
                // - deactivate the network group after the frame loop
                // - then release the HEF and device in the outer finally block
                //
                // This ordering mirrors the lifecycle of a long-running inference service.
                checkStatus(hailo_deactivate_network_group(activatedNetworkGroup), "deactivate network group");
            }
        } finally {
            checkStatus(hailo_release_hef(hef), "release HEF");
            checkStatus(hailo_release_device(device), "release device");
        }
    }

    private static void prepareInputFrame(Mat sourceFrame, Mat resizedFrame, Mat rgbFrame, Size inputSize,
                                          byte[] reusableRgbBytes, BytePointer inputBufferPointer) {
        resize(sourceFrame, resizedFrame, inputSize, 0.0, 0.0, INTER_LINEAR);

        int channels = resizedFrame.channels();
        if (channels == 3) {
            cvtColor(resizedFrame, rgbFrame, COLOR_BGR2RGB);
        } else if (channels == 4) {
            cvtColor(resizedFrame, rgbFrame, COLOR_BGRA2RGB);
        } else if (channels == 1) {
            cvtColor(resizedFrame, rgbFrame, COLOR_GRAY2RGB);
        } else {
            throw new IllegalStateException("Unsupported frame channel count: " + channels);
        }

        try (UByteIndexer indexer = rgbFrame.createIndexer()) {
            int width = rgbFrame.cols();
            int height = rgbFrame.rows();
            int offset = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    reusableRgbBytes[offset++] = (byte) indexer.get(y, x, 0);
                    reusableRgbBytes[offset++] = (byte) indexer.get(y, x, 1);
                    reusableRgbBytes[offset++] = (byte) indexer.get(y, x, 2);
                }
            }
        }

        inputBufferPointer.position(0);
        inputBufferPointer.put(reusableRgbBytes, 0, reusableRgbBytes.length);
        inputBufferPointer.position(0);
    }

    private static List<Detection> collectFrameDetections(int frameIndex, List<String> outputNames, List<hailo_vstream_info_t> outputInfos,
                                                          List<Pointer> ownedOutputPointers, List<Long> outputSizes) {
        List<Detection> allDetections = new ArrayList<Detection>();
        boolean printedSummary = false;
        for (int i = 0; i < outputNames.size(); i++) {
            hailo_vstream_info_t outputInfo = outputInfos.get(i);
            Pointer outputPointer = ownedOutputPointers.get(i);
            long outputSize = outputSizes.get(i);

            if (outputInfo.format().order() == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS && outputPointer instanceof FloatPointer) {
                DetectionSummary summary = decodeNmsByClass(outputPointer, outputSize, outputInfo.nms_shape());
                System.out.printf("Frame %d: %d detections on %s (countEncoding=%s)%n",
                        frameIndex, summary.totalBoxes, outputNames.get(i), summary.countEncoding);
                for (String line : summary.previewLines) {
                    System.out.println(line);
                }
                if (summary.totalBoxes > summary.previewLines.size()) {
                    System.out.println("  Printed the first " + summary.previewLines.size() + " detections.");
                }
                allDetections.addAll(summary.detections);
                printedSummary = true;
            }
        }

        if (!printedSummary) {
            System.out.println("Frame " + frameIndex + ": no HAILO_NMS_BY_CLASS output was found.");
        }
        return allDetections;
    }

    private static DetectionSummary decodeNmsByClass(Pointer output, long outputSize, hailo_nms_shape_t nmsShape) {
        BytePointer bytes = new BytePointer(output);
        int classes = nmsShape.number_of_classes();
        int maxBoxesPerClass = nmsShape.max_bboxes_per_class();
        int bytesPerClass = (int) (outputSize / Math.max(classes, 1));

        DetectionSummary summary = tryDecodeNmsByClass(bytes, classes, maxBoxesPerClass, bytesPerClass, true);
        if (summary != null) {
            return summary;
        }

        summary = tryDecodeNmsByClass(bytes, classes, maxBoxesPerClass, bytesPerClass, false);
        if (summary != null) {
            return summary;
        }

        throw new IllegalStateException("Could not confidently decode NMS output layout on the host side.");
    }

    private static DetectionSummary tryDecodeNmsByClass(BytePointer bytes, int classes, int maxBoxesPerClass, int bytesPerClass,
                                                        boolean countIsFloat32) {
        int totalBoxes = 0;
        List<String> previewLines = new ArrayList<String>();
        List<Detection> detections = new ArrayList<Detection>();

        for (int classId = 0; classId < classes; classId++) {
            int classOffset = classId * bytesPerClass;
            int bboxCount = countIsFloat32 ? Math.round(bytes.getFloat(classOffset)) : (bytes.getShort(classOffset) & 0xFFFF);
            int headerSize = countIsFloat32 ? Float.BYTES : 4;
            if (bboxCount < 0 || bboxCount > maxBoxesPerClass) {
                return null;
            }

            totalBoxes += bboxCount;
            for (int i = 0; i < bboxCount; i++) {
                int boxOffset = classOffset + headerSize + (i * (5 * Float.BYTES));
                float yMin = bytes.getFloat(boxOffset);
                float xMin = bytes.getFloat(boxOffset + Float.BYTES);
                float yMax = bytes.getFloat(boxOffset + (2 * Float.BYTES));
                float xMax = bytes.getFloat(boxOffset + (3 * Float.BYTES));
                float score = bytes.getFloat(boxOffset + (4 * Float.BYTES));

                if (!isReasonableBox(yMin, xMin, yMax, xMax, score)) {
                    return null;
                }

                if (previewLines.size() < MAX_DETECTIONS_TO_PRINT) {
                    previewLines.add(String.format("  class=%d score=%.4f box=[%.4f, %.4f, %.4f, %.4f]",
                            classId, score, yMin, xMin, yMax, xMax));
                }
                detections.add(new Detection(classId, score, yMin, xMin, yMax, xMax));
            }
        }

        return new DetectionSummary(countIsFloat32 ? "float32" : "uint16+padded", totalBoxes, previewLines, detections);
    }

    private static void openWriter(VideoWriter writer, String outputVideoPath, int width, int height, double fps) {
        int fourcc = VideoWriter.fourcc((byte) 'm', (byte) 'p', (byte) '4', (byte) 'v');
        if (!writer.open(outputVideoPath, fourcc, fps, new Size(width, height), true)) {
            throw new IllegalStateException("Could not open output video writer: " + outputVideoPath);
        }
    }

    private static double calculateProcessingFps(long frameStartNanos, long frameEndNanos) {
        double elapsedSeconds = Math.max((frameEndNanos - frameStartNanos) / 1_000_000_000.0, 1.0e-9);
        return 1.0 / elapsedSeconds;
    }

    private static void annotateFrame(Mat frame, List<Detection> detections, double sourceFps, double processingFps, int frameIndex) {
        Scalar boxColor = new Scalar(0.0, 255.0, 0.0, 0.0);
        Scalar textColor = new Scalar(255.0, 255.0, 255.0, 0.0);
        Scalar labelBackground = new Scalar(0.0, 140.0, 0.0, 0.0);
        Scalar hudBackground = new Scalar(32.0, 32.0, 32.0, 0.0);
        int width = frame.cols();
        int height = frame.rows();

        // Keep the FPS overlay visible on every frame so video review shows both the
        // source cadence and the approximate end-to-end throughput of this sample.
        rectangle(frame, new Point(8, 8), new Point(Math.min(width - 8, 260), Math.min(height - 8, 76)), hudBackground, -1, LINE_AA, 0);
        putText(frame, String.format("frame=%d", frameIndex), new Point(16, 30),
                FONT_HERSHEY_SIMPLEX, 0.55, textColor, 1, LINE_AA, false);
        putText(frame, String.format("video fps=%.1f", sourceFps), new Point(16, 50),
                FONT_HERSHEY_SIMPLEX, 0.55, textColor, 1, LINE_AA, false);
        putText(frame, String.format("proc fps=%.1f", processingFps), new Point(16, 70),
                FONT_HERSHEY_SIMPLEX, 0.55, textColor, 1, LINE_AA, false);

        for (Detection detection : detections) {
            int left = clamp(scaleCoordinate(detection.xMin, width), 0, Math.max(width - 1, 0));
            int top = clamp(scaleCoordinate(detection.yMin, height), 0, Math.max(height - 1, 0));
            int right = clamp(scaleCoordinate(detection.xMax, width), 0, Math.max(width - 1, 0));
            int bottom = clamp(scaleCoordinate(detection.yMax, height), 0, Math.max(height - 1, 0));

            if (right <= left || bottom <= top) {
                continue;
            }

            rectangle(frame, new Point(left, top), new Point(right, bottom), boxColor, 2, LINE_AA, 0);

            String label = String.format("class=%d score=%.2f", detection.classId, detection.score);
            int labelBottom = Math.max(top, 18);
            int labelTop = Math.max(labelBottom - 18, 0);
            int labelRight = Math.min(left + Math.max(140, label.length() * 8), Math.max(width - 1, 0));
            rectangle(frame, new Point(left, labelTop), new Point(labelRight, labelBottom), labelBackground, -1, LINE_AA, 0);
            putText(frame, label, new Point(left + 3, Math.max(labelBottom - 5, 0)),
                    FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, LINE_AA, false);
        }
    }

    private static int scaleCoordinate(float value, int size) {
        if (value >= -2.0f && value <= 2.0f) {
            return Math.round(value * size);
        }
        return Math.round(value);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static boolean isReasonableBox(float yMin, float xMin, float yMax, float xMax, float score) {
        return Float.isFinite(yMin)
                && Float.isFinite(xMin)
                && Float.isFinite(yMax)
                && Float.isFinite(xMax)
                && Float.isFinite(score)
                && score >= -1.0f
                && score <= 2.0f
                && yMin >= -8192.0f
                && xMin >= -8192.0f
                && yMax >= -8192.0f
                && xMax >= -8192.0f
                && yMin <= 8192.0f
                && xMin <= 8192.0f
                && yMax <= 8192.0f
                && xMax <= 8192.0f
                && yMin <= yMax
                && xMin <= xMax;
    }

    private static void zeroOutputBuffers(List<Pointer> ownedOutputPointers, List<Long> outputSizes) {
        for (int i = 0; i < ownedOutputPointers.size(); i++) {
            zeroBuffer(ownedOutputPointers.get(i), outputSizes.get(i));
        }
    }

    private static void zeroBuffer(Pointer output, long sizeInBytes) {
        BytePointer bytes = new BytePointer(output);
        for (long i = 0; i < sizeInBytes; i++) {
            bytes.put(i, (byte) 0);
        }
    }

    private static hailo_device_id_t selectDevice(hailo_device_id_t scannedIds, long deviceCount, String requestedDeviceId) {
        if (requestedDeviceId == null || requestedDeviceId.isBlank()) {
            return scannedIds.getPointer(0);
        }
        for (int i = 0; i < deviceCount; i++) {
            hailo_device_id_t candidate = scannedIds.getPointer(i);
            if (requestedDeviceId.equals(candidate.id().getString())) {
                return candidate;
            }
        }
        throw new IllegalArgumentException("Requested device ID not found: " + requestedDeviceId);
    }

    private static String getFirstNetworkGroupName(_hailo_hef hef) {
        hailo_network_group_info_t groups = new hailo_network_group_info_t(HAILO_MAX_NETWORK_GROUPS);
        SizeTPointer count = new SizeTPointer(1).put(HAILO_MAX_NETWORK_GROUPS);
        checkStatus(hailo_get_network_groups_infos(hef, groups, count), "enumerate network groups");
        if (count.get() < 1) {
            throw new IllegalStateException("The HEF did not expose any network groups.");
        }
        return fixedLengthString(groups.getPointer(0).name(), HAILO_MAX_NETWORK_GROUP_NAME_SIZE);
    }

    private static hailo_vstream_info_t findVStreamInfo(hailo_vstream_info_t allVstreams, long count, String name) {
        for (int i = 0; i < count; i++) {
            hailo_vstream_info_t candidate = allVstreams.getPointer(i);
            if (name.equals(fixedLengthString(candidate.name(), HAILO_MAX_STREAM_NAME_SIZE))) {
                return candidate;
            }
        }
        throw new IllegalArgumentException("Could not find vstream info for " + name);
    }

    private static long getFrameSize(hailo_vstream_info_t info, hailo_format_t userFormat) {
        SizeTPointer frameSize = new SizeTPointer(1);
        checkStatus(hailo_get_vstream_frame_size(info, userFormat, frameSize),
                "resolve vstream frame size for " + fixedLengthString(info.name(), HAILO_MAX_STREAM_NAME_SIZE));
        return frameSize.get();
    }

    private static void populateBufferByName(hailo_stream_raw_buffer_by_name_t target, String name, Pointer buffer, long size) {
        copyCString(target.name(), name, HAILO_MAX_STREAM_NAME_SIZE);
        hailo_stream_raw_buffer_t rawBuffer = target.raw_buffer();
        rawBuffer.buffer(buffer);
        rawBuffer.size(size);
    }

    private static void copyCString(BytePointer target, String value, int maxLength) {
        byte[] bytes = value.getBytes(StandardCharsets.US_ASCII);
        int limit = Math.min(bytes.length, maxLength - 1);
        for (int i = 0; i < maxLength; i++) {
            target.put(i, (byte) 0);
        }
        for (int i = 0; i < limit; i++) {
            target.put(i, bytes[i]);
        }
    }

    private static String fixedLengthString(BytePointer bytes, int length) {
        byte[] copy = new byte[length];
        bytes.get(copy);
        int actualLength = 0;
        while (actualLength < copy.length && copy[actualLength] != 0) {
            actualLength++;
        }
        return new String(copy, 0, actualLength, StandardCharsets.US_ASCII);
    }

    private static void checkStatus(int status, String action) {
        if (status != HAILO_SUCCESS) {
            throw new IllegalStateException(action + " failed with Hailo status " + status);
        }
    }

    private static final class DetectionSummary {
        private final String countEncoding;
        private final int totalBoxes;
        private final List<String> previewLines;
        private final List<Detection> detections;

        private DetectionSummary(String countEncoding, int totalBoxes, List<String> previewLines, List<Detection> detections) {
            this.countEncoding = countEncoding;
            this.totalBoxes = totalBoxes;
            this.previewLines = previewLines;
            this.detections = detections;
        }
    }

    private static final class Detection {
        private final int classId;
        private final float score;
        private final float yMin;
        private final float xMin;
        private final float yMax;
        private final float xMax;

        private Detection(int classId, float score, float yMin, float xMin, float yMax, float xMax) {
            this.classId = classId;
            this.score = score;
            this.yMin = yMin;
            this.xMin = xMin;
            this.yMax = yMax;
            this.xMax = xMax;
        }
    }
}
