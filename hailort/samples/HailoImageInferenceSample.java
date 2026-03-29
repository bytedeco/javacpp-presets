import org.bytedeco.hailort._hailo_configured_network_group;
import org.bytedeco.hailort._hailo_device;
import org.bytedeco.hailort._hailo_hef;
import org.bytedeco.hailort._hailo_activated_network_group;
import org.bytedeco.hailort.hailo_activate_network_group_params_t;
import org.bytedeco.hailort.hailo_3d_image_shape_t;
import org.bytedeco.hailort.hailo_bbox_float32_t;
import org.bytedeco.hailort.hailo_configure_params_t;
import org.bytedeco.hailort.hailo_device_id_t;
import org.bytedeco.hailort.hailo_format_t;
import org.bytedeco.hailort.hailo_network_group_info_t;
import org.bytedeco.hailort.hailo_nms_shape_t;
import org.bytedeco.hailort.hailo_stream_raw_buffer_by_name_t;
import org.bytedeco.hailort.hailo_stream_raw_buffer_t;
import org.bytedeco.hailort.hailo_version_t;
import org.bytedeco.hailort.hailo_vstream_info_t;
import org.bytedeco.hailort.hailo_input_vstream_params_by_name_t;
import org.bytedeco.hailort.hailo_output_vstream_params_by_name_t;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteOrder;
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

public class HailoImageInferenceSample {
    private static final String DEFAULT_HEF = "C:\\Users\\BarryPitman\\work\\javacpp-presets\\hailort\\models\\yolov8s_hailo8l_v2.11.0.hef";
    private static final int MAX_DEVICES = 16;
    private static final int MAX_DETECTIONS_TO_PRINT = 10;

    public static void main(String[] args) throws Exception {
        Loader.load(org.bytedeco.hailort.global.hailort.class);

        String hefPath = args.length > 0 ? args[0] : DEFAULT_HEF;
        String imagePath = args.length > 1 ? args[1] : null;
        String requestedDeviceId = args.length > 2 ? args[2] : null;

        File hefFile = new File(hefPath);
        if (!hefFile.isFile()) {
            throw new IllegalArgumentException("HEF file was not found: " + hefFile.getAbsolutePath());
        }

        hailo_version_t version = new hailo_version_t();
        checkStatus(hailo_get_library_version(version), "load HailoRT version");
        System.out.printf("HailoRT %d.%d.%d%n", version.major(), version.minor(), version.revision());
        System.out.println("Using HEF: " + hefFile.getAbsolutePath());

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
        // Reuse for video/multi-image workloads:
        // - Keep the device open across many frames.
        // - Keep the HEF loaded while you are serving one model.
        checkStatus(hailo_create_device_by_id(selectedId, device), "open device");
        checkStatus(hailo_create_hef_file(hef, hefFile.getAbsolutePath()), "open HEF");

        try {
            String networkGroupName = getFirstNetworkGroupName(hef);
            hailo_vstream_info_t allVstreams = new hailo_vstream_info_t(HAILO_MAX_STREAMS_COUNT);
            SizeTPointer vstreamCountPtr = new SizeTPointer(1).put(HAILO_MAX_STREAMS_COUNT);
            checkStatus(hailo_hef_get_all_vstream_infos(hef, networkGroupName, allVstreams, vstreamCountPtr), "enumerate vstreams");

            hailo_configure_params_t configureParams = new hailo_configure_params_t();
            checkStatus(hailo_init_configure_params_by_device(hef, device, configureParams), "init configure params");

            _hailo_configured_network_group networkGroup = new _hailo_configured_network_group();
            SizeTPointer networkGroupCountPtr = new SizeTPointer(1).put(1);
            // Reuse for video/multi-image workloads:
            // - Configure once per model/device combination.
            // - Keep the configured network group alive while processing a stream of frames.
            checkStatus(hailo_configure_device(device, hef, configureParams, networkGroup, networkGroupCountPtr), "configure device");
            if (networkGroupCountPtr.get() != 1) {
                throw new IllegalStateException("Expected exactly one configured network group but got " + networkGroupCountPtr.get());
            }

            // HailoRT requires the configured network group to be activated before hailo_infer().
            // Reuse for video/multi-image workloads:
            // - Activate once before your frame loop.
            // - Deactivate after the loop completes or when switching models/network groups.
            hailo_activate_network_group_params_t activationParams = new hailo_activate_network_group_params_t();
            _hailo_activated_network_group activatedNetworkGroup = new _hailo_activated_network_group();
            checkStatus(hailo_activate_network_group(networkGroup, activationParams, activatedNetworkGroup), "activate network group");

            try {
                hailo_input_vstream_params_by_name_t inputParams = new hailo_input_vstream_params_by_name_t(HAILO_MAX_STREAMS_COUNT);
                SizeTPointer inputCountPtr = new SizeTPointer(1).put(HAILO_MAX_STREAMS_COUNT);
                // Reuse for video/multi-image workloads:
                // - Build input/output vstream params once.
                // - They describe buffer formats and do not need to be rebuilt for every frame.
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

                byte[] inputImage = createInputImage(inputInfo.shape(), imagePath);
                if (inputImage.length != inputFrameSize) {
                    throw new IllegalStateException("Prepared input image is " + inputImage.length + " bytes, expected " + inputFrameSize);
                }

                BytePointer inputBufferPointer = new BytePointer(inputFrameSize);
                // Reuse for video/multi-image workloads:
                // - Keep this host buffer allocated and overwrite its contents for each new frame.
                // - The same idea applies to any BufferedImage/scaling scratch space you manage outside this sample.
                inputBufferPointer.put(inputImage);
                hailo_stream_raw_buffer_by_name_t inputBuffers = new hailo_stream_raw_buffer_by_name_t(1);
                // Reuse for video/multi-image workloads:
                // - The by-name wrapper can stay allocated as long as the stream names and backing buffers stay the same.
                // - For each frame you typically only update the payload inside the existing buffers.
                populateBufferByName(inputBuffers.getPointer(0), inputName, inputBufferPointer, inputFrameSize);

                List<Pointer> ownedOutputPointers = new ArrayList<Pointer>();
                List<Long> outputSizes = new ArrayList<Long>();
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
                    // Some no-detection frames leave unused tail bytes untouched, so zero the host buffer
                    // first to keep output decoding deterministic across repeated runs.
                    zeroBuffer(outputPointer, outputFrameSize);

                    ownedOutputPointers.add(outputPointer);
                    outputSizes.add(outputFrameSize);
                    populateBufferByName(outputBuffers.getPointer(i), outputName, outputPointer, outputFrameSize);
                }

                // For a video pipeline, everything above this point can be moved out of the frame loop.
                // The per-frame work is then:
                // 1. preprocess/copy the next image into the existing input buffer
                // 2. optionally clear/reuse the output buffers
                // 3. call hailo_infer(...)
                // 4. decode the output buffers
                System.out.println("Running one inference...");
                checkStatus(hailo_infer(networkGroup, inputParams, inputBuffers, inputCount, outputParams, outputBuffers, outputCount, 1), "run inference");
                System.out.println("Inference completed.");

                for (int i = 0; i < outputCount; i++) {
                    hailo_output_vstream_params_by_name_t outputParam = outputParams.getPointer(i);
                    String outputName = fixedLengthString(outputParam.name(), HAILO_MAX_STREAM_NAME_SIZE);
                    hailo_vstream_info_t outputInfo = findVStreamInfo(allVstreams, vstreamCountPtr.get(), outputName);
                    Pointer outputPointer = ownedOutputPointers.get(i);
                    long outputSize = outputSizes.get(i);

                    System.out.printf("Output %d: %s, size=%d bytes, format=%s%n",
                            i,
                            outputName,
                            outputSize,
                            describeFormat(outputInfo.format()));

                    if (outputInfo.format().order() == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS && outputPointer instanceof FloatPointer) {
                        printRawBytePreview(outputPointer, (int) Math.min(outputSize, 32));
                        printNmsByClassSummary(outputPointer, outputSize, outputInfo.nms_shape());
                    } else if (outputPointer instanceof FloatPointer) {
                        printFloatPreview((FloatPointer) outputPointer, Math.min((int) (outputSize / Float.BYTES), 16));
                    } else if (outputPointer instanceof BytePointer) {
                        printBytePreview((BytePointer) outputPointer, (int) Math.min(outputSize, 32));
                    }
                }
            } finally {
                checkStatus(hailo_deactivate_network_group(activatedNetworkGroup), "deactivate network group");
            }
        } finally {
            checkStatus(hailo_release_hef(hef), "release HEF");
            checkStatus(hailo_release_device(device), "release device");
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
        checkStatus(hailo_get_vstream_frame_size(info, userFormat, frameSize), "resolve vstream frame size for " + fixedLengthString(info.name(), HAILO_MAX_STREAM_NAME_SIZE));
        return frameSize.get();
    }

    private static byte[] createInputImage(hailo_3d_image_shape_t shape, String imagePath) throws IOException {
        int width = shape.width();
        int height = shape.height();
        int channels = shape.features();
        if (channels != 3) {
            throw new IllegalStateException("This sample currently expects a 3-channel input, got " + channels);
        }

        BufferedImage source;
        if (imagePath != null && !imagePath.isBlank()) {
            File imageFile = new File(imagePath);
            if (!imageFile.isFile()) {
                throw new IllegalArgumentException("Image file was not found: " + imageFile.getAbsolutePath());
            }
            source = ImageIO.read(imageFile);
            if (source == null) {
                throw new IllegalArgumentException("Could not decode image: " + imageFile.getAbsolutePath());
            }
            System.out.println("Using image: " + imageFile.getAbsolutePath());
        } else {
            source = createSyntheticImage(width, height);
            System.out.println("Using generated synthetic test image.");
        }

        BufferedImage scaled = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = scaled.createGraphics();
        try {
            graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            graphics.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            graphics.drawImage(source, 0, 0, width, height, null);
        } finally {
            graphics.dispose();
        }

        byte[] bytes = new byte[width * height * channels];
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = scaled.getRGB(x, y);
                bytes[index++] = (byte) ((rgb >> 16) & 0xFF);
                bytes[index++] = (byte) ((rgb >> 8) & 0xFF);
                bytes[index++] = (byte) (rgb & 0xFF);
            }
        }
        return bytes;
    }

    private static BufferedImage createSyntheticImage(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (x * 255) / Math.max(width - 1, 1);
                int g = (y * 255) / Math.max(height - 1, 1);
                int dx = x - (width / 2);
                int dy = y - (height / 2);
                int distance = Math.min(255, (int) Math.sqrt((dx * dx) + (dy * dy)));
                int b = 255 - distance;
                image.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return image;
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

    private static void printNmsByClassSummary(Pointer output, long outputSize, hailo_nms_shape_t nmsShape) {
        BytePointer bytes = new BytePointer(output);
        int classes = nmsShape.number_of_classes();
        int maxBoxesPerClass = nmsShape.max_bboxes_per_class();
        int bytesPerClass = (int) (outputSize / Math.max(classes, 1));

        System.out.printf("Decoded NMS output: %d classes, up to %d boxes per class%n", classes, maxBoxesPerClass);
        // For this Hailo 4 YOLOv8 HEF, the host-side NMS buffer is organized per class and the counter
        // behaves like a float32 value in practice. We still keep a padded uint16 fallback because the
        // Hailo headers describe multiple related NMS layouts and future HEFs may differ.
        if (tryPrintNmsByClassSummary(bytes, classes, maxBoxesPerClass, bytesPerClass, true)) {
            return;
        }
        if (tryPrintNmsByClassSummary(bytes, classes, maxBoxesPerClass, bytesPerClass, false)) {
            return;
        }
        System.out.println("Could not confidently decode NMS output layout on the host side yet.");
    }

    private static boolean tryPrintNmsByClassSummary(BytePointer bytes, int classes, int maxBoxesPerClass, int bytesPerClass, boolean countIsFloat32) {
        int totalBoxes = 0;
        int printed = 0;

        for (int classId = 0; classId < classes; classId++) {
            int classOffset = classId * bytesPerClass;
            int bboxCount = countIsFloat32 ? Math.round(bytes.getFloat(classOffset)) : (bytes.getShort(classOffset) & 0xFFFF);
            // The fallback path assumes a 16-bit count plus padding so each class block still stays
            // 4-byte aligned before the float bbox payload begins.
            int headerSize = countIsFloat32 ? Float.BYTES : 4;
            if (bboxCount < 0 || bboxCount > maxBoxesPerClass) {
                return false;
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
                    return false;
                }
                if (printed < MAX_DETECTIONS_TO_PRINT) {
                    System.out.printf("  class=%d score=%.4f box=[%.4f, %.4f, %.4f, %.4f]%n",
                            classId, score, yMin, xMin, yMax, xMax);
                    printed++;
                }
            }
        }

        System.out.println("NMS count encoding: " + (countIsFloat32 ? "float32" : "uint16+padded"));
        System.out.println("Total decoded boxes: " + totalBoxes);
        if (printed == 0) {
            System.out.println("No detections were produced for this image.");
        } else if (totalBoxes > printed) {
            System.out.println("Printed the first " + printed + " detections.");
        }
        return true;
    }

    private static boolean isReasonableBox(float yMin, float xMin, float yMax, float xMax, float score) {
        // We allow slightly out-of-range coordinates because the decoded YOLOv8 NMS output can land
        // just outside [0,1] after post-processing, and some models may choose pixel coordinates.
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

    private static void printFloatPreview(FloatPointer output, int count) {
        StringBuilder builder = new StringBuilder("Float preview:");
        for (int i = 0; i < count; i++) {
            builder.append(' ').append(String.format("%.5f", output.get(i)));
        }
        System.out.println(builder.toString());
    }

    private static void printBytePreview(BytePointer output, int count) {
        StringBuilder builder = new StringBuilder("Byte preview:");
        for (int i = 0; i < count; i++) {
            builder.append(' ').append(output.get(i) & 0xFF);
        }
        System.out.println(builder.toString());
    }

    private static void printRawBytePreview(Pointer output, int count) {
        BytePointer bytes = new BytePointer(output);
        StringBuilder builder = new StringBuilder("Raw byte preview:");
        for (int i = 0; i < count; i++) {
            builder.append(' ').append(bytes.get(i) & 0xFF);
        }
        builder.append(" (nativeOrder=").append(ByteOrder.nativeOrder()).append(')');
        System.out.println(builder.toString());
    }

    private static void zeroBuffer(Pointer output, long sizeInBytes) {
        BytePointer bytes = new BytePointer(output);
        for (long i = 0; i < sizeInBytes; i++) {
            bytes.put(i, (byte) 0);
        }
    }

    private static String describeFormat(hailo_format_t format) {
        return "type=" + format.type() + ", order=" + format.order();
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
}
