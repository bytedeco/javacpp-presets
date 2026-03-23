import org.bytedeco.hailort._hailo_hef;
import org.bytedeco.hailort.hailo_3d_image_shape_t;
import org.bytedeco.hailort.hailo_format_t;
import org.bytedeco.hailort.hailo_network_group_info_t;
import org.bytedeco.hailort.hailo_network_info_t;
import org.bytedeco.hailort.hailo_nms_shape_t;
import org.bytedeco.hailort.hailo_quant_info_t;
import org.bytedeco.hailort.hailo_stream_info_t;
import org.bytedeco.hailort.hailo_version_t;
import org.bytedeco.hailort.hailo_vstream_info_t;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.SizeTPointer;

import java.io.File;
import java.nio.charset.StandardCharsets;

import static org.bytedeco.hailort.global.hailort.HAILO_D2H_STREAM;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_HAILO_NMS;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_HAILO_NMS_WITH_BYTE_MASK;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_NCHW;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_NC;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_NHCW;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_NHWC;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_NV12;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_NV21;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_RGB888;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_ORDER_YUY2;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_TYPE_AUTO;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_TYPE_FLOAT32;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_TYPE_UINT16;
import static org.bytedeco.hailort.global.hailort.HAILO_FORMAT_TYPE_UINT8;
import static org.bytedeco.hailort.global.hailort.HAILO_H2D_STREAM;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_NETWORK_GROUPS;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_NETWORK_GROUP_NAME_SIZE;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_NETWORK_NAME_SIZE;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_NETWORKS_IN_NETWORK_GROUP;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_STREAM_NAME_SIZE;
import static org.bytedeco.hailort.global.hailort.HAILO_MAX_STREAMS_COUNT;
import static org.bytedeco.hailort.global.hailort.HAILO_SUCCESS;
import static org.bytedeco.hailort.global.hailort.hailo_create_hef_file;
import static org.bytedeco.hailort.global.hailort.hailo_get_library_version;
import static org.bytedeco.hailort.global.hailort.hailo_get_network_groups_infos;
import static org.bytedeco.hailort.global.hailort.hailo_hef_get_all_stream_infos;
import static org.bytedeco.hailort.global.hailort.hailo_hef_get_all_vstream_infos;
import static org.bytedeco.hailort.global.hailort.hailo_hef_get_network_infos;
import static org.bytedeco.hailort.global.hailort.hailo_release_hef;

public class HefInspectionSample {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Usage: java HefInspectionSample <path-to-model.hef>");
            System.exit(2);
        }

        File hefFile = new File(args[0]);
        if (!hefFile.isFile()) {
            System.err.println("HEF file was not found: " + hefFile.getAbsolutePath());
            System.exit(2);
        }

        Loader.load(org.bytedeco.hailort.global.hailort.class);

        hailo_version_t version = new hailo_version_t();
        checkStatus(hailo_get_library_version(version), "load HailoRT version");
        System.out.printf("HailoRT %d.%d.%d%n", version.major(), version.minor(), version.revision());
        System.out.println("Inspecting HEF: " + hefFile.getAbsolutePath());

        _hailo_hef hef = new _hailo_hef();
        checkStatus(hailo_create_hef_file(hef, hefFile.getAbsolutePath()), "open HEF");

        try {
            printNetworkGroups(hef);
        } finally {
            checkStatus(hailo_release_hef(hef), "release HEF");
        }
    }

    private static void printNetworkGroups(_hailo_hef hef) {
        hailo_network_group_info_t groups = new hailo_network_group_info_t(HAILO_MAX_NETWORK_GROUPS);
        SizeTPointer count = new SizeTPointer(1).put(HAILO_MAX_NETWORK_GROUPS);
        checkStatus(hailo_get_network_groups_infos(hef, groups, count), "enumerate network groups");

        long groupCount = count.get();
        System.out.println("Network groups: " + groupCount);
        for (int i = 0; i < groupCount; i++) {
            hailo_network_group_info_t group = groups.getPointer(i);
            String groupName = fixedLengthString(group.name(), HAILO_MAX_NETWORK_GROUP_NAME_SIZE);
            System.out.printf("  [%d] %s (multi-context=%s)%n", i, groupName, group.is_multi_context());
            printNetworks(hef, groupName);
            printStreams(hef, groupName);
            printVStreams(hef, groupName);
        }
    }

    private static void printNetworks(_hailo_hef hef, String groupName) {
        hailo_network_info_t networks = new hailo_network_info_t(HAILO_MAX_NETWORKS_IN_NETWORK_GROUP);
        SizeTPointer count = new SizeTPointer(1).put(HAILO_MAX_NETWORKS_IN_NETWORK_GROUP);
        checkStatus(hailo_hef_get_network_infos(hef, groupName, networks, count), "enumerate networks for " + groupName);

        long networkCount = count.get();
        System.out.println("    Networks: " + networkCount);
        for (int i = 0; i < networkCount; i++) {
            hailo_network_info_t network = networks.getPointer(i);
            System.out.println("      - " + fixedLengthString(network.name(), HAILO_MAX_NETWORK_NAME_SIZE));
        }
    }

    private static void printStreams(_hailo_hef hef, String groupName) {
        hailo_stream_info_t streams = new hailo_stream_info_t(HAILO_MAX_STREAMS_COUNT);
        SizeTPointer count = new SizeTPointer(1);
        count.put(HAILO_MAX_STREAMS_COUNT);
        checkStatus(hailo_hef_get_all_stream_infos(hef, groupName, streams, HAILO_MAX_STREAMS_COUNT, count),
                "enumerate streams for " + groupName);

        long streamCount = count.get();
        System.out.println("    Streams: " + streamCount);
        for (int i = 0; i < streamCount; i++) {
            hailo_stream_info_t stream = streams.getPointer(i);
            System.out.printf("      - %s [%s] %s, hwFrameSize=%d, hwDataBytes=%d, quant=%s%n",
                    fixedLengthString(stream.name(), HAILO_MAX_STREAM_NAME_SIZE),
                    directionName(stream.direction()),
                    describeFormat(stream.format()),
                    stream.hw_frame_size(),
                    stream.hw_data_bytes(),
                    describeQuant(stream.quant_info()));
        }
    }

    private static void printVStreams(_hailo_hef hef, String groupName) {
        hailo_vstream_info_t vstreams = new hailo_vstream_info_t(HAILO_MAX_STREAMS_COUNT);
        SizeTPointer count = new SizeTPointer(1).put(HAILO_MAX_STREAMS_COUNT);
        checkStatus(hailo_hef_get_all_vstream_infos(hef, groupName, vstreams, count), "enumerate vstreams for " + groupName);

        long vstreamCount = count.get();
        System.out.println("    VStreams: " + vstreamCount);
        for (int i = 0; i < vstreamCount; i++) {
            hailo_vstream_info_t vstream = vstreams.getPointer(i);
            String name = fixedLengthString(vstream.name(), HAILO_MAX_STREAM_NAME_SIZE);
            String networkName = fixedLengthString(vstream.network_name(), HAILO_MAX_NETWORK_NAME_SIZE);
            System.out.printf("      - %s (network=%s) [%s] %s, quant=%s%n",
                    name,
                    networkName,
                    directionName(vstream.direction()),
                    describeVStreamShape(vstream),
                    describeQuant(vstream.quant_info()));
        }
    }

    private static String describeVStreamShape(hailo_vstream_info_t vstream) {
        hailo_format_t format = vstream.format();
        if (isNmsOrder(format.order())) {
            hailo_nms_shape_t nmsShape = vstream.nms_shape();
            return String.format("%s, nms=%d classes, %d boxes/class, %d total boxes",
                    describeFormat(format),
                    nmsShape.number_of_classes(),
                    nmsShape.max_bboxes_per_class(),
                    nmsShape.max_bboxes_total());
        }

        hailo_3d_image_shape_t shape = vstream.shape();
        return String.format("%s, shape=%dx%dx%d",
                describeFormat(format),
                shape.height(),
                shape.width(),
                shape.features());
    }

    private static String describeFormat(hailo_format_t format) {
        return "type=" + formatTypeName(format.type()) + ", order=" + formatOrderName(format.order());
    }

    private static String describeQuant(hailo_quant_info_t quantInfo) {
        return String.format("scale=%.6f zp=%.6f", quantInfo.qp_scale(), quantInfo.qp_zp());
    }

    private static boolean isNmsOrder(int order) {
        return order == HAILO_FORMAT_ORDER_HAILO_NMS
                || order == HAILO_FORMAT_ORDER_HAILO_NMS_WITH_BYTE_MASK
                || order == HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP
                || order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS
                || order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE;
    }

    private static String directionName(int direction) {
        if (direction == HAILO_H2D_STREAM) {
            return "input";
        }
        if (direction == HAILO_D2H_STREAM) {
            return "output";
        }
        return "unknown(" + direction + ")";
    }

    private static String formatTypeName(int type) {
        if (type == HAILO_FORMAT_TYPE_AUTO) {
            return "AUTO";
        }
        if (type == HAILO_FORMAT_TYPE_UINT8) {
            return "UINT8";
        }
        if (type == HAILO_FORMAT_TYPE_UINT16) {
            return "UINT16";
        }
        if (type == HAILO_FORMAT_TYPE_FLOAT32) {
            return "FLOAT32";
        }
        return Integer.toString(type);
    }

    private static String formatOrderName(int order) {
        if (order == HAILO_FORMAT_ORDER_NHWC) {
            return "NHWC";
        }
        if (order == HAILO_FORMAT_ORDER_NHCW) {
            return "NHCW";
        }
        if (order == HAILO_FORMAT_ORDER_NCHW) {
            return "NCHW";
        }
        if (order == HAILO_FORMAT_ORDER_NC) {
            return "NC";
        }
        if (order == HAILO_FORMAT_ORDER_RGB888) {
            return "RGB888";
        }
        if (order == HAILO_FORMAT_ORDER_NV12) {
            return "NV12";
        }
        if (order == HAILO_FORMAT_ORDER_NV21) {
            return "NV21";
        }
        if (order == HAILO_FORMAT_ORDER_YUY2) {
            return "YUY2";
        }
        if (order == HAILO_FORMAT_ORDER_HAILO_NMS) {
            return "HAILO_NMS";
        }
        if (order == HAILO_FORMAT_ORDER_HAILO_NMS_WITH_BYTE_MASK) {
            return "HAILO_NMS_WITH_BYTE_MASK";
        }
        if (order == HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP) {
            return "HAILO_NMS_ON_CHIP";
        }
        if (order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS) {
            return "HAILO_NMS_BY_CLASS";
        }
        if (order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE) {
            return "HAILO_NMS_BY_SCORE";
        }
        return Integer.toString(order);
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
