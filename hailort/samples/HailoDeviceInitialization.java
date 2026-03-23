import org.bytedeco.hailort._hailo_device;
import org.bytedeco.hailort.hailo_device_id_t;
import org.bytedeco.hailort.hailo_device_identity_t;
import org.bytedeco.hailort.hailo_extended_device_information_t;
import org.bytedeco.hailort.hailo_version_t;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.SizeTPointer;

import java.nio.charset.StandardCharsets;

import static org.bytedeco.hailort.global.hailort.HAILO_SUCCESS;
import static org.bytedeco.hailort.global.hailort.hailo_create_device_by_id;
import static org.bytedeco.hailort.global.hailort.hailo_get_device_id;
import static org.bytedeco.hailort.global.hailort.hailo_get_extended_device_information;
import static org.bytedeco.hailort.global.hailort.hailo_get_library_version;
import static org.bytedeco.hailort.global.hailort.hailo_identify;
import static org.bytedeco.hailort.global.hailort.hailo_release_device;
import static org.bytedeco.hailort.global.hailort.hailo_scan_devices;

public class HailoDeviceInitialization {
    private static final int MAX_DEVICES = 16;

    public static void main(String[] args) {
        Loader.load(org.bytedeco.hailort.global.hailort.class);

        hailo_version_t version = new hailo_version_t();
        checkStatus(hailo_get_library_version(version), "load HailoRT version");
        System.out.printf("HailoRT %d.%d.%d%n", version.major(), version.minor(), version.revision());

        hailo_device_id_t scannedIds = new hailo_device_id_t(MAX_DEVICES);
        SizeTPointer count = new SizeTPointer(1).put(MAX_DEVICES);
        checkStatus(hailo_scan_devices(null, scannedIds, count), "scan devices");

        long deviceCount = count.get();
        if (deviceCount == 0) {
            System.out.println("No Hailo devices were found.");
            System.out.println("On Windows, confirm the Hailo driver is installed and that the M.2 accelerator is visible through your USB adapter.");
            return;
        }

        System.out.println("Detected devices:");
        for (int i = 0; i < deviceCount; i++) {
            hailo_device_id_t current = scannedIds.getPointer(i);
            System.out.printf("  [%d] %s%n", i, current.id().getString());
        }

        String requestedDeviceId = args.length > 0 ? args[0] : null;
        hailo_device_id_t selectedId = selectDevice(scannedIds, deviceCount, requestedDeviceId);
        System.out.println("Opening device: " + selectedId.id().getString());

        _hailo_device device = new _hailo_device();
        checkStatus(hailo_create_device_by_id(selectedId, device), "open device");

        try {
            hailo_device_id_t resolvedId = new hailo_device_id_t();
            checkStatus(hailo_get_device_id(device, resolvedId), "read device id");

            hailo_device_identity_t identity = new hailo_device_identity_t();
            checkStatus(hailo_identify(device, identity), "identify device");

            hailo_extended_device_information_t extendedInfo = new hailo_extended_device_information_t();
            checkStatus(hailo_get_extended_device_information(device, extendedInfo), "read extended device information");

            System.out.println("Connected to: " + resolvedId.id().getString());
            System.out.println("  Product: " + fixedLengthString(identity.product_name(), identity.product_name_length() & 0xFF));
            System.out.println("  Part number: " + fixedLengthString(identity.part_number(), identity.part_number_length() & 0xFF));
            System.out.println("  Serial: " + fixedLengthString(identity.serial_number(), identity.serial_number_length() & 0xFF));
            System.out.println("  Firmware: " + identity.fw_version().major() + "." + identity.fw_version().minor() + "." + identity.fw_version().revision());
            System.out.println("  Core clock (MHz): " + extendedInfo.neural_network_core_clock_rate());
            System.out.println("  PCIe supported: " + extendedInfo.supported_features().pcie());
            System.out.println("  Ethernet supported: " + extendedInfo.supported_features().ethernet());
        } finally {
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

    private static String fixedLengthString(org.bytedeco.javacpp.BytePointer bytes, int length) {
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
