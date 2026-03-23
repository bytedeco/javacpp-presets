package org.bytedeco.hailort;

import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.hailort.presets.hailort.class)
public class HailoHelper {

    // use a regex search & replace, i.e. replace this:
    // HAILO_STATUS__X\((\d+),\s+(\w+)\s+(.*)\)\\
    // this this:
    // public static int $2 = $1; $3
    
    public static int HAILO_SUCCESS = 0; /*!< Success - No error */
    public static int HAILO_UNINITIALIZED = 1; /*!< No error code was initialized */
    public static int HAILO_INVALID_ARGUMENT = 2; /*!< Invalid argument passed to function */
    public static int HAILO_OUT_OF_HOST_MEMORY = 3; /*!< Cannot allocate more memory at host */
    public static int HAILO_TIMEOUT = 4; /*!< Received a timeout */
    public static int HAILO_INSUFFICIENT_BUFFER = 5; /*!< Buffer is insufficient */
    public static int HAILO_INVALID_OPERATION = 6; /*!< Invalid operation */
    public static int HAILO_NOT_IMPLEMENTED = 7; /*!< Code has not been implemented */
    public static int HAILO_INTERNAL_FAILURE = 8; /*!< Unexpected internal failure */
    public static int HAILO_DATA_ALIGNMENT_FAILURE = 9; /*!< Data is not aligned */
    public static int HAILO_CHUNK_TOO_LARGE = 10; /*!< Chunk too large */
    public static int HAILO_INVALID_LOGGER_LEVEL = 11; /*!< Used non-compiled level */
    public static int HAILO_CLOSE_FAILURE = 12; /*!< Failed to close fd */
    public static int HAILO_OPEN_FILE_FAILURE = 13; /*!< Failed to open file */
    public static int HAILO_FILE_OPERATION_FAILURE = 14; /*!< File operation failure */
    public static int HAILO_UNSUPPORTED_CONTROL_PROTOCOL_VERSION = 15; /*!< Unsupported control protocol version */
    public static int HAILO_UNSUPPORTED_FW_VERSION = 16; /*!< Unsupported firmware version */
    public static int HAILO_INVALID_CONTROL_RESPONSE = 17; /*!< Invalid control response */
    public static int HAILO_FW_CONTROL_FAILURE = 18; /*!< Control failed in firmware */
    public static int HAILO_ETH_FAILURE = 19; /*!< Ethernet operation has failed */
    public static int HAILO_ETH_INTERFACE_NOT_FOUND = 20; /*!< Ethernet interface not found */
    public static int HAILO_ETH_RECV_FAILURE = 21; /*!< Ethernet failed at recv operation */
    public static int HAILO_ETH_SEND_FAILURE = 22; /*!< Ethernet failed at send operation */
    public static int HAILO_INVALID_FIRMWARE = 23; /*!< Firmware bin is invalid */
    public static int HAILO_INVALID_CONTEXT_COUNT = 24; /*!< Host build too many contexts */
    public static int HAILO_INVALID_FRAME = 25; /*!< Part or all of the result data is invalid */
    public static int HAILO_INVALID_HEF = 26; /*!< Invalid HEF */
    public static int HAILO_PCIE_NOT_SUPPORTED_ON_PLATFORM = 27; /*!< PCIe not supported on platform */
    public static int HAILO_INTERRUPTED_BY_SIGNAL = 28; /*!< Blocking syscall was interrupted by a signal */
    public static int HAILO_START_VDMA_CHANNEL_FAIL = 29; /*!< Starting VDMA channel failure */
    public static int HAILO_SYNC_VDMA_BUFFER_FAIL = 30; /*!< Synchronizing VDMA buffer failure */
    public static int HAILO_STOP_VDMA_CHANNEL_FAIL = 31; /*!< Stopping VDMA channel failure */
    public static int HAILO_CLOSE_VDMA_CHANNEL_FAIL = 32; /*!< Closing VDMA channel failure */
    public static int HAILO_ATR_TABLES_CONF_VALIDATION_FAIL = 33; /*!< Validating address translation tables failure, for FW control use */
    public static int HAILO_EVENT_CREATE_FAIL = 34; /*!< Creating event failure */
    public static int HAILO_READ_EVENT_FAIL = 35; /*!< Reading event failure */
    public static int HAILO_DRIVER_OPERATION_FAILED = 36; /*!< Driver operation (i.e ioctl) returned failure. Read driver log for more info (dmesg for linux) */
    public static int HAILO_DRIVER_FAIL = HAILO_DRIVER_OPERATION_FAILED; /*!< Driver failure */
    public static int HAILO_INVALID_FIRMWARE_MAGIC = 37; /*!< Invalid FW magic */
    public static int HAILO_INVALID_FIRMWARE_CODE_SIZE = 38; /*!< Invalid FW code size */
    public static int HAILO_INVALID_KEY_CERTIFICATE_SIZE = 39; /*!< Invalid key certificate size */
    public static int HAILO_INVALID_CONTENT_CERTIFICATE_SIZE = 40; /*!< Invalid content certificate size */
    public static int HAILO_MISMATCHING_FIRMWARE_BUFFER_SIZES = 41; /*!< FW buffer sizes mismatch */
    public static int HAILO_INVALID_FIRMWARE_CPU_ID = 42; /*!< Invalid CPU ID in FW */
    public static int HAILO_CONTROL_RESPONSE_MD5_MISMATCH = 43; /*!< MD5 of control response does not match expected MD5 */
    public static int HAILO_GET_CONTROL_RESPONSE_FAIL = 44; /*!< Get control response failed */
    public static int HAILO_GET_D2H_EVENT_MESSAGE_FAIL = 45; /*!< Reading device-to-host message failure */
    public static int HAILO_MUTEX_INIT_FAIL = 46; /*!< Mutex initialization failure */
    public static int HAILO_OUT_OF_DESCRIPTORS = 47; /*!< Cannot allocate more descriptors */
    public static int HAILO_UNSUPPORTED_OPCODE = 48; /*!< Unsupported opcode was sent to device */
    public static int HAILO_USER_MODE_RATE_LIMITER_NOT_SUPPORTED = 49; /*!< User mode rate limiter not supported on platform */
    public static int HAILO_RATE_LIMIT_MAXIMUM_BANDWIDTH_EXCEEDED = 50; /*!< Rate limit exceeded HAILO_DEFAULT_MAX_ETHERNET_BANDWIDTH_BYTES_PER_SEC */
    public static int HAILO_ANSI_TO_UTF16_CONVERSION_FAILED = 51; /*!< Failed converting ANSI string to UNICODE */
    public static int HAILO_UTF16_TO_ANSI_CONVERSION_FAILED = 52; /*!< Failed converting UNICODE string to ANSI */
    public static int HAILO_UNEXPECTED_INTERFACE_INFO_FAILURE = 53; /*!< Failed retrieving interface info */
    public static int HAILO_UNEXPECTED_ARP_TABLE_FAILURE = 54; /*!< Failed retrieving arp table */
    public static int HAILO_MAC_ADDRESS_NOT_FOUND = 55; /*!< MAC address not found in the arp table */
    public static int HAILO_NO_IPV4_INTERFACES_FOUND = 56; /*!< No interfaces found with an IPv4 address */
    public static int HAILO_SHUTDOWN_EVENT_SIGNALED = 57; /*!< A shutdown event has been signaled */
    public static int HAILO_THREAD_ALREADY_ACTIVATED = 58; /*!< The given thread has already been activated */
    public static int HAILO_THREAD_NOT_ACTIVATED = 59; /*!< The given thread has not been activated */
    public static int HAILO_THREAD_NOT_JOINABLE = 60; /*!< The given thread is not joinable */
    public static int HAILO_NOT_FOUND = 61; /*!< Could not find element */
    public static int HAILO_COMMUNICATION_CLOSED = 62; /*!< The communication between endpoints is closed */
    public static int HAILO_STREAM_ABORT = 63; /*!< Stream recv/send was aborted */
    public static int HAILO_DRIVER_NOT_INSTALLED = 64; /*!< Driver is not installed/running on the system. */
    public static int HAILO_PCIE_DRIVER_NOT_INSTALLED = HAILO_DRIVER_NOT_INSTALLED; /*!< Pcie driver is not installed */
    public static int HAILO_NOT_AVAILABLE = 65; /*!< Component is not available */
    public static int HAILO_TRAFFIC_CONTROL_FAILURE = 66; /*!< Traffic control failure */
    public static int HAILO_INVALID_SECOND_STAGE = 67; /*!< Second stage bin is invalid */
    public static int HAILO_INVALID_PIPELINE = 68; /*!< Pipeline is invalid */
    public static int HAILO_NETWORK_GROUP_NOT_ACTIVATED = 69; /*!< Network group is not activated */
    public static int HAILO_VSTREAM_PIPELINE_NOT_ACTIVATED = 70; /*!< VStream pipeline is not activated */
    public static int HAILO_OUT_OF_FW_MEMORY = 71; /*!< Cannot allocate more memory at fw */
    public static int HAILO_STREAM_NOT_ACTIVATED = 72; /*!< Stream is not activated */
    public static int HAILO_DEVICE_IN_USE = 73; /*!< The device is already in use */
    public static int HAILO_OUT_OF_PHYSICAL_DEVICES = 74; /*!< There are not enough physical devices */
    public static int HAILO_INVALID_DEVICE_ARCHITECTURE = 75; /*!< Invalid device architecture */
    public static int HAILO_INVALID_DRIVER_VERSION = 76; /*!< Invalid driver version */
    public static int HAILO_RPC_FAILED = 77; /*!< RPC failed */
    public static int HAILO_INVALID_SERVICE_VERSION = 78; /*!< Invalid service version */
    public static int HAILO_NOT_SUPPORTED = 79; /*!< Not supported operation */
    public static int HAILO_NMS_BURST_INVALID_DATA = 80; /*!< Invalid data in NMS burst */
    public static int HAILO_OUT_OF_HOST_CMA_MEMORY = 81; /*!< Cannot allocate more CMA memory at host */
    public static int HAILO_QUEUE_IS_FULL = 82; /*!< Cannot push more items into the queue */
    public static int HAILO_DMA_MAPPING_ALREADY_EXISTS = 83; /*!< DMA mapping already exists */
    public static int HAILO_CANT_MEET_BUFFER_REQUIREMENTS = 84; /*!< can't meet buffer requirements */
    public static int HAILO_DRIVER_INVALID_RESPONSE = 85; /*!< Driver returned invalid response. Make sure the driver version is the same as libhailort  */
    public static int HAILO_DRIVER_INVALID_IOCTL = 86; /*!< Driver cannot handle ioctl. Can happen on libhailort vs driver version mismatch or when ioctl function is not supported */
    public static int HAILO_DRIVER_TIMEOUT = 87; /*!< Driver operation returned a timeout. Device reset may be required. */
    public static int HAILO_DRIVER_INTERRUPTED = 88; /*!< Driver operation interrupted by system request (i.e can happen on application exit) */
    public static int HAILO_CONNECTION_REFUSED = 89; /*!< Connection was refused by other side */
    public static int HAILO_DRIVER_WAIT_CANCELED = 90; /*!< Driver operation was canceled */
    public static int HAILO_HEF_FILE_CORRUPTED = 91; /*!< HEF file is corrupted */
    public static int HAILO_HEF_NOT_SUPPORTED = 92; /*!< HEF file is not supported. Make sure the DFC version is compatible. */
    public static int HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE = 93; /*!< HEF file is not compatible with device. */
    public static int HAILO_INVALID_HEF_USE = 94; /*!< Invalid HEF use (i.e. when using HEF from a file path without first copying it's content to a mapped buffer while shared_weights is enabled) */
    public static int HAILO_OPERATION_ABORTED = 95; /*!< Operation was aborted */
    public static int HAILO_DEVICE_NOT_CONNECTED = 96; /*!< Device is not connected */
    public static int HAILO_DEVICE_TEMPORARILY_UNAVAILABLE = 97; /*!< Device is temporarily unavailable, try again later */


    public static final int INT_MAX = 2147483647;
    public static final long UINT32_MAX = 0xFFFFFFFFL;
}
