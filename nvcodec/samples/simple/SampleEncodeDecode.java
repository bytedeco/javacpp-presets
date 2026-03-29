import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.nvcodec.nvcuvid.CUVIDDECODECAPS;

import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.nvcodec.global.nvcuvid.*;
import static org.bytedeco.nvcodec.global.nvencodeapi.*;

public class SampleEncodeDecode {
    public static void checkEncodeApiCall(String functionName, int result) {
        if (result != NV_ENC_SUCCESS) {
            throw new IllegalStateException(String.format("%s returned '%d'", functionName, result));
        }
    }

    public static void checkCudaApiCall(String functionName, int result) {
        if (result != CUDA_SUCCESS) {
            throw new IllegalStateException(String.format("%s returned '%d'", functionName, result));
        }
    }

    public static void main(String[] args) {
        int targetGpu = 0; // If you use NVIDIA GPU not '0', changing it.

        CUctx_st cuContext = new CUctx_st();

        checkCudaApiCall("cuInit", cuInit(0));
        checkCudaApiCall("cuCtxCreate", cuCtxCreate(cuContext, null, 0, targetGpu));
        try {
            // Check encoder max supported version
            try (IntPointer version = new IntPointer(1)) {
                checkEncodeApiCall("NvEncodeAPIGetMaxSupportedVersion", NvEncodeAPIGetMaxSupportedVersion(version));

                System.out.printf("Encoder Max Supported Version\t : %d \r\n", version.get());
            }

            // Query decoder capability 'H.264' codec
            try (CUVIDDECODECAPS decodeCaps = new CUVIDDECODECAPS()) {
                decodeCaps.eCodecType(cudaVideoCodec_H264);
                decodeCaps.eChromaFormat(cudaVideoChromaFormat_420);
                decodeCaps.nBitDepthMinus8(0);

                checkCudaApiCall("cuvidGetDecoderCaps", cuvidGetDecoderCaps(decodeCaps));

                System.out.printf("Decoder Capability H.264 Codec\t : %s \r\n", (decodeCaps.bIsSupported() != 0));
            }
        } finally {
            checkCudaApiCall("cuCtxDestroy", cuCtxDestroy(cuContext));
        }
    }
}
