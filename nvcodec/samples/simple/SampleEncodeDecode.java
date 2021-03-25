import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.nvcodec.nvcuvid.CUVIDDECODECAPS;

import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.nvcodec.global.nvcuvid.*;
import static org.bytedeco.nvcodec.global.nvencodeapi.*;

public class SampleEncodeDecode {
    public static void checkEncodeApiCall(String functionName, int result) {
        if (result != NV_ENC_SUCCESS) {
            System.err.printf("ERROR: %s returned '%d' \r\n", functionName, result);
            System.exit(-1);
        }
    }

    public static void checkCudaApiCall(String functionName, int result) {
        if (result != CUDA_SUCCESS) {
            System.err.printf("ERROR: %s returned '%d' \r\n", functionName, result);
            System.exit(-1);
        }
    }

    public static void main(String[] args) {
        int targetGpu = 0; // If you use NVIDIA GPU not '0', changing it.

        CUctx_st cuContext = new CUctx_st();

        checkCudaApiCall("cuInit", cuInit(0));
        checkCudaApiCall("cuCtxCreate", cuCtxCreate(cuContext, 0, targetGpu));
        // Check encoder max supported version
        {
            IntPointer version = new IntPointer(1);

            checkEncodeApiCall("NvEncodeAPIGetMaxSupportedVersion", NvEncodeAPIGetMaxSupportedVersion(version));

            System.out.printf("Encoder Max Supported Version\t : %d \r\n", version.get());
        }

        // Query decoder capability 'MPEG-1' codec
        {
            CUVIDDECODECAPS decodeCaps = new CUVIDDECODECAPS();
            decodeCaps.eCodecType(cudaVideoCodec_HEVC);
            decodeCaps.eChromaFormat(cudaVideoChromaFormat_420);
            decodeCaps.nBitDepthMinus8(2); // 10 bit

            checkCudaApiCall("cuvidGetDecoderCaps", cuvidGetDecoderCaps(decodeCaps));

            System.out.printf("Decoder Capability MPEG-1 Codec\t : %s \r\n", (decodeCaps.bIsSupported() != 0));
        }
    }
}
