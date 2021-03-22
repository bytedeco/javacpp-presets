package org.bytedeco.nvcodec.sample.util;

import org.jcodec.common.Codec;
import org.bytedeco.nvcodec.sample.exceptions.InvalidArgument;
import org.bytedeco.nvcodec.sample.exceptions.NvCodecException;
import org.bytedeco.nvcodec.nvencodeapi.GUID;

import static org.bytedeco.nvcodec.global.nvcuvid.cudaVideoCodec_enum.*;
import static org.bytedeco.nvcodec.global.nvencodeapi.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.cuda.global.cudart.CUDA_SUCCESS;
import static org.bytedeco.nvcodec.global.nvcuvid.*;
import static org.jcodec.common.Codec.*;

public class NvCodecUtil {
    public static void checkNvCodecApiCall(int result) throws NvCodecException {
        if (result != CUDA_SUCCESS) {
            throw new NvCodecException(result);
        }
    }

    public static boolean checkInputFile(String inFilePath) {
        try (FileInputStream inputFileStream = new FileInputStream(inFilePath)) {
            if (0 < inputFileStream.available()) {
                return true;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return false;
    }

    public static void validateResolution(int width, int height) throws InvalidArgument {
        if (width <= 0 || height <= 0) {
            StringBuilder sb = new StringBuilder();
            sb.append("Please specify positive non zero resolution as -s WxH. Current resolution is ").append(width).append("x").append(height).append("\n");

            throw new InvalidArgument(sb.toString());
        }
    }

    public final static Map<Codec, cudaVideoCodec_enum> codecs = new HashMap<Codec, cudaVideoCodec_enum>();

    static {
        codecs.put(H264, cudaVideoCodec_H264);
        codecs.put(MPEG2, cudaVideoCodec_MPEG2);
        codecs.put(MPEG4, cudaVideoCodec_MPEG4);
        codecs.put(VC1, cudaVideoCodec_VC1);
        codecs.put(JPEG, cudaVideoCodec_JPEG);
        codecs.put(VP8, cudaVideoCodec_VP8);
        codecs.put(VP9, cudaVideoCodec_VP9);
    }

    public static cudaVideoCodec_enum convertToNvCodec(Codec codec) throws NvCodecException {
        cudaVideoCodec_enum nvCodec = codecs.get(codec);

        if (nvCodec == null) {
            throw new NvCodecException(String.format("%s codec not supported on NvCodec. %n", codec.name()), -1);
        }

        return nvCodec;
    }

    public static _NVENCSTATUS getNVENCSTATUSByValue(int value) {
        _NVENCSTATUS[] statuses = _NVENCSTATUS.values();

        for (_NVENCSTATUS status : statuses) {
            if (status.value == value) {
                return status;
            }
        }

        return null;
    }

    public static cudaVideoCodec_enum getVideoCodecByValue(int value) {
        cudaVideoCodec_enum[] codecs = cudaVideoCodec_enum.values();

        for (cudaVideoCodec_enum codec : codecs) {
            if (codec.value == value) {
                return codec;
            }
        }

        return null;
    }

    public static cudaVideoChromaFormat_enum getVideoChromaFormatByValue(int value) {
        cudaVideoChromaFormat_enum[] codecs = cudaVideoChromaFormat_enum.values();

        for (cudaVideoChromaFormat_enum codec : codecs) {
            if (codec.value == value) {
                return codec;
            }
        }

        return null;
    }

    public static boolean compareGUID(GUID guid1, GUID guid2) {
        if (guid1.Data1() == guid2.Data1() && guid1.Data2() == guid2.Data2() && guid1.Data3() == guid2.Data3()) {
            return true;
        }

        return false;
    }
}
