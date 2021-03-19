package org.nvcodec.sample.util;

import org.nvcodec.sample.exceptions.InvalidArgument;
import org.nvcodec.sample.exceptions.NvCodecException;
import org.bytedeco.nvcodec.nvencodeapi.GUID;

import static org.bytedeco.nvcodec.global.nvencodeapi.*;

import java.io.FileInputStream;
import java.io.IOException;

import static org.bytedeco.cuda.global.cudart.CUDA_SUCCESS;
import static org.bytedeco.nvcodec.global.nvcuvid.*;

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
