import java.io.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.ffmpeg.avcodec.*;
import org.bytedeco.ffmpeg.avformat.*;
import org.bytedeco.ffmpeg.avutil.*;
import org.bytedeco.ffmpeg.swscale.*;
import static org.bytedeco.ffmpeg.global.avcodec.*;
import static org.bytedeco.ffmpeg.global.avformat.*;
import static org.bytedeco.ffmpeg.global.avutil.*;
import static org.bytedeco.ffmpeg.global.swscale.*;

public class ReadFewFrame {
    /**
    * Write image data using simple image format ppm
    * @see https://en.wikipedia.org/wiki/Netpbm_format
    */
    static void save_frame(AVFrame pFrame, int width, int height, int f_idx) throws IOException {
        // Open file
        String szFilename = String.format("frame%d_.ppm", f_idx);
        OutputStream pFile = new FileOutputStream(szFilename);

        // Write header
        pFile.write(String.format("P6\n%d %d\n255\n", width, height).getBytes());

        // Write pixel data
        BytePointer data = pFrame.data(0);
        byte[] bytes = new byte[width * 3];
        int l = pFrame.linesize(0);
        for(int y = 0; y < height; y++) {
            data.position(y * l).get(bytes);
            pFile.write(bytes);
        }

        // Close file
        pFile.close();
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Read few frame and write to image");
        if (args.length < 1) {
            System.out.println("Missing input video file");
            System.exit(-1);
        }
        int ret = -1, i = 0, v_stream_idx = -1;
        String vf_path = args[0];
        AVFormatContext fmt_ctx = new AVFormatContext(null);
        AVPacket pkt = new AVPacket();

        ret = avformat_open_input(fmt_ctx, vf_path, null, null);
        if (ret < 0) {
            System.out.printf("Open video file %s failed \n", vf_path);
            throw new IllegalStateException();
        }

        // i dont know but without this function, sws_getContext does not work
        if (avformat_find_stream_info(fmt_ctx, (PointerPointer)null) < 0) {
            System.exit(-1);
        }

        av_dump_format(fmt_ctx, 0, args[0], 0);

        for (i = 0; i < fmt_ctx.nb_streams(); i++) {
            if (fmt_ctx.streams(i).codecpar().codec_type() == AVMEDIA_TYPE_VIDEO) {
                v_stream_idx = i;
                break;
            }
        }
        if (v_stream_idx == -1) {
            System.out.println("Cannot find video stream");
            throw new IllegalStateException();
        } else {
            System.out.printf("Video stream %d with resolution %dx%d\n", v_stream_idx,
                    fmt_ctx.streams(i).codecpar().width(),
                    fmt_ctx.streams(i).codecpar().height());
        }

        AVCodecContext codec_ctx = avcodec_alloc_context3(null);
        avcodec_parameters_to_context(codec_ctx, fmt_ctx.streams(v_stream_idx).codecpar());

        AVCodec codec = avcodec_find_decoder(codec_ctx.codec_id());
        if (codec == null) {
            System.out.println("Unsupported codec for video file");
            throw new IllegalStateException();
        }
        ret = avcodec_open2(codec_ctx, codec, (PointerPointer)null);
        if (ret < 0) {
            System.out.println("Can not open codec");
            throw new IllegalStateException();
        }

        AVFrame frm = av_frame_alloc();

        // Allocate an AVFrame structure
        AVFrame pFrameRGB = av_frame_alloc();
        if (pFrameRGB == null) {
            System.exit(-1);
        }

        // Determine required buffer size and allocate buffer
        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, codec_ctx.width(),
                        codec_ctx.height(), 1);
        BytePointer buffer = new BytePointer(av_malloc(numBytes));

        SwsContext sws_ctx = sws_getContext(
                codec_ctx.width(),
                codec_ctx.height(),
                codec_ctx.pix_fmt(),
                codec_ctx.width(),
                codec_ctx.height(),
                AV_PIX_FMT_RGB24,
                SWS_BILINEAR,
                null,
                null,
                (DoublePointer)null
            );

        if (sws_ctx == null) {
            System.out.println("Can not use sws");
            throw new IllegalStateException();
        }

        av_image_fill_arrays(pFrameRGB.data(), pFrameRGB.linesize(),
                buffer, AV_PIX_FMT_RGB24, codec_ctx.width(), codec_ctx.height(), 1);

        i = 0;
        int ret1 = -1, ret2 = -1, fi = -1;
        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt.stream_index() == v_stream_idx) {
                ret1 = avcodec_send_packet(codec_ctx, pkt);
                ret2 = avcodec_receive_frame(codec_ctx, frm);
                System.out.printf("ret1 %d ret2 %d\n", ret1, ret2);
                // avcodec_decode_video2(codec_ctx, frm, fi, pkt);
            }
            // if not check ret2, error occur [swscaler @ 0x1cb3c40] bad src image pointers
            // ret2 same as fi
            // if (fi && ++i <= 5) {
            if (ret2 >= 0 && ++i <= 5) {
                sws_scale(
                        sws_ctx,
                        frm.data(),
                        frm.linesize(),
                        0,
                        codec_ctx.height(),
                        pFrameRGB.data(),
                        pFrameRGB.linesize()
                    );

                save_frame(pFrameRGB, codec_ctx.width(), codec_ctx.height(), i);
                // save_frame(frm, codec_ctx.width(), codec_ctx.height(), i);
            }
            av_packet_unref(pkt);
            if (i >= 5) {
                break;
            }
        }

        av_frame_free(frm);

        avcodec_close(codec_ctx);
        avcodec_free_context(codec_ctx);

        avformat_close_input(fmt_ctx);
        System.out.println("Shutdown");
        System.exit(0);
    }
}
