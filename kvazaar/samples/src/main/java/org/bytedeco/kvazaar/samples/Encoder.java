package org.bytedeco.kvazaar.samples;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import static org.bytedeco.kvazaar.global.kvazaar.KVZ_CSP_400;
import static org.bytedeco.kvazaar.global.kvazaar.KVZ_CSP_420;
import static org.bytedeco.kvazaar.global.kvazaar.KVZ_INTERLACING_NONE;
import static org.bytedeco.kvazaar.global.kvazaar.kvz_api_get;
import org.bytedeco.kvazaar.kvz_api;
import org.bytedeco.kvazaar.kvz_config;
import org.bytedeco.kvazaar.kvz_data_chunk;
import org.bytedeco.kvazaar.kvz_encoder;
import org.bytedeco.kvazaar.kvz_frame_info;
import org.bytedeco.kvazaar.kvz_picture;
import org.bytedeco.kvazaar.samples.yuv.YuvFrame;
import org.bytedeco.kvazaar.samples.yuv.YuvReader;

public class Encoder {

    private final kvz_api api;
    private final kvz_config config;
    private final kvz_encoder enc;
    private final YuvReader input;
    private final Path output;

    public Encoder(YuvReader yuvReader, Path outputPath) {
        input = yuvReader;
        output = outputPath;
        api = kvz_api_get(8);
        config = api.config_alloc().call();
        api.config_init().call(config);
        config.width(input.getFrameWidth());
        config.height(input.getFrameHeight());
        config.framerate_num(input.getFrameRateNumerator());
        config.framerate_denom(input.getFrameRateDenominator());
        config.threads(0);
        config.owf(4);
        enc = api.encoder_open().call(config);
    }

    public void encodeFrames() throws IOException {
        int numFrames = 0;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        while (input.hasMoreFrames()) {
            YuvFrame frame = input.getNextFrame();
            kvz_picture pic_in = api.picture_alloc_csp().call(KVZ_CSP_420, input.getFrameWidth(), input.getFrameHeight());
            pic_in.interlacing(KVZ_INTERLACING_NONE);
            BytePointer inputFramePointer = new BytePointer(frame.getLumaPlane());
            inputFramePointer.position(0);
            pic_in.y(inputFramePointer);
            pic_in.pts(numFrames);
            pic_in.width(input.getFrameWidth());
            pic_in.height(input.getFrameHeight());
            pic_in.u(new BytePointer(frame.getCrPlane()));
            pic_in.v(new BytePointer(frame.getCbPlane()));
            PointerPointer<kvz_data_chunk> data_out = new PointerPointer<>(1);
            IntPointer len_out = new IntPointer(1);
            len_out.put(0);
            PointerPointer pic_out = null;
            PointerPointer src_out = null;
            kvz_frame_info info_out = new kvz_frame_info();
            api.encoder_encode().call(enc, pic_in, data_out, len_out, pic_out, src_out, info_out);
            kvz_data_chunk chunk = data_out.get(kvz_data_chunk.class);
            while (chunk != null) {
                ByteBuffer bb = chunk.asByteBuffer();
                byte[] chunkBytes = new byte[chunk.len()];
                bb.get(chunkBytes);
                baos.writeBytes(chunkBytes);
                chunk = chunk.next();
            }
            numFrames += 1;
        }
        IntPointer len_out = new IntPointer(1);
        do {
            PointerPointer<kvz_data_chunk> data_out = new PointerPointer<>(1);
            api.encoder_encode().call(enc, null, data_out, len_out, null, null, null);
            System.out.println("trailing len_out: " + len_out.get());
            kvz_data_chunk chunk = data_out.get(kvz_data_chunk.class);
            while (chunk != null) {
                ByteBuffer bb = chunk.asByteBuffer();
                byte[] chunkBytes = new byte[chunk.len()];
                bb.get(chunkBytes);
                baos.writeBytes(chunkBytes);
                chunk = chunk.next();
            }
        } while (len_out.get() != 0);
        System.out.println("processed frames: " + numFrames);
        Files.write(output, baos.toByteArray(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    public void cleanup() {
        api.encoder_close().call(enc);
        api.config_destroy().call(config);
    }

}
