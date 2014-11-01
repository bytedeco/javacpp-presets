package jjybdx4il.javacv.examples;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import org.bytedeco.javacpp.avcodec;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.FrameRecorder;
import org.junit.Test;

public class FFmpegTest extends Base {

    public static final int w = 640;
    public static final int h = 480;

    @Test
    public void test() throws FrameRecorder.Exception {
        File outFile = getTempFile(FFmpegTest.class.getName() + ".avi");
        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(outFile, w, h);

        recorder.setVideoCodec(13);
        recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);
        recorder.setFormat("mp4");
        recorder.setPixelFormat(0); //PIX_FMT_YUV420P
        recorder.setFrameRate(30000.0/1001.0);
        recorder.setVideoBitrate(10 * 1024 * 1024);

        recorder.start();
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = (Graphics2D) img.getGraphics();
        g.setColor(Color.WHITE);
        IplImage ipl = IplImage.createFrom(img);
        final int steps = 300;
        for (int i = 0; i <= steps; i++) {
            g.fillRect(0, 0, w * i / steps, h * i / steps);
            ipl.copyFrom(img);
            recorder.record(ipl);
        }
        recorder.stop();
    }
}
