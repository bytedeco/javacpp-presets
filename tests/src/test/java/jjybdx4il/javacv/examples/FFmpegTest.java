package jjybdx4il.javacv.examples;

import java.awt.image.BufferedImage;
import java.io.File;
import org.bytedeco.javacpp.avcodec;
import org.bytedeco.javacpp.avutil;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.FrameRecorder;
import org.junit.Ignore;
import org.junit.Test;

public class FFmpegTest extends Base {

    public static final int w = 640;
    public static final int h = 480;

    /**
     *
     * Error with maven central linux-x86_64 jar on Ubuntu 12.04.05 LTS amd64:
     *
     * <pre>test(jjybdx4il.javacv.examples.FFmpegTest)  Time elapsed: 0.046 sec  &lt;&lt;&lt; ERROR!
     * java.lang.UnsatisfiedLinkError: no jniavutil in java.library.path
     * at java.lang.ClassLoader.loadLibrary(ClassLoader.java:1886)
     * at java.lang.Runtime.loadLibrary0(Runtime.java:849)
     * at java.lang.System.loadLibrary(System.java:1088)
     * at org.bytedeco.javacpp.Loader.loadLibrary(Loader.java:535)
     * at org.bytedeco.javacpp.Loader.load(Loader.java:410)
     * at org.bytedeco.javacpp.Loader.load(Loader.java:353)
     * at org.bytedeco.javacpp.avutil.&lt;clinit>(avutil.java:10)
     * at java.lang.Class.forName0(Native Method)
     * at java.lang.Class.forName(Class.java:274)
     * at org.bytedeco.javacpp.Loader.load(Loader.java:385)
     * at org.bytedeco.javacpp.Loader.load(Loader.java:353)
     * at org.bytedeco.javacpp.avformat.&lt;clinit>(avformat.java:13)
     * at org.bytedeco.javacv.FFmpegFrameRecorder.&lt;clinit>(FFmpegFrameRecorder.java:106)
     * at jjybdx4il.javacv.examples.FFmpegTest.test(FFmpegTest.java:20)
     * at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
     * at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:57)
     * at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
     * at java.lang.reflect.Method.invoke(Method.java:606)
     * at org.junit.runners.model.FrameworkMethod$1.runReflectiveCall(FrameworkMethod.java:47)
     * at org.junit.internal.runners.model.ReflectiveCallable.run(ReflectiveCallable.java:12)
     * at org.junit.runners.model.FrameworkMethod.invokeExplosively(FrameworkMethod.java:44)
     * at org.junit.internal.runners.statements.InvokeMethod.evaluate(InvokeMethod.java:17)
     * at org.junit.runners.ParentRunner.runLeaf(ParentRunner.java:271)
     * at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:70)
     * at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:50)
     * at org.junit.runners.ParentRunner$3.run(ParentRunner.java:238)
     * at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:63)
     * at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:236)
     * at org.junit.runners.ParentRunner.access$000(ParentRunner.java:53)
     * at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:229)
     * at org.junit.runners.ParentRunner.run(ParentRunner.java:309)
     * at org.apache.maven.surefire.junit4.JUnit4Provider.execute(JUnit4Provider.java:252)
     * at org.apache.maven.surefire.junit4.JUnit4Provider.executeTestSet(JUnit4Provider.java:141)
     * at org.apache.maven.surefire.junit4.JUnit4Provider.invoke(JUnit4Provider.java:112)
     * at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
     * at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:57)
     * at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
     * at java.lang.reflect.Method.invoke(Method.java:606)
     * at org.apache.maven.surefire.util.ReflectionUtils.invokeMethodWithArray(ReflectionUtils.java:189)
     * at org.apache.maven.surefire.booter.ProviderFactory$ProviderProxy.invoke(ProviderFactory.java:165)
     * at org.apache.maven.surefire.booter.ProviderFactory.invokeProvider(ProviderFactory.java:85)
     * at org.apache.maven.surefire.booter.ForkedBooter.runSuitesInProcess(ForkedBooter.java:115)
     * at org.apache.maven.surefire.booter.ForkedBooter.main(ForkedBooter.java:75)
     * Caused by: java.lang.UnsatisfiedLinkError: /tmp/javacpp341118125695795/libjniavutil.so: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.17' not found (required by /tmp/javacpp341118125695795/libavutil.so.52)
     * at java.lang.ClassLoader$NativeLibrary.load(Native Method)
     * at java.lang.ClassLoader.loadLibrary1(ClassLoader.java:1965)
     * at java.lang.ClassLoader.loadLibrary0(ClassLoader.java:1890)
     * at java.lang.ClassLoader.loadLibrary(ClassLoader.java:1851)
     * at java.lang.Runtime.load0(Runtime.java:795)
     * at java.lang.System.load(System.java:1062)
     * at org.bytedeco.javacpp.Loader.loadLibrary(Loader.java:524)
     * ... 39 more
     * </pre>
     *
     * @throws org.bytedeco.javacv.FrameRecorder.Exception
     */
    @Ignore
    @Test
    public void test() throws FrameRecorder.Exception {
        File outFile = getTempFile(FFmpegTest.class.getName() + ".avi");
        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(outFile, w, h);

        recorder.setVideoCodec(13);
        recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);
        recorder.setFormat("mp4");
        recorder.setPixelFormat(avutil.PIX_FMT_YUV420P9);
        recorder.setFrameRate(30);
        recorder.setVideoBitrate(10 * 1024 * 1024);

        recorder.start();
        BufferedImage img = new BufferedImage(BufferedImage.TYPE_3BYTE_BGR, w, h);
        IplImage ipl = IplImage.createFrom(img);
        recorder.record(ipl);
        recorder.stop();
    }
}
