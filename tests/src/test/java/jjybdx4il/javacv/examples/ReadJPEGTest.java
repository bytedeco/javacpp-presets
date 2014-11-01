package jjybdx4il.javacv.examples;

import java.io.File;
import java.io.IOException;
import org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_highgui.cvLoadImage;
import static org.junit.Assert.*;
import org.junit.Test;

/**
 * @see <a href="https://github.com/bytedeco/javacv/blob/master/samples/BlobDemo.java">BlobDema.java @
 * github</a>
 */
public class ReadJPEGTest extends Base {

    @Test
    public void test() throws IOException {
        File outFile = createTestPNGFile(ReadJPEGTest.class.getName() + ".png");

        IplImage RawImage = cvLoadImage(outFile.getAbsolutePath());
        assertNotNull(RawImage);
        ShowImage(RawImage, "RawImage", 512);
    }
}
