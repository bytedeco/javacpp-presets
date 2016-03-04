
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

import javax.imageio.ImageIO;

import org.apache.commons.exec.CommandLine;
import org.apache.commons.exec.DefaultExecutor;
import static org.junit.Assert.assertEquals;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.assertEquals;

public class StitchingIT {

    private static final File WORKDIR = new File(System.getProperty("basedir"), "target");
    public static final String EXAMPLE_IMG_RES_LOC
            = "/org/openimaj/image/contour/aestheticode/aestheticode.jpg";
    private static File imgInputPart1;
    private static File imgInputPart2;
    private static int inputImageWidth;
    private static int inputImageHeight;

    /**
     * Split the sample image into two parts.
     *
     * @throws IOException
     * @throws URISyntaxException
     */
    @BeforeClass
    public static void beforeClass() throws IOException, URISyntaxException {
        BufferedImage image = ImageIO.read(StitchingIT.class.getResourceAsStream(EXAMPLE_IMG_RES_LOC));
        inputImageWidth = image.getWidth();
        inputImageHeight = image.getHeight();
        final int overlap = inputImageWidth / 20;
        imgInputPart1 = new File(WORKDIR, "splitOutput1.jpg");
        ImageIO.write(crop(image,
                0, 0, inputImageWidth / 2 + overlap, inputImageHeight), "jpg", imgInputPart1);
        imgInputPart2 = new File(WORKDIR, "splitOutput2.jpg");
        ImageIO.write(crop(image,
                inputImageWidth / 2 - overlap, 0, inputImageWidth / 2 + overlap, inputImageHeight),
                "jpg", imgInputPart2);
    }

    @Test
    public void testStitchingSample() throws IOException {
        File resultImgFile = new File(WORKDIR, "stitched.jpg");

        // run the stitching sample
        DefaultExecutor executor = new DefaultExecutor();
        executor.setWorkingDirectory(new File(System.getProperty("basedir"), "sample"));
        assertEquals(0, executor.execute(new CommandLine("mvn").addArguments(new String[]{
            "compile",
            "exec:java",
            "-Dexec.args=--output "+resultImgFile.getAbsolutePath() + " "
            + imgInputPart1.getAbsolutePath() + " " + imgInputPart2.getAbsolutePath()
        })));

        BufferedImage image = ImageIO.read(resultImgFile);
        assertEquals(inputImageWidth, image.getWidth(), 5d);
        assertEquals(inputImageHeight, image.getHeight(), 5d);
    }

    private static BufferedImage crop(BufferedImage img, int x, int y, int w, int h) {
        BufferedImage imgPart = new BufferedImage(w, h, img.getType());
        Graphics2D gr = imgPart.createGraphics();
        gr.drawImage(img, 0, 0, w, h, x, y, x + w, y + h, null);
        gr.dispose();
        return imgPart;
    }
}
