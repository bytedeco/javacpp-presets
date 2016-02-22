package com.github.jjYBdx4IL.maven.examples.javacpp;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import org.bytedeco.javacpp.opencv_stitching.Stitcher;
import static org.junit.Assert.*;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author jjYBdx4IL
 */
public class StitchingTest {

    private static final Logger log = LoggerFactory.getLogger(StitchingTest.class);
    private static final File WORKDIR = new File(System.getProperty("basedir"), "target");
    public static final String EXAMPLE_IMG_RES_LOC
            = "/org/openimaj/image/contour/aestheticode/aestheticode.jpg";

    @BeforeClass
    public static void beforeClass() {
        File libDir = new File(System.getProperty("load.libs.from.dir"));
        log.info("loading all shared libs from " + libDir.getAbsolutePath());
        log.info("java.library.path=" + System.getProperty("java.library.path"));

        List<String> toLoad = new ArrayList<String>();
        for (File f : libDir.listFiles()) {
            if (!f.getName().startsWith("lib")) {
                continue;
            }
            toLoad.add(f.getAbsolutePath());
        }

        boolean somethingDone;
        do {
            somethingDone = false;
            for (String f : toLoad) {
                try {
                    System.load(f);
                    log.info("loaded " + f);
                    somethingDone = true;
                    toLoad.remove(f);
                    break;
                } catch (UnsatisfiedLinkError ex) {
                }
            }
        } while (somethingDone);

        for (String f : toLoad) {
            log.warn("failed to load " + f);
        }
    }

    public static BufferedImage crop(BufferedImage img, int x, int y, int w, int h) {
        BufferedImage imgPart = new BufferedImage(w, h, img.getType());
        Graphics2D gr = imgPart.createGraphics();
        gr.drawImage(img, 0, 0, w, h, x, y, x + w, y + h, null);
        gr.dispose();
        return imgPart;
    }

    @Test
    public void testStitching() throws URISyntaxException, IOException {
        assertEquals("slf4j", System.getProperty("org.bytedeco.javacpp.logger"));

        File imgInputFile = new File(getClass().getResource(EXAMPLE_IMG_RES_LOC).toURI());

        // split input image
        BufferedImage image = ImageIO.read(imgInputFile);
        final int w = image.getWidth();
        final int h = image.getHeight();
        final int overlap = w/20;
        File outFile1 = new File(WORKDIR, "splitOutput1.jpg");
        ImageIO.write(crop(image, 0, 0, w/2+overlap, h), "jpg", outFile1);
        File outFile2 = new File(WORKDIR, "splitOutput2.jpg");
        ImageIO.write(crop(image, w/2-overlap, 0, w/2+overlap, h), "jpg", outFile2);

        MatVector imgs = new MatVector();

        Mat img = imread(outFile1.getAbsolutePath());
        assertFalse(img.empty());
        log.info(img.toString());
        imgs.resize(imgs.size() + 1);
        imgs.put(imgs.size() - 1, img);

        img = imread(outFile2.getAbsolutePath());
        assertFalse(img.empty());
        log.info(img.toString());
        imgs.resize(imgs.size() + 1);
        imgs.put(imgs.size() - 1, img);

        Mat pano = new Mat();
        Stitcher stitcher = Stitcher.createDefault(false);
        int status = stitcher.stitch(imgs, pano);

        assertEquals("stitching images", Stitcher.OK, status);

        File resultImgFile = new File(WORKDIR, "stitched.jpg");
        assertTrue(imwrite(resultImgFile.getAbsolutePath(), pano));
        log.info("output written to " + resultImgFile.getAbsolutePath());

        image = ImageIO.read(resultImgFile);
        assertEquals(w, image.getWidth(), 5d);
        assertEquals(h, image.getHeight(), 5d);
    }

}
