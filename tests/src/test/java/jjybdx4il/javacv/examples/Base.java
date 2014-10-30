package jjybdx4il.javacv.examples;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacv.CanvasFrame;
import static org.junit.Assert.*;

/**
 * Derived from
 * <a href="https://github.com/bytedeco/javacv/blob/master/samples/BlobDemo.java">BlobDema.java</a>.
 * Run in headless mode to avoid X11 output and associated delays.
 */
public class Base {

    private static final GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
    protected static final File tempDir = new File("target", "temp");

    protected static File getTempFile(String filename) {
        File f = new File(tempDir, filename);
        if (!f.getParentFile().exists()) {
            assertTrue(f.getParentFile().mkdirs());
        }
        if (f.exists()) {
            assertTrue(f.delete());
        }
        return f;
    }

    protected static File createTestPNGFile(String filename) throws IOException {
        BufferedImage img = new BufferedImage(3, 3, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = (Graphics2D) img.getGraphics();
        g.setColor(Color.BLUE);
        g.fillRect(0, 0, img.getWidth(), img.getHeight());
        g.setColor(Color.RED);
        g.fillRect(1, 1, 1, 1);
        File outFile = getTempFile(filename);
        ImageIO.write(img, "png", outFile);
        return outFile;
    }

    // Versions with 2, 3, and 4 parms respectively
    public static void ShowImage(opencv_core.IplImage image, String caption) {
        opencv_core.CvMat mat = image.asCvMat();
        int width = mat.cols();
        if (width < 1) {
            width = 1;
        }
        int height = mat.rows();
        if (height < 1) {
            height = 1;
        }
        double aspect = 1.0 * width / height;
        if (height < 128) {
            height = 128;
            width = (int) (height * aspect);
        }
        if (width < 128) {
            width = 128;
        }
        height = (int) (width / aspect);
        ShowImage(image, caption, width, height);
    }

    public static void ShowImage(opencv_core.IplImage image, String caption, int size) {
        if (size < 128) {
            size = 128;
        }
        opencv_core.CvMat mat = image.asCvMat();
        int width = mat.cols();
        if (width < 1) {
            width = 1;
        }
        int height = mat.rows();
        if (height < 1) {
            height = 1;
        }
        double aspect = 1.0 * width / height;
        if (height != size) {
            height = size;
            width = (int) (height * aspect);
        }
        if (width != size) {
            width = size;
        }
        height = (int) (width / aspect);
        ShowImage(image, caption, width, height);
    }

    public static void ShowImage(opencv_core.IplImage image, String caption, int width, int height) {
        if (ge.isHeadlessInstance()) {
            return;
        }

        CanvasFrame canvas = new CanvasFrame(caption, 1);   // gamma=1
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        canvas.setCanvasSize(width, height);
        canvas.showImage(image);

        try {
            Thread.sleep(1000L);
        } catch (InterruptedException ex) {
            throw new IllegalStateException(ex);
        }
    }

    public static void Highlight(opencv_core.IplImage image, int[] inVec) {
        Highlight(image, inVec[0], inVec[1], inVec[2], inVec[3], 1);
    }

    public static void Highlight(opencv_core.IplImage image, int[] inVec, int Thick) {
        Highlight(image, inVec[0], inVec[1], inVec[2], inVec[3], Thick);
    }

    public static void Highlight(opencv_core.IplImage image, int xMin, int yMin, int xMax, int yMax) {
        Highlight(image, xMin, yMin, xMax, yMax, 1);
    }

    public static void Highlight(opencv_core.IplImage image, int xMin, int yMin, int xMax, int yMax, int Thick) {
        opencv_core.CvPoint pt1 = cvPoint(xMin, yMin);
        opencv_core.CvPoint pt2 = cvPoint(xMax, yMax);
        opencv_core.CvScalar color = cvScalar(255, 0, 0, 0);       // blue [green] [red]
        cvRectangle(image, pt1, pt2, color, Thick, 4, 0);
    }

    public static void PrintGrayImage(opencv_core.IplImage image, String caption) {
        int size = 512; // impractical to print anything larger
        opencv_core.CvMat mat = image.asCvMat();
        int cols = mat.cols();
        if (cols < 1) {
            cols = 1;
        }
        int rows = mat.rows();
        if (rows < 1) {
            rows = 1;
        }
        double aspect = 1.0 * cols / rows;
        if (rows > size) {
            rows = size;
            cols = (int) (rows * aspect);
        }
        if (cols > size) {
            cols = size;
        }
        rows = (int) (cols / aspect);
        PrintGrayImage(image, caption, 0, cols, 0, rows);
    }

    public static void PrintGrayImage(opencv_core.IplImage image, String caption, int MinX, int MaxX, int MinY, int MaxY) {
        int size = 512; // impractical to print anything larger
        opencv_core.CvMat mat = image.asCvMat();
        int cols = mat.cols();
        if (cols < 1) {
            cols = 1;
        }
        int rows = mat.rows();
        if (rows < 1) {
            rows = 1;
        }

        if (MinX < 0) {
            MinX = 0;
        }
        if (MinX > cols) {
            MinX = cols;
        }
        if (MaxX < 0) {
            MaxX = 0;
        }
        if (MaxX > cols) {
            MaxX = cols;
        }
        if (MinY < 0) {
            MinY = 0;
        }
        if (MinY > rows) {
            MinY = rows;
        }
        if (MaxY < 0) {
            MaxY = 0;
        }
        if (MaxY > rows) {
            MaxY = rows;
        }

        System.out.println("\n" + caption);
        System.out.print("   +");
        for (int icol = MinX; icol < MaxX; icol++) {
            System.out.print("-");
        }
        System.out.println("+");

        for (int irow = MinY; irow < MaxY; irow++) {
            if (irow < 10) {
                System.out.print(" ");
            }
            if (irow < 100) {
                System.out.print(" ");
            }
            System.out.print(irow);
            System.out.print("|");
            for (int icol = MinX; icol < MaxX; icol++) {
                int val = (int) mat.get(irow, icol);
                String C = " ";
                if (val == 0) {
                    C = "*";
                }
                System.out.print(C);
            }
            System.out.println("|");
        }
        System.out.print("   +");
        for (int icol = MinX; icol < MaxX; icol++) {
            System.out.print("-");
        }
        System.out.println("+");
    }

    public static void PrintImageProperties(opencv_core.IplImage image) {
        opencv_core.CvMat mat = image.asCvMat();
        int cols = mat.cols();
        int rows = mat.rows();
        int depth = mat.depth();
        System.out.println("ImageProperties for " + image + " : cols=" + cols + " rows=" + rows + " depth=" + depth);
    }

    public static float BinaryHistogram(opencv_core.IplImage image) {
        opencv_core.CvScalar Sum = cvSum(image);
        float WhitePixels = (float) (Sum.getVal(0) / 255);
        opencv_core.CvMat mat = image.asCvMat();
        float TotalPixels = mat.cols() * mat.rows();
        //float BlackPixels = TotalPixels - WhitePixels;
        return WhitePixels / TotalPixels;
    }

    // Counterclockwise small angle rotation by skewing - Does not stretch border pixels
    public static opencv_core.IplImage SkewGrayImage(opencv_core.IplImage Src, double angle) // angle is in radians
    {
        //double radians = - Math.PI * angle / 360.0;   // Half because skew is horizontal and vertical
        double sin = -Math.sin(angle);
        double AbsSin = Math.abs(sin);

        int nChannels = Src.nChannels();
        if (nChannels != 1) {
            System.out.println("ERROR: SkewGrayImage: Require 1 channel: nChannels=" + nChannels);
            System.exit(1);
        }

        opencv_core.CvMat SrcMat = Src.asCvMat();
        int SrcCols = SrcMat.cols();
        int SrcRows = SrcMat.rows();

        double WidthSkew = AbsSin * SrcRows;
        double HeightSkew = AbsSin * SrcCols;

        int DstCols = (int) (SrcCols + WidthSkew);
        int DstRows = (int) (SrcRows + HeightSkew);

        opencv_core.CvMat DstMat = cvCreateMat(DstRows, DstCols, CV_8UC1);  // Type matches IPL_DEPTH_8U
        cvSetZero(DstMat);
        cvNot(DstMat, DstMat);

        for (int irow = 0; irow < DstRows; irow++) {
            int dcol = (int) (WidthSkew * irow / SrcRows);
            for (int icol = 0; icol < DstCols; icol++) {
                int drow = (int) (HeightSkew - HeightSkew * icol / SrcCols);
                int jrow = irow - drow;
                int jcol = icol - dcol;
                if (jrow < 0 || jcol < 0 || jrow >= SrcRows || jcol >= SrcCols) {
                    DstMat.put(irow, icol, 255);
                } else {
                    DstMat.put(irow, icol, (int) SrcMat.get(jrow, jcol));
                }
            }
        }

        opencv_core.IplImage Dst = cvCreateImage(cvSize(DstCols, DstRows), IPL_DEPTH_8U, 1);
        Dst = DstMat.asIplImage();
        return Dst;
    }

    public static opencv_core.IplImage TransposeImage(opencv_core.IplImage SrcImage) // angle is in radians
    {
        opencv_core.CvMat mat = SrcImage.asCvMat();
        int cols = mat.cols();
        int rows = mat.rows();
        opencv_core.IplImage DstImage = cvCreateImage(cvSize(rows, cols), IPL_DEPTH_8U, 1);
        cvTranspose(SrcImage, DstImage);
        cvFlip(DstImage, DstImage, 1);
        return DstImage;
    }

    public Base() {
    }

}
