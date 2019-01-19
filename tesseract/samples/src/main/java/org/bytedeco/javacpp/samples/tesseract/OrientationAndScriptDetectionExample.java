
package org.bytedeco.javacpp.samples.tesseract;

import static org.bytedeco.javacpp.lept.*;
import static org.bytedeco.javacpp.tesseract.*;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.tesseract;

/**
 * To run this program, you need to configure:
 * <ul>
 * <li>An environment variable pointing to the dictionaries installed on the system
 * TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00</li>
 * <li>An environment variable to tweak the Locale
 * LC_ALL=C</li>
 * </ul>
 */
public class OrientationAndScriptDetectionExample {
    public static void main(String[] args) {
        BytePointer outText;

        TessBaseAPI api = new TessBaseAPI();
        // Initialize tesseract-ocr with English, intializing tessdata path with the standard ENV variable
        if (api.Init(System.getenv("TESSDATA_PREFIX") + "/tessdata", "eng") != 0) {
            System.err.println("Could not initialize tesseract.");
            System.exit(1);
        }

        // Open input image with leptonica library
        PIX image = pixRead("src/main/resources/org/bytedeco/javacpp/samples/tesseract/Wikipedia-Computer_modern_sample.png");
        api.SetPageSegMode(PSM_AUTO_OSD);
        api.SetImage(image);
        tesseract.ETEXT_DESC reco = TessMonitorCreate();
        api.Recognize(reco);

        tesseract.PageIterator iterator = api.AnalyseLayout();
        int[] orientation = new int[1];
        int[] writing_direction = new int[1];
        int[] textline_order = new int[1];
        float[] deskew_angle = new float[1];

        iterator.Orientation(orientation, writing_direction, textline_order, deskew_angle);
        String osdInformation = String.format("Orientation: %d;\nWritingDirection: %d\nTextlineOrder: %d\nDeskew angle: %.4f\n",
                orientation[0], writing_direction[0], textline_order[0], deskew_angle[0]);
        System.out.println(osdInformation);

        // Destroy used object and release memory
        api.End();
        pixDestroy(image);
    }
}
