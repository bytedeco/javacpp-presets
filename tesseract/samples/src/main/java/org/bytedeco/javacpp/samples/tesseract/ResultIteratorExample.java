
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
public class ResultIteratorExample {
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
        api.SetImage(image);

        tesseract.ETEXT_DESC recoc = TessMonitorCreate();
        api.Recognize(recoc);

        tesseract.ResultIterator ri = api.GetIterator();
        int pageIteratorLevel = RIL_WORD;
        if (ri != null) {
            do {
                outText = ri.GetUTF8Text(pageIteratorLevel);
                float conf = ri.Confidence(pageIteratorLevel);
                int[] x1 = new int[1], y1 = new int[1], x2 = new int[1], y2 = new int[1];
                ri.BoundingBox(pageIteratorLevel, x1, y1, x2, y2);
                String riInformation = String.format("word: '%s';  \tconf: %.2f; BoundingBox: %d,%d,%d,%d;\n", outText.getString(), conf, x1[0], y1[0], x2[0], y2[0]);
                System.out.println(riInformation);

                outText.deallocate();
            } while (ri.Next(pageIteratorLevel));
        }

        // Destroy used object and release memory
        api.End();
        pixDestroy(image);
    }
}
