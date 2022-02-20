import java.io.File;
import java.net.URL;
import org.bytedeco.javacpp.*;
import org.bytedeco.leptonica.*;
import org.bytedeco.tesseract.*;
import static org.bytedeco.leptonica.global.lept.*;
import static org.bytedeco.tesseract.global.tesseract.*;

/**
 * To run this program, you need to configure:
 * <ul>
 * <li>An environment variable pointing to the dictionaries installed on the system
 * TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00</li>
 * <li>An environment variable to tweak the Locale
 * LC_ALL=C</li>
 * </ul>
 *
 * @author Arnaud Jeansen
 */
public class OrientationAndScriptDetectionExample {
    public static void main(String[] args) throws Exception {
        BytePointer outText;

        TessBaseAPI api = new TessBaseAPI();
        // Initialize tesseract-ocr with English, initializing tessdata path with the standard ENV variable
        if (api.Init(System.getenv("TESSDATA_PREFIX") + "/tessdata", "eng") != 0) {
            System.err.println("Could not initialize tesseract.");
            System.exit(1);
        }

        // Open input image with leptonica library
        URL url = new URL("https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Computer_modern_sample.svg/1920px-Computer_modern_sample.svg.png");
        File file = Loader.cacheResource(url);
        PIX image = pixRead(file.getAbsolutePath());
        api.SetPageSegMode(PSM_AUTO_OSD);
        api.SetImage(image);
        ETEXT_DESC reco = TessMonitorCreate();
        api.Recognize(reco);

        PageIterator iterator = api.AnalyseLayout();
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
