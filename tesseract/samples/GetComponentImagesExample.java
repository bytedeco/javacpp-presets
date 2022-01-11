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
 *     <li>An environment variable pointing to the dictionaries installed on the system
 *     TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00</li>
 * </ul>
 *
 * @author Arnaud Jeansen
 */
public class GetComponentImagesExample {
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
        api.SetImage(image);

        // Lookup all component images
        int[] blockIds = {};
        BOXA boxes = api.GetComponentImages(RIL_TEXTLINE, true, null, blockIds);

        for (int i = 0; i < boxes.n(); i++) {
            // For each image box, OCR within its area
            BOX box = boxes.box(i);
            api.SetRectangle(box.x(), box.y(), box.w(), box.h());
            outText = api.GetUTF8Text();
            String ocrResult = outText.getString();
            int conf = api.MeanTextConf();

            String boxInformation = String.format("Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s", i, box.x(), box.y(), box.w(), box.h(), conf, ocrResult);
            System.out.println(boxInformation);

            outText.deallocate();
        }

        // Destroy used object and release memory
        api.End();
        pixDestroy(image);
    }
}
