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
public class IteratorOverClassifierChoicesExample {
    public static void main(String[] args) throws Exception {
        BytePointer outText;
        BytePointer choiceText;

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

        ETEXT_DESC recoc = TessMonitorCreate();
        api.Recognize(recoc);

        ResultIterator ri = api.GetIterator();
        int pageIteratorLevel = RIL_SYMBOL;
        if (ri != null) {
            do {
                outText = ri.GetUTF8Text(pageIteratorLevel);
                float conf = ri.Confidence(pageIteratorLevel);
                String symbolInformation = String.format("symbol: '%s';  \tconf: %.2f", outText.getString(), conf);
                System.out.println(symbolInformation);
                boolean indent = false;
                ChoiceIterator ci = TessResultIteratorGetChoiceIterator(ri);
                do {
                    if (indent)
                        System.out.print("\t\t");
                    System.out.print("\t-");
                    choiceText = ci.GetUTF8Text();
                    System.out.println(String.format("%s conf: %f", choiceText.getString(), ci.Confidence()));
                    indent = true;
                    choiceText.deallocate();
                } while (ci.Next());

                outText.deallocate();
            } while (ri.Next(pageIteratorLevel));
        }

        // Destroy used object and release memory
        api.End();
        pixDestroy(image);
    }
}
