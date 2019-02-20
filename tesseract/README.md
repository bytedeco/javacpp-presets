JavaCPP Presets for Tesseract
=============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Tesseract 4.0.0  https://github.com/tesseract-ocr

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/tesseract/apidocs/


Sample Usage
------------
Here is a simple example of Tesseract ported to Java from this C++ source file and for this data:

 * https://github.com/tesseract-ocr/tesseract/wiki/APIExample
 * https://github.com/tesseract-ocr/tessdata

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `BasicExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="path/to/image/file"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.tesseract</groupId>
    <artifactId>BasicExample</artifactId>
    <version>1.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>BasicExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tesseract-platform</artifactId>
            <version>4.0.0-1.5-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `BasicExample.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.leptonica.*;
import org.bytedeco.tesseract.*;
import static org.bytedeco.leptonica.global.lept.*;
import static org.bytedeco.tesseract.global.tesseract.*;

public class BasicExample {
    public static void main(String[] args) {
        BytePointer outText;

        TessBaseAPI api = new TessBaseAPI();
        // Initialize tesseract-ocr with English, without specifying tessdata path
        if (api.Init(null, "eng") != 0) {
            System.err.println("Could not initialize tesseract.");
            System.exit(1);
        }

        // Open input image with leptonica library
        PIX image = pixRead(args.length > 0 ? args[0] : "/usr/src/tesseract/testing/phototest.tif");
        api.SetImage(image);
        // Get OCR result
        outText = api.GetUTF8Text();
        System.out.println("OCR output:\n" + outText.getString());

        // Destroy used object and release memory
        api.End();
        outText.deallocate();
        pixDestroy(image);
    }
}
```
