JavaCPP Presets for libpostal
=============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * libpostal 1.1-alpha  https://github.com/openvenues/libpostal
 
libpostal is a C library for parsing/normalizing street addresses around the world using statistical NLP and open data.
The goal of this project is to understand location-based strings in every language, everywhere.

Data Files
----------

libpostal needs to download a few gigabytes of data from S3. The basic files are on-disk representations of the data structures necessary to perform expansion.
For address parsing, since model training takes a few days, the libpostal team publishes the fully trained model to S3 and will update it automatically as new addresses get added to OSM, OpenAddresses, etc.
Same goes for the language classifier model. Data files are automatically downloaded when you run the build with enabled data download.
To check for and download any new data files, you can either run ```make```, or run:

```libpostal_data download all $YOUR_DATA_DIR```

Replace $YOUR_DATA_DIR with the path where the training data should be stored.

Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libpostal/apidocs/


Sample Usage
------------
Here is a simple example of the libpostal parser and normalization functionality.

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries.
To run this sample code, after creating the `pom.xml` and `Example.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="/PATH_TO_LIBPOSTAL_TRAINING_DATA_DIRECTORY"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.libpostal</groupId>
    <artifactId>example</artifactId>
    <version>1.5.3</version>
    <properties>
        <exec.mainClass>Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>libpostal-platform</artifactId>
            <version>1.1-alpha-1.5.3</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `Example.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.libpostal.*;
import static org.bytedeco.libpostal.global.postal.*;

public class Example {
    public static void main(String[] args) throws Exception {
        String dataDir = args.length >= 1 ? new String(args[0]) : "/PATH_TO_LIBPOSTAL_TRAINING_DATA_DIRECTORY";
        boolean setup1 = libpostal_setup_datadir(dataDir);
        boolean setup2 = libpostal_setup_parser_datadir(dataDir);
        boolean setup3 = libpostal_setup_language_classifier_datadir(dataDir);
        if (setup1 && setup2 && setup3) {
            libpostal_address_parser_options_t options = libpostal_get_address_parser_default_options();
            BytePointer address = new BytePointer("781 Franklin Ave Crown Heights Brooklyn NYC NY 11216 USA", "UTF-8");
            libpostal_address_parser_response_t response = libpostal_parse_address(address, options);
            long count = response.num_components();
            for (int i = 0; i < count; i++) {
                System.out.println(response.labels(i).getString() + " " + response.components(i).getString());
            }
            libpostal_normalize_options_t normalizeOptions = libpostal_get_default_options();
            SizeTPointer sizeTPointer = new SizeTPointer(0);
            address = new BytePointer("Quatre vingt douze Ave des Champs-Élysées", "UTF-8");
            PointerPointer result = libpostal_expand_address(address, normalizeOptions, sizeTPointer);
            long t_size = sizeTPointer.get(0);
            for (long i = 0; i < t_size; i++) {
                System.out.println(result.getString(i));
            }
            libpostal_teardown();
            libpostal_teardown_parser();
            libpostal_teardown_language_classifier();
            System.exit(0);
        } else {
            System.out.println("Cannot setup libpostal, check if the training data is available at the specified path!");
            System.exit(-1);
        }
    }
}
```
