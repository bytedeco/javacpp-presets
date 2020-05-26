JavaCPP Presets for Hyperscan
=============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Hyperscan 5.3.0  https://www.hyperscan.io

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/hyperscan/apidocs/


Sample Usage
------------
Here is a simple example of Hyperscan

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `HyperscanTest.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

Be aware that flag `-Djavacpp.platform` can be used to specify the target platform when running the maven command above, for example:
```bash
 $ mvn compile exec:java -Djavacpp.platform=linux-x86_64
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.hyperscan</groupId>
    <artifactId>HyperscanTest</artifactId>
    <version>1.5.4-SNAPSHOT</version>
    <properties>
        <exec.mainClass>HyperscanTest</exec.mainClass>
        <maven.compiler.target>1.7</maven.compiler.target>
        <maven.compiler.source>1.7</maven.compiler.source>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>hyperscan-platform</artifactId>
            <version>5.3.0-1.5.4-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `HyperscanTest.java` source file
```java
import org.bytedeco.hyperscan.global.hyperscan;
import org.bytedeco.hyperscan.hs_compile_error_t;
import org.bytedeco.hyperscan.hs_database_t;
import org.bytedeco.hyperscan.hs_scratch_t;
import org.bytedeco.hyperscan.match_event_handler;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.annotation.Cast;

import static org.bytedeco.hyperscan.global.hyperscan.HS_FLAG_SINGLEMATCH;
import static org.bytedeco.hyperscan.global.hyperscan.HS_MODE_BLOCK;

public class HyperscanTest {

    public static void main(String[] args) {
        Loader.load(hyperscan.class);

        String[] patterns = { "abc1", "asa", "dab" };
        hs_database_t database_t = null;
        match_event_handler matchEventHandler = null;
        hs_scratch_t scratchSpace = new hs_scratch_t();
        hs_compile_error_t compile_error_t;

        try(PointerPointer<hs_database_t> database_t_p = new PointerPointer<hs_database_t>(1);
            PointerPointer<hs_compile_error_t> compile_error_t_p = new PointerPointer<hs_compile_error_t>(1);
            IntPointer compileFlags = new IntPointer(HS_FLAG_SINGLEMATCH, HS_FLAG_SINGLEMATCH, HS_FLAG_SINGLEMATCH);
            IntPointer patternIds = new IntPointer(1, 1, 1);
            PointerPointer expressionsPointer = new PointerPointer<BytePointer>(patterns)
        ) {

            matchEventHandler = new match_event_handler() {
                @Override
                public int call(@Cast("unsigned int") int id,
                        @Cast("unsigned long long") long from,
                        @Cast("unsigned long long") long to,
                        @Cast("unsigned int") int flags, Pointer context) {
                    System.out.println(from + "-" + to);
                    System.out.println(id);
                    return 0;
                }
            };

            int result = hyperscan.hs_compile_multi(expressionsPointer, compileFlags, patternIds, 3, HS_MODE_BLOCK,
                    null, database_t_p, compile_error_t_p);

            database_t = new hs_database_t(database_t_p.get(0));
            compile_error_t = new hs_compile_error_t(compile_error_t_p.get(0));
            if (result != 0) {
                System.out.println(compile_error_t.message().getString());
                System.exit(1);
            }
            result = hyperscan.hs_alloc_scratch(database_t, scratchSpace);
            if (result != 0) {
                System.out.println("Error during scratch space allocation");
                System.exit(1);
            }

            String textToSearch = "-21dasaaadabcaaa";
            hyperscan.hs_scan(database_t, textToSearch, textToSearch.length(), 0, scratchSpace, matchEventHandler, expressionsPointer);

        } finally {
            hyperscan.hs_free_scratch(scratchSpace);
            if (database_t != null) {
                hyperscan.hs_free_database(database_t);
            }
            if (matchEventHandler != null) {
                matchEventHandler.close();
            }
        }
    }
}
```
