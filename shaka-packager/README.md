JavaCPP Presets for shaka-packager
=============================


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * shaka-packager 3.2.0  https://github.com/shaka-project/shaka-packager.git
 
Shaka Packager is a tool and a media packaging SDK for DASH and HLS packaging and encryption. It can prepare and package media content for online streaming.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/shaka-packager/apidocs/


Sample Usage
------------
Here is a simple example of the shaka-packager.

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries.
To run this sample code, after creating the `pom.xml` and `Example.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.libpostal</groupId>
    <artifactId>example</artifactId>
    <version>1.5.11-SNAPSHOT</version>
    <properties>
        <exec.mainClass>Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>shaka-packager-platform</artifactId>
            <version>3.2.0-1.5.8</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `Example.java` source file
```java
import org.bytedeco.shakapackager.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class Example {
  

    public static void main(String[] args) throws Exception {

        List<StreamDescriptor> streams = new ArrayList<>();

        TestParams ts = new TestParams();
        ts.dump_stream_info(true);

        StreamDescriptor st = new StreamDescriptor();
        st.input("test.mp4");
        st.stream_selector("video");
        st.output("output_video.mp4");
        streams.add(st);

        StreamDescriptor ste = new StreamDescriptor();
        ste.input("test.mp4");
        ste.stream_selector("audio");
        ste.output("output_audio.mp4");
        streams.add(ste);

        final StreamDescriptor rectsPointer = new StreamDescriptor(streams.size());
        for (int i=0; i<streams.size(); ++i)
            rectsPointer.position(i).put(streams.get(i));


        Packager packager = new Packager();
        System.out.println("Shaka Packager version : " + Packager.GetLibraryVersion());
        PackagingParams packaging_params = new PackagingParams();
        packaging_params.test_params(ts);
        packaging_params.chunking_params(new ChunkingParams().segment_duration_in_seconds(5.0));
        Status s = packager.Initialize(packaging_params,rectsPointer);
        s = packager.Run();
        System.out.println("test status : " + s.ToString());
    }
}

```
