JavaCPP Presets for SentencePiece
=================================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/sentencepiece/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/sentencepiece) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/sentencepiece.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![sentencepiece](https://github.com/bytedeco/javacpp-presets/workflows/sentencepiece/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Asentencepiece)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * SentencePiece 0.1.99  https://github.com/google/sentencepiece

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/sentencepiece/apidocs/


Sample Usage
------------
Here is a simple example of SentencePiece ported to Java from this C++ example:

 * https://github.com/google/sentencepiece/blob/v0.1.99/doc/api.md

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SentencePieceExample.java` source files below, simply execute on the command line:
```bash
$ wget https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs10000.model
$ mvn compile exec:java exec.args="en.wiki.bpe.vs10000.model"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.sentencepiece</groupId>
    <artifactId>sentencepiece-example</artifactId>
    <version>1.5.10</version>
    <properties>
        <exec.mainClass>SentencePieceExample</exec.mainClass>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>sentencepiece-platform</artifactId>
            <version>0.1.99-1.5.10</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SentencePieceExample.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.sentencepiece.*;

/**
 * To try encoding you can download an existing model, i.e.
 * wget https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs10000.model
 * mvn compile exec:java exec.args="en.wiki.bpe.vs10000.model"
 */
public final class SentencePieceExample {
    public static void main(String[] args) {
        SentencePieceProcessor processor = new SentencePieceProcessor();
        Status status = processor.Load(args[0]);
        if (!status.ok()) {
            throw new RuntimeException(status.ToString());
        }

        IntVector ids = new IntVector();
        processor.Encode("hello world!", ids);

        for (int id : ids.get()) {
            System.out.print(id + " ");
        }
        System.out.println();

        BytePointer text = new BytePointer("");
        processor.Decode(ids, text);
        System.out.println(text.getString());
    }
}
```
