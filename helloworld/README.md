JavaCPP Presets for helloworld
==============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * helloworld 1.0  https://github.com/bytedeco/helloworld

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.



Sample Usage
------------
Here is a simple example of helloworld ported to Java from this C source file:

 * https://github.com/bytedeco/helloworld/#example

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `HelloWorldTest.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.helloworld</groupId>
    <artifactId>helloworld-test</artifactId>
    <version>1.5.1-SNAPSHOT</version>
    <properties>
        <exec.mainClass>HelloWorldTest</exec.mainClass>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>helloworld-platform</artifactId>
            <version>1.0-1.5.1-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `HelloWorldTest.java` source file
```java
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import static org.bytedeco.helloworld.global.helloworld.*;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.CharPointer;

public class HelloWorldTest {

    public static void main(String[] args) throws UnsupportedEncodingException {
        System.out.println("READ -------------------------------------------------");

        boolean b = getBool();
        System.out.printf("J boolean = %b%n", b);

        byte by = getByte();
        System.out.printf("J byte = %d%n", by);

        char c = getChar();
        System.out.printf("J char = \\u%04X%n", (int) c);

        short s = getShort();
        System.out.printf("J short = %d%n", s);

        int i = getInt();
        System.out.printf("J int = %d%n", i);

        long l = getLong();
        System.out.printf("J long = %d%n", l);

        BytePointer bp1 = getAsciiString();
        System.out.printf("J AsciiString = %s%n", bp1.getString());

        CharPointer cp1 = getUnicodeString();
        System.out.printf("J UnicodeString = %s%n", cp1.getString());

        System.out.println("WRITE ------------------------------------------------");
        printBool(true);
        printBool(false);
        printByte(Byte.MIN_VALUE);
        printByte(Byte.MAX_VALUE);
        printChar(Character.MIN_VALUE);
        printChar(Character.MAX_VALUE);
        printShort(Short.MIN_VALUE);
        printShort(Short.MAX_VALUE);
        printInt(Integer.MIN_VALUE);
        printInt(Integer.MAX_VALUE);
        printLong(Long.MIN_VALUE);
        printLong(Long.MAX_VALUE);

        // byte array
        byte[] tmp = "Hello byte array!".getBytes(Charset.forName("US-ASCII"));
        byte[] array = new byte[tmp.length + 1];
        System.arraycopy(tmp, 0, array, 0, tmp.length);
        array[array.length - 1] = 0;
        printAsciiString(array);

        // ByteBuffer
        final byte[] bytes = "Hello ByteBuffer!".getBytes(Charset.forName("US-ASCII"));
        ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length + 1);
        buf.put(bytes);
        buf.put((byte) 0);
        buf.rewind();
        printAsciiString(buf);

        // ByteBuffer
        BytePointer bp2 = new BytePointer("Hello BytePointer!", "US-ASCII");
        printAsciiString(bp2);

        // CharPointer
        String str = "Hello CharPointer!";
        CharPointer cp2 = new CharPointer(str.length() + 1);
        cp2.putString(str);
        printUnicodeString(cp2);
    }
}
```

