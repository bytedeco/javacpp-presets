JavaCPP Presets for helloworld
==============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * helloworld 2.0  https://github.com/bytedeco/helloworld

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/helloworld/apidocs/


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
    <version>1.5.2-SNAPSHOT</version>
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
            <version>2.0-1.5.2-SNAPSHOT</version>
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
import org.bytedeco.helloworld.PersonTypePtr;
import org.bytedeco.javacpp.BytePointer;
import static org.bytedeco.helloworld.global.helloworld.*;

public class HelloWorldTest {

    public static void main(String[] args) throws UnsupportedEncodingException {
        {
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

            BytePointer bp1 = getUtf8String();
            System.out.printf("J UTF-8 String = %s%n", bp1.getString());

            {
                // Person
                PersonTypePtr person = getPerson();
                System.out.printf("J PersonType.firstname = %s%n", person.firstname().getString());
                System.out.printf("J PersonType.lastname  = %s%n", person.lastname().getString());

                // PersonPtr
                PersonTypePtr personPtr = getPersonPtr();
                System.out.printf("J PersonType.firstname = %s%n", personPtr.firstname().getString());
                System.out.printf("J PersonType.lastname  = %s%n", personPtr.lastname().getString());

                // PersonType
                PersonTypePtr personType = getPersonType();
                System.out.printf("J PersonType.firstname = %s%n", personType.firstname().getString());
                System.out.printf("J PersonType.lastname  = %s%n", personType.lastname().getString());

                // PersonTypePtr
                PersonTypePtr personTypePtr = getPersonTypePtr();
                System.out.printf("J PersonType.firstname = %s%n", personTypePtr.firstname().getString());
                System.out.printf("J PersonType.lastname  = %s%n", personTypePtr.lastname().getString());
            }
        }
        {
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
            byte[] tmp = ("Hello z\u00df\u6c34" + new String(Character.toChars(0x0001F34C)) + " byte array!").getBytes(Charset.forName("UTF-8"));
            byte[] array = new byte[tmp.length + 1];
            System.arraycopy(tmp, 0, array, 0, tmp.length);
            array[array.length - 1] = 0;
            printUtf8String(array);

            // ByteBuffer
            final byte[] bytes = ("Hello z\u00df\u6c34" + new String(Character.toChars(0x0001F34C)) + " ByteBuffer!").getBytes(Charset.forName("UTF-8"));
            ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length + 1);
            buf.put(bytes);
            buf.put((byte) 0);
            buf.rewind();
            printUtf8String(buf);

            // ByteBuffer
            BytePointer bp2 = new BytePointer("Hello z\u00df\u6c34" + new String(Character.toChars(0x1F34C)) + " BytePointer!", "UTF-8");
            printUtf8String(bp2);

            {
                // Person
                PersonTypePtr person = new PersonTypePtr();
                person.firstname(new BytePointer("John"));
                person.lastname(new BytePointer("Doe"));
                printPerson(person);

                // PersonPtr
                PersonTypePtr personPtr = new PersonTypePtr();
                personPtr.firstname(new BytePointer("John"));
                personPtr.lastname(new BytePointer("Doe"));
                printPersonPtr(personPtr);

                // PersonType
                PersonTypePtr personType = new PersonTypePtr();
                personType.firstname(new BytePointer("John"));
                personType.lastname(new BytePointer("Doe"));
                printPersonType(personType);

                // PersonTypePtr
                PersonTypePtr personTypePtr = new PersonTypePtr();
                personTypePtr.firstname(new BytePointer("John"));
                personTypePtr.lastname(new BytePointer("Doe"));
                printPersonTypePtr(personTypePtr);
            }
        }
    }
}
```

