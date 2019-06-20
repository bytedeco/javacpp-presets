import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.CharPointer;
import static org.bytedeco.helloworld.global.helloworld.*;

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
