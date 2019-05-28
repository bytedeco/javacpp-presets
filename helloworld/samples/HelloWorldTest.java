import org.bytedeco.helloworld.global.helloworld;
import org.bytedeco.javacpp.BytePointer;

public class HelloWorldTest {

    public static void main(String[] args) {
        BytePointer pointer = helloworld.getHelloWorldMessage();
        System.out.println(pointer.getString());
    }
}
