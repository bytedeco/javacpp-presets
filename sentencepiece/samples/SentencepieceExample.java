import java.nio.ByteBuffer;
import org.bytedeco.javacpp.*;
import org.bytedeco.sentencepiece.*;

public final class SentencepieceExample {

    public static void main(String[] args) {

        SentencePieceProcessor processor = new SentencePieceProcessor();
        Status status = processor.Load(args[0]);
        if (!status.ok()) {
            throw new RuntimeException(status.ToString().getString());
        }

        IntVector ids = new IntVector();
        processor.Encode("hello world!", ids);

        for (int id : ids.get()) {
            System.out.print(id + " ");
        }
    }

}
