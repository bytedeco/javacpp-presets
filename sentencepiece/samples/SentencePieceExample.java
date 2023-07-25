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
