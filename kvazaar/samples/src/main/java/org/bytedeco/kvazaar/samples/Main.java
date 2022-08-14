package org.bytedeco.kvazaar.samples;

import java.io.File;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.bytedeco.kvazaar.samples.yuv.Y4mReader;
import org.bytedeco.kvazaar.samples.yuv.YuvReader;

public class Main {

    public static void main(String[] args) {

        try {
            // TODO: use commons-cli for proper arg parsing
            if (args.length < 2) {
                System.err.println("Usage: Encode inputfile outputfile");
                System.exit(1);
            }
            YuvReader input = getInputFileReader(args);
            Path outputPath = Path.of(args[1]);
            if ((input != null) && (outputPath != null)){
                Encoder encoder = new Encoder(input, outputPath);
                try {
                    encoder.encodeFrames();
                } finally {
                    encoder.cleanup();
                }
            }
        } catch (IOException ex) {
            System.out.println("Input / output problem: " + ex.getMessage());
            System.exit(1);
        }
    }

    private static YuvReader getInputFileReader(String[] args) throws IOException {
        String infile = args[0];
        if (infile.toLowerCase().endsWith(".y4m")) {
            System.out.println("Attempting to parse as Y4M (YUV4MPEG2)");
            File inputFile = new File(infile);
            SeekableByteChannel channel = FileChannel.open(inputFile.toPath(), StandardOpenOption.READ);
            YuvReader reader = new Y4mReader(channel);
            reader.readHeader();
            return reader;
        } else if (infile.toLowerCase().endsWith(".yuv")) {
            System.out.println("Attempting to parse as raw YUV");
            System.err.println("However, RAW YUV isn't implemented yet");
            System.exit(1);
        } else {
            System.err.println("Only Y4M and YUV input files are supported");
            System.exit(1);
        }
        return null;
    }
}
