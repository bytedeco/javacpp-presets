import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Verifies that every generated fluent Options setter is a native JavaCPP binding.
 * Run after {@code mvn generate-sources} from the pytorch module directory.
 */
public class BenchmarkNNOptionsFluentSetters {
    private static final Pattern CLASS_PATTERN = Pattern.compile("public class (\\w+) extends \\w+");
    private static final Pattern NATIVE_SETTER_PATTERN =
            Pattern.compile("public native(?: @\\w+(?:\\([^)]*\\))?)* \\w+Options \\w+\\([^;]* setter\\);");
    private static final Pattern JAVA_SETTER_PATTERN =
            Pattern.compile("public \\w+Options \\w+\\([^)]* setter[^)]*\\) \\{");

    public static void main(String[] args) throws IOException {
        Path sourceDirectory = args.length == 0
                ? Paths.get("src/gen/java/org/bytedeco/pytorch")
                : Paths.get(args[0]);
        int optionClasses = 0;
        int setters = 0;
        long start = System.nanoTime();

        try (DirectoryStream<Path> files = Files.newDirectoryStream(sourceDirectory, "*Options.java")) {
            for (Path file : files) {
                String source = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
                Matcher classMatcher = CLASS_PATTERN.matcher(source);
                if (!classMatcher.find()) {
                    throw new AssertionError("Missing Options class declaration in " + file);
                }
                optionClasses++;
                String className = classMatcher.group(1);
                if (JAVA_SETTER_PATTERN.matcher(source).find()) {
                    throw new AssertionError(file + " contains a Java fluent setter implementation");
                }
                Matcher setterMatcher = NATIVE_SETTER_PATTERN.matcher(source);
                while (setterMatcher.find()) {
                    setters++;
                }
            }
        }

        if (optionClasses == 0 || setters == 0) {
            throw new AssertionError("No native Options fluent setters found");
        }
        System.out.println("Options native fluent setter benchmark: " + setters + " setters in "
                + optionClasses + " classes, " + ((System.nanoTime() - start) / 1_000_000L) + " ms");
    }
}
