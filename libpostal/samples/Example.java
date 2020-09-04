import org.bytedeco.javacpp.*;
import org.bytedeco.libpostal.*;
import static org.bytedeco.libpostal.global.postal.*;

public class Example {
    public static void main(String[] args) throws Exception {
        String dataDir = args.length >= 1 ? new String(args[0]) : "data/";
        String libpostal_data = Loader.load(org.bytedeco.libpostal.libpostal_data.class);
        ProcessBuilder pb = new ProcessBuilder("bash", libpostal_data, "download", "all", dataDir);
        pb.inheritIO().start().waitFor();

        boolean setup1 = libpostal_setup_datadir(dataDir);
        boolean setup2 = libpostal_setup_parser_datadir(dataDir);
        boolean setup3 = libpostal_setup_language_classifier_datadir(dataDir);
        if (setup1 && setup2 && setup3) {
            libpostal_address_parser_options_t options = libpostal_get_address_parser_default_options();
            BytePointer address = new BytePointer("781 Franklin Ave Crown Heights Brooklyn NYC NY 11216 USA", "UTF-8");
            libpostal_address_parser_response_t response = libpostal_parse_address(address, options);
            long count = response.num_components();
            for (int i = 0; i < count; i++) {
                System.out.println(response.labels(i).getString() + " " + response.components(i).getString());
            }
            libpostal_normalize_options_t normalizeOptions = libpostal_get_default_options();
            SizeTPointer sizeTPointer = new SizeTPointer(0);
            address = new BytePointer("Quatre vingt douze Ave des Champs-Élysées", "UTF-8");
            PointerPointer result = libpostal_expand_address(address, normalizeOptions, sizeTPointer);
            long t_size = sizeTPointer.get(0);
            for (long i = 0; i < t_size; i++) {
                System.out.println(result.getString(i));
            }
            libpostal_teardown();
            libpostal_teardown_parser();
            libpostal_teardown_language_classifier();
            System.exit(0);
        } else {
            System.out.println("Cannot setup libpostal, check if the training data is available at the specified path!");
            System.exit(-1);
        }
    }
}
