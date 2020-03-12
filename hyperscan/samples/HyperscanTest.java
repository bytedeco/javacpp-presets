import org.bytedeco.hyperscan.global.hyperscan;
import org.bytedeco.hyperscan.hs_compile_error_t;
import org.bytedeco.hyperscan.hs_database_t;
import org.bytedeco.hyperscan.hs_scratch_t;
import org.bytedeco.hyperscan.match_event_handler;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.annotation.Cast;

import static org.bytedeco.hyperscan.global.hyperscan.HS_FLAG_SINGLEMATCH;
import static org.bytedeco.hyperscan.global.hyperscan.HS_MODE_BLOCK;

public class HyperscanTest {

    public static void main(String[] args) {
        Loader.load(hyperscan.class);

        String[] patterns = { "abc1", "asa", "dab" };
        hs_database_t database_t = null;
        match_event_handler matchEventHandler = null;
        hs_scratch_t scratchSpace = new hs_scratch_t();
        hs_compile_error_t compile_error_t;

        try(PointerPointer<hs_database_t> database_t_p = new PointerPointer<hs_database_t>(1);
            PointerPointer<hs_compile_error_t> compile_error_t_p = new PointerPointer<hs_compile_error_t>(1);
            IntPointer compileFlags = new IntPointer(HS_FLAG_SINGLEMATCH, HS_FLAG_SINGLEMATCH, HS_FLAG_SINGLEMATCH);
            IntPointer patternIds = new IntPointer(1, 1, 1);
            PointerPointer expressionsPointer = new PointerPointer<BytePointer>(patterns)
        ) {

            matchEventHandler = new match_event_handler() {
                @Override
                public int call(@Cast("unsigned int") int id,
                        @Cast("unsigned long long") long from,
                        @Cast("unsigned long long") long to,
                        @Cast("unsigned int") int flags, Pointer context) {
                    System.out.println(from + "-" + to);
                    System.out.println(id);
                    return 0;
                }
            };

            int result = hyperscan.hs_compile_multi(expressionsPointer, compileFlags, patternIds, 3, HS_MODE_BLOCK,
                    null, database_t_p, compile_error_t_p);

            database_t = new hs_database_t(database_t_p.get(0));
            compile_error_t = new hs_compile_error_t(compile_error_t_p.get(0));
            if (result != 0) {
                System.out.println(compile_error_t.message().getString());
                System.exit(1);
            }
            result = hyperscan.hs_alloc_scratch(database_t, scratchSpace);
            if (result != 0) {
                System.out.println("Error during scratch space allocation");
                System.exit(1);
            }

            String textToSearch = "-21dasaaadabcaaa";
            hyperscan.hs_scan(database_t, textToSearch, textToSearch.length(), 0, scratchSpace, matchEventHandler, expressionsPointer);

        } finally {
            hyperscan.hs_free_scratch(scratchSpace);
            if (database_t != null) {
                hyperscan.hs_free_database(database_t);
            }
            if (matchEventHandler != null) {
                matchEventHandler.close();
            }
        }
    }
}
