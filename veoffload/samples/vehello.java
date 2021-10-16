import org.bytedeco.javacpp.*;
import org.bytedeco.veoffload.*;
import static org.bytedeco.veoffload.global.veo.*;

public class vehello {
    public static void main(String[] args) {
        /* Load "vehello" on VE node 0 */
        veo_proc_handle proc = veo_proc_create(0);
        long handle = veo_load_library(proc, "./libvehello.so");
        veo_thr_ctxt ctx = veo_context_open(proc);

        veo_args argp = veo_args_alloc();
        long id = veo_call_async_by_name(ctx, handle, "hello", argp);
        long retval[] = {0};
        veo_call_wait_result(ctx, id, retval);
        veo_args_free(argp);
        veo_context_close(ctx);
        veo_proc_destroy(proc);
        System.exit(0);
    }
}
