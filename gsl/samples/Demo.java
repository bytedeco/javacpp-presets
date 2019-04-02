import org.bytedeco.javacpp.*;
import org.bytedeco.gsl.*;
import static org.bytedeco.gsl.global.gsl.*;

public class Demo {
    public static void main(String[] args) {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        gsl_rng_type T;
        gsl_rng r;

        int n = 10;
        double mu = 3.0;

        /* create a generator chosen by the 
           environment variable GSL_RNG_TYPE */

        gsl_rng_env_setup();

        T = gsl_rng_default();
        r = gsl_rng_alloc(T);

        /* print n random variates chosen from 
           the poisson distribution with mean 
           parameter mu */

        for (int i = 0; i < n; i++) {
            int k = gsl_ran_poisson(r, mu);
            System.out.printf(" %d", k);
        }

        System.out.println();
        gsl_rng_free(r);
        System.exit(0);
    }
}
