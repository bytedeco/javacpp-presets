/* Start reading here */

import org.bytedeco.javacpp.*;
import static java.lang.Math.*;
import static org.bytedeco.fftw.global.fftw3.*;

public class Example {

    static final int NUM_POINTS = 64;


    /* Never mind this bit */

    static final int REAL = 0;
    static final int IMAG = 1;

    static void acquire_from_somewhere(DoublePointer signal) {
        /* Generate two sine waves of different frequencies and amplitudes. */

        double[] s = new double[(int)signal.capacity()];
        for (int i = 0; i < NUM_POINTS; i++) {
            double theta = (double)i / (double)NUM_POINTS * PI;

            s[2 * i + REAL] = 1.0 * cos(10.0 * theta) +
                              0.5 * cos(25.0 * theta);

            s[2 * i + IMAG] = 1.0 * sin(10.0 * theta) +
                              0.5 * sin(25.0 * theta);
        }
        signal.put(s);
    }

    static void do_something_with(DoublePointer result) {
        double[] r = new double[(int)result.capacity()];
        result.get(r);
        for (int i = 0; i < NUM_POINTS; i++) {
            double mag = sqrt(r[2 * i + REAL] * r[2 * i + REAL] +
                              r[2 * i + IMAG] * r[2 * i + IMAG]);

            System.out.println(mag);
        }
    }


    /* Resume reading here */

    public static void main(String args[]) {
        Loader.load(org.bytedeco.fftw.global.fftw3.class);

        DoublePointer signal = new DoublePointer(2 * NUM_POINTS);
        DoublePointer result = new DoublePointer(2 * NUM_POINTS);

        fftw_plan plan = fftw_plan_dft_1d(NUM_POINTS, signal, result,
                                          FFTW_FORWARD, (int)FFTW_ESTIMATE);

        acquire_from_somewhere(signal);
        fftw_execute(plan);
        do_something_with(result);

        fftw_destroy_plan(plan);
    }
}
