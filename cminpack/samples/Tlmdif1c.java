/*     driver for lmdif1 example. */

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;

import static java.lang.Math.*;
import static org.bytedeco.cminpack.global.cminpack.*;

public class Tlmdif1c {
  public static void main(String[] args)
  {
    Loader.load(org.bytedeco.cminpack.global.cminpack.class);

    int info, lwa, iwa[] = new int[3];
    double tol, fnorm, x[] = new double[3], fvec[] = new double[15], wa[] = new double[75];
    int m = 15;
    int n = 3;
    /* auxiliary data (e.g. measurements) */
    double[] y = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
                    3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};
    /* the following struct defines the data points */
    DoublePointer data = new DoublePointer(y);

    /* the following starting values provide a rough fit. */

    x[0] = 1.;
    x[1] = 1.;
    x[2] = 1.;

    lwa = 75;

    /* set tol to the square root of the machine precision.  unless high
       precision solutions are required, this is the recommended
       setting. */

    tol = sqrt(dpmpar(1));

    info = lmdif1(new Fcn(), data, m, n, x, fvec, tol, iwa, wa, lwa);

    fnorm = enorm(m, fvec);

    System.out.printf("      final l2 norm of the residuals%15.7g\n\n",(double)fnorm);
    System.out.printf("      exit parameter                %10d\n\n", info);
    System.out.printf("      final approximate solution\n\n %15.7g%15.7g%15.7g\n",
          (double)x[0], (double)x[1], (double)x[2]);
    System.exit(0);
  }

  public static class Fcn extends cminpack_func_mn {
    @Override public int call(Pointer p, int m, int n, DoublePointer x, DoublePointer fvec, int iflag)
    {
      /* function fcn for lmdif1 example */

      int i;
      double tmp1,tmp2,tmp3;
      DoublePointer y = new DoublePointer(p);
      assert m == 15 && n == 3;

      DoubleIndexer xIdx = DoubleIndexer.create(x.capacity(n));
      DoubleIndexer yIdx = DoubleIndexer.create(y.capacity(m));
      DoubleIndexer fvecIdx = DoubleIndexer.create(fvec.capacity(m));
      for (i = 0; i < 15; ++i)
        {
          tmp1 = i + 1;
          tmp2 = 15 - i;
          tmp3 = (i > 7) ? tmp2 : tmp1;
          fvecIdx.put(i, y.get(i) - (x.get(0) + tmp1/(x.get(1)*tmp2 + x.get(2)*tmp3)));
        }
      return 0;
    }
  }
}
