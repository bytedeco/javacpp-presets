/*
 * This example demonstrates the use of ISO_C_BINDING to call arpack (portability).
 *
 * Just use arpack as you would have normally done, but, use *[ae]upd_c instead of *[ae]upd_.
 * The main advantage is that compiler checks (arguments) are performed at build time.
 */

import org.bytedeco.javacpp.*;
import static org.bytedeco.arpackng.global.arpack.*;

/* test program to solve for the 9 largest eigenvalues of
 * A*x = lambda*x where A is the diagonal matrix
 * with entries 1000, 999, ... , 2, 1 on the diagonal.
 * */
public class IcbArpackC {
    static void dMatVec(double[] workd, int x, int y) {
      int i;
      for ( i = 0; i < 1000; ++i) {
        workd[y + i] = ((double) (i+1))*workd[x + i];
      }
    }

    static int ds() {
      int[] ido = {0};
      byte[] bmat = {'I'};
      int N = 1000;
      byte[] which = {'L', 'M'};
      int nev = 9;
      double tol = 0;
      double[] resid = new double[N];
      int ncv = 2*nev+1;
      double[] V = new double[ncv*N];
      int ldv = N;
      int[] iparam = new int[11];
      int[] ipntr = new int[14];
      double[] workd = new double[3*N];
      int rvec = 1;
      byte[] howmny = {'A'};
      double[] d = new double[nev+1];
      int[] select = new int[ncv];
      double[] z = new double[(N+1)*(nev+1)];
      int ldz = N+1;
      double sigma=0;
      int k;
      for (k=0; k < 3*N; ++k )
        workd[k] = 0;
      double[] workl = new double[3*(ncv*ncv) + 6*ncv];
      for (k=0; k < 3*(ncv*ncv) + 6*ncv; ++k )
        workl[k] = 0;
      int lworkl = 3*(ncv*ncv) + 6*ncv;
      int[] info = {0};

      iparam[0] = 1;
      iparam[2] = 10*N;
      iparam[3] = 1;
      iparam[4] = 0; // number of ev found by arpack.
      iparam[6] = 1;

      while(ido[0] != 99) {
        /* call arpack like you would have, but, use dsaupd_c instead of dsaupd_ */
        dsaupd_c(ido, bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
                 workd, workl, lworkl, info);

        dMatVec(workd, ipntr[0]-1, ipntr[1]-1);
      }
      if (iparam[4] != nev) return 1; // check number of ev found by arpack.

      /* call arpack like you would have, but, use dseupd_c instead of dseupd_ */
      dseupd_c(rvec, howmny, select, d, z, ldz, sigma,
               bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
               workd, workl, lworkl, info);
      int i;
      for (i = 0; i < nev; ++i) {
        System.out.printf("%f\n", d[i]);
        if(Math.abs(d[i] - (double)(1000-(nev-1)+i))>1e-6){
          return 1;
        }
      }
      return 0;
    }

    static void zMatVec(double[] workd, int x, int y) {
      int i;
      for (i = 0; i < 1000; ++i) {
        workd[2 * (y + i)    ] = (i+1.)*workd[2 * (x + i)    ];
        workd[2 * (y + i) + 1] = (i+1.)*workd[2 * (x + i) + 1];
      }
    }

    static int zn() {
      int[] ido = {0};
      byte[] bmat = {'I'};
      int N = 1000;
      byte[] which = {'L', 'M'};
      int nev = 9;
      double tol = 0;
      double[] resid = new double[2 * N];
      int ncv = 2*nev+1;
      double[] V = new double[2 * (ncv*N)];
      int ldv = N;
      int[] iparam = new int[11];
      int[] ipntr = new int[14];
      double[] workd = new double[2 * 3*N];
      int rvec = 1;
      byte[] howmny = {'A'};
      double[] d = new double[2 * (nev+1)];
      int[] select = new int[ncv];
      double[] z = new double[2 * (N+1)*(nev+1)];
      int ldz = N+1;
      double[] sigma = {0, 0};
      int k;
      for (k=0; k < 3*N; ++k )
        workd[k] = 0;
      double[] workl = new double[2 * (3*(ncv*ncv) + 6*ncv)];
      for (k=0; k < 2 * (3*(ncv*ncv) + 6*ncv); ++k )
        workl[k] = 0;
      int lworkl = 3*(ncv*ncv) + 6*ncv;
      double[] rwork = new double[ncv];
      double[] workev = new double[2 * 2*ncv];
      int[] info = {0};

      iparam[0] = 1;
      iparam[2] = 10*N;
      iparam[3] = 1;
      iparam[4] = 0; // number of ev found by arpack.
      iparam[6] = 1;

      while(ido[0] != 99) {
        /* call arpack like you would have, but, use znaupd_c instead of znaupd_ */
        znaupd_c(ido, bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
                 workd, workl, lworkl, rwork, info);

        zMatVec(workd, ipntr[0]-1, ipntr[1]-1);
      }
      if (iparam[4] != nev) return 1; // check number of ev found by arpack.

      /* call arpack like you would have, but, use zneupd_c instead of zneupd_ */
      zneupd_c(rvec, howmny, select, d, z, ldz, sigma, workev,
               bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
               workd, workl, lworkl, rwork, info);
      int i;
      for (i = 0; i < nev; ++i) {
        System.out.printf("%f %f\n", d[2 * i], d[2 * i + 1]);
        if(Math.abs(d[2 * i] - (double)(1000-i))>1e-6 || Math.abs(d[2 * i + 1] - (double)(1000-i))>1e-6){
          return 1;
        }
      }
      return 0;
    }

    public static void main(String[] args) {
      sstats_c();
      int rc = ds(); // arpack without debug.
      if (rc != 0) System.exit(rc);
      int[] nopx_c = {0}, nbx_c = {0}, nrorth_c = {0}, nitref_c = {0}, nrstrt_c = {0};
      float[] tsaupd_c = {0}, tsaup2_c = {0}, tsaitr_c = {0}, tseigt_c = {0}, tsgets_c = {0}, tsapps_c = {0}, tsconv_c = {0};
      float[] tnaupd_c = {0}, tnaup2_c = {0}, tnaitr_c = {0}, tneigt_c = {0}, tngets_c = {0}, tnapps_c = {0}, tnconv_c = {0};
      float[] tcaupd_c = {0}, tcaup2_c = {0}, tcaitr_c = {0}, tceigt_c = {0}, tcgets_c = {0}, tcapps_c = {0}, tcconv_c = {0};
      float[] tmvopx_c = {0}, tmvbx_c = {0}, tgetv0_c = {0}, titref_c = {0}, trvec_c = {0};
      stat_c(  nopx_c,    nbx_c, nrorth_c, nitref_c, nrstrt_c,
             tsaupd_c, tsaup2_c, tsaitr_c, tseigt_c, tsgets_c, tsapps_c, tsconv_c,
             tnaupd_c, tnaup2_c, tnaitr_c, tneigt_c, tngets_c, tnapps_c, tnconv_c,
             tcaupd_c, tcaup2_c, tcaitr_c, tceigt_c, tcgets_c, tcapps_c, tcconv_c,
             tmvopx_c,  tmvbx_c, tgetv0_c, titref_c,  trvec_c);
      System.out.printf("Timers : nopx %d, tmvopx %f - nbx %d, tmvbx %f\n", nopx_c[0], tmvopx_c[0], nbx_c[0], tmvbx_c[0]);

      System.out.printf("------\n");

      debug_c(6, -6, 1,
              1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1); // set debug flags.
      rc = zn(); // arpack with debug.

      System.out.printf("------\n");
      System.exit(rc);
    }
}
