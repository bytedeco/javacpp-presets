import org.bytedeco.javacpp.*;
import org.bytedeco.hdf5_plugins.HDF5Plugins;

import static org.bytedeco.hdf5.global.hdf5.*;

/** Writes and reads back a Zstandard-compressed HDF5 dataset. */
public class ZstdSample {
    public static void main(String[] args) throws Exception {
        HDF5Plugins.load(); // lets HDF5 find the bundled filter plugins

        int n = 1024;
        int[] values = new int[n];
        for (int i = 0; i < n; i++) {
            values[i] = i / 16;
        }

        long file = H5Fcreate("zstd.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        long space = H5Screate_simple(1, new long[] {n}, null);
        long dcpl = H5Pcreate(H5P_DATASET_CREATE());
        H5Pset_chunk(dcpl, 1, new long[] {n});
        H5Pset_filter(dcpl, HDF5Plugins.H5Z_FILTER_ZSTD, H5Z_FLAG_MANDATORY, 1, new int[] {5}); // compression level
        long dset = H5Dcreate2(file, "data", H5T_NATIVE_INT(), space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_INT(), H5S_ALL, H5S_ALL, H5P_DEFAULT, new IntPointer(values));
        H5Dclose(dset);
        H5Pclose(dcpl);
        H5Sclose(space);
        H5Fclose(file);

        file = H5Fopen("zstd.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
        dset = H5Dopen2(file, "data", H5P_DEFAULT);
        IntPointer out = new IntPointer(n);
        H5Dread(dset, H5T_NATIVE_INT(), H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
        System.out.println("values[1000] = " + out.get(1000));
        H5Dclose(dset);
        H5Fclose(file);
    }
}
