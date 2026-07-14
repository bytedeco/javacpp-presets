package org.bytedeco.pytorch;

/** Internal package-private helper. Some Tensor accessor methods
 *  (sizes / scalar_type / device) are inherited from TensorBase
 *  and not explicitly re-declared in the generated Tensor.java. To
 *  let ModulePrinter reach them we expose a tiny static facade here. */
final class ModulePrinterShapeHelper {
    private ModulePrinterShapeHelper() {}

    static long[] sizesAsArray(org.bytedeco.pytorch.LongHeaderOnlyArrayRef ref) {
        long len = ref.size();
        if (len == 0) return new long[0];
        return ref.vec().get();
    }
}
