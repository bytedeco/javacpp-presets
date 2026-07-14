package org.bytedeco.pytorch;

/** Internal tiny wrapper around ModuleAsHelper for ModulePrinter.
 *  Reflectively calls ModuleAsHelper.recover(Module) to look up the
 *  typed wrapper (e.g. LinearImpl) in the registry, so ModulePrinter
 *  can show the concrete type name without calling m.name() (which
 *  SIGSEGVs in some libtorch 2.12 builds on stale Module-typed refs). */
final class ModuleAsHelperRecover {
    private ModuleAsHelperRecover() {}

    static Module recover(Module m) {
        try {
            Class<?> helper = Class.forName("org.bytedeco.pytorch.ModuleAsHelper");
            java.lang.reflect.Method mth = helper.getDeclaredMethod("recover", Module.class);
            mth.setAccessible(true);
            return (Module) mth.invoke(null, m);
        } catch (Throwable t) {
            return null;
        }
    }
}
