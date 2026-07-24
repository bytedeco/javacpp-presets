package org.bytedeco.pytorch.nn;


/** Internal tiny wrapper around ModuleAsHelper for ModulePrinter. */
public final class ModuleAsHelperRecover {
    private ModuleAsHelperRecover() {}

    static Module recover(Module m) {
        try {
            Class<?> helper = Class.forName("org.bytedeco.pytorch.nn.ModuleAsHelper");
            java.lang.reflect.Method mth = helper.getDeclaredMethod("recover", Module.class);
            mth.setAccessible(true);
            return (Module) mth.invoke(null, m);
        } catch (Throwable t) {
            return null;
        }
    }
}
