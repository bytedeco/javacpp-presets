package org.bytedeco.pytorch;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.bytedeco.javacpp.Pointer;

/**
 * Small reflection helper backing the generic
 * {@link Module#as(Class) Module.as(Class)} method and the generic
 * {@link SequentialImpl#push_back(Module)} entry point.
 *
 * <p>Both rely on the same idea: the C++ side does not let us instantiate
 * the {@code push_back<Module>} template because the base {@code Module}
 * class has no {@code forward()} method (see the
 * {@code has_forward<Module>} static_assert in {@code AnyModule}). We work
 * around that by performing the dispatch on the <em>Java</em> side via
 * the typed {@code asXxx()} methods JavaCPP already generates for the
 * 80+ built-in layers. The C++ JNI proxy is therefore only asked to
 * invoke instantiations of the template with concrete {@code *Impl} types
 * — the path the existing typed push_back overloads already exercise.
 *
 * <p>This class is internal and intentionally minimal — the goal is to
 * keep the byte-code injected by the preset as small as possible.
 */
final class ModuleAsHelper {

    private ModuleAsHelper() {}

    private static final ConcurrentMap<Long, Module> REGISTRY =
            new ConcurrentHashMap<Long, Module>();

    private static long keyOf(Module module) {
        long id = module.moduleObjectId();
        return id != 0L ? id : module.address();
    }

    static void remember(Module module) {
        if (module == null || module.isNull()) {
            return;
        }
        long key = keyOf(module);
        Module current = REGISTRY.get(key);
        if (current != null) {
            // Keep the most specific Java type for a given native pointer.
            if (current.getClass() != Module.class && module.getClass() == Module.class) {
                return;
            }
            if (current.getClass().isAssignableFrom(module.getClass())) {
                REGISTRY.put(key, module);
            }
            return;
        }
        REGISTRY.put(key, module);
    }

    static Module recover(Module module) {
        if (module == null || module.isNull()) {
            return module;
        }
        Module known = REGISTRY.get(keyOf(module));
        return known != null ? known : module;
    }

    static boolean hasForwardOverride(Module module, Class<?>... parameterTypes) {
        if (module == null || module.getClass() == Module.class) {
            return false;
        }
        try {
            Method method = module.getClass().getMethod("forward", parameterTypes);
            return method.getDeclaringClass() != Module.class;
        } catch (NoSuchMethodException e) {
            return false;
        }
    }

    /**
     * Creates a fresh instance of {@code targetClass} wrapping
     * {@code module}'s native pointer when needed.
     */
    static <T extends Module> T wrap(Module module, Class<T> targetClass) {
        try {
            long key = keyOf(module);
            Module known = REGISTRY.get(key);
            if (targetClass.isInstance(known)) {
                return targetClass.cast(known);
            }
            Constructor<T> ctor = targetClass.getDeclaredConstructor(Pointer.class);
            ctor.setAccessible(true);
            T wrapped = ctor.newInstance(module);
            remember(wrapped);
            return wrapped;
        } catch (NoSuchMethodException e) {
            throw new IllegalArgumentException(
                    targetClass.getName()
                    + " cannot be wrapped from this native pointer; call as() on instances created from Java modules of the same class",
                    e);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(
                    "Failed to wrap pointer with " + targetClass.getName(), e);
        }
    }

    /**
     * Pushes {@code module} into {@code seq} by recovering its actual
     * concrete C++ type via the typed {@code as<Impl>()} helpers and
     * forwarding to the matching typed {@code push_back<Impl>()} overload.
     * The {@code name} parameter is optional — pass {@code null} to
     * auto-generate one (matching the C++ template's default behaviour).
     *
     * <p>This supports two flavours of {@link Module} arguments:
     * <ol>
     *   <li>Built-in {@code *Impl} classes — recovered via
     *       {@code getSimpleName()} reflection in O(1).</li>
     *   <li>User-defined {@link Module} subclasses that wrap a known
     *       C++ type — recovered by walking the asXxx() helpers
     *       until one returns non-null. This is the path the
     *       "complex custom class" requirement exercises.</li>
     * </ol>
     */
    static void pushBackModule(SequentialImpl seq, Module module, String name) {
        remember(module);
        // Fast path: derive the asXxx method name from the simple class
        // name. No registry lookup, no array iteration — just one
        // reflection-based asXxx invocation.
        Class<?> moduleClass = module.getClass();
        String simpleName = moduleClass.getSimpleName();
        String asName = "as" + (simpleName.endsWith("Impl")
                ? simpleName.substring(0, simpleName.length() - "Impl".length())
                : simpleName);
        try {
            Method asMethod = Module.class.getMethod(asName);
            Object cast = asMethod.invoke(module);
            if (cast != null) {
                invokePushBack(seq, cast, cast.getClass(), name);
                return;
            }
            // If the asXxx returned null, the underlying C++ object is
            // not actually a module of the requested type — fall through
            // to the slower walk that tries every recognised *Impl.
        } catch (NoSuchMethodException e) {
            // No asXxx for this simple class name (custom Java class)
            // — fall through to the as-walk below.
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(
                    "Failed to push_back " + module.getClass().getName(), e);
        }

        // Fallback for user-defined Module subclasses: walk the
        // built-in asXxx helpers until one returns a non-null wrapper
        // that matches a recognised built-in layer. The C++ side's
        // `as<T>()` performs the dynamic_cast so we don't need to know
        // the actual type ahead of time.
        Object[] result = findUnderlyingType(module);
        if (result == null) {
            // Keep user-defined Module subclasses as a single unit.
            // This path intentionally stores the base Module pointer.
            if (name == null) {
                seq.push_back(module);
            } else {
                seq.push_back(name, module);
            }
            return;
        }
        invokePushBack(seq, result[0], (Class<?>) result[1], name);
    }

    /**
     * Walks the 80+ built-in asXxx() helpers and returns the first
     * one that produces a non-null wrapper. Returns {@code null} when
     * the underlying C++ object isn't a recognised built-in type.
     * The result is a 2-element array: {@code [cast, implClass]}.
     */
    private static Object[] findUnderlyingType(Module module) {
        Method[] methods = Module.class.getMethods();
        // Filter to asXxx methods that return a Module subclass and
        // have no parameters — that's the shape the JavaCPP generator
        // emits for every built-in *Impl.
        for (Method m : methods) {
            if (!m.getName().startsWith("as")
                    || m.getName().equals("as")
                    || m.getParameterCount() != 0) {
                continue;
            }
            Class<?> ret = m.getReturnType();
            if (!Module.class.isAssignableFrom(ret) || ret == Module.class) {
                continue;
            }
            try {
                Object cast = m.invoke(module);
                if (cast != null) {
                    return new Object[]{cast, ret};
                }
            } catch (ReflectiveOperationException ignored) {
                // skip methods we can't invoke
            }
        }
        return null;
    }

    /** Helper that performs the reflective push_back invocation. */
    private static void invokePushBack(SequentialImpl seq, Object cast,
                                       Class<?> implClass, String name) {
        try {
            if (name == null) {
                Method pushBack = SequentialImpl.class.getMethod(
                        "push_back", implClass);
                pushBack.invoke(seq, cast);
            } else {
                Method pushBack = SequentialImpl.class.getMethod(
                        "push_back", String.class, implClass);
                pushBack.invoke(seq, name, cast);
            }
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(
                    "Failed to push_back " + cast.getClass().getName(), e);
        }
    }

    // Fallback list of known *Impl class simple names. Used only when
    // the fast path (module.getClass().getSimpleName()) doesn't match
    // a built-in layer name.
    private static final String[] keys = {
            "LinearImpl", "Conv1dImpl", "Conv2dImpl", "Conv3dImpl",
            "ConvTranspose1dImpl", "ConvTranspose2dImpl", "ConvTranspose3dImpl",
            "BatchNorm1dImpl", "BatchNorm2dImpl", "BatchNorm3dImpl",
            "ReLUImpl", "ReLU6Impl", "LeakyReLUImpl", "ELUImpl",
            "DropoutImpl", "Dropout2dImpl", "Dropout3dImpl",
            "FlattenImpl", "IdentityImpl", "EmbeddingImpl",
            "SequentialImpl", "ModuleListImpl", "ModuleDictImpl",
    };
}