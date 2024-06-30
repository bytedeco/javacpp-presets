package org.bytedeco.pytorch.helper;

import org.bytedeco.javacpp.annotation.Adapter;

import java.lang.annotation.*;

@Documented @Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Adapter("WeakPtrAdapter")
public @interface WeakPtr {
    String value() default "";
}
