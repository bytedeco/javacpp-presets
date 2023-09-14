package org.bytedeco.pytorch;

import org.bytedeco.javacpp.annotation.Adapter;
import org.bytedeco.javacpp.annotation.Cast;

import java.lang.annotation.*;

@Documented @Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Cast("c10::string_view&") @Adapter("StringViewAdapter")
public @interface StringView {
}
