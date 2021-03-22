JavaCPP Presets for Skia
========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/skia/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/skia) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/skia.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![skia](https://github.com/bytedeco/javacpp-presets/workflows/skia/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Askia)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Mono/Skia 2.80.2  https://github.com/mono/skia

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/skia/apidocs/

&lowast; Bindings are currently available only for the C API of Mono/Skia.


Sample Usage
------------
Here is a simple example of Mono/Skia ported to Java from this C source file:

 * https://github.com/mono/skia/blob/xamarin-mobile-bindings/experimental/c-api-example/skia-c-example.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SkiaCExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.skia</groupId>
    <artifactId>skiacexample</artifactId>
    <version>1.5.5</version>
    <properties>
        <exec.mainClass>SkiaCExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>skia-platform</artifactId>
            <version>2.80.2-1.5.5</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SkiaCExample.java` source file
```java
import java.io.FileOutputStream;
import java.io.IOException;

import org.bytedeco.javacpp.*;

import org.bytedeco.skia.*;
import static org.bytedeco.skia.global.Skia.*;

public class SkiaCExample {
    private static sk_surface_t makeSurface(int w, int h) {
        sk_imageinfo_t info = new sk_imageinfo_t();
        info.width(w);
        info.height(h);
        info.colorType(BGRA_8888_SK_COLORTYPE);
        info.alphaType(PREMUL_SK_ALPHATYPE);
        return sk_surface_new_raster(info, 0, null);
    }

    private static void emitPng(String path, sk_surface_t surface) throws IOException {
        sk_image_t image = sk_surface_new_image_snapshot(surface);
        sk_data_t data = sk_image_encode(image);
        sk_image_unref(image);
        Pointer pointer = sk_data_get_data(data).limit(sk_data_get_size(data));
        FileOutputStream out = new FileOutputStream(path);
        out.getChannel().write(pointer.asByteBuffer());
        out.close();
    }

    private static void draw(sk_canvas_t canvas) {
        sk_paint_t fill = sk_paint_new();
        sk_paint_set_color(fill, 0xFF0000FF);
        sk_canvas_draw_paint(canvas, fill);

        sk_paint_set_color(fill, 0xFF00FFFF);
        sk_rect_t rect = new sk_rect_t();
        rect.left(100.0f);
        rect.top(100.0f);
        rect.right(540.0f);
        rect.bottom(380.0f);
        sk_canvas_draw_rect(canvas, rect, fill);

        sk_paint_t stroke = sk_paint_new();
        sk_paint_set_color(stroke, 0xFFFF0000);
        sk_paint_set_antialias(stroke, true);
        sk_paint_set_style(stroke, STROKE_SK_PAINT_STYLE);
        sk_paint_set_stroke_width(stroke, 5.0f);
        sk_path_t path = sk_path_new();

        sk_path_move_to(path, 50.0f, 50.0f);
        sk_path_line_to(path, 590.0f, 50.0f);
        sk_path_cubic_to(path, -490.0f, 50.0f, 1130.0f, 430.0f, 50.0f, 430.0f);
        sk_path_line_to(path, 590.0f, 430.0f);
        sk_canvas_draw_path(canvas, path, stroke);

        sk_paint_set_color(fill, 0x8000FF00);
        sk_rect_t rect2 = new sk_rect_t();
        rect2.left(120.0f);
        rect2.top(120.0f);
        rect2.right(520.0f);
        rect2.bottom(360.0f);
        sk_canvas_draw_oval(canvas, rect2, fill);

        sk_path_delete(path);
        sk_paint_delete(stroke);
        sk_paint_delete(fill);
    }

    public static void main (String[] args) throws IOException {
        sk_surface_t surface = makeSurface(640, 480);
        sk_canvas_t canvas = sk_surface_get_canvas(surface);
        draw(canvas);
        emitPng("skia-c-example.png", surface);
        sk_surface_unref(surface);
    }
}
```
