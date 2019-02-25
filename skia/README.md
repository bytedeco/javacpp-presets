JavaCPP Presets for Skia
========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * [mono/skia](https://github.com/mono/skia) branch `update-master` as of 2017-05-11  https://skia.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/skia/apidocs/

&lowast; Bindings are currently available only for the C API of Skia.

Sample Usage
------------
Here is a simple example of Skia ported to Java from this C source file:

 * https://github.com/mono/skia/blob/update-master/experimental/c-api-example/skia-c-example.c

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
    <version>1.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>SkiaCExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>skia-platform</artifactId>
            <version>20170511-53d6729-1.5-SNAPSHOT</version>
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
        info.colorType(sk_colortype_get_default_8888());
        info.alphaType(PREMUL_SK_ALPHATYPE);
        return sk_surface_new_raster(info, null);
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
