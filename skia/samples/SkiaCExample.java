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
