import java.io.File;
import org.bytedeco.javacpp.*;
import org.bytedeco.qt.Qt5Core.*;
import org.bytedeco.qt.Qt5Gui.*;
import org.bytedeco.qt.Qt5Widgets.*;
import static org.bytedeco.qt.global.Qt5Core.*;
import static org.bytedeco.qt.global.Qt5Gui.*;
import static org.bytedeco.qt.global.Qt5Widgets.*;

public class GettingStarted {
    private static IntPointer argc;
    private static PointerPointer argv;

    public static void main(String[] args) {
        String path = Loader.load(org.bytedeco.qt.global.Qt5Core.class);
        argc = new IntPointer(new int[]{3});
        argv = new PointerPointer("gettingstarted", "-platformpluginpath", new File(path).getParent(), null);

        QApplication app = new QApplication(argc, argv);

        QTextEdit textEdit = new QTextEdit();
        textEdit.show();

        System.exit(app.exec());
    }
}
