import java.io.File;
import org.bytedeco.javacpp.*;
import org.bytedeco.qt.QtCore.*;
import org.bytedeco.qt.QtGui.*;
import org.bytedeco.qt.QtWidgets.*;
import static org.bytedeco.qt.global.QtCore.*;
import static org.bytedeco.qt.global.QtGui.*;
import static org.bytedeco.qt.global.QtWidgets.*;

public class GettingStarted {
    private static IntPointer argc;
    private static PointerPointer argv;

    public static void main(String[] args) {
        String path = Loader.load(org.bytedeco.qt.global.QtCore.class);
        argc = new IntPointer(new int[]{3});
        argv = new PointerPointer("gettingstarted", "-platformpluginpath", new File(path).getParent(), null);

        QApplication app = new QApplication(argc, argv);

        QTextEdit textEdit = new QTextEdit();
        textEdit.show();

        System.exit(app.exec());
    }
}
