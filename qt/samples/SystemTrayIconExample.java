import java.io.File;
import java.io.IOException;
import java.net.URL;
import org.bytedeco.javacpp.*;
import org.bytedeco.qt.Qt5Core.*;
import org.bytedeco.qt.Qt5Gui.*;
import org.bytedeco.qt.Qt5Widgets.*;
import static org.bytedeco.qt.global.Qt5Core.*;
import static org.bytedeco.qt.global.Qt5Gui.*;
import static org.bytedeco.qt.global.Qt5Widgets.*;

/**
 * @author Greg Hart
 */
public class SystemTrayIconExample {

  private static IntPointer argc;
  private static PointerPointer argv;
  private static QApplication app;
  private static Window window;

  public static void main(String[] args) {
    String path = Loader.load(org.bytedeco.qt.global.Qt5Core.class);
    argc = new IntPointer(new int[]{3});
    argv = new PointerPointer("systray", "-platformpluginpath", new File(path).getParent(), null);

    app = new QApplication(argc, argv);

    if (!QSystemTrayIcon.isSystemTrayAvailable()) {
      QMessageBox.critical(null, QObject.tr("Systray"),
          QObject.tr("I couldn't detect any system tray on this system."));
      System.exit(1);
    }
    QApplication.setQuitOnLastWindowClosed(false);

    window = new Window();
    window.show();
    System.exit(QApplication.exec());
  }

  public static class Window extends QDialog {

    public Window() {
      createIconGroupBox();
      createMessageGroupBox();

      iconLabel.setMinimumWidth(durationLabel.sizeHint().width());

      createActions();
      createTrayIcon();

      ClickedCallback showMessageCallback = new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          showMessage();
        }
      };
      QAbstractButton_clicked(showMessageButton, this, showMessageCallback, showMessageCallback.id);
      ToggledCallback setVisableCallback = new ToggledCallback() {
        @Override
        public void toggled(boolean checked) {
          trayIcon.setVisible(checked);
        }
      };
      QAbstractButton_toggled(showIconCheckBox, trayIcon, setVisableCallback, setVisableCallback.id);
      CurrentIndexChangedCallback1 setIconCallback = new CurrentIndexChangedCallback1() {
        @Override
        public void currentIndexChanged(int index) {
          setIcon(index);
        }
      };
      QComboBox_currentIndexChanged(iconComboBox, this, setIconCallback, setIconCallback.id);
      MessageClickedCallback messageClickedCallback = new MessageClickedCallback() {
        @Override
        public void messageClicked() {
          Window.this.messageClicked();
        }
      };
      QSystemTrayIcon_messageClicked(trayIcon, this, messageClickedCallback, messageClickedCallback.id);
      ActivatedCallback iconActivatedCallback = new ActivatedCallback() {
        @Override
        public void activated(QSystemTrayIcon.ActivationReason reason) {
          iconActivated(reason);
        }
      };
      QSystemTrayIcon_activated(trayIcon, this, iconActivatedCallback, iconActivatedCallback.id);

      QVBoxLayout mainLayout = new QVBoxLayout();
      mainLayout.addWidget(iconGroupBox);
      mainLayout.addWidget(messageGroupBox);
      setLayout(mainLayout);

      iconComboBox.setCurrentIndex(1);
      trayIcon.show();

      setWindowTitle(tr("Systray"));
      resize(400, 300);
    }

    @Override
    public void setVisible(boolean visible) {
      minimizeAction.setEnabled(visible);
      maximizeAction.setEnabled(!isMaximized());
      restoreAction.setEnabled(isMaximized() || !visible);
      super.setVisible(visible);
    }

//    @Override
    protected void closeEvent(QCloseEvent event) {
      if (trayIcon.isVisible()) {
        QMessageBox.information(this, tr("Systray"),
            tr("The program will keep running in the " +
                "system tray. To terminate the program, " +
                "choose <b>Quit</b> in the context menu " +
                "of the system tray entry."));
        hide();
        event.ignore();
      }
    }

    private void setIcon(int index) {
      QIcon icon = iconComboBox.itemIcon(index);
      trayIcon.setIcon(icon);
      setWindowIcon(icon);

      trayIcon.setToolTip(iconComboBox.itemText(index));
    }
    private void iconActivated(QSystemTrayIcon.ActivationReason reason) {
      switch (reason) {
        case Trigger:
        case DoubleClick:
          iconComboBox.setCurrentIndex((iconComboBox.currentIndex() + 1) % iconComboBox.count());
          break;
        case MiddleClick:
          showMessage();
          break;
        default:
          ;
      }
    }
    private void showMessage() {
      showIconCheckBox.setChecked(true);
      QSystemTrayIcon.MessageIcon msgIcon = QSystemTrayIcon.MessageIcon.values()[
          typeComboBox.itemData(typeComboBox.currentIndex()).toInt()];
      if (msgIcon == QSystemTrayIcon.MessageIcon.NoIcon) {
        QIcon icon = new QIcon(iconComboBox.itemIcon(iconComboBox.currentIndex()));
        trayIcon.showMessage(titleEdit.text(), bodyEdit.toPlainText(), icon,
            durationSpinBox.value() * 1000);
      } else {
        trayIcon.showMessage(titleEdit.text(), bodyEdit.toPlainText(), msgIcon,
            durationSpinBox.value() * 1000);
      }
    }
    private void messageClicked() {
      QMessageBox.information(null, tr("Systray"),
          tr("Sorry, I already gave what help I could.\n" +
              "Maybe you should try asking a human?"));
    }

    private QHBoxLayout iconLayout;

    private void createIconGroupBox() {
      iconGroupBox = new QGroupBox(tr("Tray Icon"));

      iconLabel = new QLabel(QString.fromStdString("Icon:"));

      iconComboBox = new QComboBox();
      try {
        iconComboBox.addItem(new QIcon(QString.fromStdString(Loader.cacheResource(new URL(
            "https://code.qt.io/cgit/qt/qtbase.git/plain/examples/widgets/desktop/systray/images/bad.png")).getAbsolutePath())),
            tr("Bad"));
        iconComboBox.addItem(new QIcon(QString.fromStdString(Loader.cacheResource(new URL(
            "https://code.qt.io/cgit/qt/qtbase.git/plain/examples/widgets/desktop/systray/images/heart.png")).getAbsolutePath())),
            tr("Heart"));
        iconComboBox.addItem(new QIcon(QString.fromStdString(Loader.cacheResource(new URL(
            "https://code.qt.io/cgit/qt/qtbase.git/plain/examples/widgets/desktop/systray/images/trash.png")).getAbsolutePath())),
            tr("Trash"));
      } catch (IOException e) {
        e.printStackTrace();
      }

      showIconCheckBox = new QCheckBox(tr("Show icon"));
      showIconCheckBox.setChecked(true);

      iconLayout = new QHBoxLayout();
      iconLayout.addWidget(iconLabel);
      iconLayout.addWidget(iconComboBox);
      iconLayout.addStretch();
      iconLayout.addWidget(showIconCheckBox);
      iconGroupBox.setLayout(iconLayout);
    }
    private void createMessageGroupBox() {
      messageGroupBox = new QGroupBox(tr("Balloon Message"));

      typeLabel = new QLabel(tr("Type:"));

      typeComboBox = new QComboBox();
      typeComboBox.addItem(tr("None"), new QVariant(QSystemTrayIcon.MessageIcon.NoIcon.value));
      typeComboBox.addItem(style().standardIcon(
          QStyle.SP_MessageBoxInformation), tr("Information"),
          new QVariant(QSystemTrayIcon.MessageIcon.Information.value));
      typeComboBox.addItem(style().standardIcon(
          QStyle.SP_MessageBoxWarning), tr("Warning"),
          new QVariant(QSystemTrayIcon.MessageIcon.Warning.value));
      typeComboBox.addItem(style().standardIcon(
          QStyle.SP_MessageBoxCritical), tr("Critical"),
          new QVariant(QSystemTrayIcon.MessageIcon.Critical.value));
      typeComboBox.addItem(new QIcon(), tr("Custom icon"),
          new QVariant(QSystemTrayIcon.MessageIcon.NoIcon.value));
      typeComboBox.setCurrentIndex(1);

      durationLabel = new QLabel(tr("Duration:"));

      durationSpinBox = new QSpinBox();
      durationSpinBox.setRange(5, 60);
      durationSpinBox.setSuffix(QString.fromStdString(" s"));
      durationSpinBox.setValue(15);

      durationWarningLabel = new QLabel(tr("(some systems might ignore this " +
          "hint)"));
      durationWarningLabel.setIndent(10);

      titleLabel = new QLabel(tr("Title:"));

      titleEdit = new QLineEdit(tr("Cannot connect to network"));

      bodyLabel = new QLabel(tr("Body:"));

      bodyEdit = new QTextEdit();
      bodyEdit.setPlainText(tr("Don't believe me. Honestly, I don't have a " +
          "clue.\nClick this balloon for details."));

      showMessageButton = new QPushButton(tr("Show Message"));
      showMessageButton.setDefault(true);

      final QGridLayout messageLayout = new QGridLayout();
      messageLayout.addWidget(typeLabel, 0, 0);
      messageLayout.addWidget(typeComboBox, 0, 1, 1, 2);
      messageLayout.addWidget(durationLabel, 1, 0);
      messageLayout.addWidget(durationSpinBox, 1, 1);
      messageLayout.addWidget(durationWarningLabel, 1, 2, 1, 3);
      messageLayout.addWidget(titleLabel, 2, 0);
      messageLayout.addWidget(titleEdit, 2, 1, 1, 4);
      messageLayout.addWidget(bodyLabel, 3, 0);
      messageLayout.addWidget(bodyEdit, 3, 1, 2, 4);
      messageLayout.addWidget(showMessageButton, 5, 4);
      messageLayout.setColumnStretch(3, 1);
      messageLayout.setRowStretch(4, 1);
      messageGroupBox.setLayout(messageLayout);
    }
    private void createActions() {
      minimizeAction = new QAction(tr("Mi&nimize"), this);
      TriggeredCallback hideCallback = new TriggeredCallback() {
        @Override
        public void triggered(boolean checked) {
          hide();
        }
      };
      QAction_triggered(minimizeAction, this, hideCallback, hideCallback.id);

      maximizeAction = new QAction(tr("Ma&ximize"), this);
      TriggeredCallback showMaximizedCallback = new TriggeredCallback() {
        @Override
        public void triggered(boolean checked) {
          showMaximized();
        }
      };
      QAction_triggered(maximizeAction, this, showMaximizedCallback, showMaximizedCallback.id);

      restoreAction = new QAction(tr("&Restore"), this);
      TriggeredCallback showNormalCallback = new TriggeredCallback() {
        @Override
        public void triggered(boolean checked) {
          showNormal();
        }
      };
      QAction_triggered(restoreAction, this, showNormalCallback, showNormalCallback.id);

      quitAction = new QAction(tr("&Quit"), this);
      TriggeredCallback quitCallback = new TriggeredCallback() {
        @Override
        public void triggered(boolean checked) {
          QCoreApplication.quit();
        }
      };
      QAction_triggered(quitAction, QApplication.instance(), quitCallback, quitCallback.id);
    }
    private void createTrayIcon() {
      trayIconMenu = new QMenu(this);
      trayIconMenu.addAction(minimizeAction);
      trayIconMenu.addAction(maximizeAction);
      trayIconMenu.addAction(restoreAction);
      trayIconMenu.addSeparator();
      trayIconMenu.addAction(quitAction);

      trayIcon = new QSystemTrayIcon(this);
      trayIcon.setContextMenu(trayIconMenu);
    }

    private QGroupBox iconGroupBox;
    private QLabel iconLabel;
    private QComboBox iconComboBox;
    private QCheckBox showIconCheckBox;

    private QGroupBox messageGroupBox;
    private QLabel typeLabel;
    private QLabel durationLabel;
    private QLabel durationWarningLabel;
    private QLabel titleLabel;
    private QLabel bodyLabel;
    private QComboBox typeComboBox;
    private QSpinBox durationSpinBox;
    private QLineEdit titleEdit;
    private QTextEdit bodyEdit;
    private QPushButton showMessageButton;

    private QAction minimizeAction;
    private QAction maximizeAction;
    private QAction restoreAction;
    private QAction quitAction;

    private QSystemTrayIcon trayIcon;
    private QMenu trayIconMenu;
  }
}
