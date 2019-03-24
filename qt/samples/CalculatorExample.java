import java.io.File;
import java.util.ArrayList;
import java.util.List;
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
public class CalculatorExample {

  private static IntPointer argc;
  private static PointerPointer argv;
  private static QApplication app;
  private static Calculator calc;

  public static void main(String[] args) {
    String path = Loader.load(org.bytedeco.qt.global.Qt5Core.class);
    argc = new IntPointer(new int[]{3});
    argv = new PointerPointer("calc", "-platformpluginpath", new File(path).getParent(), null);

    app = new QApplication(argc, argv);
    calc = new Calculator();
    calc.show();
    System.exit(QApplication.exec());
  }

  public static class Button extends QToolButton {

    public Button(QString text) {
      this(text, null);
    }

    public Button(QString text, QWidget parent) {
      super(parent);

      setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred);
      setText(text);
    }

    @Override
    public QSize sizeHint() {
      QSize size = super.sizeHint();
      IntPointer rheight = size.rheight();
      rheight.put(rheight.get() + 20);
      size.rwidth().put(Math.max(size.width(), size.height()));
      return size;
    }
  }

  public static class Calculator extends QWidget {

    public Calculator() {
      this(null);
    }

    public Calculator(QWidget parent) {
      sumInMemory = 0.0;
      sumSoFar = 0.0;
      factorSoFar = 0.0;
      waitingForOperand = true;

      display = new QLineEdit(QString.fromStdString("0"));
      display.setReadOnly(true);
      display.setAlignment(AlignRight);
      display.setMaxLength(15);

      QFont font = display.font();
      font.setPointSize(font.pointSize() + 8);
      display.setFont(font);

      for (int i = 0; i < NumDigitButtons; ++i) {
        final QString text = QString.number(i);
        digitButtons[i] = createButton(text, new ClickedCallback() {
          @Override
          public void clicked(boolean checked) {
            digitClicked(text);
          }
        });
      }

      pointButton = createButton(tr("."), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          pointClicked();
        }
      });
      changeSignButton = createButton(tr("\u00b1"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          changeSignClicked();
        }
      });

      backspaceButton = createButton(tr("Backspace"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          backspaceClicked();
        }
      });
      clearButton = createButton(tr("Clear"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          clear();
        }
      });
      clearAllButton = createButton(tr("Clear All"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          clearAll();
        }
      });

      clearMemoryButton = createButton(tr("MC"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          clearMemory();
        }
      });
      readMemoryButton = createButton(tr("MR"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          readMemory();
        }
      });
      setMemoryButton = createButton(tr("MS"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          setMemory();
        }
      });
      addToMemoryButton = createButton(tr("M+"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          addToMemory();
        }
      });

      divisionButton = createButton(tr("\u00f7"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          multiplicativeOperatorClicked(tr("\u00f7"));
        }
      });
      timesButton = createButton(tr("\u00d7"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          multiplicativeOperatorClicked(tr("\u00d7"));
        }
      });
      minusButton = createButton(tr("-"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          additiveOperatorClicked(tr("-"));
        }
      });
      plusButton = createButton(tr("+"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          additiveOperatorClicked(tr("+"));
        }
      });

      squareRootButton = createButton(tr("Sqrt"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          unaryOperatorClicked(tr("Sqrt"));
        }
      });
      powerButton = createButton(tr("x\u00b2"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          unaryOperatorClicked(tr("x\u00b2"));
        }
      });
      reciprocalButton = createButton(tr("1/x"), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          unaryOperatorClicked(tr("1/x"));
        }
      });
      equalButton = createButton(tr("="), new ClickedCallback() {
        @Override
        public void clicked(boolean checked) {
          equalClicked();
        }
      });

      QGridLayout mainLayout = new QGridLayout();
      mainLayout.setSizeConstraint(QLayout.SetFixedSize);
      mainLayout.addWidget(display, 0, 0, 1, 6);
      mainLayout.addWidget(backspaceButton, 1, 0, 1, 2);
      mainLayout.addWidget(clearButton, 1, 2, 1, 2);
      mainLayout.addWidget(clearAllButton, 1, 4, 1, 2);

      mainLayout.addWidget(clearMemoryButton, 2, 0);
      mainLayout.addWidget(readMemoryButton, 3, 0);
      mainLayout.addWidget(setMemoryButton, 4, 0);
      mainLayout.addWidget(addToMemoryButton, 5, 0);

      for (int i = 1; i < NumDigitButtons; ++i) {
        int row = ((9 - i) / 3) + 2;
        int column = ((i - 1) % 3) + 1;
        mainLayout.addWidget(digitButtons[i], row, column);
      }

      mainLayout.addWidget(digitButtons[0], 5, 1);
      mainLayout.addWidget(pointButton, 5, 2);
      mainLayout.addWidget(changeSignButton, 5, 3);

      mainLayout.addWidget(divisionButton, 2, 4);
      mainLayout.addWidget(timesButton, 3, 4);
      mainLayout.addWidget(minusButton, 4, 4);
      mainLayout.addWidget(plusButton, 5, 4);

      mainLayout.addWidget(squareRootButton, 2, 5);
      mainLayout.addWidget(powerButton, 3, 5);
      mainLayout.addWidget(reciprocalButton, 4, 5);
      mainLayout.addWidget(equalButton, 5, 5);
      setLayout(mainLayout);

      setWindowTitle(tr("Calculator"));
    }

    private void digitClicked(QString text) {
      int digitValue = text.toInt();
      if (display.text().equals("0") && digitValue == 0.0)
        return;

      if (waitingForOperand) {
        display.clear();
        waitingForOperand = false;
      }
      display.setText(display.text().add(QString.number(digitValue)));
    }
    private void unaryOperatorClicked(QString clickedOperator) {
      double operand = display.text().toDouble();
      double result = 0.0;

      if (clickedOperator.equals(tr("Sqrt"))) {
        if (operand < 0.0) {
          abortOperation();
          return;
        }
        result = Math.sqrt(operand);
      } else if (clickedOperator.equals(tr("x\u00b2"))) {
        result = Math.pow(operand, 2.0);
      } else if (clickedOperator.equals(tr("1/x"))) {
        if (operand == 0.0) {
          abortOperation();
          return;
        }
        result = 1.0 / operand;
      }
      display.setText(QString.number(result));
      waitingForOperand = true;
    }
    private void additiveOperatorClicked(QString clickedOperator) {
      double operand = display.text().toDouble();

      if (!pendingMultiplicativeOperator.isEmpty()) {
        if (!calculate(operand, pendingMultiplicativeOperator)) {
          abortOperation();
          return;
        }
        display.setText(QString.number(factorSoFar));
        operand = factorSoFar;
        factorSoFar = 0.0;
        pendingMultiplicativeOperator.clear();
      }

      if (!pendingAdditiveOperator.isEmpty()) {
        if (!calculate(operand, pendingAdditiveOperator)) {
          abortOperation();
          return;
        }
        display.setText(QString.number(sumSoFar));
      } else {
        sumSoFar = operand;
      }

      pendingAdditiveOperator = clickedOperator;
      waitingForOperand = true;
    }
    private void multiplicativeOperatorClicked(QString clickedOperator) {
      double operand = display.text().toDouble();

      if (!pendingMultiplicativeOperator.isEmpty()) {
        if (!calculate(operand, pendingMultiplicativeOperator)) {
          abortOperation();
          return;
        }
        display.setText(QString.number(factorSoFar));
      } else {
        factorSoFar = operand;
      }

      pendingMultiplicativeOperator = clickedOperator;
      waitingForOperand = true;
    }
    private void equalClicked() {
      double operand = display.text().toDouble();

      if (!pendingMultiplicativeOperator.isEmpty()) {
        if (!calculate(operand, pendingMultiplicativeOperator)) {
          abortOperation();
          return;
        }
        operand = factorSoFar;
        factorSoFar = 0.0;
        pendingMultiplicativeOperator.clear();
      }
      if (!pendingAdditiveOperator.isEmpty()) {
        if (!calculate(operand, pendingAdditiveOperator)) {
          abortOperation();
          return;
        }
        pendingAdditiveOperator.clear();
      } else {
        sumSoFar = operand;
      }

      display.setText(QString.number(sumSoFar));
      sumSoFar = 0.0;
      waitingForOperand = true;
    }
    private void pointClicked() {
      if (waitingForOperand)
        display.setText(QString.fromStdString("0"));
      if (!display.text().contains(QString.fromStdString(".")))
        display.setText(display.text().add(tr(".")));
      waitingForOperand = false;
    }
    private void changeSignClicked() {
      QString text = display.text();
      double value = text.toDouble();

      if (value > 0.0) {
        text.prepend(tr("-"));
      } else if (value < 0.0) {
        text.remove(0, 1);
      }
      display.setText(text);
    }
    private void backspaceClicked() {
      if (waitingForOperand)
        return;

      QString text = display.text();
      text.chop(1);
      if (text.isEmpty()) {
        text = QString.fromStdString("0");
        waitingForOperand = true;
      }
      display.setText(text);
    }
    private void clear() {
      if (waitingForOperand)
        return;

      display.setText(QString.fromStdString("0"));
      waitingForOperand = true;
    }
    private void clearAll() {
      sumSoFar = 0.0;
      factorSoFar = 0.0;
      pendingAdditiveOperator.clear();
      pendingMultiplicativeOperator.clear();
      display.setText(QString.fromStdString("0"));
      waitingForOperand = true;
    }
    private void clearMemory() {
      sumInMemory = 0.0;
    }
    private void readMemory() {
      display.setText(QString.number(sumInMemory));
      waitingForOperand = true;
    }
    private void setMemory() {
      equalClicked();
      sumInMemory = display.text().toDouble();
    }
    private void addToMemory() {
      equalClicked();
      sumInMemory += display.text().toDouble();
    }
    List<ClickedCallback> callbacks = new ArrayList<ClickedCallback>();
    private Button createButton(QString text, ClickedCallback member) {
      Button button = new Button(text);
      QAbstractButton_clicked(button, this, member, member.id);
      callbacks.add(member);
      return button;
    }
    private void abortOperation() {
      clearAll();
      display.setText(tr("####"));
    }
    private boolean calculate(double rightOperand, QString pendingOperator) {
      if (pendingOperator.equals(tr("+"))) {
        sumSoFar += rightOperand;
      } else if (pendingOperator.equals(tr("-"))) {
        sumSoFar -= rightOperand;
      } else if (pendingOperator.equals(tr("\u00d7"))) {
        factorSoFar *= rightOperand;
      } else if (pendingOperator.equals(tr("\u00f7"))) {
        if (rightOperand == 0.0)
          return false;
        factorSoFar /= rightOperand;
      }
      return true;
    }

    private double sumInMemory;
    private double sumSoFar;
    private double factorSoFar;
    private QString pendingAdditiveOperator = new QString();
    private QString pendingMultiplicativeOperator = new QString();
    private boolean waitingForOperand;

    private QLineEdit display;

    private static int NumDigitButtons = 10;
    private Button[] digitButtons = new Button[NumDigitButtons];

    private Button pointButton;
    private Button changeSignButton;

    private Button backspaceButton;
    private Button clearButton;
    private Button clearAllButton;

    private Button clearMemoryButton;
    private Button readMemoryButton;
    private Button setMemoryButton;
    private Button addToMemoryButton;

    private Button divisionButton;
    private Button timesButton;
    private Button minusButton;
    private Button plusButton;

    private Button squareRootButton;
    private Button powerButton;
    private Button reciprocalButton;
    private Button equalButton;
  }
}
