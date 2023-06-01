# Machine learning and graphics for financial markets and beyond.

The following code is used for machine learning (classification) and financial market charts.
A file is used for all called classes "new_scatter_2period_.py". Price data is requested from yahoo.

The old codes are saved in the file: **old_scatter_2period_.py**. New code that has been added and will be updated: **new_scatter_2period_.py**.

The following libraries are required: 
[PyQt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/introduction.html#pyqt5-components),
[numpy](https://numpy.org/),
[pandas](https://pandas.pydata.org/),
[sklearn](https://scikit-learn.org/stable/index.html),
[scipy](https://scipy.org/),
[TA-Lib](https://ta-lib.github.io/ta-lib-python/doc_index.html),
[yfinance](https://github.com/ranaroussi/yfinance)

You also need to install: PyQtChart.
```
python -m pip install PyQtChart 
```
You can see more detailed work of the scripts [here](https://quant12345.github.io/index.html)

![Visually, everything looks like this](https://github.com/quant12345/Machine-Learning-in-Finance/blob/980b7b23d86cad6019950e8c586983d6a88336d1/chart.gif)

The following modules are used to access classes:
```
from PyQt5 import QtWidgets
import yfinance as yf
import new_scatter_2period_
```

**class Main2Period:** 
Class that launches a chart in which you move the indicator period with the slider and select the one you need from the drop-down menu. The graph is immediately redrawn and you can see how the data changes interactively. 

The indicator values are taken with a shift of 1. If the candle closed above or at the same level as the open, then the dot is green, if the close was with a decrease, then its color is blue.

All you need to do is just pass a dataframe that has columns: 'Open','High', 'Low', 'Close'. Most indicators from the TA-LIB library are calculated: with one parameter(RSI, CCI, Momentum, SMA, EMA and others).
```
from PyQt5 import QtWidgets
import yfinance as yf
import new_scatter_2period_

df = yf.download('GE', start='2007-05-15', end='2021-10-01')

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = new_scatter_2period_.Main2Period(df_=df)
    w.show()
    sys.exit(app.exec_())
```
**class Main2Market:**
Similar to the previous version. But, in the second dataframe, another financial instrument is used. In this case, these are S&P 500(^GSPC). The class has been improved and now the second tool works much faster. If the value was growing or equally, then the point green, the falling point value will be red.

Added various indicators from the library: TA-Lib (which are calculated only by one column, unlike the previous class).

To work, you need the main 'df' dataframe, which should have columns: 'Open', 'Close'. Dataframe 'df1' for another financial instrument. You must also specify which columns to calculate the indicators for: cl1_='Close', cl2_='Close'. In this example, a [1-minute timeframe](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) is used (by default, it is equal to the daily timeframe). If you are using a daily timeframe, then you can leave it out. The daily timeframe example lines are commented out (you can try them).
```
from PyQt5 import QtWidgets
import yfinance as yf
import new_scatter_2period_


#df = yf.download('BAC', start='2007-05-15', end='2021-10-01')
#df1 = yf.download('^GSPC', start='2007-05-15', end='2021-10-01')

df = yf.download('BAC', period='5d', interval='1m')
df1 = yf.download('^GSPC', period='5d', interval='1m')


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = new_scatter_2period_.Main2Market(df_=df, df1_=df1, cl1_='Close', cl2_='Close', timeframe_='T')
    w.show()
    sys.exit(app.exec_())
```
**class curve_prec_rec:**
For classification, a one-period increment with a shift of one is used. The "GradientBoostingClassifier" model is used.
"decision_function()" is used to get the sum of points from "sklearn".By adjusting the "threshold" we get the optimal ratio of "precisions" and "recalls".
report=True(prints "classification_report"), split - how much in % training data, class_= 1 or class_= 0
```
df = yf.download('GE', start='2007-05-15', end='2021-10-01')

if __name__ == "__main__":
   import sys

   app = QtWidgets.QApplication(sys.argv)
   w = new_scatter_2period_.curve_prec_rec(report=False, class_=1, df=df, split=70, depth=7, tree=100, L=0.01)
   w.show()
   sys.exit(app.exec_())
```
**class curve_roc_:**
For classification, a one-period increment with a shift of one is used. The "GradientBoostingClassifier" model is used.
By adjusting the slider, select the desired ratio of fpr and tpr. Split - how much in % training data, class_= 1 or class_= 0

```
df = yf.download('GE', start='2007-05-15', end='2021-10-01')

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = new_scatter_2period_.curve_roc_(class_=0, df=df, split=70, depth=3, tree=100, L=0.3)
    w.show()
    sys.exit(app.exec_())
```
**class candle_pyqt:**
Creates a candlestick chart with the ability to change the scale and scroll it. The "open_clicked" button creates a one-period increment with a shift of one.
Training is conducted during the training period and further classification using the "GradientBoostingClassifier". The green dot classifies the growth, the red dot classifies the fall, the bold (large) designation of the training period. Also, a window with a balance chart is added at the bottom, which is based on the inputs. The "close_clicked" button closes all graphical series except the candlestick chart.
parametr = [7, 50, 0.01]#parametr[0]-depth, parametr[1]-tree, parametr[2]-learning.
```
df = yf.download('GE', start='2007-05-15', end='2021-10-01')
parametr = [7, 50, 0.01]

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = new_scatter_2period_.candle_pyqt(df_=df, parametr_=parametr, split=70)
    w.show()
    sys.exit(app.exec_())
```
