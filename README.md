# Machine learning and graphics for financial markets and beyond.

The following code is used for machine learning (classification) and financial market charts.
A file is used for all called classes "scatter_2period_.py ". Price data is requested from yahoo.
The following libraries are required: PyQt5, numpy, sklearn, scipy, pandas_datareader.

![Visually, everything looks like this](https://github.com/quant12345/Machine-Learning-in-Finance/blob/980b7b23d86cad6019950e8c586983d6a88336d1/chart.gif)

The following modules are used to access classes:
```
from PyQt5 import QtWidgets
import scatter_2period_
import pandas_datareader.data as web
```

**class Main2Period:** 
creates two increments from a financial instrument with different and dynamically variable
periods with a shift of one.One increment on the x axis, the other on the y axis. If the value was growing, then the dot is green, the falling value
of the dot will be blue.
```
from PyQt5 import QtWidgets
import pandas_datareader.data as web
import scatter_2period_

df = web.DataReader('^GSPC', 'yahoo', start='2010-05-15', end='2021-10-01')

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = scatter_2period_.Main2Period(df_=df)
    w.show()
    sys.exit(app.exec_())
```
**class Main2Market:**
Similar to the previous version. But, in the second increment, another financial instrument is used. In this case, these are bonds(^TNX). The second increment
works much slower, since synchronization with the first increment is used. If the value was growing, then the point green, the falling
point value will be red.
```
df1 = web.DataReader('^GSPC', 'yahoo', start='2010-05-15', end='2021-10-01')
df2 = web.DataReader('^TNX', 'yahoo', start='2010-05-15', end='2021-10-01')

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = scatter_2period_.Main2Market(df1_=df1, df2_=df2)
    w.show()
    sys.exit(app.exec_())
```
**class curve_prec_rec:**
For classification, a one-period increment with a shift of one is used. The "GradientBoostingClassifier" model is used.
"decision_function()" is used to get the sum of points from "sklearn".By adjusting the "threshold" we get the optimal ratio of "precisions" and "recalls".
report=True(prints "classification_report"), split - how much in % training data, class_= 1 or class_= 0
```
if __name__ == "__main__":
   import sys

   app = QtWidgets.QApplication(sys.argv)
   w = scatter_2period_.curve_prec_rec(report=False, class_=1, df=df, split=70, depth=7, tree=100, L=0.01)
   w.show()
   sys.exit(app.exec_())
```
**class curve_roc_:**
For classification, a one-period increment with a shift of one is used. The "GradientBoostingClassifier" model is used.
By adjusting the slider, select the desired ratio of fpr and tpr. Split - how much in % training data, class_= 1 or class_= 0

```
df = web.DataReader('^GSPC', 'yahoo', start='2020-05-15', end='2021-10-01')

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = scatter_2period_.curve_roc_(class_=0, df=df, split=70, depth=3, tree=100, L=0.3)
    w.show()
    sys.exit(app.exec_())
```
**class candle_pyqt:**
Creates a candlestick chart with the ability to change the scale and scroll it. The "open_clicked" button creates a one-period increment with a shift of one.
Training is conducted during the training period and further classification using the "GradientBoostingClassifier". The green dot classifies the growth, the red dot classifies the fall, the bold (large) designation of the training period. Also, a window with a balance chart is added at the bottom, which is based on the inputs. The "close_clicked" button closes all graphical series except the candlestick chart.
parametr = [7, 50, 0.01]#parametr[0]-depth, parametr[1]-tree, parametr[2]-learning.
```
df = web.DataReader('^GSPC', 'yahoo', start='2010-05-15', end='2021-10-01')
parametr = [7, 50, 0.01]

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = scatter_2period_.candle_pyqt(df_=df, parametr_=parametr, split=70)
    w.show()
    sys.exit(app.exec_())
```
