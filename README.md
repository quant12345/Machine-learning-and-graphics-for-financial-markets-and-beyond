# Machine-Learning-in-Finance

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
```
if __name__ == "__main__":
   import sys

   app = QtWidgets.QApplication(sys.argv)
   w = scatter_2period_.curve_prec_rec(report=False, class_=1, df=df, split=70, depth=7, tree=100, L=0.01)#split-how much in % training data, class_=1 or class_=0
   w.show()
   sys.exit(app.exec_())
```

