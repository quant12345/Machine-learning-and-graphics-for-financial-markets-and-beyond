import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets, QtChart
from PyQt5.QtWidgets import (QHBoxLayout, QLabel)
from PyQt5.QtCore import *
from PyQt5.QtChart import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report, roc_curve
from scipy.ndimage.interpolation import shift
from sklearn import metrics
from talib import abstract


class Main2Period(QtWidgets.QMainWindow):
    def __init__(self, parent=None, df_=pd.DataFrame()):
        super().__init__(parent)
        self.df = df_
        self.x = len(self.df['Open'])
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        """
        Momentum(mom) indicators with period1=1 and period2=2 
        are initially selected
        """

        self.period1 = 1
        self.period2 = 2
        self.name1 = 'mom'
        self.name2 = 'mom'
        self.index1 = 20
        self.index2 = 20

        self.iplus = np.where(self.df['Close'] >= self.df['Open'])[0]
        self.iplus = np.delete(self.iplus, np.where(self.iplus == self.x - 1))
        # Indices where the closing price is greater than or equal to the opening

        self.iminus = np.where(self.df['Close'] < self.df['Open'])[0]
        self.iminus = np.delete(self.iminus, np.where(self.iminus == self.x - 1))
        # Indices where the closing price is less than the opening


        self.indicator = ['atr', 'natr', 'trange', 'typprice', 'wclprice', 'adx', 'adxr', 'cci', 'dx',
                          'minus_di', 'plus_di', 'willr', 'sma', 'ema', 'dema', 'kama', 'tema', 'trima', 'wma',
                          'cmo', 'mom', 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'trix', 'ht_trendline', 'ht_dcperiod',
                          'ht_dcphase', 'midpoint', 'linearreg', 'linearreg_angle', 'linearreg_intercept',
                          'linearreg_slope',
                          'tsf', 'atan', 'ceil', 'cos', 'exp', 'floor', 'ln', 'log10', 'sin', 'sinh', 'sqrt', 'tan',
                          'max', 'maxindex', 'min', 'minindex', 'sum']

        self._chart_view = QtChart.QChartView()

        self.button1 = QtWidgets.QPushButton('select indicator N 1')
        self.button1.clicked.connect(self.on_clicked1)  # Subscribing to the indicator selection event №1
        self.button2 = QtWidgets.QPushButton('select indicator N 2')
        self.button2.clicked.connect(self.on_clicked2)  # Subscribing to the indicator selection event №2

        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setRange(1, 150)
        self.slider1.setTickInterval(1)
        self.slider1.valueChanged.connect(self.updateLabel1)  # Subscribing to the indicator period selection event №1

        self.label1 = QLabel('period1=1', self)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumWidth(80)

        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider2.setRange(1, 150)
        self.slider2.setTickInterval(1)
        self.slider2.valueChanged.connect(self.updateLabel2)  # Subscribing to the indicator period selection event №2

        self.label2 = QLabel('period2=1', self)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumWidth(80)

        for w in (self.button1, self.slider1, self.label1):
            hbox1.addWidget(w)

        for w in (self.button2, self.slider2, self.label2):
            hbox2.addWidget(w)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        lay = QtWidgets.QVBoxLayout(central_widget)
        lay.addWidget(self._chart_view)
        lay.insertLayout(1, hbox1)
        lay.insertLayout(2, hbox2)

        self.array1 = np.zeros(self.x)
        self.array2 = np.zeros(self.x)
        self.adjust_axes()

    def updateLabel1(self, value):
        self.label1.setText('period1=' + str(value))
        self.data(1, value, self.name1, self.index1)
        self.Update(1)
        self.period1 = value

    def updateLabel2(self, value):
        self.label2.setText('period2=' + str(value))
        self.data(2, value, self.name2, self.index2)
        self.Update(2)
        self.period2 = value

    def data(self, type_, period, name, index):
        if index <= 11:
            ind = abstract.Function(name, timeperiod=period)
            try:
                arr = ind(self.df['High'], self.df['Low'], self.df['Close'])
            except:
                print(name, 'The indicator is not calculated. Set the period longer.')
                return
        else:
            ind = abstract.Function(name, timeperiod=period)
            try:
                arr = ind(self.df['Close'])
            except:
                print(name, 'The indicator is not calculated. Set the period longer.')
                return

        i = np.where(np.isnan(arr))[0]
        if len(i) > 0:
            i = i[-1]
            arr[:i + 1] = arr[i + 1]  # All empty values are replaced by the nearest non-empty value

        if type_ == 1:
            self.array1 = arr

        if type_ == 2:
            self.array2 = arr

    def adjust_axes(self):
        """
        Series are created here: self.iplus values of the previous (shift by one)
        indicator are taken, the same principle for self.iminus.
        """
        self.data(1, 1, self.name1, 20)
        self.data(2, 2, self.name2, 20)

        chart = QtChart.QChart()
        scatterU = QtChart.QScatterSeries()
        scatterD = QtChart.QScatterSeries()
        scatterU.setColor(Qt.darkGreen)#darkGreen
        scatterD.setColor(Qt.darkBlue)
        scatterU.setMarkerSize(10.0)
        scatterD.setMarkerSize(10.0)
        scatterU.setName('scatterU')
        scatterD.setName('scatterD')

        scatterU.append([QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]) for i in self.iplus])
        scatterD.append([QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]) for i in self.iminus])
        chart.addSeries(scatterU)
        chart.addSeries(scatterD)
        chart.createDefaultAxes()
        self._chart_view.setChart(chart)

    def Update(self, aaa):
        # Here the series is replaced with new indicator values and the scale is set.

        chart = self._chart_view.chart().series()
        su = 0
        sd = 0
        for serie in chart:
            if serie.name() == 'scatterU':
                su = serie
            if serie.name() == 'scatterD':
                sd = serie

        su.replace([QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]) for i in self.iplus])
        sd.replace([QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]) for i in self.iminus])
        if aaa == 1:
            maxX = np.max(self.array1)
            minX = np.min(self.array1)
            self._chart_view.chart().axisX(su).setRange(minX, maxX)
            self._chart_view.chart().axisX(sd).setRange(minX, maxX)
        if aaa == 2:
            maxY = np.max(self.array2)
            minY = np.min(self.array2)
            self._chart_view.chart().axisY(su).setRange(minY, maxY)
            self._chart_view.chart().axisY(sd).setRange(minY, maxY)

        self._chart_view.chart().show()

    def on_clicked1(self):
        dialog = QtWidgets.QInputDialog()
        dialog.setWindowTitle('select indicator')
        dialog.setLabelText('select an indicator from the drop-down menu')
        dialog.setOkButtonText('&Enter')
        dialog.setCancelButtonText('&Cancel')

        dialog.setOption(QtWidgets.QInputDialog.UseListViewForComboBoxItems,
                         on=True)
        dialog.setComboBoxItems(self.indicator)
        dialog.setComboBoxEditable(True)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self.name1 = dialog.textValue()
            self.index1 = dialog.comboBoxItems().index(dialog.textValue())
            self.updateLabel1(self.period1)

    def on_clicked2(self):
        dialog = QtWidgets.QInputDialog()
        dialog.setWindowTitle('select indicator')
        dialog.setLabelText('select an indicator from the drop-down menu')
        dialog.setOkButtonText('&Enter')
        dialog.setCancelButtonText('&Cancel')

        dialog.setOption(QtWidgets.QInputDialog.UseListViewForComboBoxItems,
                         on=True)
        dialog.setComboBoxItems(self.indicator)
        dialog.setComboBoxEditable(True)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self.name2 = dialog.textValue()
            self.index2 = dialog.comboBoxItems().index(dialog.textValue())
            self.updateLabel2(self.period2)



class Main2Market(QtWidgets.QMainWindow):
    def __init__(self, parent=None, df_=pd.DataFrame(), df1_=pd.DataFrame(), cl1_='Close', cl2_='Close', timeframe_='D'):
        super().__init__(parent)
        self.df = df_
        self.df1 = df1_
        self.cl1 = cl1_
        self.cl2 = cl2_
        self.timeframe = timeframe_
        self.x = len(self.df['Open'])
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        """
        Momentum(mom) indicators with period1=1 and period2=2 
        are initially selected
        """

        self.period1 = 1
        self.period2 = 2
        self.name1 = 'mom'
        self.name2 = 'mom'
        self.index1 = 12
        self.index2 = 12

        start = self.df.index[0]
        stop = self.df.index[-1]
        self.df1 = self.df1[(self.df1.index >= start) & (self.df1.index <= stop)].copy()
        #select the necessary rows by datetime based on the self.df range.
        self.df2 = self.df1.reindex(pd.date_range(start, stop, freq=self.timeframe))
        #Expand self.df1 indexes.
        self.iplus = np.where(self.df['Close'] >= self.df['Open'])[0]
        self.iplus = np.delete(self.iplus, np.where(self.iplus == self.x - 1))
        # Indices where the closing price is greater than or equal to the opening

        self.iminus = np.where(self.df['Close'] < self.df['Open'])[0]
        self.iminus = np.delete(self.iminus, np.where(self.iminus == self.x - 1))
        # Indices where the closing price is less than the opening

        self.indicator = ['sma', 'ema', 'dema', 'kama', 'tema', 'trima', 'wma',
                          'cmo', 'roc', 'rocp', 'rocr', 'rocr100', 'mom', 'rsi', 'trix', 'ht_trendline', 'ht_dcperiod',
                          'ht_dcphase', 'midpoint', 'linearreg', 'linearreg_angle', 'linearreg_intercept',
                          'linearreg_slope',
                          'tsf', 'atan', 'ceil', 'cos', 'exp', 'floor', 'ln', 'log10', 'sin', 'sinh', 'sqrt', 'tan',
                          'max', 'maxindex', 'min', 'minindex', 'sum']

        self._chart_view = QtChart.QChartView()

        self.button1 = QtWidgets.QPushButton('select indicator N 1')
        self.button1.clicked.connect(self.on_clicked1)  # Subscribing to the indicator selection event №1
        self.button2 = QtWidgets.QPushButton('select indicator N 2')
        self.button2.clicked.connect(self.on_clicked2)  # Subscribing to the indicator selection event №2

        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setRange(1, 150)
        self.slider1.setTickInterval(1)
        self.slider1.valueChanged.connect(self.updateLabel1)  # Subscribing to the indicator period selection event №1

        self.label1 = QLabel('period1=1', self)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumWidth(80)

        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider2.setRange(1, 150)
        self.slider2.setTickInterval(1)
        self.slider2.valueChanged.connect(self.updateLabel2)  # Subscribing to the indicator period selection event №2

        self.label2 = QLabel('period2=1', self)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumWidth(80)

        for w in (self.button1, self.slider1, self.label1):
            hbox1.addWidget(w)

        for w in (self.button2, self.slider2, self.label2):
            hbox2.addWidget(w)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        lay = QtWidgets.QVBoxLayout(central_widget)
        lay.addWidget(self._chart_view)
        lay.insertLayout(1, hbox1)
        lay.insertLayout(2, hbox2)

        self.array1 = np.zeros(self.x)
        self.array2 = np.zeros(self.x)
        self.adjust_axes()

    def updateLabel1(self, value):
        self.label1.setText('period1=' + str(value))
        self.data(1, value, self.name1, self.index1)
        self.Update(1)
        self.period1 = value

    def updateLabel2(self, value):
        self.label2.setText('period2=' + str(value))
        self.data(2, value, self.name2, self.index2)
        self.Update(2)
        self.period2 = value

    def data(self, type_, period, name, index):
        if type_ == 1:
            ind = abstract.Function(name, timeperiod=period)
            try:
                arr = ind(self.df[self.cl1])
            except:
                print(name, 'The indicator is not calculated. Set the period longer.')
                return

            i = np.where(np.isnan(arr))[0]
            if len(i) > 0:
                i = i[-1]
                arr[:i + 1] = arr[i + 1]  # All empty values are replaced by the nearest non-empty value

            self.array1 = arr

        if type_ == 2:
            ind = abstract.Function(name, timeperiod=period)
            try:
                arr = ind(self.df1[self.cl2])
            except:
                print(name, 'The indicator is not calculated. Set the period longer.')
                return

            i = np.where(np.isnan(arr))[0]
            if len(i) > 0:
                i = i[-1]
                arr[:i + 1] = arr[i + 1]  # All empty values are replaced by the nearest non-empty value

            self.df2[self.cl2] = np.nan
            self.df2.loc[self.df1.index, self.cl2] = arr
            self.df2 = self.df2.fillna(method='ffill').fillna(method='bfill')

            self.array2 = self.df2.loc[self.df.index, self.cl2]


    def adjust_axes(self):
        """
        Series are created here: self.iplus values of the previous (shift by one)
        indicator are taken, the same principle for self.iminus.
        """
        self.data(1, 1, self.name1, 12)
        self.data(2, 2, self.name2, 12)

        chart = QtChart.QChart()
        scatterU = QtChart.QScatterSeries()
        scatterD = QtChart.QScatterSeries()
        scatterU.setColor(Qt.darkGreen)
        scatterD.setColor(Qt.red)
        scatterU.setMarkerSize(10.0)
        scatterD.setMarkerSize(10.0)
        scatterU.setName('scatterU')
        scatterD.setName('scatterD')

        scatterU.append([QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]) for i in self.iplus])
        scatterD.append([QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]) for i in self.iminus])
        chart.addSeries(scatterU)
        chart.addSeries(scatterD)
        chart.createDefaultAxes()
        self._chart_view.setChart(chart)

    def Update(self, aaa):
        # Here the series is replaced with new indicator values and the scale is set.

        chart = self._chart_view.chart().series()
        su = 0
        sd = 0
        for serie in chart:
            if serie.name() == 'scatterU':
                su = serie
            if serie.name() == 'scatterD':
                sd = serie

        su.replace([QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]) for i in self.iplus])
        sd.replace([QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]) for i in self.iminus])
        if aaa == 1:
            maxX = np.max(self.array1)
            minX = np.min(self.array1)
            self._chart_view.chart().axisX(su).setRange(minX, maxX)
            self._chart_view.chart().axisX(sd).setRange(minX, maxX)
        if aaa == 2:
            maxY = np.max(self.array2)
            minY = np.min(self.array2)
            self._chart_view.chart().axisY(su).setRange(minY, maxY)
            self._chart_view.chart().axisY(sd).setRange(minY, maxY)

        self._chart_view.chart().show()

    def on_clicked1(self):
        dialog = QtWidgets.QInputDialog()
        dialog.setWindowTitle('select indicator')
        dialog.setLabelText('select an indicator from the drop-down menu')
        dialog.setOkButtonText('&Enter')
        dialog.setCancelButtonText('&Cancel')

        dialog.setOption(QtWidgets.QInputDialog.UseListViewForComboBoxItems,
                         on=True)
        dialog.setComboBoxItems(self.indicator)
        dialog.setComboBoxEditable(True)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self.name1 = dialog.textValue()
            self.index1 = dialog.comboBoxItems().index(dialog.textValue())
            self.updateLabel1(self.period1)

    def on_clicked2(self):
        dialog = QtWidgets.QInputDialog()
        dialog.setWindowTitle('select indicator')
        dialog.setLabelText('select an indicator from the drop-down menu')
        dialog.setOkButtonText('&Enter')
        dialog.setCancelButtonText('&Cancel')

        dialog.setOption(QtWidgets.QInputDialog.UseListViewForComboBoxItems,
                         on=True)
        dialog.setComboBoxItems(self.indicator)
        dialog.setComboBoxEditable(True)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self.name2 = dialog.textValue()
            self.index2 = dialog.comboBoxItems().index(dialog.textValue())
            self.updateLabel2(self.period2)



class curve_prec_rec(QtWidgets.QMainWindow):
    def __init__(self, parent=None, report=False, class_=1, df=0, split=70, depth=5, tree=50, L=0.0):
        super().__init__(parent)
        self.report = report
        self.class_ = class_
        self.df = df
        close = self.df['Close'].values
        open = self.df['Open'].values
        x = len(close)
        self.split = int((x/100.0)*split)

        array = np.zeros(x)
        array[1:] = (close[1:] - close[:x - 1]) / close[:x - 1]
        array[:1] = 0.0
        array = shift(array, 1, cval=0.0)

        X = array.reshape(-1, 1)
        if self.class_ == 1:
            self.y = np.where(close > open, 1, 0)
        else:
            self.y = np.where(close > open, 0, 1)

        clf = GradientBoostingClassifier(max_depth=depth, max_features="sqrt", n_estimators=tree,
                                         learning_rate=L, random_state=0)
        clf.fit(X[:self.split], self.y[:self.split])
        self.y_scores_test = clf.decision_function(X[self.split:])

        self.precisions, self.recalls, self.thresholds = precision_recall_curve(self.y[self.split:], self.y_scores_test)

        hbox1 = QHBoxLayout()
        self._chart_view = QtChart.QChartView()

        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setRange(0, len(self.thresholds) - 2)
        self.slider1.valueChanged.connect(self.updateLabel)

        self.label1 = QLabel(self)
        self.label1.setAlignment(Qt.AlignLeft)
        self.label1.setMinimumWidth(80)

        hbox1.addWidget(self.slider1)
        hbox1.addWidget(self.label1)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        lay = QtWidgets.QVBoxLayout(central_widget)
        lay.addWidget(self._chart_view)
        lay.insertLayout(1, hbox1)

        chart = QtChart.QChart()
        scatter = QtChart.QScatterSeries()
        scatter.setMarkerSize(12.0)
        line_precisions = QtChart.QLineSeries()
        line_recalls = QtChart.QLineSeries()
        line_precisions.setName('precisions')  # precisions
        line_recalls.setName('recalls')
        scatter.setName('scatter')

        for i in range(0, len(self.precisions) - 1):
            line_precisions.append(QtCore.QPointF(self.thresholds[i], self.precisions[i + 1]))
            line_recalls.append(QtCore.QPointF(self.thresholds[i], self.recalls[i + 1]))


        scatter.append(self.thresholds[0], self.recalls[0])
        scatter.append(self.thresholds[0], self.precisions[0])
        chart.addSeries(scatter)
        chart.addSeries(line_precisions)
        chart.addSeries(line_recalls)
        chart.createDefaultAxes()
        self._chart_view.setChart(chart)

    def updateLabel(self, value):
        u = []
        u.append(QtCore.QPointF(self.thresholds[value], self.recalls[value + 1]))
        u.append(QtCore.QPointF(self.thresholds[value], self.precisions[value + 1]))
        chart = self._chart_view.chart().series()
        for serie in chart:
            if serie.name() == 'scatter':
                serie.replace(u)

        y_ = np.where(self.y_scores_test > self.thresholds[value], 1, 0)
        con = confusion_matrix(self.y[self.split:], y_)
        if self.report:
            print(classification_report(self.y[self.split:], y_))

        pr = (con[0][0] / (con[0][0] + con[1][0]), con[1][1] / (con[0][1] + con[1][1]))
        rec = (con[0][0] / (con[0][0] + con[0][1]), con[1][1] / (con[1][0] + con[1][1]))
        self.label1.setText('thresholds=' + str(round(self.thresholds[value], 5)) + '\n'
                            + 'precision 0=' + str(round(pr[0], 2)) + '\n' + 'precision 1=' + str(round(pr[1], 2))
                            + '\n' + 'recalls 0=' + str(round(rec[0], 2)) + '\n' + 'recalls 1=' + str(round(rec[1], 2)))


class candle_pyqt(QtWidgets.QMainWindow):
    def __init__(self, parent=None, df_=0, parametr_=0, split=0):
        super().__init__(parent)
        self.df = df_
        self.x = len(self.df['Close'].values)
        self.x_ = self.x - 1
        self.step = int((self.x / 100.0) * 0.5)
        self.split = int((self.x / 100) * split)
        self.parametr = parametr_
        self._chart_view = QtChart.QChartView()

        self.open = QtWidgets.QPushButton()
        self.open.setText("Open")
        self.open.setMaximumWidth(70)
        self.open.clicked.connect(self.open_clicked)

        self.close = QtWidgets.QPushButton()
        self.close.setText("Close")
        self.close.setMaximumWidth(70)
        self.close.clicked.connect(self.close_clicked)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal,
                                              sliderMoved=self.AxisSlider,
                                              pageStep=self.step,
                                              )
        self.scrollbar.setStyleSheet("background : silver;")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal,
                                        sliderMoved=self.ZoomSlider)
        self.slider.setStyleSheet("background : silver;")

        self.scrollbar.setRange(0, self.x_)
        self.slider.setRange(1, 300)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.open)
        hbox.addWidget(self.close)
        hbox.setAlignment(Qt.AlignLeft)

        lay = QtWidgets.QVBoxLayout(central_widget)
        lay.insertLayout(0, hbox)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        for w in (self._chart_view, self.splitter, self.scrollbar, self.slider):
            lay.addWidget(w)

        self.resize(640, 480)

        self._chart = QtChart.QChart()

        self._candlestick_serie = QtChart.QCandlestickSeries()
        self._candlestick_serie.setDecreasingColor(Qt.red)
        self._candlestick_serie.setIncreasingColor(Qt.green)

        for i in range(0, self.x):
            o_ = self.df['Open'].values[i]
            h_ = self.df['High'].values[i]
            l_ = self.df['Low'].values[i]
            c_ = self.df['Close'].values[i]
            self._candlestick_serie.append(QtChart.QCandlestickSet(o_, h_, l_, c_, float(i)))

        self._chart.addSeries(self._candlestick_serie)

        axisX = QValueAxis()
        axisX.setTickCount(5)
        self._chart.addAxis(axisX, Qt.AlignBottom)
        self._candlestick_serie.attachAxis(axisX)

        axisY = QValueAxis()
        self._chart.addAxis(axisY, Qt.AlignLeft)
        self._candlestick_serie.attachAxis(axisY)

        self.splitter.addWidget(self._chart_view)
        self._chart_view.setChart(self._chart)
        self.AxisSlider(self.scrollbar.value())

    def adjust_axes(self, value_min, value_max):  # свечной график тип 12, линейный 0, скатер 6
        if value_min >= 0 and value_max > 0 and value_max > value_min:
            ymin = np.amin(self.df['Low'].values[int(value_min): int(value_max)])
            ymax = np.amax(self.df['High'].values[int(value_min): int(value_max)])
            for i in range(0, self.splitter.count()):
                chart_view = self.splitter.widget(i)
                if isinstance(chart_view, QtChart.QChartView):
                    chart = chart_view.chart()
                    for serie in chart.series():
                        if serie.type() != 0:
                            self._chart.axisX(serie).setRange(value_min, value_max)
                            self._chart.axisY(serie).setRange(ymin, ymax)
                        if serie.type() == 0:
                            ymin = np.amin(self.depo[int(value_min): int(value_max)])
                            ymax = np.amax(self.depo[int(value_min): int(value_max)])
                            self._chart.axisX(serie).setRange(value_min, value_max)
                            self._chart.axisY(serie).setRange(ymin, ymax)

    def AxisSlider(self, value):
        value2 = value + self.step
        value1 = value
        if value2 >= self.x_:
            value2 = self.x_
            value1 = value2 - self.step
        self.adjust_axes(int(value1), int(value2))

    def ZoomSlider(self, value):
        self.step = int(self.x_ / value)
        self.AxisSlider(self.scrollbar.value())

    def open_clicked(self):
        chart_view = self.splitter.widget(0)
        chart = chart_view.chart()
        r = len(chart.series())
        if r < 2:
            y_pred = pred_class(self.df, self.parametr, self.split)
            x = len(self.df['Close'].values)
            scatterU = QtChart.QScatterSeries()
            scatterD = QtChart.QScatterSeries()
            scatter = QtChart.QScatterSeries()

            for i in range(0, x):
                if y_pred[i] == 0:
                    scatterU.append(float(i), self.df['Low'].values[i])
                else:
                    scatterD.append(float(i), self.df['High'].values[i])

            scatter.append(self.split, (self.df['High'].values[self.split] + self.df['Low'].values[self.split])/2.0)
            scatter.setMarkerSize(30.0)
            scatterU.setColor(Qt.darkGreen)
            scatterD.setColor(Qt.red)

            chart.addSeries(scatterU)
            chart.addSeries(scatterD)
            chart.addSeries(scatter)
            chart_view.setChart(chart)
            self.splitter.addWidget(chart_view)

            axisX = QValueAxis()
            chart.addAxis(axisX, Qt.AlignBottom)
            scatterU.attachAxis(axisX)

            axisX = QValueAxis()
            chart.addAxis(axisX, Qt.AlignBottom)
            scatterD.attachAxis(axisX)

            axisX = QValueAxis()
            chart.addAxis(axisX, Qt.AlignBottom)
            scatter.attachAxis(axisX)

            axisY = QValueAxis()
            chart.addAxis(axisY, Qt.AlignLeft)
            scatterU.attachAxis(axisY)

            axisY = QValueAxis()
            chart.addAxis(axisY, Qt.AlignLeft)
            scatterD.attachAxis(axisY)

            axisY = QValueAxis()
            chart.addAxis(axisY, Qt.AlignLeft)
            scatter.attachAxis(axisY)

            chart.axisX(scatterU).setVisible(False)
            chart.axisY(scatterU).setVisible(False)
            chart.axisX(scatterD).setVisible(False)
            chart.axisY(scatterD).setVisible(False)
            chart.axisX(scatter).setVisible(False)
            chart.axisY(scatter).setVisible(False)

            chart_view = QtChart.QChartView()
            chart = QtChart.QChart()
            line_serie = QtChart.QLineSeries()
            self.depo = balance(y_pred, self.df['Open'].values, x)

            for i in range(0, x):
                line_serie.append(QtCore.QPointF(i, self.depo[i]))

            chart.addSeries(line_serie)
            chart.createDefaultAxes()
            chart.legend().hide()
            chart_view.setChart(chart)
            self.splitter.addWidget(chart_view)
            self.AxisSlider(self.scrollbar.value())

    def close_clicked(self):
        chart_view = self.splitter.widget(0)
        chart = chart_view.chart()
        r = chart.series()
        if len(r) >= 2:
            for i in range(1, len(r)):
                delete = r[i]
                delete.deleteLater()

        count = self.splitter.count()
        if count > 1:
            w = self.splitter.widget(count - 1)
            if w is not None:
                w.deleteLater()


def balance(y, o, x):
    depozit = []
    depozit.append(1000.0)
    buy = 0
    sell = 0
    for i in range(1, x):
        perekl = 0
        if buy == 0.0 and sell == 0:
            depozit.append(depozit[len(depozit) - 1])
            if y[i] == 0:
                buy = o[i]
                perekl = 1
            if y[i] == 1:
                sell = o[i]
                perekl = 1

        if buy != 0.0:
            if y[i] == 0 and perekl == 0:
                depozit.append(depozit[len(depozit) - 1])
            if y[i] == 1 or i == (x - 1):
                depozit.append(depozit[len(depozit) - 1] + (o[i] - buy))
                buy = 0.0

        if sell != 0.0:
            if y[i] == 1 and perekl == 0:
                depozit.append(depozit[len(depozit) - 1])
            if y[i] == 0 or i == (x - 1):
                depozit.append(depozit[len(depozit) - 1] + (sell - o[i]))
                sell = 0.0

    return depozit


def pred_class(df, parametr, split):
    close = df['Close'].values
    open = df['Open'].values
    x = len(close)
    array = np.zeros(x)
    array[1:] = (close[1:] - close[:x - 1]) / close[:x - 1]
    array[:1] = 0.0
    array = shift(array, 1, cval=0.0)

    X = array.reshape(-1, 1)
    y = np.where(close > open, 0, 1)
    clf = GradientBoostingClassifier(max_depth=parametr[0], max_features="sqrt", n_estimators=parametr[1],
                                     learning_rate=parametr[2], random_state=0)
    clf.fit(X[:split], y[:split])
    y_pred = clf.predict(X)

    return y_pred


class curve_roc_(QtWidgets.QMainWindow):
    def __init__(self, parent=None, class_=1, df=0, split=0, depth=0, tree=0, L=0.0):
        super().__init__(parent)
        self.class_ = class_
        self.df = df
        close = self.df['Close'].values
        open = self.df['Open'].values
        x = len(close)
        self.split = int((x/100.0)*split)

        array = np.zeros(x)
        array[1:] = (close[1:] - close[:x - 1]) / close[:x - 1]
        array[:1] = 0.0
        array = shift(array, 1, cval=0.0)

        X = array.reshape(-1, 1)
        if self.class_ == 1:
            self.y = np.where(close > open, 1, 0)
        else:
            self.y = np.where(close > open, 0, 1)

        clf = GradientBoostingClassifier(max_depth=depth, max_features="sqrt", n_estimators=tree,
                                         learning_rate=L, random_state=0)
        clf.fit(X[:self.split], self.y[:self.split])
        self.y_scores_test = clf.decision_function(X[self.split:])

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.y[self.split:], self.y_scores_test)

        hbox1 = QHBoxLayout()
        self._chart_view = QtChart.QChartView()

        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setRange(0, len(self.thresholds) - 1)
        self.slider1.valueChanged.connect(self.updateLabel)

        self.label1 = QLabel(self)
        self.label1.setAlignment(Qt.AlignLeft)
        self.label1.setMinimumWidth(80)

        hbox1.addWidget(self.slider1)
        hbox1.addWidget(self.label1)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        lay = QtWidgets.QVBoxLayout(central_widget)
        lay.addWidget(self._chart_view)
        lay.insertLayout(1, hbox1)

        chart = QtChart.QChart()
        scatter = QtChart.QScatterSeries()
        scatter.setMarkerSize(12.0)
        line = QtChart.QLineSeries()
        line.setName('line')
        scatter.setName('scatter')

        for i in range(0, len(self.fpr)):
            line.append(QtCore.QPointF(self.fpr[i], self.tpr[i]))


        scatter.append(self.fpr[0], self.tpr[0])
        chart.addSeries(scatter)
        chart.addSeries(line)
        chart.createDefaultAxes()
        self._chart_view.setChart(chart)

    def updateLabel(self, value):
        u = []
        u.append(QtCore.QPointF(self.fpr[value], self.tpr[value]))
        chart = self._chart_view.chart().series()
        for serie in chart:
            if serie.name() == 'scatter':
                serie.replace(u)

        self.label1.setText('fpr=' + str(round(self.fpr[value], 5)) + '\n'
                            + 'tpr=' + str(round(self.tpr[value], 5)))