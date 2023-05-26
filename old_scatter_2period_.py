import numpy as np
from PyQt5 import QtCore, QtWidgets, QtChart
from PyQt5.QtWidgets import (QHBoxLayout, QLabel)
from PyQt5.QtCore import *
from PyQt5.QtChart import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.ndimage.interpolation import shift
from sklearn import metrics


class Main2Period(QtWidgets.QMainWindow):
    def __init__(self, parent=None, df_=0):
        super().__init__(parent)
        self.df = df_
        self.close = self.df['Close'].values
        self.open = self.df['Open'].values
        self.x = len(self.close)
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()

        self._chart_view = QtChart.QChartView()

        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setRange(1, 150)
        self.slider1.setTickInterval(1)
        self.slider1.valueChanged.connect(self.updateLabel1)

        self.label1 = QLabel('period1=1', self)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumWidth(80)

        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider2.setRange(2, 150)
        self.slider2.setTickInterval(1)
        self.slider2.valueChanged.connect(self.updateLabel2)

        self.label2 = QLabel('period2=2', self)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumWidth(80)

        hbox1.addWidget(self.slider1)
        hbox1.addWidget(self.label1)
        hbox2.addWidget(self.slider2)
        hbox2.addWidget(self.label2)

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
        self.data(1, value)
        self.Update(1)

    def updateLabel2(self, value):
        self.label2.setText('period2=' + str(value))
        self.data(2, value)
        self.Update(2)

    def data(self, type_, period):
        if type_ == 1:
            self.array1[period:] = (self.close[period:] - self.close[:self.x - period]) / self.close[:self.x - period]
            self.array1[:period] = 0.0
        if type_ == 2:
            self.array2[period:] = (self.close[period:] - self.close[:self.x - period]) / self.close[:self.x - period]
            self.array2[:period] = 0.0

    def adjust_axes(self):
        self.data(1, 1)
        self.data(2, 2)

        chart = QtChart.QChart()
        scatterU = QtChart.QScatterSeries()
        scatterD = QtChart.QScatterSeries()
        scatterU.setColor(Qt.darkGreen)
        scatterD.setColor(Qt.darkBlue)
        scatterU.setMarkerSize(10.0)
        scatterD.setMarkerSize(10.0)
        scatterU.setName('scatterU')
        scatterD.setName('scatterD')

        u = []
        d = []
        for i in range(0, self.x):
            if self.close[i] >= self.open[i]:
                u.append(QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]))
            if self.close[i] < self.open[i]:
                d.append(QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]))

        scatterU.append(u)
        scatterD.append(d)
        chart.addSeries(scatterU)
        chart.addSeries(scatterD)
        chart.createDefaultAxes()
        self._chart_view.setChart(chart)

    def Update(self, zn):
        chart = self._chart_view.chart().series()
        su = 0
        sd = 0
        for serie in chart:
            if serie.name() == 'scatterU':
                su = serie
            if serie.name() == 'scatterD':
                sd = serie

        u = []
        d = []
        for i in range(0, self.x):
            if self.close[i] >= self.open[i]:
                u.append(QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]))
            if self.close[i] < self.open[i]:
                d.append(QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]))

        su.replace(u)
        sd.replace(d)
        if zn == 1:
            maxX = np.max(self.array1)
            minX = np.min(self.array1)
            self._chart_view.chart().axisX(su).setRange(minX, maxX)
            self._chart_view.chart().axisX(sd).setRange(minX, maxX)
        if zn == 2:
            maxY = np.max(self.array2)
            minY = np.min(self.array2)
            self._chart_view.chart().axisY(su).setRange(minY, maxY)
            self._chart_view.chart().axisY(sd).setRange(minY, maxY)

        self._chart_view.chart().show()


def syncing(n, dv, dn):
    N = len(dv)
    N1 = len(dn)
    z = np.zeros(N)
    x = N1 - 1
    global u
    u = 0
    ferst_index = 500000

    for q in range(0, N):
        k = u
        for i in range(k, N1):
            if dn[i - 1] <= dv[q] and dn[i] > dv[q]:
                u = i
                z[q] = n[i - 1]
                if ferst_index == 500000:
                    ferst_index = q
                break
            if x == i and dv[q] >= dn[i]:
                u = i
                z[q] = n[i]
                break

    if ferst_index >= 1:
        z[:ferst_index] = z[ferst_index]

    return z


class Main2Market(QtWidgets.QMainWindow):
    def __init__(self, parent=None, df1_=0, df2_=0):
        super().__init__(parent)
        self.df1 = df1_
        self.df2 = df2_
        self.close1 = self.df1['Close'].values
        self.open1 = self.df1['Open'].values
        self.time1 = self.df1.index
        self.close2 = self.df2['Close'].values
        self.time2 = self.df2.index
        self.x = len(self.close1)
        self.x_ = len(self.close2)
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()

        self._chart_view = QtChart.QChartView()

        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setRange(1, 150)
        self.slider1.setTickInterval(1)
        self.slider1.valueChanged.connect(self.updateLabel1)

        self.label1 = QLabel('period1=1', self)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumWidth(80)

        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider2.setRange(2, 150)
        self.slider2.setTickInterval(1)
        self.slider2.valueChanged.connect(self.updateLabel2)

        self.label2 = QLabel('period2=2', self)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumWidth(80)

        hbox1.addWidget(self.slider1)
        hbox1.addWidget(self.label1)
        hbox2.addWidget(self.slider2)
        hbox2.addWidget(self.label2)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        lay = QtWidgets.QVBoxLayout(central_widget)
        lay.addWidget(self._chart_view)
        lay.insertLayout(1, hbox1)
        lay.insertLayout(2, hbox2)

        self.array1 = np.zeros(self.x)
        self.array2 = np.zeros(self.x)
        self.array = np.zeros(self.x_)
        self.adjust_axes()

    def updateLabel1(self, value):
        self.label1.setText('period1=' + str(value))
        self.data(1, value)
        self.Update(1)

    def updateLabel2(self, value):
        self.label2.setText('period2=' + str(value))
        self.data(2, value)
        self.Update(2)

    def data(self, type_, period):
        if type_ == 1:
            self.array1[period:] = (self.close1[period:] - self.close1[:self.x - period]) / self.close1[:self.x - period]
            self.array1[:period] = 0.0
        if type_ == 2:
            self.array[period:] = (self.close2[period:] - self.close2[:self.x_ - period]) / self.close2[:self.x_ - period]
            self.array[:period] = 0.0
            self.array2 = syncing(self.array, self.time1, self.time2)

    def adjust_axes(self):
        self.data(1, 1)
        self.data(2, 2)

        chart = QtChart.QChart()
        scatterU = QtChart.QScatterSeries()
        scatterD = QtChart.QScatterSeries()
        scatterU.setColor(Qt.darkGreen)
        scatterD.setColor(Qt.red)
        scatterU.setMarkerSize(10.0)
        scatterD.setMarkerSize(10.0)
        scatterU.setName('scatterU')
        scatterD.setName('scatterD')

        u = []
        d = []
        for i in range(0, self.x):
            if self.close1[i] >= self.open1[i]:
                u.append(QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]))
            if self.close1[i] < self.open1[i]:
                d.append(QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]))

        scatterU.append(u)
        scatterD.append(d)
        chart.addSeries(scatterU)
        chart.addSeries(scatterD)
        chart.createDefaultAxes()
        self._chart_view.setChart(chart)

    def Update(self, zn):
        chart = self._chart_view.chart().series()
        su = 0
        sd = 0
        for serie in chart:
            if serie.name() == 'scatterU':
                su = serie
            if serie.name() == 'scatterD':
                sd = serie

        u = []
        d = []
        for i in range(0, self.x):
            if self.close1[i] >= self.open1[i]:
                u.append(QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]))
            if self.close1[i] < self.open1[i]:
                d.append(QtCore.QPointF(self.array1[i - 1], self.array2[i - 1]))

        su.replace(u)
        sd.replace(d)
        if zn == 1:
            maxX = np.max(self.array1)
            minX = np.min(self.array1)
            self._chart_view.chart().axisX(su).setRange(minX, maxX)
            self._chart_view.chart().axisX(sd).setRange(minX, maxX)
        if zn == 2:
            maxY = np.max(self.array2)
            minY = np.min(self.array2)
            self._chart_view.chart().axisY(su).setRange(minY, maxY)
            self._chart_view.chart().axisY(sd).setRange(minY, maxY)

        self._chart_view.chart().show()


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