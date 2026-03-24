import numpy as np
from qtpy import QtGui, QtCore, QtWidgets


__all__ = ['backgroundAdvDiagnostics', 'plotAdvDiagnostics']


class backgroundAdvDiagnostics(QtWidgets.QFrame):
    '''
    Draw the background frame for the advanced diagnostics inset.
    '''
    def __init__(self):
        super(backgroundAdvDiagnostics, self).__init__()
        self.initUI()

    def initUI(self):
        self.setStyleSheet("QFrame {"
            "  background-color: rgb(0, 0, 0);"
            "  border-width: 1px;"
            "  border-style: solid;"
            "  border-color: #3399CC;}")

        self.lpad = 6
        self.rpad = 6
        self.tpad = 4
        self.bpad = 4
        self.wid = self.size().width()
        self.hgt = self.size().height()
        self.brx = self.wid - self.rpad
        self.bry = self.hgt - self.bpad

        font_size = max(8, round(self.hgt * 0.05))
        title_size = max(10, font_size + 2)
        self.title_font = QtGui.QFont('Helvetica', title_size)
        self.label_font = QtGui.QFont('Helvetica', font_size)
        self.value_font = QtGui.QFont('Helvetica', font_size)
        self.value_font.setBold(True)

        self.title_metrics = QtGui.QFontMetrics(self.title_font)
        self.label_metrics = QtGui.QFontMetrics(self.label_font)
        self.title_height = self.title_metrics.height()
        self.label_height = self.label_metrics.height()
        self.row_spacing = max(4, round(self.hgt * 0.03))

        self.plotBitMap = QtGui.QPixmap(self.width() - 2, self.height() - 2)
        self.plotBitMap.fill(self.bg_color)
        self.plotBackground()

    def resizeEvent(self, e):
        self.initUI()

    def draw_frame(self, qp):
        pen = QtGui.QPen(self.fg_color, 1, QtCore.Qt.SolidLine)
        qp.setPen(pen)

        qp.setFont(self.title_font)
        title_rect = QtCore.QRect(0, self.tpad, self.wid, self.title_height)
        qp.drawText(title_rect, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, 'Advanced Diagnostics')

        header_y = self.tpad + self.title_height + self.row_spacing
        qp.drawLine(0, header_y, self.wid, header_y)

    def plotBackground(self):
        qp = QtGui.QPainter()
        qp.begin(self.plotBitMap)
        self.draw_frame(qp)
        qp.end()


class plotAdvDiagnostics(backgroundAdvDiagnostics):
    '''
    Plot the advanced diagnostics values.
    '''
    def __init__(self):
        self.bg_color = QtGui.QColor('#000000')
        self.fg_color = QtGui.QColor('#ffffff')
        self.use_left = False
        self.prof = None
        self.rows = []
        super(plotAdvDiagnostics, self).__init__()

    def _get_values(self):
        if self.prof is None:
            return []

        prefix = 'left' if self.use_left else 'right'
        rows = [
            ('SFC-100m SRH', getattr(self.prof, '%s_srh100m' % prefix)[0]),
            ('SFC-250m SRH', getattr(self.prof, '%s_srh250m' % prefix)[0]),
            ('SFC-500m SRH', getattr(self.prof, '%s_srh500m' % prefix)[0]),
        ]
        return rows

    def _format_value(self, value):
        if np.ma.isMaskedArray(value):
            value = value.item(0) if value.size == 1 else value
        if value is np.ma.masked or np.ma.is_masked(value) or not np.isfinite(value):
            return 'M'
        return str(int(np.rint(value)))

    def setProf(self, prof):
        self.prof = prof
        self.rows = self._get_values()
        self.clearData()
        self.plotBackground()
        self.plotData()
        self.update()

    def setPreferences(self, update_gui=True, **prefs):
        self.bg_color = QtGui.QColor(prefs['bg_color'])
        self.fg_color = QtGui.QColor(prefs['fg_color'])

        if update_gui and self.prof is not None:
            self.rows = self._get_values()
            self.clearData()
            self.plotBackground()
            self.plotData()
            self.update()

    def setDeviant(self, deviant):
        self.use_left = deviant == 'left'
        if self.prof is not None:
            self.rows = self._get_values()
            self.clearData()
            self.plotBackground()
            self.plotData()
            self.update()

    def resizeEvent(self, e):
        super(plotAdvDiagnostics, self).resizeEvent(e)
        self.plotData()

    def paintEvent(self, e):
        super(plotAdvDiagnostics, self).paintEvent(e)
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.drawPixmap(1, 1, self.plotBitMap)
        qp.end()

    def clearData(self):
        self.plotBitMap = QtGui.QPixmap(self.width() - 2, self.height() - 2)
        self.plotBitMap.fill(self.bg_color)

    def plotData(self):
        if self.prof is None:
            return

        qp = QtGui.QPainter()
        qp.begin(self.plotBitMap)
        qp.setRenderHint(qp.TextAntialiasing)

        pen = QtGui.QPen(self.fg_color, 1, QtCore.Qt.SolidLine)
        qp.setPen(pen)

        start_y = self.tpad + self.title_height + self.row_spacing * 2
        label_width = int(self.wid * 0.68)
        value_width = self.wid - label_width - self.rpad

        for idx, (label, value) in enumerate(self.rows):
            y = start_y + idx * (self.label_height + self.row_spacing)
            qp.setFont(self.label_font)
            label_rect = QtCore.QRect(self.lpad, y, label_width, self.label_height)
            qp.drawText(label_rect, QtCore.Qt.TextDontClip | QtCore.Qt.AlignLeft, label)

            qp.setFont(self.value_font)
            value_rect = QtCore.QRect(label_width, y, value_width, self.label_height)
            qp.drawText(value_rect, QtCore.Qt.TextDontClip | QtCore.Qt.AlignRight, self._format_value(value))

        qp.end()
