from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal


#===============================================================================
# GLOBAL VARIABLES
#===============================================================================
IMG_BOMB = QImage("./imagfes/bomb.png")
IMG_FLAG = QImage("./images/flag.png")
IMG_START = QImage("./images/rocket.png")
IMG_CLOCK = QImage("./images/clock-select.png")
NUM_COLORS = {
    0: QColor('#4CAF50'),
    1: QColor('#00f3ff'),
    2: QColor('#03A9F4'),
    3: QColor('#3F51B5'),
    4: QColor('#8a00d4'),
    5: QColor('#d400ab'),
    6: QColor('#ff0000'),
    7: QColor('#FF9800'),
    8: QColor('#fff600')
}
STATUS_READY = 0
STATUS_PLAYING = 1
STATUS_FAILED = 2
STATUS_SUCCESS = 3

#===============================================================================
# TILE OF BOARD AND ASSOCIATED FUNCTION
#===============================================================================
"""
Class use to represent title on the Minesweeper board
"""
class Tile(QWidget):
    expandable = pyqtSignal(int, int)
    clicked = pyqtSignal()
    ohno = pyqtSignal()

    """
    Initialize the board, choose the size base on the LEVELS selected
    """
    def __init__(self, x, y, level, lst_revealed, scr, *args, **kwargs):
        super(Tile, self).__init__(*args, **kwargs)
        self.setFixedSize(QSize(60, 60))
        self.x = x
        self.y = y
        self.boardsize = level
        self.current_revealed = lst_revealed
        self.score = scr
    """
    Reset the boolean flag of a tile
    """
    def reset(self):
        self.is_start = False
        self.is_mine = False
        self.adjacent_n = 0
        self.is_revealed = False
        self.is_flagged = False
        self.update()

    """
    Update the information containt in the tile
    """
    def updatedata(self, lst_revealed, scr):
        self.current_revealed = lst_revealed
        self.score = scr

    """
    Drawing stuff about the tile
    """
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = event.rect()
        if self.is_revealed:
            color = self.palette().color(QPalette.Background)
            outer, inner = color, color
        else:
            outer, inner = QColor('#878787'), QColor('#202020')
        p.fillRect(r, QBrush(inner))
        pen = QPen(outer)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(r)
        if self.is_revealed:
            if self.is_start:
                p.drawPixmap(r, QPixmap(IMG_START))
            elif self.is_mine:
                p.drawPixmap(r, QPixmap(IMG_BOMB))
            elif self.adjacent_n >= 0:
                pen = QPen(NUM_COLORS[self.adjacent_n])
                p.setPen(pen)
                f = p.font()
                f.setBold(True)
                p.setFont(f)
                p.drawText(r, Qt.AlignHCenter | Qt.AlignVCenter, str(self.adjacent_n))
        elif self.is_flagged:
            p.drawPixmap(r, QPixmap(IMG_FLAG))

    """
    Put a flag (=marker) on a tile to point out a mine (right click)
    """
    def flag(self):
        self.is_flagged = True
        self.update()
        self.clicked.emit()

    """
    Return the value of the tile
    """
    def get_value(self):
        if(self.is_start):
            return -2
        elif(self.is_mine):
            return -1
        else:
            return self.adjacent_n
    """
    Reveal the tiles
    """
    def reveal(self):
        self.is_revealed = True
        self.update()

    """
    Manage the actions caused by a click on a tile
    """
    def click(self):
        if not self.is_revealed:
            self.reveal()
            self.current_revealed.append((self.x, self.y))
            if self.adjacent_n == 0:
                self.expandable.emit(self.x, self.y)
        self.clicked.emit()

    """
    Handle the mouse action on a tile
    """
    def mouseReleaseEvent(self, e):
        if (e.button() == Qt.RightButton and not self.is_revealed):
            self.flag()
        elif (e.button() == Qt.LeftButton):
            self.click()
            if self.is_mine:
                self.ohno.emit()
            else:
                self.score += 1
