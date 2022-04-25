from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from ai import AI

import random
import time
import numpy as np

#TODO : poetry package

global SCORE
SCORE = 0

IMG_BOMB = QImage("./images/bomb.png")
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

LEVELS = [
    (8, 10),
    (16, 40),
    (24, 99)
]

STATUS_READY = 0
STATUS_PLAYING = 1
STATUS_FAILED = 2
STATUS_SUCCESS = 3

supersmart = AI()

class Pos(QWidget):
    """
    Class use to represent title on the Minesweeper board
    """
    expandable = pyqtSignal(int, int)
    clicked = pyqtSignal()
    ohno = pyqtSignal()

    def __init__(self, x, y, *args, **kwargs):
        super(Pos, self).__init__(*args, **kwargs)
        self.setFixedSize(QSize(60, 60))
        self.x = x
        self.y = y

    def reset(self):
        self.is_start = False
        self.is_mine = False
        self.adjacent_n = 0
        self.is_revealed = False
        self.is_flagged = False
        self.update()

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

    def flag(self):
        self.is_flagged = True
        self.update()
        self.clicked.emit()

    def get_value(self):
        if(self.is_start):
            return -2
        elif(self.is_mine):
            return -1
        else:
            return self.adjacent_n

    def reveal(self):
        self.is_revealed = True
        self.update()

    def click(self):
        if not self.is_revealed:
            self.reveal()
            if self.adjacent_n == 0:
                self.expandable.emit(self.x, self.y)
        self.clicked.emit()

    def mouseReleaseEvent(self, e):
        global SCORE
        if (e.button() == Qt.RightButton and not self.is_revealed):
            self.flag()

        elif (e.button() == Qt.LeftButton):
            self.click()

            if self.is_mine:
                self.ohno.emit()
            else:
                SCORE += 1

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        """Long-running task."""
        print("Yo")
        self.train_AI(1)
        for i in range(5):
            sleep(1)
            self.progress.emit(i + 1)
        self.finished.emit()


class MainWindow(QMainWindow):
    """
    Main class use for GUI
    """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Minesweeper AI")
        self.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint);
        self.setWindowIcon(QIcon("./images/bomb.png"))

        self.b_size, self.n_mines = LEVELS[1]

        w = QWidget()
        hb = QHBoxLayout()

        self.score = QLabel()
        self.score.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.clock = QLabel()
        self.clock.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        f = self.score.font()
        f.setPointSize(10)
        f.setWeight(75)
        self.score.setFont(f)
        self.clock.setFont(f)

        self._timer = QTimer()
        self._timer.timeout.connect(self.update_timer)
        self._timer.start(1000)  # 1 second timer

        self.score.setText(str(SCORE))
        self.clock.setText("000")

        self.button = QPushButton("RESTART")
        self.button.pressed.connect(self.button_pressed)

        self.button_AI_learn = QPushButton("LEARN AI")
        self.button_AI_learn.pressed.connect(self.button_AI_learn_pressed)

        self.button_AI_play = QPushButton("PLAY AI")
        self.button_AI_play.pressed.connect(self.button_AI_play_pressed)

        score = QLabel("Score : ")
        time = QLabel("Time : ")

        hb.addWidget(self.button_AI_learn)
        hb.addWidget(self.button_AI_play)
        hb.addWidget(self.button)
        hb.addWidget(score)
        hb.addWidget(self.score)
        hb.addWidget(time)
        hb.addWidget(self.clock)

        vb = QVBoxLayout()
        vb.addLayout(hb)

        self.grid = QGridLayout()
        self.grid.setSpacing(5)

        vb.addLayout(self.grid)
        w.setLayout(vb)
        self.setCentralWidget(w)

        self.init_map()
        self.update_status(STATUS_READY)

        self.reset_map()
        self.update_status(STATUS_READY)

        # Initialize Agent
        supersmart.setboardsize(self.b_size)
        supersmart.setProba(self.b_size)
        supersmart.setPayoffs(self.get_payoff())
        self.show()

    def init_map(self):
        # Add positions to the map
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = Pos(x, y)
                self.grid.addWidget(w, y, x)
                # Connect signal to handle expansion.
                w.clicked.connect(self.trigger_start)
                w.expandable.connect(self.expand_reveal)
                w.ohno.connect(self.game_over)

    def reset_map(self):
        # Clear all mine positions
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.reset()

        # Add mines to the positions
        positions = []
        while len(positions) < self.n_mines:
            x, y = random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1)
            if (x, y) not in positions:
                w = self.grid.itemAtPosition(y, x).widget()
                w.is_mine = True
                positions.append((x, y))

        def get_adjacency_n(x, y):
            positions = self.get_surrounding(x, y)
            n_mines = sum(1 if w.is_mine else 0 for w in positions)

            return n_mines

        # Add adjacencies to the positions
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.adjacent_n = get_adjacency_n(x, y)

        # Place starting marker
        while True:
            x, y = random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1)
            w = self.grid.itemAtPosition(y, x).widget()
            # We don't want to start on a mine.
            if (x, y) not in positions:
                w = self.grid.itemAtPosition(y, x).widget()
                w.is_start = True

                # Reveal all positions around this, if they are not mines either.
                for w in self.get_surrounding(x, y):
                    if not w.is_mine:
                        w.click()
                break

    def get_surrounding(self, x, y):
        positions = []

        for xi in range(max(0, x - 1), min(x + 2, self.b_size)):
            for yi in range(max(0, y - 1), min(y + 2, self.b_size)):
                positions.append(self.grid.itemAtPosition(yi, xi).widget())

        return positions

    # Button menu for new game
    def button_pressed(self):
        global SCORE
        SCORE = 0
        self.score.setText(str(SCORE))

        if self.status == STATUS_PLAYING:
            self.update_status(STATUS_FAILED)
            self.reveal_map()
        elif self.status == STATUS_READY:
            self.update_status(STATUS_FAILED)
            self.reveal_map()
        elif self.status == STATUS_FAILED:
            self.update_status(STATUS_READY)
            SCORE = 0
            self.reset_map()
        elif self.status == STATUS_SUCCESS:
            self.update_status(STATUS_READY)
            SCORE = 0
            self.reset_map()


    def button_AI_learn_pressed(self):
        #TRAIN AI
        # Make this function run as parallal
        self.train_AI(1000)

    def train_AI(self, episodes):
        action_probabilities = None
        global SCORE
        avg_score = 0
        for i in range(0, episodes):
            #stat = self.get_status()
            OLDSCORE = 0
            while(self.get_status() != STATUS_FAILED):
                #print("     Episode :", i," | Status: ", self.get_status(), "| Score", SCORE)
                x, y = supersmart.act()
                OLDSCORE = SCORE
                #self.button_pressed()
                self.update_status(self.get_status())
                if(self.get_status() == STATUS_SUCCESS):
                    self.AI_turn(x, y)
                    """
                    stimuli = 10
                    payoffs = self.get_payoff()
                    supersmart.updatePayoffs(payoffs)
                    action_probabilities = supersmart.learn(stimuli, x, y)
                    """
                    self.update_status(STATUS_READY)
                    SCORE = 0
                    self.reset()
                    #self.button_pressed()
                    #print(action_probabilities)
                else:
                    stimuli = self.AI_turn(x, y)
                    """
                    payoffs = self.get_payoff()
                    supersmart.updatePayoffs(payoffs)
                    action_probabilities = supersmart.learn(stimuli, x, y)
                    """

            avg_score += OLDSCORE
                #stat = self.get_status()
            # If failed
            #print("Episode :",i)
            #stimuli = -1
            #payoffs = self.get_payoff()
            #stimuli = self.compute_stimuli()
            #action_probabilities = supersmart.learn(stimuli, x, y)
            #print(action_probabilities)
            self.update_status(STATUS_READY)
            SCORE = 0
            self.reset()

        print("Avg. score :", avg_score/episodes)
        action_probabilities = supersmart.learn(stimuli, x, y)
        print(action_probabilities)

    def AI_turn(self, x, y):
        global SCORE
        stimuli = 0
        w = self.grid.itemAtPosition(x, y).widget()
        if(not w.is_revealed):
            w.click()
            w.reveal()
            #time.sleep(0.5)
            self.show()
            #print("AI CLICK")
            if w.is_mine: #GAMEOVER
                w.ohno.emit()
                stimuli = -10
                #SCORE = 0
                self.update_status(STATUS_FAILED)
            else:
                stimuli = 1
                SCORE += 1
                self.score.setText(str(SCORE))
            return stimuli
        else:
            return -0.5 # Si mauvaise position


    # Button for luch the AI algorithm
    def button_AI_play_pressed(self):
        global SCORE
        avg_score = 0
        self.update_status(STATUS_READY)
        print("Let's go AI")
        for i in range(0, 1000):
            OLDSCORE = 0
            while(self.get_status() != STATUS_FAILED):
                #print(" Status: ", self.get_status(), "| Score", SCORE)
                x, y = supersmart.act()
                #self.button_pressed()
                OLDSCORE = SCORE
                self.AI_turn(x, y)
                #self.update_status(self.get_status())
            #self.AI_turn(x, y)
            #self.update_status(STATUS_READY)
            #SCORE = 0
            #self.reset()
            avg_score += OLDSCORE
            self.update_status(STATUS_READY)
            SCORE = 0
            self.reset()

        #avg_score += 1#SCORE - OLDSCORE
        #print(avg_score)
        print("Avg. score : ", avg_score/1000)

        """
        if(act==0):
            x = random.randint(0, boardsize) # assurer que x, y pas déjà uncover
            y = random.randint(0, boardsize)
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = self.grid.itemAtPosition(x, y).widget()
                if(not w.is_mine):
                    w.click()
                    SCORE += 1
                    self.score.setText(str(SCORE))
                # w.ohno.emit() # cas ou on click sur un bombe
        """
    def get_payoff(self):
        payoffs = np.zeros((self.b_size,self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = self.grid.itemAtPosition(x, y).widget()
                if(w.is_revealed):
                    payoffs[x,y]= w.get_value() # Revoir la mannière d'attribuer les payoffs
                else:
                    payoffs[x,y]=0
        #print(payoffs)
        return payoffs

    def reset(self):
        global SCORE
        SCORE = 0
        self.score.setText(str(SCORE))
        #self.update_status(STATUS_READY)
        self.reset_map()
        self.show()


    def reveal_map(self):
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.reveal()

    def expand_reveal(self, x, y):
        for xi in range(max(0, x - 1), min(x + 2, self.b_size)):
            for yi in range(max(0, y - 1), min(y + 2, self.b_size)):
                w = self.grid.itemAtPosition(yi, xi).widget()
                if not w.is_mine:
                    w.click()

    def trigger_start(self, *args):
        if self.status != STATUS_PLAYING:
            # First click.
            self.update_status(STATUS_PLAYING)
            # Start timer.
            self._timer_start_nsecs = int(time.time())

    def update_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def update_timer(self):
        self.score.setText(str(SCORE))
        if self.status == STATUS_PLAYING:
            n_secs = int(time.time()) - self._timer_start_nsecs
            self.clock.setText("%03d" % n_secs)

    """
    def runLongTask(self):
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        #self.worker.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        #self.longRunningBtn.setEnabled(False)
        self.thread.finished.connect(
            lambda: print("FINI")
        )
        self.thread.finished.connect(
            lambda :print("FINI2")
            #lambda: self.stepLabel.setText("Long-Running Step: 0")
        )
    """

    # Emit ohno
    def game_over(self):
        global SCORE
        print("SCORE : ", SCORE)
        SCORE = 0
        #print("     STATUS : ", self.get_status())
        self.reveal_map()
        #time.sleep(0.5)
        self.update_status(STATUS_READY)
        self.reset()


if __name__ == '__main__':
    app = QApplication([])
    app.setStyle("Fusion")

    QToolTip.setFont(QFont('SansSerif', 5))

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(200, 50, 20)) # Rouge normalement
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(218, 218, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    window = MainWindow()
    app.exec_()
