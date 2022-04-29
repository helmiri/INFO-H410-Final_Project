from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from ai import AI
from tile import Tile
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
import random
import time
import numpy as np

#To add a package to the project : poetry add 'package_name'


#===============================================================================
# GLOBAL VARIABLES
#===============================================================================
global SCORE, CURRENT_REVEALED, model, LEVEL
SCORE = 0
CURRENT_REVEALED = []
LEVELS = [
    (8, 10),
    (16, 40),
    (24, 99)
]
LEVEL = LEVELS[1] # TODO : Bug for other level than 1

model = Sequential()
supersmart = AI()


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
# GUI AND AI
#===============================================================================
"""
Main class use for GUI and the AI managment
"""
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        global LEVEL
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Minesweeper AI")
        self.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint);
        self.setWindowIcon(QIcon("./images/bomb.png"))
        self.b_size, self.n_mines = LEVEL

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
        self.show()

    """
    Init the board and connect signal of each tile to the correct function
    """
    def init_map(self):
        global LEVEL, CURRENT_REVEALED, SCORE
        # Add positions to the map
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = Tile(x, y, LEVEL, CURRENT_REVEALED, SCORE)
                self.grid.addWidget(tile, y, x)
                # Connect signal to handle expansion.
                tile.clicked.connect(self.trigger_start)
                tile.expandable.connect(self.expand_reveal)
                tile.ohno.connect(self.game_over)

    """
    Reset all the board, choose random positions for mine and give new value to each tiles
    """
    def reset_map(self):
        global SCORE, CURRENT_REVEALED
        # Clear all mine positions
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                tile.reset()
                tile.updatedata(CURRENT_REVEALED, SCORE)

        # Add mines to the positions
        positions = []
        while len(positions) < self.n_mines:
            x, y = random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1)
            if (x, y) not in positions:
                tile = self.grid.itemAtPosition(y, x).widget()
                tile.is_mine = True
                positions.append((x, y))

        # Give number of mines surrounding a tile
        def get_adjacency_n(x, y):
            positions = self.get_surrounding(x, y)
            n_mines = sum(1 if tile.is_mine else 0 for tile in positions)
            return n_mines

        # Add adjacencies to the positions
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                tile.adjacent_n = get_adjacency_n(x, y)

        # Place starting marker
        while True:
            x, y = random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1)
            tile = self.grid.itemAtPosition(y, x).widget()
            # We don't want to start on a mine.
            if (x, y) not in positions:
                tile = self.grid.itemAtPosition(y, x).widget()
                tile.is_start = True
                # Reveal all positions around this, if they are not mines either.
                for tile in self.get_surrounding(x, y):
                    if not tile.is_mine:
                        tile.click()
                break

    """
    Return all the tiles around a give tile at position (x,y)
    """
    def get_surrounding(self, x, y):
        positions = []
        for xi in range(max(0, x - 1), min(x + 2, self.b_size)):
            for yi in range(max(0, y - 1), min(y + 2, self.b_size)):
                positions.append(self.grid.itemAtPosition(yi, xi).widget())
        return positions

    """
    Allow us to get the x,y positions of all the tiles around the set of revealed tiles
    """
    def get_perimeter(self):
        global CURRENT_REVEALED
        #print("current_revealed", len(CURRENT_REVEALED))
        perimeter = []
        neighbors = []

        for pos in CURRENT_REVEALED:
            neighb_tmp = self.get_surrounding(pos[0], pos[1])
            for elem in neighb_tmp:
                if((elem.x, elem.y) not in CURRENT_REVEALED):
                    perimeter.append((elem.x, elem.y))

        perimeter = list(dict.fromkeys(perimeter))
        return perimeter

    """
    Get the perimeter og the reveal mine with 1 if it's a mine 0 otherwise
    """
    def get_bombe_peri(self, peri):
        peri_bomb = []
        for pos in peri:
            tile = self.grid.itemAtPosition(pos[0], pos[1]).widget()
            if(tile.is_mine == True):
                peri_bomb.append(1)
            else:
                peri_bomb.append(0)
        return peri_bomb

    """
    Give the list of the value of the revealed tiles
    """
    def get_input_from_revealed(self):
        global CURRENT_REVEALED
        value_revealed = []
        for pos in CURRENT_REVEALED:
            tile = self.grid.itemAtPosition(pos[0], pos[1]).widget()
            value_revealed.append(tile.get_value())
        return value_revealed

    """
    Return the matrix of all the tile's value on the board
    """
    def get_tiles_value(self):
        value_mat = np.zeros((self.b_size,self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(x, y).widget()
                value_mat[x,y]= tile.get_value()
        return value_mat

    """
    Return the matrix of all the tile's value known on the board
    """
    def get_tiles_revealed_value(self):
        value_mat = np.zeros((self.b_size,self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(x, y).widget()
                if(tile.is_revealed):
                    value_mat[x,y]= tile.get_value()
                else:
                    value_mat[x,y]= 10
        return value_mat

    """
    Return the matrix with the positions of mine, 1 if it's a mine, 0 otherwise
    """
    def get_all_mine(self):
        mines = np.zeros((self.b_size,self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(x, y).widget()
                if(tile.is_mine):
                    mines[x,y]= 1
                else:
                    mines[x,y]= 0
        return mines

    """
    Reset all the global value and the board
    """
    def reset(self):
        global SCORE, CURRENT_REVEALED
        CURRENT_REVEALED = []
        SCORE = 0
        self.score.setText(str(SCORE))
        self.reset_map()
        self.show()

    """
    Reveal all the tiles on the board
    """
    def reveal_map(self):
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                tile.reveal()

    """
    Reveal all the tile which are not mine around a position (x,y)
    """
    def expand_reveal(self, x, y):
        for xi in range(max(0, x - 1), min(x + 2, self.b_size)):
            for yi in range(max(0, y - 1), min(y + 2, self.b_size)):
                tile = self.grid.itemAtPosition(yi, xi).widget()
                if not tile.is_mine:
                    tile.click()
    """
    Start the timer in at the first click and update the current status
    """
    def trigger_start(self, *args):
        if self.status != STATUS_PLAYING:
            self.update_status(STATUS_PLAYING)
            self._timer_start_nsecs = int(time.time())
    """
    Update the current status of the player
    """
    def update_status(self, status):
        self.status = status

    """
    Return the current status
    """
    def get_status(self):
        return self.status

    """
    Update the in game timer text value
    """
    def update_timer(self):
        self.score.setText(str(SCORE))
        if self.status == STATUS_PLAYING:
            n_secs = int(time.time()) - self._timer_start_nsecs
            self.clock.setText("%03d" % n_secs)

    """
    Code execute when the game emit the 'ohno' signal which and the game and restart a new one
    """
    def game_over(self):
        global SCORE
        print("SCORE : ", SCORE)
        SCORE = 0
        self.reveal_map()
        self.update_status(STATUS_READY)
        self.reset()

    """
    Code execute when the user click on the restart button
    """
    def button_pressed(self):
        global SCORE, CURRENT_REVEALED
        SCORE = 0
        CURRENT_REVEALED = []
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

    """
    Code execute when the user click on the learn AI button
    """
    def button_AI_learn_pressed(self):
        # TODO : Make this function run as parallal
        self.train_AI(100)

    """
    Define the architecture of the neuronal network
    """
    def set_model(self, n_inputs, n_outputs):
        global model
        model.add(Dense(10, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        #model.add(Dense(20, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
        model.compile(loss='mean_squared_error', optimizer='adam')

    """
    Steps to do in order to train the model with all the different game
    """
    def train_AI(self, episodes):
        global SCORE, model
        action_probabilities = None
        avg_score = 0

        # get_tiles_value : give the value of each tile on the board
        Xfin = np.array(self.get_tiles_value())
        yfin = np.array(self.get_all_mine())

        # Create multiple beginning of game (=episodes) and add them to the input list
        for i in range(0, episodes):
            print("Creating game #",i+1)
            Xfin = np.vstack([Xfin, self.get_tiles_value()])
            yfin = np.vstack([yfin, self.get_all_mine()])
            self.update_status(STATUS_READY)
            self.reset()

        # Train the model with all the game in the input list
        n_inputs, n_outputs = Xfin.shape[1], yfin.shape[1]
        self.set_model(n_inputs, n_outputs)
        model.fit(Xfin, yfin, verbose=1, epochs=200)
        print("Number of generated game",len(Xfin))
        print("Number of generated solution",len(yfin))


    """
    Code execute to test the prediction made by the model
    """
    def button_AI_play_pressed(self):
        global SCORE, model
        avg_score = 0
        self.update_status(STATUS_READY)
        nb_test_run = 10
        for i in range(0, nb_test_run):
            OLDSCORE = 0
            while(self.get_status() != STATUS_FAILED):
                testX = self.get_tiles_revealed_value()
                # Given the current board the model predict the prob of mine with yhat
                yhat = model.predict(testX)
                #print(yhat)
                # Give the positions of tile around the revealed tiles
                peri = self.get_perimeter()
                #print(peri)
                # Choose the best position to click given the prediction and the perimeter
                x, y = supersmart.act(yhat, peri)
                #print(x, y)
                OLDSCORE = SCORE
                self.AI_turn(x, y)

            avg_score += OLDSCORE
            self.update_status(STATUS_READY)
            SCORE = 0
            self.reset()
            supersmart.reset()

        print("Avg. score : ", avg_score/nb_test_run)


    """
    Make the different action of a normal turn in game like it is a human who is playing (click etc)
    """
    def AI_turn(self, x, y):
        global SCORE, CURRENT_REVEALED

        tile = self.grid.itemAtPosition(x, y).widget()
        tile.updatedata(CURRENT_REVEALED, SCORE)

        if(not tile.is_revealed):
            tile.click()
            tile.reveal()
            self.show()
            if tile.is_mine: #GAMEOVER
                tile.ohno.emit()
                self.update_status(STATUS_FAILED)
            else:
                SCORE += 1
                self.score.setText(str(SCORE))



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
