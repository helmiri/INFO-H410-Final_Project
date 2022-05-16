from asyncio import sleep
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from sklearn import neighbors
from ai import AI
from solver import naive, rule_1, rule_2
from rl import QAgent
from tile import Tile
from numpy import asarray

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow import keras

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#print(device_lib.list_local_devices())
# 2.10.0.dev20220427

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
LEVEL = LEVELS[0]

model = Sequential()
supersmart = AI()
seed = 7
np.random.seed(seed)
random.seed(seed)


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
        #self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        #self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        screen_size = QApplication.primaryScreen().availableSize()
        tilesize = screen_size.height()//20
        #self.setFixedHeight(min(LEVEL[0]*tilesize, screen_size.height()))
        #self.setFixedWidth(min(LEVEL[0]*tilesize, screen_size.width()))

        self.b_size, self.n_mines = LEVEL

        w = QWidget()
        hb = QHBoxLayout()
        hb1 = QHBoxLayout()

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

        self.button = QPushButton("Restart")
        self.button.pressed.connect(self.button_pressed)

        self.button_AI_learn = QPushButton("Learn")
        self.button_AI_learn.pressed.connect(self.button_AI_learn_pressed)

        self.button_AI_play = QPushButton("Play")
        self.button_AI_play.pressed.connect(self.button_AI_play_pressed)

        self.button_AI_test = QPushButton("Test")
        self.button_AI_test.pressed.connect(self.button_AI_test_pressed)

        self.button_AI_solve = QPushButton("Solve")
        self.button_AI_solve.pressed.connect(self.button_solve_pressed)

        self.button2 = QPushButton("RL learn")
        self.button2.pressed.connect(self.rl_learn)

        self.button3 = QPushButton("RL play")
        self.button3.pressed.connect(self.rl_play)

        score = QLabel("Score : ")
        time = QLabel("Time : ")

        hb.addWidget(self.button_AI_learn)
        hb.addWidget(self.button_AI_play)
        hb.addWidget(self.button)
        hb.addWidget(score)
        hb.addWidget(self.score)
        hb.addWidget(time)
        hb.addWidget(self.clock)
        hb.addWidget(self.button_AI_test)
        hb.addWidget(self.button_AI_solve)

        hb1.addWidget(self.button2)
        hb1.addWidget(self.button3)

        vb = QVBoxLayout()
        vb.addLayout(hb)
        vb.addLayout(hb1)

        self.grid = QGridLayout()
        self.grid.setSpacing(10)

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
                tile = Tile(x, y, LEVEL)
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
                #tile.updatedata(CURRENT_REVEALED, SCORE)

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
                        if((x,y) not in CURRENT_REVEALED):
                            CURRENT_REVEALED.append((x,y))
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
        CURRENT_REVEALED = self.get_pos_of_revealed()
        perimeter = set()
        for pos in CURRENT_REVEALED:
            neighb_tmp = self.get_surrounding(pos[0], pos[1])
            for elem in neighb_tmp:
                if((elem.x, elem.y) not in CURRENT_REVEALED):
                    perimeter.add((elem.x, elem.y))

        return list(perimeter)

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
    Return the matrix of all the tile's value on the board
    """
    def get_tiles_value(self):
        value_mat = np.zeros((self.b_size,self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                value_mat[x,y]= tile.get_value()
        return value_mat


    """
    Return the matrix of the tile's value known on the board
    """
    def get_tiles_revealed_value(self):
        value_mat = np.zeros((self.b_size,self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if(tile.is_revealed):
                        value_mat[x,y]= tile.get_value()
                else:
                    value_mat[x,y]= -8
        return value_mat

    """
    Return a list of all the tile's positions
    """
    def get_pos_of_revealed(self):
        lst_revealed = []
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if(tile.is_revealed):
                    lst_revealed.append((x, y))
        return lst_revealed

    """
    Return the matrix of all the tile's positions
    """
    def get_revealed_tiles(self):
        lst_revealed = []
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if tile.is_revealed:
                    lst_revealed.append(tile)
        return lst_revealed

    """
    Return the matrix with the positions of mine, 1 if it's a mine, 0 otherwise
    """
    def get_all_mine(self):
        mines = np.zeros((self.b_size,self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if(tile.is_mine):
                    mines[x,y]= 1
                else:
                    mines[x,y]= 0
        return mines

    """
    Reset all the global value and the board
    """
    def reset(self):
        global SCORE
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
    Update the in game timer value and the score
    """
    def update_timer(self):
        global SCORE
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
        self.update_status(STATUS_FAILED)
        self.reset()

    """
    Code execute when the user click on the restart button
    """
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
            self.reset_map()
        elif self.status == STATUS_SUCCESS:
            self.update_status(STATUS_READY)
            self.reset_map()

    def win(self):
        cond = False
        if self.get_status() == STATUS_FAILED:
            return True

        if len(self.get_revealed_tiles()) == (LEVEL[0] ** 2 ) - LEVEL[1]:
            cond = True
            self.update_status(STATUS_SUCCESS)
        elif self.get_status() == STATUS_FAILED:
            self.reveal_mines()
            return True
        return cond

    """
    Reveal all the mine on theb board
    """
    def reveal_mines(self):
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if tile.is_mine:
                    tile.reveal()

    """
    Code execute when the user click on the learn AI button
    """
    def button_AI_learn_pressed(self):
        self.train_AI(500000) #500000

    """
    Save the model of NN
    """
    def save_model(self, model):
        model.save('model')

    """
    Load the model of
    """
    def load_model(self):
        global model
        model = keras.models.load_model('model')
        #https://www.tensorflow.org/guide/keras/save_and_serialize

    """
    Define the architecture of the neuronal network
    """
    def set_model(self, n_inputs, n_outputs, episodes):
        global model
        matrixSize = n_inputs

        # CNN model test
        num_filters = 32
        filter_size = 2
        pool_size = 1

        model = keras.models.Sequential([
          keras.layers.Conv2D(32, filter_size, input_shape=(matrixSize,matrixSize, 1)),
          keras.layers.MaxPooling2D(pool_size=pool_size),
          keras.layers.Conv2D(16, filter_size),
          keras.layers.MaxPooling2D(pool_size=pool_size),
          keras.layers.Flatten(),
          keras.layers.Dense((matrixSize*matrixSize)*64, activation="sigmoid"),
          keras.layers.Dropout(0.2),
          keras.layers.Dense((matrixSize*matrixSize)*32, activation="sigmoid"),
          keras.layers.Dropout(0.05),
          keras.layers.Dense((matrixSize*matrixSize), activation="sigmoid"),
          keras.layers.Reshape((matrixSize, matrixSize))
        ])
        """
        model = keras.models.Sequential([
            keras.layers.Dense((matrixSize*matrixSize)*2, input_shape=(matrixSize,matrixSize), activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Flatten(),
            keras.layers.Dense((matrixSize*matrixSize)*4, activation="sigmoid"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense((matrixSize*matrixSize)*4, activation="sigmoid"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense((matrixSize*matrixSize)*4, activation="sigmoid"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense((matrixSize*matrixSize)*4, activation="sigmoid"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense((matrixSize*matrixSize)*4, activation="sigmoid"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(matrixSize*matrixSize, activation="sigmoid"),
            keras.layers.Reshape((matrixSize, matrixSize))
        ])
        """


        model.compile(optimizer="adam",loss="mean_squared_error", metrics=["accuracy"])
        model.summary()

    def rl_learn(self):
        alpha = 0.1
        epsilon_max = 0.9
        epsilon_min = 0.1
        epsilon_decay = 0.99

        #action : 0 = click ; 1 = flag ; 2 = ignore
        self.agent = QAgent(self.b_size*self.b_size, alpha, epsilon_max, epsilon_min, epsilon_decay)
        self.run_episode(True)

    def rl_play(self):
        self.reset_map()
        self.run_episode(False)

    def run_episode(self, training):
        wins = 0
        nb_game = 1000
        for episode in range(nb_game):
            while not self.win():
                QApplication.processEvents()

                #random tile
                x, y = random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1)
                tile = self.grid.itemAtPosition(y, x).widget()
                #print(tile.x + (self.b_size * tile.y))
                action = self.agent.act(tile.x + (self.b_size * tile.y), training)
                print(action)

                #tile clicked
                if action == 0:
                    tile.click()
                    tile.reveal()
                #tile flagged
                """
                elif action == 1:
                    tile.flag()
                """
                #action == 2 -> tile ingored

                #click + not mine
                if not tile.is_mine and tile.is_revealed:
                    if training:
                        self.agent.learn(tile.x + (self.b_size * tile.y), 0, 1, False)
                #click + mine
                elif tile.is_mine and tile.is_revealed:
                    if training:
                        self.agent.learn(tile.x + (self.b_size * tile.y), 0, -1, True)
                    self.update_status(STATUS_FAILED)
                    break
                #ignore + not mine
                elif not tile.is_mine and not tile.is_revealed:
                    if training:
                        self.agent.learn(tile.x + (self.b_size * tile.y), 1, -0.1, False)
                #ignore + mine
                elif tile.is_mine and not tile.is_revealed:
                    if training:
                        self.agent.learn(tile.x + (self.b_size * tile.y), 1, 5, False)
                """
                #flag + not mine
                elif not tile.is_mine and tile.is_flagged:
                    if training:
                        self.agent.learn(tile.x + (self.b_size * tile.y), 1, -0.5, False)
                #flag + mine
                elif tile.is_mine and tile.is_flagged:
                    if training:
                        self.agent.learn(tile.x + (self.b_size * tile.y), 1, 10, False)
                """


            if self.get_status() == STATUS_SUCCESS:
                wins += 1
            self.reset_map()
            print("WINS/TOTAL: " + str(wins) + "/" + str(episode+1))
        print("WIN RATE:" + str(wins/nb_game*100))
        print(self.agent.q_table)


    """
    Steps to do in order to train the model with all the different game
    """
    def train_AI(self, datasetSize):
        global SCORE, model
        avg_score = 0
        episodes = 25

        # get_tiles_value : give the value of each tile on the board
        Xfin = []
        yfin = []

        # Create multiple beginning of game (=episodes) and add them to the input list
        # TODO: Apprendre des parties complètes pas juste des débuts de game sur base du solver
        print("Generating", datasetSize,"games :")

        # Play solver and keep only win game
        """
        nb_win_game = 0
        while nb_win_game < datasetSize:
            tile = None
            Xtmp = []
            ytmp = []
            while not self.win():
                if tile is not None and tile.is_mine and tile.is_revealed:
                    self.update_status(STATUS_FAILED)
                    break
                revealed = self.get_revealed_tiles()

                tmp = list()
                for item in revealed:
                    tmp.append(self.get_surrounding(item.x, item.y))

                neighborhoods = [[] for i in range(len(tmp))]
                for i, neighborhood in enumerate(tmp):
                    for neighbor in neighborhood:
                        if not neighbor.is_revealed:
                            neighborhoods[i].append(neighbor)

                tile = rule_1(revealed, neighborhoods)
                if tile != None:
                    tile.flag()
                    continue
                tile = rule_2(revealed, neighborhoods)
                if tile != None:
                    tile.click()
                    tile.reveal()
                    continue

                perimeter = dict.fromkeys(self.get_perimeter(), 0)
                coords = naive(revealed, neighborhoods, perimeter)
                item = random.choice(coords)
                tile = self.grid.itemAtPosition(item[1], item[0]).widget()
                tile.click()
                tile.reveal()
                Xtmp.append(self.get_tiles_revealed_value())
                ytmp.append(self.get_all_mine())
            if self.get_status() == STATUS_SUCCESS:
                nb_win_game += len(Xtmp)
                print(nb_win_game)
                Xfin += Xtmp
                yfin += ytmp
            self.reset_map()
        """

        nb_board_game = 0
        while nb_board_game < datasetSize:
            while not self.win():
                Xfin.append(self.get_tiles_revealed_value())
                yfin.append(self.get_all_mine())
                #yfin.append(self.get_tiles_value())
                x = random.randint(0, LEVEL[0]-1)
                y = random.randint(0, LEVEL[0]-1)
                #print(x, y)
                self.AI_turn(x, y)
                nb_board_game+=1
            self.update_status(STATUS_READY)
            self.reset()

        """
        for i in tqdm(range(0, datasetSize)):
            #QApplication.processEvents()
            Xfin.append(self.get_tiles_revealed_value())
            yfin.append(self.get_all_mine())
            #yfin.append(self.get_tiles_value())
            #x = random.randint(0, LEVEL[0]-1)
            #y = random.randint(0, LEVEL[0]-1)
            #print(x, y)
            #self.AI_turn(x, y)
            self.update_status(STATUS_READY)
            self.reset()
        """

        # Train the model with all the game in the input list
        n_inputs, n_outputs = len(Xfin[0]), len(yfin[0])
        self.set_model(n_inputs, n_inputs, episodes)

        #seed = 7
        #np.random.seed(seed)
        X_train, X_test, Y_train, Y_test = train_test_split(np.array(Xfin), np.array(yfin), test_size=0.1, random_state=seed)

        #es = EarlyStopping(monitor='loss', mode='min', verbose=1, min_delta=0.01, patience=episodes)

        #self.save_model(model)


        print("SIZE X TRAIN", X_train.shape)
        X_train = X_train.reshape((X_train.shape[0], 8, 8, 1))
        print("SIZE X TRAIN", X_train.shape)

        history = model.fit(X_train, Y_train, batch_size=200, shuffle=True, epochs=episodes, validation_split=0.1, validation_data=(X_test, Y_test))

        X_test = X_test.reshape((X_test.shape[0], 8, 8, 1))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        """
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        """
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # 0.0811249986290931 (10000 20 relu relu sigmoid)
        # 0.0488750003278255 (10000 20 relu relu relu)
        # 0.0823125019669532 (10000 20 relu sigmoid sigmoid)

    """
    Code execute to test the prediction made by the model
    """
    def button_AI_play_pressed(self):
        global SCORE, model
        avg_score = 0
        self.update_status(STATUS_READY)

        #self.load_model()

        nb_test_run = 500
        wins = 0
        for i in range(0, nb_test_run):
            OLDSCORE = 0
            while not self.win():

                QApplication.processEvents()
                testX = np.array([self.get_tiles_revealed_value()])

                # CNN model
                test_x = np.array(testX[0].transpose())
                test_x = test_x.reshape(1, 8, 8, 1)
                yhat = model.predict(test_x)


                # Given the current board the model predict the prob of mine with yhat
                #yhat = model.predict(np.array([testX[0].transpose()]))

                yhat = np.array([yhat[0].transpose()])
                #print(yhat)
                # Give the positions of tile around the revealed tiles
                peri = self.get_perimeter()
                CURRENT_REVEALED = self.get_pos_of_revealed()
                # Choose the best position to click given the prediction and the perimeter
                x, y = supersmart.act(yhat, peri, CURRENT_REVEALED)
                fx, fy = supersmart.flag(yhat, peri, CURRENT_REVEALED)
                #print(fx, fy)
                ftile = self.grid.itemAtPosition(fy, fx).widget()
                ftile.flag()

                OLDSCORE = SCORE
                self.AI_turn(x, y)


            if self.get_status() == STATUS_SUCCESS:
                wins += 1
                print("WIN !")
            else:
                print("LOSE :(")
            #print("Status : ", self.get_status())
            avg_score += OLDSCORE
            self.update_status(STATUS_READY)
            SCORE = 0
            self.reset()
            supersmart.reset()

        print("WIN RATE:" + str(wins/nb_test_run*100)+ "%")
        print("Avg. score : ", avg_score/nb_test_run)

    """
    Make the different action of a normal turn in game like it is a human who is playing (click etc)
    """
    def AI_turn(self, x, y):
        global SCORE
        tile = self.grid.itemAtPosition(y, x).widget()
        #tile.updatedata(CURRENT_REVEALED, SCORE)
        if(not tile.is_revealed):
            tile.click()
            tile.reveal()
            #self.show()
            if tile.is_mine: #GAMEOVER
                tile.ohno.emit()
                self.update_status(STATUS_FAILED)
            else:
                SCORE += 1
                self.score.setText(str(SCORE))

    """
    Simple function link to a button to test stuff about the AI
    """
    def button_AI_test_pressed(self):
        global model
        #self.load_model()

        CURRENT_REVEALED = self.get_pos_of_revealed()
        testX = np.array([self.get_tiles_revealed_value()])

        # For CNN model
        test_x = np.array(testX[0].transpose())
        test_x = test_x.reshape(1, 8, 8, 1)
        print(test_x.shape)
        print(test_x)
        yhat = model.predict(test_x)

        # Given the current board the model predict the prob of mine with yhat
        #yhat = model.predict(np.array([testX[0].transpose()]))
        #yhat = np.array([yhat[0].transpose()])

        ytrue = self.get_all_mine()
        ytrue = np.array([ytrue.transpose()])

        print(yhat)
        print(ytrue)
        #plt.imshow(ytrue, cmap='hot', interpolation='nearest')
        plt.imshow(yhat[0], cmap='hot', interpolation='nearest')
        plt.show()

        # Give the positions of tile around the revealed tiles
        peri = self.get_perimeter()

        # Choose the best position to click given the prediction and the perimeter
        x, y = supersmart.act(yhat, peri, CURRENT_REVEALED)
        print(x, y)

        #OLDSCORE = SCORE
        #self.AI_turn(x, y)        #self.AI_turn(x, y)

    def button_solve_pressed(self):
        wins = 0
        nb_game = 1000
        for episode in range(nb_game):
            tile = None
            while not self.win():
                QApplication.processEvents()

                if tile is not None and tile.is_mine and tile.is_revealed:
                    self.update_status(STATUS_FAILED)
                    break
                #time.sleep(0.15)
                revealed = self.get_revealed_tiles()

                tmp = list()
                for item in revealed:
                    tmp.append(self.get_surrounding(item.x, item.y))

                neighborhoods = [[] for i in range(len(tmp))]
                for i, neighborhood in enumerate(tmp):
                    for neighbor in neighborhood:
                        if not neighbor.is_revealed:
                            neighborhoods[i].append(neighbor)

                tile = rule_1(revealed, neighborhoods)
                if tile != None:
                    tile.flag()
                    continue
                tile = rule_2(revealed, neighborhoods)
                if tile != None:
                    tile.click()
                    tile.reveal()
                    continue

                perimeter = dict.fromkeys(self.get_perimeter(), 0)
                coords = naive(revealed, neighborhoods, perimeter)
                item = random.choice(coords)
                tile = self.grid.itemAtPosition(item[1], item[0]).widget()
                tile.click()
                tile.reveal()
            if self.get_status() == STATUS_SUCCESS:
                wins += 1
            self.reset_map()
            print("WINS/TOTAL: " + str(wins) + "/" + str(episode))
        print("WIN RATE:" + str(wins/nb_game*100))


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
