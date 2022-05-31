from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from sklearn import neighbors
from ai import AI
from solver import *
from rl import QAgent
from tile import Tile

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow import keras

import pickle
import random
import time
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from asyncio import sleep

# ===============================================================================
# GLOBAL VARIABLES
# ===============================================================================
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
SEED = 7

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

# ===============================================================================
# GUI
# ===============================================================================
class DifficultySelector(QWidget):
    """
    Difficulty selection prompt
    """
    def __init__(self) -> None:
        global LEVEL
        super().__init__()
        self.setWindowTitle("Minesweeper AI")
        self.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("./images/bomb.png"))
        self.button_confirm = QPushButton("Confirm")
        self.button_confirm.pressed.connect(self.button_confirm_pressed)
        self.label = QLabel("Difficulty")
        self.seedLabel = QLabel("Seed")
        self.dropdown = QComboBox()
        self.dropdown.addItems(["Beginner (8x8)", "Advanced (16x16)", "Expert(24x24)"])
        self.seedBox = QSpinBox()
        vb = QVBoxLayout()
        difficultyBox = QHBoxLayout()
        difficultyBox.addWidget(self.label)
        difficultyBox.addWidget(self.dropdown)
        seedBox = QHBoxLayout()
        seedBox.addWidget(self.seedLabel)
        seedBox.addWidget(self.seedBox)
        vb.addLayout(difficultyBox)
        vb.addLayout(seedBox)
        vb.addWidget(self.button_confirm)
        self.setLayout(vb)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.setFixedWidth(400)
        self.show()
        self.setFixedSize(self.size())

    def button_confirm_pressed(self):
        global LEVEL
        global LEVELS
        global SEED
        LEVEL = LEVELS[self.dropdown.currentIndex()]
        seed = self.seedBox.value()
        np.random.seed(seed)
        random.seed(seed)
        self.close()

class MainWindow(QMainWindow):
    """
    Main class use for GUI and the AI managment
    """
    def __init__(self, *args, **kwargs):
        global LEVEL
        super(MainWindow, self).__init__(*args, **kwargs)

        difficulty_selector = DifficultySelector()
        # Wait until closed
        loop = QEventLoop()
        difficulty_selector.destroyed.connect(loop.quit)
        loop.exec()

        self.setWindowTitle("Minesweeper AI")
        self.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("./images/bomb.png"))
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.b_size, self.n_mines = LEVEL
        self.agent = None
        self.manual_play = False

        w = QWidget()
        vb = QVBoxLayout()
        hb = QHBoxLayout()
        hb0 = QHBoxLayout()
        hb1 = QHBoxLayout()
        hb2 = QHBoxLayout()

        self.cb = QComboBox()
        self.cb.addItems(["50 games", "100 games", "500 games", "1000 games"])
        self.cb.setToolTip("Number of games for the training phase.")
        self.button_solve = QPushButton("Solver Play")
        self.button_solve.pressed.connect(self.button_solve_pressed)
        self.button_AI_learn = QPushButton("CNN Learn")
        self.button_AI_learn.pressed.connect(self.button_AI_learn_pressed)
        self.button_AI_play = QPushButton("CNN Play")
        self.button_AI_play.pressed.connect(self.button_AI_play_pressed)
        self.button_RL_learn = QPushButton("RL learn")
        self.button_RL_learn.pressed.connect(self.rl_learn)
        self.button_RL_play = QPushButton("RL play")
        self.button_RL_play.pressed.connect(self.rl_play)
        self.score = QLabel()
        self.score.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.winrate = QLabel()
        self.winrate.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        f = self.score.font()
        f.setPointSize(10)
        f.setWeight(75)
        self.score.setFont(f)
        self.winrate.setFont(f)
        self.score.setText(str(SCORE))
        self.winrate.setText("0%")
        self.score_text = QLabel("Score : ")
        self.winrate_text = QLabel("Win rate : ")
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.hide()

        hb.addWidget(self.score_text)
        hb.addWidget(self.score)
        hb.addWidget(self.winrate_text)
        hb.addWidget(self.winrate)
        hb0.addWidget(self.cb)
        hb0.addWidget(self.button_solve)
        hb1.addWidget(self.button_AI_learn)
        hb1.addWidget(self.button_AI_play)
        hb2.addWidget(self.button_RL_learn)
        hb2.addWidget(self.button_RL_play)
        vb.addLayout(hb0)
        vb.addLayout(hb1)
        vb.addLayout(hb2)
        vb.addLayout(hb)
        self.grid = QGridLayout()
        self.grid.setSpacing(5)
        vb.addLayout(self.grid)
        w.setLayout(vb)
        vb.addWidget(self.pbar)
        self.setCentralWidget(w)

        self.winMsg = QMessageBox()
        self.winMsg.setWindowTitle("Congratulations !")
        self.winMsg.setText("<p align='center'; style='font-size:11pt'> You won !")
        self.winMsg.setStyleSheet("QLabel{min-width: 200px;}")

        self.init_map()
        self.update_status(STATUS_READY)
        self.reset_map()
        self.update_status(STATUS_READY)

        supersmart.setboardsize(self.b_size)
        self.show()
        self.setFixedSize(self.size())

    def init_map(self):
        """
        Init the board and connect signal of each tile to the correct function
        """
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
                tile.score.connect(self.update_score)
                tile.manual.connect(self.update_manual)

    def reset_map(self):
        """
        Reset all the board, choose random positions for mine and give new value to each tiles
        """
        global SCORE, CURRENT_REVEALED
        self.manual_play = False
        # Clear all mine positions
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                tile.reset()
        # Add mines to the positions
        positions = []
        while len(positions) < self.n_mines:
            x, y = random.randint(
                0, self.b_size - 1), random.randint(0, self.b_size - 1)
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
                tile.neighbors = self.get_surrounding(tile.x, tile.y)
        # Place starting marker
        while True:
            x, y = random.randint(
                0, self.b_size - 1), random.randint(0, self.b_size - 1)
            tile = self.grid.itemAtPosition(y, x).widget()
            # We don't want to start on a mine.
            if (x, y) not in positions:
                tile = self.grid.itemAtPosition(y, x).widget()
                tile.is_start = True
                # Reveal all positions around this, if they are not mines either.
                for tile in self.get_surrounding(x, y):
                    if not tile.is_mine:
                        tile.click()
                        if((x, y) not in CURRENT_REVEALED):
                            CURRENT_REVEALED.append((x, y))
                break
        self.update_score()

    def get_surrounding(self, x, y):
        """
        Return all the tiles around a give tile at position (x,y)

        Parameters
        ----------
        x : int
            x position of the tile
        y : int
            y position of the tile
        """
        positions = []
        for xi in range(max(0, x - 1), min(x + 2, self.b_size)):
            for yi in range(max(0, y - 1), min(y + 2, self.b_size)):
                positions.append(self.grid.itemAtPosition(yi, xi).widget())
        return positions

    def get_perimeter(self):
        """
        Return the list of the x,y positions of all the tiles around the set of revealed tiles
        """
        CURRENT_REVEALED = self.get_pos_of_revealed()
        perimeter = set()
        for pos in CURRENT_REVEALED:
            neighb_tmp = self.get_surrounding(pos[0], pos[1])
            for elem in neighb_tmp:
                if((elem.x, elem.y) not in CURRENT_REVEALED):
                    perimeter.add((elem.x, elem.y))
        return list(perimeter)

    def get_perim_as_tile(self):
        """
        Return the list the tiles in the perimeter of the set of revealed tiles
        """
        CURRENT_REVEALED = self.get_pos_of_revealed()
        perimeter = set()
        for pos in CURRENT_REVEALED:
            neighb_tmp = self.get_surrounding(pos[0], pos[1])
            for elem in neighb_tmp:
                if elem not in CURRENT_REVEALED:
                    perimeter.add(elem)
        return list(perimeter)

    def get_bombe_peri(self, peri):
        """
        Return the perimeter og the reveal mine with 1 if it's a mine 0 otherwise

        Parameters
        ----------
        peri : list
            peri is a list of (x,y) position of tiles
        """
        peri_bomb = []
        for pos in peri:
            tile = self.grid.itemAtPosition(pos[0], pos[1]).widget()
            if(tile.is_mine == True):
                peri_bomb.append(1)
            else:
                peri_bomb.append(0)
        return peri_bomb

    def get_tiles_value(self):
        """
        Return the matrix of all the tile's value on the board
        """
        value_mat = np.zeros((self.b_size, self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                value_mat[x, y] = tile.get_value()
        return value_mat

    def get_tiles_revealed_value(self):
        """
        Return the matrix of the tile's value known on the board
        """
        value_mat = np.zeros((self.b_size, self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if(tile.is_revealed):
                    if(tile.get_value()==-2): # Si start position
                        value_mat[x,y]=0
                    else:
                        value_mat[x,y]= tile.get_value()
                else:
                    value_mat[x,y]= -1
        return value_mat

    def get_pos_of_revealed(self):
        """
        Return a list of all the tile's positions
        """
        lst_revealed = []
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if(tile.is_revealed):
                    lst_revealed.append((x, y))
        return lst_revealed

    def get_revealed_tiles(self):
        """
        Return the matrix of all the tile's positions
        """
        lst_revealed = []
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if tile.is_revealed:
                    lst_revealed.append(tile)
        return lst_revealed

    def get_mine_peri(self):
        """
        Return the matrix with the position of mine in the perieter of revealed tiles
        """
        mines = np.zeros((self.b_size, self.b_size))
        peri = self.get_perimeter()
        for pos in peri:
            x, y = pos[0], pos[1]
            tile = self.grid.itemAtPosition(y, x).widget()
            if(tile.is_mine):
                mines[x, y] = 1
            else:
                mines[x, y] = 0
        return mines

    def reset(self):
        """
        Reset all the global value and the board
        """
        global SCORE
        SCORE = 0
        self.score.setText(str(SCORE))
        self.reset_map()
        self.show()

    def reveal_map(self):
        """
        Reveal all the tiles on the board
        """
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                tile.reveal()

    def expand_reveal(self, x, y):
        """
        Reveal all the tile which are not mine around a position (x,y)

        Parameters
        ----------
        x : int
            x position of the tile
        y : int
            y position of the tile
        """
        for xi in range(max(0, x - 1), min(x + 2, self.b_size)):
            for yi in range(max(0, y - 1), min(y + 2, self.b_size)):
                tile = self.grid.itemAtPosition(yi, xi).widget()
                if not tile.is_mine:
                    tile.click()

    def trigger_start(self, *args):
        """
        Start the game and update the current status
        """
        if self.status != STATUS_PLAYING:
            self.update_status(STATUS_PLAYING)

    def update_status(self, status):
        """
        Update the current status of the player
        """
        self.status = status

    def get_status(self):
        """
        Return the current status
        """
        return self.status

    def update_score(self):
        """
        Update the score
        """
        global SCORE
        revealed = self.get_revealed_tiles()
        SCORE = len(revealed)
        self.score.setText(str(SCORE))
        if self.manual_play and self.win():
            self.winMsg.exec_()
            self.reset_map()

    def update_manual(self):
        """
        Update the manual play boolean
        """
        self.manual_play = True
        self.score_text.setText("Score : ")
        self.score.setText("0")
        self.winrate.setText("0%")

    def game_over(self):
        """
        Code execute when a tile emit the 'ohno' signal which end the game and restart a new one
        """
        global SCORE
        if self.manual_play:
            print("SCORE : ", SCORE)
        SCORE = 0
        self.reveal_map()
        self.update_status(STATUS_FAILED)
        self.reset()

    def win(self):
        """
        Check if a game is win or not
        """
        cond = False
        if self.get_status() == STATUS_FAILED:
            return True
        if len(self.get_revealed_tiles()) == (LEVEL[0] ** 2) - LEVEL[1]:
            cond = True
            self.update_status(STATUS_SUCCESS)
        elif self.get_status() == STATUS_FAILED:
            self.reveal_mines()
            return True
        return cond

    def reveal_mines(self):
        """
        Reveal all the mine on theb board
        """
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if tile.is_mine:
                    tile.reveal()

    def get_number_of_play(self):
        """
        Get the number of game to play
        """
        text = self.cb.currentText()
        if "50 games" in text:
            return 50
        elif "100 games" in text:
            return 100
        elif "500 games" in text:
            return 500
        elif "1000 games" in text:
            return 1000

    def warning_before_learn(self):
        """
        Warn the user before lunching a learning process
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning !")
        msg.setText("Warning !")
        msg.setInformativeText("Attention, you will delete the previous saved model and create a new one, do you want to continue? ")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = msg.exec_()
        return retval

    def update_pbar(self, value, reset=False):
        """
        Update the progress bar while doing training/playing

        Parameters
        ----------
        value : int
            value to add to the progress bar
        reset : bool
            if True reset the progress bar
        """
        if(not reset):
            self.pbar.show()
            self.pbar.setValue(int(value))
        else:
            self.pbar.setValue(0)
            self.pbar.hide()

# ===============================================================================
# REINFORCEMENT LEARNING
# ===============================================================================
    def rl_save(self):
        """
        Save the trained RL agent
        """
        with open('model/q_agent_config.pickle', 'wb') as config_agent:
            pickle.dump(self.agent, config_agent)

    def rl_learn(self):
        """
        Create a new RL agent and train it
        """
        res = self.warning_before_learn()
        if(res == 1024):
            alpha = 0.1; epsilon_max = 0.9; epsilon_min = 0.1; epsilon_decay = 0.99
            self.agent = QAgent(alpha, epsilon_max, epsilon_min, epsilon_decay)
            self.run_episode(True, 1000000)
            self.rl_save()

    def rl_play(self):
        """
        Play using the trained RL agent : load a trained agent if no new agent has been created
        """
        if self.agent == None:
            with open('model/q_agent_config_1M_run.pickle', 'rb') as config_agent:
                self.agent = pickle.load(config_agent)
        self.reset_map()
        self.run_episode(False, self.get_number_of_play())

    def run_episode(self, training, nb_game):
        """
        Play the game with a RL agent

        Parameters
        ----------
        training : bool
            if True train the agent, else test the agent
        nb_game : int
            number of game the agent has to learn or play
        """
        wins = 0; avg_score = 0
        if training:
            description = 'Training progress'
        else:
            description = 'Testing progress'
        for episode in tqdm(range(nb_game), desc = description):
            self.update_pbar(episode/nb_game*100, False)
            unproductive_moves = 0; score = 0
            while not self.win():
                QApplication.processEvents()
                if unproductive_moves > 200: #force game over after more than 200 unproductive moves
                    self.update_status(STATUS_FAILED)
                    break
                #select a random tile
                x, y = random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1)
                tile = self.grid.itemAtPosition(y, x).widget()
                if tile.is_revealed: #tile has been already picked
                    unproductive_moves += 1
                #get a 3x3 cluster around the tile
                cluster = self.get_surrounding(x, y)
                for i in range(len(cluster)): #get current state of the cluster
                    if cluster[i].is_revealed:
                        cluster[i] = cluster[i].get_value()
                    else:
                        cluster[i] = -3
                #get the agent action
                action = self.agent.act(cluster, training)
                if action == 1: #tile clicked
                    tile.click()
                    tile.reveal()
                if not tile.is_mine and tile.is_revealed: #click + not mine
                    if training:
                        self.agent.learn(cluster, 1, 1, False)
                elif tile.is_mine and tile.is_revealed: #click + mine
                    if training:
                        self.agent.learn(cluster, 1, -1, True)
                    self.update_status(STATUS_FAILED)
                    break
                elif not tile.is_mine and not tile.is_revealed: #ignore + not mine
                    if training:
                        self.agent.learn(cluster, 2, -1, False)
                elif tile.is_mine and not tile.is_revealed: #ignore + mine
                    if training:
                        self.agent.learn(cluster, 2, 1, False)
                revealed = self.get_revealed_tiles()
                score = len(revealed)
            if self.get_status() == STATUS_SUCCESS:
                wins += 1
            avg_score += score
            self.reset_map()
            self.winrate.setText(str(round(wins/nb_game*100,2))+"%")
            self.score_text.setText("Average score : ")
            self.score.setText(str(round(avg_score/nb_game, 2)))
        self.update_pbar(0, True)

# ===============================================================================
# CONVOLUTIONAL NEURAL NETWORK
# ===============================================================================
    def save_model(self, model):
        """
        Save the model of CNN

        Parameters
        ----------
        model : tensorflow
            the tensorflow model with the value learned
        """
        model.save('model/model_cnn')

    def load_model(self):
        """
        Load the model of CNN
        """
        model = keras.models.load_model('model/model_cnn')
        return model

    def set_model(self, matrixSize):
        """
        Define the architecture of the CNN

        Parameters
        ----------
        matrixSize : int
            size of the input matrix, i.e. the size of the game board
        """
        filter_size = 2
        pool_size = 1
        model = keras.models.Sequential([
          keras.layers.Conv2D(96, filter_size, input_shape=(matrixSize,matrixSize, 1), activation="relu"),
          keras.layers.MaxPooling2D(pool_size=pool_size),
          keras.layers.Conv2D(40, filter_size, activation="relu"),
          keras.layers.Conv2D(128, filter_size, activation="relu"),
          keras.layers.Conv2D(40, filter_size, activation="relu"),
          keras.layers.MaxPooling2D(pool_size=pool_size),
          keras.layers.Flatten(),
          keras.layers.Dense((matrixSize*matrixSize)*40, activation="sigmoid"),
          keras.layers.Dense((matrixSize*matrixSize)*40, activation="sigmoid"),
          keras.layers.Dense((matrixSize*matrixSize), activation="sigmoid"),
          keras.layers.Reshape((matrixSize, matrixSize))
        ])
        model.compile(optimizer="adam",loss="mean_squared_error", metrics=["accuracy"])
        model.summary()
        return model

    def create_game(self, datasetSize):
        """
        Generate different sets of boards

        Parameters
        ----------
        datasetSize : int
            the number of game boards we want to generate
        """
        Xfin = []; yfin = []
        nb_board_game = 0
        pbar = tqdm(total = datasetSize,  desc = "Creating winning games")
        while nb_board_game < datasetSize:
            nb_temp_game = 0
            cnt = 0
            while not self.win():
                if(cnt%20==0):
                    Xfin.append(self.get_tiles_revealed_value())
                    yfin.append(self.get_mine_peri())
                    nb_temp_game +=1
                x = random.randint(0, LEVEL[0]-1)
                y = random.randint(0, LEVEL[0]-1)
                while(self.grid.itemAtPosition(y, x).widget().is_mine):
                    x = random.randint(0, LEVEL[0]-1)
                    y = random.randint(0, LEVEL[0]-1)
                self.AI_turn(x, y)
                cnt+=1
            nb_board_game+= nb_temp_game
            self.update_status(STATUS_READY)
            self.reset()
            pbar.update(nb_temp_game)
        pbar.close()
        return Xfin, yfin

    def button_AI_learn_pressed(self):
        """
        Steps to do in order to train the model with all the different game
        """
        episodes = 10; datasetSize = 5000000
        res = self.warning_before_learn()
        if(res == 1024): # +/- 7h of training for 8x8
            try:
                shutil.rmtree('model/model_cnn')
            except:
                pass
            Xfin, yfin = self.create_game(datasetSize)
            n_inputs, n_outputs = len(Xfin[0]), len(yfin[0])
            model = self.set_model(n_inputs)
            X_train, X_test, Y_train, Y_test = train_test_split(np.array(Xfin, dtype="float64"), np.array(yfin, dtype="float64"), test_size=0.1, random_state=seed)
            X_train = X_train.reshape((X_train.shape[0], LEVEL[0], LEVEL[0], 1))
            X_test = X_test.reshape((X_test.shape[0], LEVEL[0], LEVEL[0], 1))
            history = model.fit(X_train, Y_train, batch_size=200, shuffle=True, epochs=episodes, validation_split=0.1, validation_data=(X_test, Y_test))
            score = model.evaluate(X_test, Y_test, verbose=0)
            self.save_model(model)
            print("Number board:", datasetSize)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])

    def button_AI_play_pressed(self):
        """
        Code execute to test the prediction made by the model
        """
        avg_score = 0; wins = 0
        model = self.load_model()
        nb_game = self.get_number_of_play()
        for i in tqdm(range(0, nb_game),  desc = "Playing games"):
            self.update_pbar(i/nb_game*100, False)
            score = 0
            while not self.win():
                QApplication.processEvents()
                testX = np.array([self.get_tiles_revealed_value()])
                test_x = np.array(testX[0].transpose())
                test_x = test_x.reshape(1, LEVEL[0], LEVEL[0], 1)
                yhat = model.predict(test_x)
                peri = self.get_perimeter()
                CURRENT_REVEALED = self.get_pos_of_revealed()
                score = len(CURRENT_REVEALED)
                x, y = supersmart.act(yhat, peri, CURRENT_REVEALED)
                fpos, mpos = supersmart.flag(yhat, peri, CURRENT_REVEALED)
                if(fpos[0]!=None):
                    ftile = self.grid.itemAtPosition(fpos[1], fpos[0]).widget()
                    mtile.unmark()
                    ftile.flag()
                if(mpos[0]!=None):
                    mtile = self.grid.itemAtPosition(mpos[1], mpos[0]).widget()
                    mtile.mark(0)
                self.AI_turn(x, y)
            if self.get_status() == STATUS_SUCCESS:
                wins += 1
            avg_score += score
            self.reset()
            supersmart.reset()
            self.winrate.setText(str(round(wins/nb_game*100,2))+"%")
            self.score_text.setText("Average score : ")
            self.score.setText(str(round(avg_score/nb_game, 2)))
        self.update_pbar(0, True)

    def AI_turn(self, x, y):
        """
        Make the different action of a normal turn in game

        Parameters
        ----------
        x : int
            x position of the next click of the AI
        y : int
            y position of the next click of the AI
        """
        tile = self.grid.itemAtPosition(y, x).widget()
        if(not tile.is_revealed):
            tile.click()
            tile.reveal()
            if tile.is_mine:  # GAMEOVER
                tile.ohno.emit()
                self.update_status(STATUS_FAILED)

# ===============================================================================
# ALGORITHMIC SOLVER
# ===============================================================================
    def button_solve_pressed(self):
        """
        Run the logical algorithmic solution to resole the game
        """
        wins = 0; previous = 0
        nb_game = self.get_number_of_play()
        max_score = LEVEL[0]*LEVEL[0] - LEVEL[1]
        scores = list()
        win_history = list()
        for episode in tqdm(range(nb_game), desc = "Solving games"):
            self.update_pbar(episode/nb_game*100, False)
            tile = None; SCORE = 0
            while not self.win():
                QApplication.processEvents()
                revealed = self.get_revealed_tiles()
                SCORE = len(revealed)
                if tile is not None and tile.is_mine and tile.is_revealed:
                    self.update_status(STATUS_FAILED)
                    break
                tmp = list()
                for item in revealed:
                    tmp.append(self.get_surrounding(item.x, item.y))
                # Filter unneeded
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
                perimeter = self.get_perim_as_tile()
                final_perimeter = set()
                for tile in perimeter:
                    check = False
                    for neighbor in tile.neighbors:
                        if not neighbor.is_revealed and not neighbor.is_flagged and not neighbor.is_start:
                            check = True
                            break
                    if check:
                        final_perimeter.add(tile)
                perimeter_neighbors = set()
                for tile in perimeter: # Preprocess neighboring tiles
                    surroundings = tile.neighbors
                    for n_tile in surroundings:
                        if n_tile.is_revealed:
                            perimeter_neighbors.add(n_tile)
                free, flags, certainty = rule3(final_perimeter, perimeter_neighbors)
                if certainty == 1:
                    for tile in free:
                        tile.click()
                        tile.reveal()
                else:
                    tile = random.choice(free)
                    tile.click()
                    tile.reveal()
                    if tile.is_mine:
                        self.update_status(STATUS_FAILED)
                        continue
                for tile in flags:
                    tile.flag()
            if self.get_status() == STATUS_SUCCESS:
                wins += 1
            self.reset_map()
            self.winrate.setText(str(round(wins/nb_game*100,2))+"%")
            self.score_text.setText("Average score : ")
            self.score.setText(str(round(previous/nb_game, 2)))
            previous += SCORE
        self.update_pbar(0, True)


if __name__ == '__main__':
    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont('SansSerif', 8))
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
