from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from sklearn import neighbors
from ai import AI
from solver import *
from rl import QAgent
from tile import Tile

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

# ===============================================================================
# GUI
# ===============================================================================
"""
Main class use for GUI and the AI managment
"""
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        global LEVEL
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Minesweeper AI")
        self.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("./images/bomb.png"))
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.b_size, self.n_mines = LEVEL
        self.agent = None
        self.manual_play = False

        w = QWidget()
        vb = QVBoxLayout()
        hb = QHBoxLayout()
        hb0 = QHBoxLayout()
        hb1 = QHBoxLayout()
        hb2 = QHBoxLayout()

        self.button_solve = QPushButton("Solve")
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
        self.status_text = QLabel()
        self.status_text.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        f = self.score.font()
        f.setPointSize(10)
        f.setWeight(75)
        self.score.setFont(f)
        self.status_text.setFont(f)
        self.score.setText(str(SCORE))
        self.status_text.setText("0%")
        score = QLabel("Score : ")
        status_text = QLabel("Win rate : ")
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.hide()

        hb.addWidget(score)
        hb.addWidget(self.score)
        hb.addWidget(status_text)
        hb.addWidget(self.status_text)
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

        self.init_map()
        self.update_status(STATUS_READY)
        self.reset_map()
        self.update_status(STATUS_READY)

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
                tile.score.connect(self.update_score)
                tile.manual.connect(self.update_manual)

    """
    Reset all the board, choose random positions for mine and give new value to each tiles
    """
    def reset_map(self):
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

    def get_perim_as_tile(self):
        CURRENT_REVEALED = self.get_pos_of_revealed()
        perimeter = set()
        for pos in CURRENT_REVEALED:
            neighb_tmp = self.get_surrounding(pos[0], pos[1])
            for elem in neighb_tmp:
                if elem not in CURRENT_REVEALED:
                    perimeter.add(elem)
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
        value_mat = np.zeros((self.b_size, self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                value_mat[x, y] = tile.get_value()
        return value_mat

    """
    Return the matrix of the tile's value known on the board
    """
    def get_tiles_revealed_value(self):
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
        mines = np.zeros((self.b_size, self.b_size))
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                tile = self.grid.itemAtPosition(y, x).widget()
                if(tile.is_mine):
                    mines[x, y] = 1
                else:
                    mines[x, y] = 0
        return mines

    """
    Return the matrix with the position of mine in the perieter of revealed tiles
    """
    def get_mine_peri(self):
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
    Update the score
    """
    def update_score(self):
        global SCORE
        revealed = self.get_revealed_tiles()
        SCORE = len(revealed)
        self.score.setText(str(SCORE))

    """
    Update the manual play boolean
    """
    def update_manual(self):
        self.manual_play = True

    """
    Code execute when the game emit the 'ohno' signal which and the game and restart a new one
    """
    def game_over(self):
        global SCORE
        if self.manual_play:
            print("SCORE : ", SCORE)
        SCORE = 0
        self.reveal_map()
        self.update_status(STATUS_FAILED)
        self.reset()

    """
    Check if a game is win or not
    """
    def win(self):
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
    # WARNING: Dialog window
    """
    def warning_before_learn(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning !")
        msg.setText("Warning !")
        msg.setInformativeText("Attention, you will delete the previous saved model and create a new one, do you want to continue? ")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = msg.exec_()
        return retval

    """
    Update the progress bar while doing training/playing
    """
    def update_pbar(self, value, reset=False):
        if(not reset):
            self.pbar.show()
            self.pbar.setValue(value)
        else:
            self.pbar.setValue(0)
            self.pbar.hide()

# ===============================================================================
# REINFORCEMENT LEARNING
# ===============================================================================
    """
    Save the trained RL agent
    """
    def rl_save(self):
        with open('model/q_agent_config.pickle', 'wb') as config_agent:
            pickle.dump(self.agent, config_agent)

    """
    Create a new RL agent and train it
    """
    def rl_learn(self):
        res = self.warning_before_learn()
        if(res == 1024):
            alpha = 0.1; epsilon_max = 0.9; epsilon_min = 0.1; epsilon_decay = 0.99
            self.agent = QAgent(alpha, epsilon_max, epsilon_min, epsilon_decay)
            self.run_episode(True, 1000000)
            self.rl_save()

    """
    Play using the trained RL agent
    """
    def rl_play(self):
        #load a trained agent if no new agent has been created
        if self.agent == None:
            with open('model/q_agent_config_1M_run.pickle', 'rb') as config_agent:
                self.agent = pickle.load(config_agent)
        self.reset_map()
        self.run_episode(False, 500)

    """
    Play the game with a RL agent
    """
    def run_episode(self, training, nb_game):
        nb_states = []
        nb_wins = []
        wins = 0
        if training:
            description = 'Training progress'
        else:
            description = 'Testing progress'
        for episode in tqdm(range(nb_game), desc = description):
            self.update_pbar(episode/nb_game*100, False)
            unproductive_moves = 0
            while not self.win():
                QApplication.processEvents()
                #force game over after more than 200 unproductive moves
                if unproductive_moves > 200:
                    self.update_status(STATUS_FAILED)
                    break
                #select a random tile
                x, y = random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1)
                tile = self.grid.itemAtPosition(y, x).widget()
                #tile has been already picked
                if tile.is_revealed:
                    unproductive_moves += 1
                #get a 3x3 cluster around the tile
                cluster = self.get_surrounding(x, y)
                #get current state of the cluster
                for i in range(len(cluster)):
                    if cluster[i].is_revealed:
                        cluster[i] = cluster[i].get_value()
                    else:
                        cluster[i] = -3
                #get the agent action
                action = self.agent.act(cluster, training)
                #tile clicked
                if action == 1:
                    tile.click()
                    tile.reveal()
                #action == 2 -> tile ignored
                #skip
                if action == -1:
                    continue
                #click + not mine
                if not tile.is_mine and tile.is_revealed:
                    if training:
                        self.agent.learn(cluster, 1, 1, False)
                #click + mine
                elif tile.is_mine and tile.is_revealed:
                    if training:
                        self.agent.learn(cluster, 1, -1, True)
                    self.update_status(STATUS_FAILED)
                    break
                #ignore + not mine
                elif not tile.is_mine and not tile.is_revealed:
                    if training:
                        self.agent.learn(cluster, 2, -1, False)
                #ignore + mine
                elif tile.is_mine and not tile.is_revealed:
                    if training:
                        self.agent.learn(cluster, 2, 1, False)
            if self.get_status() == STATUS_SUCCESS:
                wins += 1
            self.status_text.setText(str(round(wins/nb_game*100,2))+"%")
            self.reset_map()
            if (episode%10000) == 0:
                nb_states.append(len(self.agent.q_table))
                nb_wins.append(wins)
        self.update_pbar(0, True)
        print("WIN RATE:" + str(wins/nb_game*100))

# ===============================================================================
# CONVOLUTIONAL NEURAL NETWORK
# ===============================================================================
    """
    Save the model of CNN
    """
    def save_model(self, model):
        model.save('model/model_cnn')

    """
    Load the model of CNN
    """
    def load_model(self):
        model = keras.models.load_model('model/model_cnn')
        return model

    """
    Define the architecture of the CNN
    """
    def set_model(self, n_inputs):
        matrixSize = n_inputs
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

    """
    Generate different sets of boards
    """
    def create_game(self, datasetSize):
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

    """
    Steps to do in order to train the model with all the different game
    """
    def button_AI_learn_pressed(self):
        avg_score = 0; episodes = 10; datasetSize = 5000000
        res = self.warning_before_learn()
        if(res == 1024): # +/- 7h of training (3000000)
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

    """
    Code execute to test the prediction made by the model
    """
    def button_AI_play_pressed(self):
        global SCORE
        avg_score = 0
        self.update_status(STATUS_READY)
        model = self.load_model()
        nb_test_run = 500
        wins = 0
        for i in tqdm(range(0, nb_test_run),  desc = "Playing games"):
            self.update_pbar(i/nb_test_run*100, False)
            OLDSCORE = 0
            while not self.win():
                QApplication.processEvents()
                testX = np.array([self.get_tiles_revealed_value()])
                test_x = np.array(testX[0].transpose())
                test_x = test_x.reshape(1, LEVEL[0], LEVEL[0], 1)
                yhat = model.predict(test_x)
                peri = self.get_perimeter()
                CURRENT_REVEALED = self.get_pos_of_revealed()
                x, y = supersmart.act(yhat, peri, CURRENT_REVEALED)
                fpos, mpos = supersmart.flag(yhat, peri, CURRENT_REVEALED)
                if(fpos[0]!=None):
                    ftile = self.grid.itemAtPosition(fpos[1], fpos[0]).widget()
                    mtile.unmark()
                    ftile.flag()
                if(mpos[0]!=None):
                    mtile = self.grid.itemAtPosition(mpos[1], mpos[0]).widget()
                    mtile.mark(0)
                OLDSCORE = SCORE
                self.AI_turn(x, y)
            if self.get_status() == STATUS_SUCCESS:
                wins += 1
            self.status_text.setText(str(round(wins/nb_test_run*100,2))+"%")
            avg_score += OLDSCORE
            self.update_status(STATUS_READY)
            SCORE = 0
            self.reset()
            supersmart.reset()
        self.update_pbar(0, True)
        print("WIN RATE:" + str(wins/nb_test_run*100)+ "%")
        print("Avg. score : ", avg_score/nb_test_run)

    """
    Make the different action of a normal turn in game
    """
    def AI_turn(self, x, y):
        global SCORE
        tile = self.grid.itemAtPosition(y, x).widget()
        if(not tile.is_revealed):
            tile.click()
            tile.reveal()
            if tile.is_mine:  # GAMEOVER
                tile.ohno.emit()
                self.update_status(STATUS_FAILED)
            else:
                SCORE += 1
                self.score.setText(str(SCORE))

# ===============================================================================
# ALGORITHMIC SOLVER
# ===============================================================================
    def button_solve_pressed(self):
        wins = 0
        nb_game = 500
        max_score = LEVEL[0]*LEVEL[0] - LEVEL[1]
        previous = 0
        scores = list()
        win_history = list()
        for episode in tqdm(range(nb_game), desc = "Solving games"):
            self.update_pbar(episode/nb_game*100, False)
            tile = None
            SCORE = 0
            while not self.win():
                QApplication.processEvents()
                revealed = self.get_revealed_tiles()
                SCORE = len(revealed)
                self.score.setText(str(SCORE))
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
                # Preprocess neighboring tiles
                for tile in perimeter:
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
            self.status_text.setText(str(round(wins/nb_game*100,2))+"%")
            previous += SCORE
            self.reset_map()
        self.update_pbar(0, True)
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
