  """
  Code execute when the user click on the learn AI button
  """
  def button_AI_learn_pressed(self):
      # TODO : Make this function run as parallal
      self.train_AI(500000)


  """
  Define the architecture of the neuronal network
  """
  def set_model(self, n_inputs, n_outputs, episodes):
      global model
      matrixSize = n_inputs

      #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.005, decay_steps=episodes, decay_rate=0.95)
      #rmsprop = keras.optimizers.RMSprop(learning_rate=lr_schedule, momentum=0.1)

      model = keras.models.Sequential([
          keras.layers.Dense((matrixSize*matrixSize)*2, input_shape=(matrixSize,matrixSize), activation="relu"),
          keras.layers.Dropout(0.3),
          keras.layers.Flatten(),
          keras.layers.Dense((matrixSize*matrixSize)*4, activation="sigmoid"),
          keras.layers.Dropout(0.2),
          keras.layers.Dense((matrixSize*matrixSize)*4, activation="sigmoid"),
          keras.layers.Dropout(0.1),
          keras.layers.Dense((matrixSize*matrixSize)*4, activation="sigmoid"),
          keras.layers.Dropout(0.05),
          keras.layers.Dense((matrixSize*matrixSize)*4, activation="relu"),
          keras.layers.Dropout(0.025),
          keras.layers.Dense(matrixSize*matrixSize, activation="sigmoid"),
          keras.layers.Reshape((matrixSize, matrixSize))
      ])
      """
      model = keras.models.Sequential([
              keras.layers.Dense((matrixSize*matrixSize), input_shape=(matrixSize,matrixSize), activation="relu"),
              keras.layers.Dropout(0.1),
              keras.layers.Flatten(),
              keras.layers.Dense((matrixSize*matrixSize)/4, activation="relu"),
              keras.layers.Dropout(0.01),
              keras.layers.Dense((matrixSize*matrixSize)/2, activation="relu"),
              keras.layers.Dropout(0.01),
              keras.layers.Dense(matrixSize*matrixSize, activation="sigmoid"),
              keras.layers.Reshape((matrixSize, matrixSize))
      ])
      """

      #model.compile(optimizer=rmsprop,loss="mean_squared_error", metrics=["accuracy"])
      model.compile(optimizer="adam",loss="mean_squared_error", metrics=["accuracy"])
      model.summary()

  """
  Steps to do in order to train the model with all the different game
  """
  def train_AI(self, datasetSize):
      global SCORE, model
      avg_score = 0
      episodes = 15

      # get_tiles_value : give the value of each tile on the board
      Xfin = []
      yfin = []

      # Create multiple beginning of game (=episodes) and add them to the input list
      # TODO: Apprendre des parties complètes pas juste des débuts de game
      print("Generating", datasetSize,"games :")
      for i in tqdm(range(0, datasetSize)):

          Xfin.append(self.get_tiles_revealed_value())
          #yfin.append(self.get_all_mine())
          yfin.append(self.get_tiles_value())
          #x = random.randint(0, LEVEL[0]-1)
          #y = random.randint(0, LEVEL[0]-1)
          #print(x, y)
          #self.AI_turn(x, y)
          self.update_status(STATUS_READY)
          self.reset()

      # Train the model with all the game in the input list
      n_inputs, n_outputs = len(Xfin[0]), len(yfin[0])
      self.set_model(n_inputs, n_inputs, episodes)

      seed = 7
      np.random.seed(seed)
      X_train, X_test, Y_train, Y_test = train_test_split(np.array(Xfin), np.array(yfin), test_size=0.1, random_state=seed)

      #es = EarlyStopping(monitor='loss', mode='min', verbose=1, min_delta=0.01, patience=episodes)



      print("SIZE X TRAIN", X_train.shape)
      history = model.fit(X_train, Y_train, batch_size=500, shuffle=True, epochs=episodes, validation_split=0.1, validation_data=(X_test, Y_test))

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
