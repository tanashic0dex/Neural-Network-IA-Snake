from keras import models, layers, optimizers, activations, losses


class QNetwork:
    """
    Represents the DQN's Neural Network.
    """
    def __init__(self, input_shape, hidden_units, output_size, learning_rate=0.01): 
        """
        Params of the Neural Network.

        :param input_shape: state size 
        :type input_shape: int
        :param hidden_units: number of neurons in each layer.
        :type hidden_units: tupple with dimension (1, 3).
        :param output_size: size of the output.
        :type output_size: int
        :param learning_rate: learning rate of the action-value neural network.
        :type learning_rate: float.
        """
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.model = self.make_model()


    def make_model(self):
        """
        Makes the action-value neural network model using Keras.
        
        :return: action-value neural network.
        :rtype: Keras' model.
        """
        model = models.Sequential() 

        model.add(layers.Dense(self.hidden_units[0], activation=activations.linear, input_dim=self.input_shape))
        model.add(layers.ReLU())
        model.add(layers.Dense(self.hidden_units[1], activation=activations.linear))
        model.add(layers.ReLU())
        model.add(layers.Dense(self.hidden_units[2], activation=activations.linear))
        model.add(layers.ReLU())
        model.add(layers.Dense(self.output_size, activation=activations.linear))

        model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=self.learning_rate))
        model.summary()

        return model


    def predict(self, state, batch_size=1):
        """
        Predict best action to take.
        
        :param state: current state.
        :type state: NumPy array with dimension (1, 18).
        :param batch_size size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        """
        return self.model.predict(state, batch_size)
    

    def train(self, states, action_values, batch_size):
        """
        Train the Neural Network.

        :param states: states taken from memory. 
        :type states: NumPy array with dimension (agen.BATCH/-SIZE, 18)
        :param action_values: values of the actions of the states
        :type action_values: NumPy array with dimension (agen.BATCH_SIZE, 4).
        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int
        """
        self.model.fit(states, action_values, batch_size=batch_size, verbose=0, epochs=1)


    def load(self, name):
        """
        Loads the neural network's weights from disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.load_weights(name)


    def save(self, name):
        """
        Saves the neural network's weights to disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.save_weights(name)
