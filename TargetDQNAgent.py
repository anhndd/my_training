from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract, Multiply
from keras.models import Model

class TargetDQNAgent:
    def __init__(self, action_size):
        self.alpha = 0.0001     # target network update rate
        self.Beta = 0.01        # Leaky ReLU
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(60, 60, 2))
        input_2 = Input(shape=(self.action_size, self.action_size))
        x1 = Conv2D(32, (4, 4), strides=(2, 2),padding='Same', activation=LeakyReLU(alpha=self.Beta))(input_1)
        x1 = Conv2D(64, (2, 2), strides=(2, 2),padding='Same', activation=LeakyReLU(alpha=self.Beta))(x1)
        x1 = Conv2D(128, (2, 2), strides=(1, 1),padding='Same', activation=LeakyReLU(alpha=self.Beta))(x1)
        x1 = Flatten()(x1)
        x1 = Dense(128, activation=LeakyReLU(alpha=self.Beta))(x1)
        x1_value = Dense(64, activation=LeakyReLU(alpha=self.Beta))(x1)
        value = Dense(1, activation=LeakyReLU(alpha=self.Beta))(x1_value)
        x1_advantage = Dense(64, activation=LeakyReLU(alpha=self.Beta))(x1)
        advantage = Dense(self.action_size, activation=LeakyReLU(alpha=self.Beta))(x1_advantage)

        A = Dot(axes=1)([input_2, advantage])
        A_subtract = Subtract()([advantage, A])

        Q_value = Add()([value, A_subtract])

        input_3 = Input(shape=(5,))
        Output = Multiply()([input_3, Q_value])

        model = Model(inputs=[input_1, input_2, input_3], outputs=[Output])
        # model.compile(optimizer= Adam(lr=self.epsilon_r), loss='mse')

        return model

    def replay(self, primary_network_weights):
        target_network_weights = self.model.get_weights()
        # print target_network_weights[len(target_network_weights)-1]
        # target_network_weights = self.alpha*target_network_weights + (1-self.alpha)*primary_network_weights

        for i in range(len(target_network_weights)):
            target_network_weights[i] = self.alpha*target_network_weights[i] + (1-self.alpha)*primary_network_weights[i]
        # print 'new weight'
        # print target_network_weights[len(target_network_weights)-1], primary_network_weights[len(target_network_weights)-1]
        self.model.set_weights(target_network_weights)
