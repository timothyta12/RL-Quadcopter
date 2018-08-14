from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.build_model()
        
    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        network_states = layers.Dense(units=400)(states)
        network_states = layers.BatchNormalization()(network_states)
        network_states = layers.Activation('relu')(network_states)
        
        network_states = layers.Dense(units=300)(states)
        network_actions = layers.Dense(units=300)(actions)
        
        network = layers.Add()([network_states, network_actions])
        network = layers.Activation('relu')(network)
        
        Q_values = layers.Dense(units=1, name='Q_value')(network)
        
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        optimizer = optimizers.Adam(0.001)
        self.model.compile(optimizer=optimizer, loss='mse')
        
        action_gradients = K.gradients(Q_values, actions)
        
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)