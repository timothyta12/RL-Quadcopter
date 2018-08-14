from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        
        self.build_model()
        
    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        
        network = layers.Dense(units=400)(states)
        network = layers.BatchNormalization()(network)
        network = layers.Activation('relu')(network)
        
        network = layers.Dense(units=300, activation='relu')(network)
        network = layers.BatchNormalization()(network)
        network = layers.Activation('relu')(network)

        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(network)
        
        actions = layers.Lambda(lambda x: (x*self.action_range) + self.action_low, name='actions')(raw_actions)
        
        self.model = models.Model(inputs=states, outputs=actions)
        
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        optimizer = optimizers.Adam(0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)