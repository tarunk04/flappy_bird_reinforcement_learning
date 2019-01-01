#numpy
import numpy as np

#Keras is a deep learning libarary
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers

#for loading pretrained model
# from keras.models import load_model
# model = load_model('models/flappy_bird_model.h5')

#importing game
import flappy_class as flappy

flappy = flappy.flappy_bird()

class ExperienceReplay(object):

    def __init__(self, max_memory=100, discount=.9):

        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # Save a state to memory
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):

        # Number of experiences
        len_memory = len(self.memory)

        # Calculate the number of actions that can possibly be taken in the game
        num_actions = model.output_shape[-1]

        # Dimensions of the game field
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        targets = np.zeros((inputs.shape[0], num_actions))

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            # add the state s to the input
            inputs[i:i + 1] = state_t

            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t)[0]

            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1)[0])

            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets

def train(model, epochs,epsilon = 0.1):
    # Train
    # Epochs is the number of games played
    checkpoint = 1
    checkpoint_score = 200
    create_checkpoint = True
    score = 0
    done_traning = False
    for e in range(epochs):
        loss = 0
        counter = 0
        if e > 25 :
            epsilon = 0.01
        # Creating the game environment
        env = flappy
        env.reset()
        game_over = False

        # get initial input
        input_t = env.currState()

        #converting current state list to numpy array
        input_t = np.array([input_t]) / 300
        input_t = np.around(np.float32(input_t), decimals= 4)

        while not game_over:
            input_tm1 = input_t

            if np.random.rand() <= epsilon:
                #Random move for exploration
                action = np.random.randint(0, 1, size=1)[0]
            else:
                #predicting the next move (q has the probablity of all possible move)
                q = model.predict(input_tm1)
                #predicting move withe highest probablity or q value
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            data = env.mainGame(action)
            input_t = np.asarray([data[0]]) / 300
            np.around(np.float32(input_t), decimals=4)
            reward = data[1]
            game_over = data[2]
            if data[3] > score :
                score = data[3]
                create_checkpoint = True

            # storing experience into ExperienceReplay
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # Load batch of experiences (in this case 1)
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # train model on experiences
            loss  += model.train_on_batch(inputs, targets)

            if score % checkpoint_score == 0 and score > 0 and create_checkpoint == True :
                if checkpoint > 3:
                    done_traning = True
                    break
                model.save('models/flappy_bird_model_'+str(checkpoint)+'.h5')
                checkpoint += 1
                create_checkpoint = False

            counter +=1
            if game_over == True:
                print("[-----[Game :  ", e,"]  [Score :  ", data[3], "]  [Max Score :  ", score,"]  [Loss :  ",loss/(batch_size * counter )," ]-----]")
        if done_traning == True:
            break

def model_t():
    model = Sequential()
    model.add(Dense(hidden_size, input_dim = input_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))

    adam = optimizers.adam()

    model.compile(optimizer=adam, loss='categorical_crossentropy')

    return model


max_memory = 1000  # Maximum number of experiences we are storing
epsilon = 0.3  # exploration
num_actions = 2  # [Jump , not Jump]
hidden_size = 64  # Size of the hidden layers
batch_size = 1  # Number of experiences we use for training per batch
input_size = 3  # Size of the Input Dimensions
epoch = 300

model = model_t()
model.summary()
exp_replay = ExperienceReplay(max_memory=max_memory)

try:
    train(model,epoch,epsilon)
except:
    #creates a HDF5 file (saving trained parameters)
    print("ERROR WHILE TRAINING!!!")
model.save('models/flappy_bird_model_final.h5')
del model

print("Traning done ... Pick one of the trained model and check its preformance by loading the model in flappy.py file.....")