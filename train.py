#for Flappy_bird game
from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *

#numpy
import numpy as np

#Keras is a deep learning libarary
from keras.models import Sequential
from keras.layers.core import Dense

#for loading pretrained model
# from keras.models import load_model
# model = load_model('models/flappy_bird_model.h5')

#modified flappy game for reinforcement learning
class flappy_bird:
    #traning at higher fps
    FPS = 300
    SCREENWIDTH = 288
    SCREENHEIGHT = 512
    # amount by which base can maximum shift to left
    PIPEGAPSIZE = 125  # gap between upper and lower part of pipe
    BASEY = SCREENHEIGHT * 0.8
    # image, sound and hitmask  dicts
    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # other variables
    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY = -9  # player's velocity along Y, default same as playerFlapped
    playerMaxVelY = 10  # max vel along Y, max descend speed
    playerMinVelY = -8  # min vel along Y, max ascend speed
    playerAccY = 1  # players downward accleration
    playerRot = 0  # player's rotation
    playerVelRot = 0  # angular speed
    playerRotThr = 20  # rotation threshold
    playerFlapAcc = -9  # players speed on flapping
    playerFlapped = False  # True when player flaps
    currPipe = 0
    reward = 0

    score = playerIndex = loopIter = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    playerx = playery = 0

    basex = 0
    baseShift = 0

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = []
    newPipe2 = []

    # list of upper pipes
    upperPipes = []

    # list of lowerpipe
    lowerPipes = []

    # list of all possible players (tuple of 3 positions of flap)
    PLAYERS_LIST = (
        # red bird
        (
            'assets/sprites/redbird-upflap.png',
            'assets/sprites/redbird-midflap.png',
            'assets/sprites/redbird-downflap.png',
        ),
        # blue bird
        (
            # amount by which base can maximum shift to left
            'assets/sprites/bluebird-upflap.png',
            'assets/sprites/bluebird-midflap.png',
            'assets/sprites/bluebird-downflap.png',
        ),
        # yellow bird
        (
            'assets/sprites/yellowbird-upflap.png',
            'assets/sprites/yellowbird-midflap.png',
            'assets/sprites/yellowbird-downflap.png',
        ),
    )

    # list of backgrounds
    BACKGROUNDS_LIST = (
        'assets/sprites/background-day.png',
        'assets/sprites/background-night.png',
    )

    # list of pipes
    PIPES_LIST = (
        'assets/sprites/pipe-green.png',
        'assets/sprites/pipe-red.png',
    )

    try:
        xrange = range
    except NameError:
        xrange = range

    def __init__(self):
        global SCREEN, FPSCLOCK
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')

        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

        # sounds
        if 'win' in sys.platform:
            soundExt = '.wav'
        else:
            soundExt = '.ogg'

        self.SOUNDS['die'] = pygame.mixer.Sound('assets/audio/die' + soundExt)
        self.SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit' + soundExt)
        self.SOUNDS['point'] = pygame.mixer.Sound('assets/audio/point' + soundExt)
        self.SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
        self.SOUNDS['wing'] = pygame.mixer.Sound('assets/audio/wing' + soundExt)

        self.reset()

    def mainGame(self, action):
        self.reward = 0
        #         while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if (event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)) :
                if self.playery > -2 * self.IMAGES['player'][0].get_height():
                    self.playerVelY = self.playerFlapAcc
                    self.playerFlapped = True
                    self.SOUNDS['wing'].play()
        if action == 1:
            if self.playery > -2 * self.IMAGES['player'][0].get_height():
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                self.SOUNDS['wing'].play()

        # check for crash here
        game_over = self.checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
                                    self.upperPipes, self.lowerPipes)[0]

        if game_over:
            self.reward = -10

        # check for score
        playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                self.reward = 40
                self.SOUNDS['point'].play()

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 0

        self.playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
        self.playery += min(self.playerVelY, self.BASEY - self.playery - self.playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # print(self.upperPipes[0]['x'])
        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.currPipe = 0
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(self.IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
        # print score so player overlaps the score
        self.showScore(self.score)

        # Player rotation has a threshold
        visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            visibleRot = self.playerRot

        playerSurface = pygame.transform.rotate(self.IMAGES['player'][self.playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (self.playerx, self.playery))

        # padding the sprites
        pad = 20

        # distance of pipe from the player bird
        distance = self.upperPipes[self.currPipe]['x'] - self.playerx - self.IMAGES['player'][0].get_width()

        # updating the current target pipe to the next pipe
        if distance + self.IMAGES['pipe'][0].get_width() + self.IMAGES['player'][0].get_width() <  0:
            self.currPipe += 1
            distance = self.upperPipes[self.currPipe]['x'] - self.playerx - self.IMAGES['player'][0].get_width()

        # distance of upper pipe from the upper part of bird
        upper_pipe_dis = (self.playery - pad ) - (self.lowerPipes[self.currPipe]['y'] - self.PIPEGAPSIZE)

        # distance of lower pipe from the lower part of bird
        lower_pipe_dis = self.lowerPipes[self.currPipe]['y'] - (self.playery + self.IMAGES['player'][0].get_width() + pad)

        # Grid formation
        if distance < 140:
            distance = int(distance) - (int(distance) % 10)
        else:
            distance = int(distance) - (int(distance) % 50)

        if upper_pipe_dis < 100 and upper_pipe_dis > 0:
            upper_pipe_dis = int(upper_pipe_dis) - (int(upper_pipe_dis) % 5)
        else:
            upper_pipe_dis = int(upper_pipe_dis) - (int(upper_pipe_dis) % 15)

        if lower_pipe_dis < 100 and lower_pipe_dis > 0:
            lower_pipe_dis = int(lower_pipe_dis) - (int(lower_pipe_dis) % 5)
        else:
            lower_pipe_dis = int(lower_pipe_dis) - (int(lower_pipe_dis) % 15)

        #rewards
        if lower_pipe_dis <= -25 or upper_pipe_dis <= -25:
            self.reward = -7
        else:
            self.reward = 5

        pygame.display.update()
        FPSCLOCK.tick(self.FPS)

        return [
            [upper_pipe_dis, lower_pipe_dis, distance],
            self.reward,
            game_over,
            self.score
        ]

    def playerShm(self, playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm['val']) == 8:
            playerShm['dir'] *= -1

        if playerShm['dir'] == 1:
            playerShm['val'] += 1
        else:
            playerShm['val'] -= 1

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        # print(gapY - pipeHeight)
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = self.SCREENWIDTH + 10
        # print(gapY - pipeHeight)
        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE},  # lower pipe
        ]

    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0  # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (self.SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASEY - 1:
            return [True, True]
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in self.xrange(rect.width):
            for y in self.xrange(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

    def reset(self):
        self.pipeVelX = -4

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY = -9  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerRot = 0  # player's rotation
        self.playerVelRot = 0  # angular speed
        self.playerRotThr = 20  # rotation threshold
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps
        self.currPipe = 0

        self.score = self.playerIndex = self.loopIter = 0
        self.playerIndexGen = cycle([0, 1, 2, 1])
        self.playerx = playery = 0

        self.basex = 0
        self.baseShift = 0

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = []
        self.newPipe2 = []

        # list of upper pipes
        self.upperPipes = []

        # list of lowerpipe
        self.lowerPipes = []

        # select random background sprites
        randBg = random.randint(0, len(self.BACKGROUNDS_LIST) - 1)
        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(self.PLAYERS_LIST) - 1)
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(self.PIPES_LIST) - 1)
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hitmask for pipes
        # print(self.IMAGES['pipe'][0])
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )

        self.playerx = int(self.SCREENWIDTH * 0.2)
        self.playery = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)
        playerShmVals = {'val': 0, 'dir': 1}
        self.playery = self.playery + playerShmVals['val']

        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]

    def clear(self):
        pygame.quit()
        sys.exit()

    def currState(self):

        #padding
        pad = 20

        #calculating distances
        distance = self.upperPipes[self.currPipe]['x'] - self.playerx - self.IMAGES['player'][0].get_width()
        upper_pipe_dis = (self.playery - pad ) - (self.lowerPipes[self.currPipe]['y'] - self.PIPEGAPSIZE)
        lower_pipe_dis =   self.lowerPipes[self.currPipe]['y'] - (self.playery + self.IMAGES['player'][0].get_width() + pad)

        #creating Grid
        if distance < 140:
            distance = int(distance) - (int(distance) % 10)
        else:
            distance = int(distance) - (int(distance) % 50)

        if upper_pipe_dis < 100 and upper_pipe_dis > 0:
            upper_pipe_dis = int(upper_pipe_dis) - (int(upper_pipe_dis) % 5)
        else:
            upper_pipe_dis = int(upper_pipe_dis) - (int(upper_pipe_dis) % 15)

        if lower_pipe_dis < 100 and lower_pipe_dis > 0:
            lower_pipe_dis = int(lower_pipe_dis) - (int(lower_pipe_dis) % 5)
        else:
            lower_pipe_dis = int(lower_pipe_dis) - (int(lower_pipe_dis) % 15)

        return [upper_pipe_dis , lower_pipe_dis, distance]


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
            # print(self.memory[idx][0])

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
    checkpoint_score = 30
    create_checkpoint = True
    score = 0
    done_traning = False
    for e in range(epochs):
        if e > 15 :
            epsilon = 0.01
        # Creating the game environment
        env = flappy_bird()
        env.reset()
        game_over = False

        # get initial input
        input_t = env.currState()

        #converting current state list to numpy array
        input_t = np.array([input_t]) / 500

        while not game_over:
            action = 0
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
            input_t = np.asarray([data[0]]) / 500
            reward = data[1]
            game_over = data[2]
            if data[3] > score :
                score = data[3]
                create_checkpoint = True

            """
            The experiences < s, a, r, s’ > we make during gameplay are our training data.
            Here we first save the last experience, and then load a batch of experiences to train our model
            """

            # storing experience into ExperienceReplay
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # Load batch of experiences (in this case 1)
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # train model on experiences
            model.train_on_batch(inputs, targets)

            if score % checkpoint_score == 0 and score > 0 and create_checkpoint == True:
                if checkpoint > 3:
                    done_traning = True
                    break
                model.save('models/flappy_bird_model_'+str(checkpoint)+'.h5')
                checkpoint += 1
                create_checkpoint = False

            if game_over == True:
                print("Game :  ", e,"  Score :  ", data[3], "  Max Score :  ", score)
        if done_traning == True:
            break

def model_t():
    model = Sequential()
    model.add(Dense(hidden_size, input_dim = input_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))
    model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
    return model


max_memory = 1000  # Maximum number of experiences we are storing
epsilon = 0.3  # exploration
num_actions = 2  # [move_left, stay, move_right]
hidden_size = 64  # Size of the hidden layers
batch_size = 1  # Number of experiences we use for training per batch
input_size = 3  # Size of the playing field

model = model_t()
model.summary()
exp_replay = ExperienceReplay(max_memory=max_memory)


epoch = 200
try:
    train(model,epoch,epsilon)
except:
    # creates a HDF5 file (saving trained parameters)
    model.save('models/flappy_bird_model_final.h5')
    del model

print("Traning done ... Pick one of the trained model and check its preformance by loading the model in flappy.py file.....")