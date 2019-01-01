#imports
from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *

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
            self.reward = -5

        # check for score
        playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                self.reward = 30
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
        if lower_pipe_dis <= -10 or upper_pipe_dis <= -10:
            self.reward = -1
        else:
            self.reward = 2


        pygame.display.update()
        FPSCLOCK.tick(self.FPS)
        #print("player_y : ",self.playery,"up_dis : " ,upper_pipe_dis , " lp_dis : ",lower_pipe_dis," dis : ",distance,"rewards : ",self.reward)
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
