from itertools import count
from re import I
import numpy as np
from numpy.random.mtrand import random
from othello.OthelloUtil import getValidMoves
from othello.bots.DeepLearning.OthelloModel import OthelloModel
from othello.OthelloGame import OthelloGame
from othello.bots.Random import BOT2
from numpy import random

BOARD_SIZE=8
bot2=BOT2(board_size=BOARD_SIZE)

# weights = [
# [ 500, -100, 100,  50,  50, 100, -100,  500],
# [-100, -200, -50, -50, -50, -50, -200, -100],
# [ 100,  -50,  60,   4,   4, 60,   -50,  100],
# [ 50,   -50,   4,   2,   2,  4,   -50,   50],
# [ 50,   -50,   4,   2,   2,  4,   -50,   50],
# [ 100,  -50,  60,   4,   4,  60,  -50,  100],
# [-100, -200, -50, -50, -50, -50, -200, -100],
# [ 500, -100, 100,  50,  50, 100, -100,  500]]
# weights = np.array(weights, dtype='float32').reshape(64)

# weights = [
# [ 500, -100, 100,  50,  50, 100, -100,  500],
# [-100, -200, -50, -50, -50, -50, -200, -100],
# [ 100,  -50,  60,   4,   4, 60,   -50,  100],
# [ 50,   -50,   4,   2,   2,  4,   -50,   50],
# [ 50,   -50,   4,   2,   2,  4,   -50,   50],
# [ 100,  -50,  60,   4,   4,  60,  -50,  100],
# [-100, -200, -50, -50, -50, -50, -200, -100],
# [ 500, -100, 100,  50,  50, 100, -100,  500]]
# weights = np.array(weights, dtype='float32').reshape(64)+201

class BOT():

    def __init__(self, board_size, *args, **kargs):
        self.board_size=board_size
        self.model = OthelloModel( input_shape=(self.board_size, self.board_size) )
        try:
            self.model.load_weights()
            print('model loaded')
        except:
            print('no model exist')
            pass
        
        self.collect_gaming_data=False
        self.history=[]
    
    def getAction(self, game, color):
        predict = self.model.predict( game )
        valid_positions=getValidMoves(game, color)
        temp = np.array(game) #converts game to ndarray; possible types: othelloGame, ndarray
        minimax = []
        # print(valid_positions)
        #gets every valid moves
        for valid_position in valid_positions:
            #if only one is possible, breaks
            if len(valid_positions)==1:
                minimax.append(1)
                break
            
            #creates clone of 8x8 othelloGame
            temp_game = OthelloGame(8)
            #set it as the same as game
            temp_game.set_board(temp)
            # temp_game.showBoard()
            # print(temp)

            x= temp_game.play_one_move(valid_position, color, verbose=False)
            minimax.append(x)
            del temp_game
        minimax = np.array(minimax)
        
        

        valids=np.zeros((game.size), dtype='int')

        count=0
        for i in valid_positions:
            valids[ [i[0]*self.board_size+i[1] ] ]=minimax[count]
            count+=1
        predict*=valids

        # print(predict)

        position = np.argmax(predict)
        
        if self.collect_gaming_data:
            tmp=np.zeros_like(predict)
            tmp[position]=1.0
            self.history.append([np.array(game.copy()), tmp, color])
        
        position=(position//self.board_size, position%self.board_size)
        # print(position)
        return position
    
    def self_play_train(self, args):
        self.collect_gaming_data=True
        def gen_data():
            def getSymmetries(board, pi):
                # mirror, rotational
                pi_board = np.reshape(pi, (len(board), len(board)))
                l = []
                for i in range(1, 5):
                    for j in [True, False]:
                        newB = np.rot90(board, i)
                        newPi = np.rot90(pi_board, i)
                        if j:
                            newB = np.fliplr(newB)
                            newPi = np.fliplr(newPi)
                        l += [( newB, list(newPi.ravel()) )]
                return l
            self.history=[]
            history=[]
            game=OthelloGame(self.board_size)
            RANDOM  = random.randint(2)
            if RANDOM==0:
                print('random0')
                game.play(self, bot2, verbose=True)#args['verbose']
            elif RANDOM==1:
                print('random1')
                game.play(bot2, self, verbose=True)
            # else:
            #     print('self play')
            #     game.play(self, self, verbose=True)
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b,p in sym:
                    history.append([b, p, player])
            self.history.clear()
            game_result=game.isEndGame()
            return [(x[0],x[1]) for x in history if (game_result==0 or x[2]==game_result)]
        
        data=[]
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print('self playing', i+1)
            data+=gen_data()
        
        self.collect_gaming_data=False
        
        self.model.fit(data, batch_size = args['batch_size'], epochs = args['epochs'])
        self.model.save_weights()
        
