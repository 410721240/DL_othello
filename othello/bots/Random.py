import numpy as np
from othello.OthelloUtil import getValidMoves
from othello.OthelloGame import OthelloGame

class BOT3():
    def __init__(self, *args, **kargs):
        pass
    
    def getAction(self, game, color):
        valid_positions=getValidMoves(game, color)
        temp = np.array(game) #converts game to ndarray; possible types: othelloGame, ndarray
        minimax = []
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

            x= temp_game.play_one_move(valid_position, color, verbose=False)
            minimax.append(x)
            del temp_game
        minimax = np.array(minimax)
