from re import VERBOSE
import numpy as np
from tensorflow.python.keras.saving.hdf5_format import preprocess_weights_for_loading
from othello.OthelloUtil import getValidMoves
from othello.OthelloGame import OthelloGame

# #minimax implementation
# class BOT3():
#     board_size=8
#     def __init__(self, *args, **kargs):
#         pass
    
#     def getAction(self, game, color):
#         valid_positions=getValidMoves(game, color)
#         temp = np.array(game) #converts game to ndarray; possible types: othelloGame, ndarray
#         minimax = []
#         #gets every valid moves
#         for valid_position in valid_positions:
#             #if only one is possible, breaks
#             if len(valid_positions)==1:
#                 return valid_position
            
#             #creates clone of 8x8 othelloGame
#             temp_game = OthelloGame(8)
#             #set it as the same as game
#             temp_game.set_board(temp)

#             x= temp_game.play_one_move(valid_position, color, verbose=False)
#             # x = temp_game.get_minimax(valid_position, -color, 2, False)
#             minimax.append(x)
#             del temp_game
#         minimax = np.array(minimax)
#         # print(minimax)

#         valids=np.zeros((game.size), dtype='int')

#         count=0
#         for i in valid_positions:
#             valids[ [i[0]*self.board_size+i[1] ] ]=minimax[count]
#             count+=1

#         position = np.argmax(valids)
        
#         position=(position//self.board_size, position%self.board_size)
#         # print(position)
#         return position

# minimax implementation
class BOT3():
    board_size=8
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
                return valid_position
            
            #creates clone of 8x8 othelloGame
            temp_game = OthelloGame(8)
            #set it as the same as game
            temp_game.set_board(temp)

            # x= temp_game.play_one_move(valid_position, color, verbose=False)
            x = temp_game.get_minimax(valid_position, color, 4, -1000, 1000, maximizingPlayer=False,verbose=False)
            minimax.append(x)
            del temp_game
        minimax = np.array(minimax)
        # print(valid_positions)
        # print(minimax)
        # print(np.argmax(minimax))

        valids=np.zeros((game.size), dtype='int')

        count=0
        for i in valid_positions:
            valids[ [i[0]*self.board_size+i[1] ] ]=minimax[count]
            count+=1

        # position = np.argmax(valids)
        position = valid_positions[np.argmax(minimax)]
        
        # position=(position//self.board_size, position%self.board_size)
        # print(position)
        return position