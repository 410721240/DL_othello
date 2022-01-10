import numpy as np
from tensorflow.python.ops.gen_math_ops import Max
from othello.OthelloUtil import getValidMoves, executeMove, isValidMove
from sklearn.preprocessing import MinMaxScaler



class OthelloGame(np.ndarray):
    BLACK = 1
    WHITE = -1
    weights = [
    [ 80, -26, 24,  -1,  -5, 28, -18,  76],
    [-23, -39, -18, -9, -6, -8, -39, -1],
    [ 46,  -16,  4,   1,   -3, 6,   -20,  52],
    [ -13,   -5,   2,   -1,   4,  3,   -12,   -2],
    [ -5,   -6,   1,   -2,   -3,  0,   -9,   -5],
    [ 48,  -13,  12,   5,   0,  5,  -24,  41],
    [-27, -53, -11, -1, -11, -16, -58, -15],
    [ 87, -25, 27,  -1,  5, 36, -3,  100]]
    weights = np.array(weights, dtype='float32').reshape(64,-1)
    weights=MinMaxScaler(feature_range=(1, 50)).fit(weights).transform(weights).reshape(64)

    
    def __new__(cls, n):
        return super().__new__(cls, shape=(n,n), dtype='int')
    
    def __init__(self, n):
        self.n=n
        self.current_player=OthelloGame.BLACK
        self[np.where(self!=0)]=0
        self[int(n/2)][int(n/2)]=OthelloGame.WHITE
        self[int(n/2)-1][int(n/2)-1]=OthelloGame.WHITE
        self[int(n/2)-1][int(n/2)]=OthelloGame.BLACK
        self[int(n/2)][int(n/2)-1]=OthelloGame.BLACK
        
    def move(self, position):
        if isValidMove(self, self.current_player, position):
            executeMove(self, self.current_player, position)
            self.current_player=-self.current_player
        else:
            raise Exception('invalid move')

    def set_board(self, board):
        self[:] = board

    def isEndGame(self):
        white_valid_moves=len(getValidMoves(self, OthelloGame.WHITE))
        black_valid_moves=len(getValidMoves(self, OthelloGame.BLACK))
        if white_valid_moves==0 and black_valid_moves==0:
            v,c=np.unique(self, return_counts=True)
            white_count=c[np.where(v==OthelloGame.WHITE)]
            black_count=c[np.where(v==OthelloGame.BLACK)]
            if white_count>black_count:
                return OthelloGame.WHITE
            elif black_count>white_count:
                return OthelloGame.BLACK
            else:
                return 0
        else:
            return None
    
    def play(self, black, white, verbose=True):
        while self.isEndGame() == None:
            if verbose:
                print('{:#^30}'.format( ' Player '+str(self.current_player)+' ' ))
                self.showBoard()
            if len(getValidMoves(self, self.current_player))==0:
                if verbose:
                    print('no valid move, next player')
                self.current_player=-self.current_player
                continue
            if self.current_player==OthelloGame.WHITE:
                position=white.getAction(self.clone(), self.current_player)
            else:
                position=black.getAction(self.clone(), self.current_player)
            try:
                self.move(position)
                
            except:
                if verbose:
                    print('invalid move', end='\n\n')
                continue
        
        if verbose:
            print('---------- Result ----------', end='\n\n')
            self.showBoard()
            print()
            print('Winner:', self.isEndGame())
        return self.isEndGame()

    def play_one_move(self, position, color, verbose=True):
        self.current_player = color
        if verbose:
                print('{:#^30}'.format( ' Player '+str(self.current_player)+' ' ))
                self.showBoard()
        if len(getValidMoves(self, self.current_player))==0:
            if verbose:
                print('no valid move, next player')
            self.current_player=-self.current_player

        # print(self.current_player)
        pre = getValidMoves(self, self.current_player)
        final_score=self.weights[pre[0][0]*8+pre[0][1]]
        self.move(position)
        #if killer move wins, use it
        white_valid_moves=len(getValidMoves(self, OthelloGame.WHITE))
        black_valid_moves=len(getValidMoves(self, OthelloGame.BLACK))
        if white_valid_moves==0 and black_valid_moves==0:
            v,c=np.unique(self, return_counts=True)
            white_count=c[np.where(v==OthelloGame.WHITE)]
            black_count=c[np.where(v==OthelloGame.BLACK)]
            if color==1 and black_count>white_count:
                return 100
            if color==-1 and black_count<white_count:
                return 100


        valid_positions = getValidMoves(self, self.current_player)
        # print(valid_positions)
        # print(self.current_player)
        # print(valid_positions)

        minimax=[]
        # print('=======in minimax 1======')
        # print(valid_positions)
        if len(valid_positions)==0:
            return final_score

        for valid_position in valid_positions:
            if len(valid_positions)==1:
                minimax.append(self.weights[valid_position[0]*8+valid_position[1]])
                break
            temp_game = self.clone()
            x= temp_game.play_max(valid_position, verbose=False)
            minimax.append(x)
            del temp_game
        # if minimax == []:
        #     for i in valid_positions:
        #         minimax.append(self.weights[i[0]*8+i[1]])
        return np.min(minimax)

    def play_max(self, position, verbose=True):
        if verbose:
                print('{:#^30}'.format( ' Player '+str(self.current_player)+' ' ))
                self.showBoard()
        if len(getValidMoves(self, self.current_player))==0:
            if verbose:
                print('no valid move, next player')
            self.current_player=-self.current_player
        self.move(position)
        valid_positions = getValidMoves(self, self.current_player)
        
        valids=np.zeros((self.size), dtype='int')
        valids[ [i[0]*8+i[1] for i in valid_positions] ]=1
        # print('amax=====')
        # print(np.amax(self.weights*valids))
        return np.amax(self.weights*valids)   
    
    def showBoard(self):
        corner_offset_format='{:^'+str(len(str(self.n))+1)+'}'
        print(corner_offset_format.format(''), end='')
        for i in range(self.n):
            print('{:^3}'.format( chr(ord('A')+i) ), end='')
        print()
        print(corner_offset_format.format(''), end='')
        for i in range(self.n):
            print('{:^3}'.format('-'), end='')
        print()
        for i in range(self.n):
            print(corner_offset_format.format(i+1), end='')
            for j in range(self.n):
                if isValidMove(self, self.current_player, (i,j)):
                    print('{:^3}'.format('∎'), end='')
                else:
                    print('{:^3}'.format(self[i][j]), end='')
            print()
    
    def clone(self):
        new=self.copy()
        new.n=self.n
        new.current_player=self.current_player
        return new
    
