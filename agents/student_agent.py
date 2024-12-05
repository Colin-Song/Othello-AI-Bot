#python simulator.py --player_1 student_agent --player_2 gpt_greedy_corners_agent --display
#python simulator.py --player_1 student_agent --player_2 random_agent --display
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        # time limit
        self.time_limit = 1.97
        # starting time
        self.start_time = None

        # tile values on the board for different sizes
        self.value_boards = {
            6: np.array([
                [120, -20,  20,  20, -20, 120],
                [-20, -40,  -5,  -5, -40, -20],
                [ 20,  -5,  15,  15,  -5,  20],
                [ 20,  -5,  15,  15,  -5,  20],
                [-20, -40,  -5,  -5, -40, -20],
                [120, -20,  20,  20, -20, 120]
            ]),
            8: np.array([
                [120, -20,  20,    5,    5,  20, -20, 120],
                [-20, -40,  -5,   -5,   -5,  -5, -40, -20],
                [ 20,  -5,  15,    3,    3,  15,  -5,  20],
                [  5,  -5,   3,   15,   15,   3,  -5,   5],
                [  5,  -5,   3,   15,   15,   3,  -5,   5],
                [ 20,  -5,  15,    3,    3,  15,  -5,  20],
                [-20, -40,  -5,   -5,   -5,  -5, -40, -20],
                [120, -20,  20,    5,    5,  20, -20, 120]
            ]),
            10: np.array([
                [120, -20,  20,   5,     5,    5,    5,   20, -20, 120],
                [-20, -40,  -5,  -5,    -5,   -5,   -5,   -5, -40, -20],
                [ 20,  -5,  15,   3,     3,    3,    3,   15,  -5,  20],
                [  5,  -5,   3,   10,   10,   10,   10,    3,  -5,   5],
                [  5,  -5,   3,   10,   15,   15,   10,    3,  -5,   5],
                [  5,  -5,   3,   10,   15,   15,   10,    3,  -5,   5],
                [  5,  -5,   3,   10,   10,   10,   10,    3,  -5,   5],
                [ 20,  -5,  15,    3,    3,    3,    3,   15,  -5,  20],
                [-20, -40,  -5,   -5,   -5,   -5,   -5,   -5, -40, -20],
                [120, -20,  20,    5,    5,    5,    5,   20, -20, 120]
            ]),
            12: np.array([
                [120, -20,  20,   5,   5,    5,    5,    5,   5,   20, -20, 120],
                [-20, -40,  -5,  -5,  -5,   -5,   -5,   -5,  -5,   -5, -40, -20],
                [ 20,  -5,  15,   3,   3,    3,    3,    3,   3,   15,  -5,  20],
                [  5,  -5,   3,   7,   7,    7,    7,    7,   7,    3,  -5,   5],
                [  5,  -5,   3,   7,  10,   10,   10,   10,   7,    3,  -5,   5],
                [  5,  -5,   3,   7,  10,   15,   15,   10,   7,    3,  -5,   5],
                [  5,  -5,   3,   7,  10,   15,   15,   10,   7,    3,  -5,   5],
                [  5,  -5,   3,   7,  10,   10,   10,   10,   7,    3,  -5,   5],
                [  5,  -5,   3,   7,   7,    7,    7,    7,   7,    3,  -5,   5],
                [ 20,  -5,  15,   3,   3,    3,    3,    3,   3,   15,  -5,  20],
                [-20, -40,  -5,  -5,  -5,   -5,   -5,   -5,  -5,   -5, -40, -20],
                [120, -20,  20,   5,   5,    5,    5,    5,   5,   20, -20, 120]
            ])
        }

    def step(self, chess_board, player, opponent):
        # set current start time
        self.start_time = time.time()
        # node class to save info for gamestates
        class Node:
            def __init__(self, parent=None, move=None):
                # parent (previous gamestate)
                self.parent = parent
                # children (valid moves in current gamestate)
                self.children = []
                # number of times the gamestate has been visited
                self.visits = 0
                # the value of the gamestate
                self.value = 0.0
                # the move that led to the gamestate
                self.move = move
                # the valid moves that have not been tried
                self.valid_moves_left = None

            # function to check if there are any valid moves that have not been tried in the gamestate
            def is_fully_expanded(self):
                return self.valid_moves_left is not None and len(self.valid_moves_left) == 0

            # function to find the best child using the UCT
            def best_child(self, c_param=1.4):
                choices_weights = [(child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits)) for child in self.children]
                return self.children[np.argmax(choices_weights)]

        # functions for MCTS
        # function for tree policy in MCTS (selection/expansion phase in MCTS)
        def tree_policy(node, board, cur_player, cur_opp):
            # while loop keeps running as long as we are within the time limit
            while (time.time() - self.start_time <= self.time_limit):
                # if the gamestate has no valid moves that we haven't tried
                if (node.valid_moves_left is None):
                    # set it to valid moves for the gamestate
                    node.valid_moves_left = get_valid_moves(board, cur_player)
                
                # if there is a move in valid_moves_left
                if node.valid_moves_left:
                    # then get a move from the list 
                    move = node.valid_moves_left.pop()
                    # execute the move
                    execute_move(board, move, cur_player)
                    # create a new node for the gamestate where we set the parent to the current node and the move to the move that led to the gamestate
                    child_node = Node(parent=node, move=move)
                    # append the child to the node
                    node.children.append(child_node)
                    # return the child, board after executing the move, current player, and current opponent
                    return child_node, board, cur_player, cur_opp
                # if there is no move in valid_moves_left
                else:
                    # if the node also has no children
                    if not node.children:
                        # return the node, board, current player, and current opponent
                        return node, board, cur_player, cur_opp
                    # if the node has children
                    else:
                        # select the best child using UCT
                        node = node.best_child()
                        # execute the move that leads to the best child
                        execute_move(board, node.move, cur_player)
                    # swap player and opponent
                    cur_player, cur_opp = cur_opp, cur_player

            # return the node, board, current player, and current opponent
            return node, board, cur_player, cur_opp

        # function for default policy in MCTS (simulation phase in MCTS)
        def default_policy(board, cur_player, cur_opp):
            player = cur_player
            # set max depth limit
            max_depth = 10
            # current depth is 0
            depth = 0
            
            # while loop keeps running while depth is less than the depth limit
            while (depth < max_depth):
                # if we're past the time limit
                if (time.time() - self.start_time >= self.time_limit):
                    # then break out of the while loop
                    break

                # get the valid moves for the player given the gamestate
                valid_moves = get_valid_moves(board, cur_player)
                # if there are valid moves available for the player
                if valid_moves:
                    # using heuristics, select the best valid move
                    move = select_move_using_heuristics(board, valid_moves)
                    # execute the move
                    execute_move(board, move, cur_player)
                # if there are no valid moves for the player
                else:
                    # swap players
                    cur_player, cur_opp = cur_opp, cur_player
                    # get the valid moves for the opponent
                    valid_moves = get_valid_moves(board, cur_player)
                    # if the opponent also has no valid moves
                    if not valid_moves:
                        # break out of the while loop
                        break

                # swap players
                cur_player, cur_opp = cur_opp, cur_player
                # increase depth by 1
                depth += 1

            # return the evaluation of the gamestate for the board
            return evaluate_gamestate(board, player, cur_opp)
        
         # function to select the best move from valid_moves using heuristics
        def select_move_using_heuristics(board, valid_moves):
            # initialize the best value to be -infinity
            best_score = -float('inf')
            # initialize the best move to be none
            best_move = None

            # iterate through valid_moves list
            for move in valid_moves:
                # get the score of the move from value_boards
                score = self.value_boards[board.shape[0]][move[0], move[1]]
                # if the score of the move is better than the current best score
                if (score > best_score):
                    # set the best score to score
                    best_score = score
                    # set the best move to move
                    best_move = move
            
            # return the best move
            return best_move
        
        # function to evaluate the gamestate
        def evaluate_gamestate(board, player, opp):
            # calculate the player's value of tiles placed based on value_boards
            player_score = np.sum(self.value_boards[board.shape[0]][board == player])
            # calculate the opponents's value of tiles placed based on value_boards
            opp_score = np.sum(self.value_boards[board.shape[0]][board == opp])
            # return the player's score - opponent's score
            return player_score - opp_score

        # function for the backpropagation phase
        def backpropagation(node, value):
            # while loop keeps running if the node exists
            while node is not None:
                # increase the node's visits by 1
                node.visits += 1
                # increase the node's value by the value
                node.value += value
                # set node to its parent to backpropagate the value
                node = node.parent

            
        # functions for Minimax
        # function to find the max value for player
        def max_value(self, board, depth, alpha, beta, player, opp):
            # if the depth is 0 or we reached the end of the game or we exceed the time limit
            if (depth == 0) or (check_endgame(board, player, opp)[0]) or (time.time() - self.start_time >= self.time_limit):
                # return the evaluation of the board given the player and opponent
                return evaluate(board, player, opp)

            # set max score to -infinity
            max_score = -float('inf')
            # get the valid moves for the player for board
            valid_moves = get_valid_moves(board, player)

            # iterate through moves in valid moves list
            for move in valid_moves:
                # make a copy of the board
                new_board = deepcopy(board)
                # execute the move on the board copy for the player
                execute_move(new_board, move, player)
                # call min_value function given the new board and depth - 1
                score = min_value(new_board, depth - 1, alpha, beta, player, opp)
                # get the max score between current max score and the score returned
                max_score = max(max_score, score)
                # if the max score is greater than or equal to beta (prune)
                if (max_score >= beta):
                    # then return max score
                    return max_score 
                # set alpha to be the max between current alpha and max score
                alpha = max(alpha, max_score)
            
            # return the max score
            return max_score

        # function to find the min value for opponent
        def min_value(self, board, depth, alpha, beta, player, opp):
            # if the depth is 0 or we reached the end of the game or we exceed the time limit
            if (depth == 0) or (check_endgame(board, player, opp)[0]) or (time.time() - self.start_time >= self.time_limit):
                # return the evaluation of the board given the player and opponent
                return evaluate(board, player, opp)

            # set min score to infinity
            min_score = float('inf')
            # get the valid moves for the opponent for board
            valid_moves = get_valid_moves(board, opp)

            # iterate thorugh moves in valid moves list
            for move in valid_moves:
                # make a copy of the board
                new_board = deepcopy(board)
                # execute the moves on the board copy for the opponent
                execute_move(new_board, move, opp)
                # call max_value function given the new board and depth - 1
                score = max_value(new_board, depth - 1, alpha, beta, player, opp)

                # get the min score between current min score and the score returned
                min_score = min(min_score, score)
                # if the min score is less than or equal to alpha (prune)
                if min_score <= alpha:
                    # then return min score 
                    return min_score  
                # set beta to be the min between current beta and the min score
                beta = min(beta, min_score)
            
            # return the min score
            return min_score

        # function to evaluate the gamestate given a board
        def evaluate(board, player, opp):
            # count number of player owned tiles on board
            player_score = np.sum(board == player)
            # count number of opponent owned tiles on board
            opp_score = np.sum(board == opp)
            # return player number of tiles - opponent number of tiles
            return player_score - opp_score
    
        
        # function for selecting which algorithm (MCTS or Minimax) to use
        def select_algo(board):
            # find remaining open tiles left on board
            remaining_moves = np.sum(board == 0)
            # if there are 15 tiles left
            if remaining_moves <= 15:
                # then use Minimax
                return "minimax"
            # use MCTS when there are more than 15 tiles lef ton board
            return "mcts"

        # set the root node (current gamestate)
        root_node = Node()
        # get the valid moves for the player for the board
        valid_moves = get_valid_moves(chess_board, player)
        # set best move to None
        best_move = None
        # if there are no valid moves
        if not valid_moves:
            # then return None
            return best_move
        
        # select the algo to use
        algo = select_algo(chess_board)
        
        # if the algo is MCTS
        if algo == 'mcts':
            # while loop runs as long as we're within time limit
            while (time.time() - self.start_time <= self.time_limit):
                # make a copy of the current board
                sim_board = chess_board.copy()
                # set node to the root node
                node = root_node
                # set current player to player
                cur_player = player
                # set current opponent to opponent
                cur_opp = opponent

                # if we go past the time limit
                if (time.time() - self.start_time >= self.time_limit):
                    # break out of while loop
                    break  
                # apply tree policy to get node, board after simulation, current player, and current opponent
                node, sim_board, cur_player, cur_opp = tree_policy(node, sim_board, cur_player, cur_opp)

                # if we got past the time limit
                if (time.time() - self.start_time >= self.time_limit):
                    # break out of while loop
                    break
                # get the value of the board from simulation by applying default policy
                value = default_policy(sim_board, cur_player, cur_opp)

                # backpropagate the value to node and its parents
                backpropagation(node, value)

            # if the root node has children
            if root_node.children:
                # then choose the node with the highest number of visits
                best_move = max(root_node.children, key=lambda c: c.visits).move
            # if the root node has no children
            else:
                # then choose a random move
                best_move = random_move(chess_board, player)
            
            # return the best move
            return best_move
        # if the algo is Minimax
        else:
            # set best score to -infinity
            best_score = -float('inf')
            # set alpha to -infinity
            alpha = -float('inf')
            # set beta to infinity
            beta = float('inf')
            # set depth to 1
            depth = 1

            # while loop keeps running as long as we're within time limit
            while (time.time() - self.start_time <= self.time_limit):
                # iterate through moves in valid moves
                for move in valid_moves:
                    # create deepcopy of board
                    new_board = deepcopy(chess_board)
                    # execute the move on the board copy
                    execute_move(new_board, move, player)
                    # get the score by calling min value
                    score = min_value(new_board, depth - 1, alpha, beta, player, opponent)

                    # if the score is greater than best score
                    if score > best_score:
                        # set best score to score
                        best_score = score
                        # set best move to move
                        best_move = move

                    # set alpha to be max between current alpha and best score
                    alpha = max(alpha, best_score)

                # increase depth by 1
                depth += 1 
            
            # return best move
            return best_move