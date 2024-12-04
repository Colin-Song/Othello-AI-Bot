from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

value_boards = {
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

@register_agent("minimax_agent")
class MinimaxAgent(Agent):
    """
    A class for your implementation. Implements MCTS for early/midgame and Minimax with Alpha-Beta Pruning for the endgame.
    """

    def __init__(self):
        super(MinimaxAgent, self).__init__()
        self.name = "MinimaxAgent"

    def step(self, board, player, opponent):
        """
        Implements MCTS for early/midgame and Minimax with Alpha-Beta Pruning for the endgame.
        """
        # Node definition for the search tree 
        class Node:
            def __init__(self, board, parent=None, move=None):
                self.board = board      # current game state
                self.parent = parent    # reference to parent node
                self.move = move        # the move that led to this state/node
                self.children = []      # list of child nodes
                self.visits = 0         # number of times this node has been visited
                self.wins = 0           # number of wins from simulations at this node

            def is_fully_expanded(self):
                """Check if all valid moves for this node have been expanded."""
                valid_moves = get_valid_moves(self.board, player)
                return len(valid_moves) == len(self.children)

            def best_child(self, exploration_weight=0):
                """
                Select the best child node using UCB or raw win rate.

                Parameters:
                - exploration_weight: Determines how much exploration impacts the score.

                Returns:
                - The best child node based on the calculated score.
                """
                best_score = -float('inf') # initialize best_score as -inf
                best_child = None   # variable for best child node
                
                for child in self.children:
                    # UCB score calculation
                    if child.visits > 0:    # only consider visited nodes
                        # exploitation: average WR of the child
                        exploitation = child.wins / child.visits    
                        # exploration: encourage it to visit less-visited nodes
                        exploration = exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
                        score = exploitation + exploration
                    else:
                        score = float('inf')  # prioritize and force unvisited nodes to be explored

                    # update best_child if the score is higher 
                    if score > best_score:
                        best_score = score
                        best_child = child
                return best_child

        def minimax(board, depth, alpha, beta, maximizing_player, player, opponent):
            """
            Minimax with Alpha-Beta Pruning for endgame.

            Parameters:
            - board: numpy.ndarray, current board state.
            - depth: int, remaining depth to explore.
            - alpha: float, alpha value for pruning.
            - beta: float, beta value for pruning.
            - maximizing_player: bool, True if maximizing player's turn.
            - player: int, current player number.
            - opponent: int, opponent's player number.

            Returns:
            - tuple: (best score, best move)
            """
            valid_moves = get_valid_moves(board, player if maximizing_player else opponent)

            # Terminal condition: game over or max depth reached
            if depth == 0 or not valid_moves:
                player_score = np.sum(board == player)
                opponent_score = np.sum(board == opponent)
                return (player_score - opponent_score, None)

            best_move = None
            if maximizing_player:
                max_eval = -float('inf')
                for move in valid_moves:
                    new_board = deepcopy(board)
                    execute_move(new_board, move, player)
                    eval, _ = minimax(new_board, depth - 1, alpha, beta, False, player, opponent)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Beta cut-off
                return (max_eval, best_move)
            else:
                min_eval = float('inf')
                for move in valid_moves:
                    new_board = deepcopy(board)
                    execute_move(new_board, move, opponent)
                    eval, _ = minimax(new_board, depth - 1, alpha, beta, True, player, opponent)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha cut-off
                return (min_eval, best_move)


        # Use Minimax for endgame
        _, best_move = minimax(board, depth=15, alpha=-float('inf'), beta=float('inf'),
                                   maximizing_player=True, player=player, opponent=opponent)

        return best_move
