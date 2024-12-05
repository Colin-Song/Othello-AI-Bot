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
        self.time_limit = 1.996

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
        

        self.start_time = time.time()
        root_node = Node(board)
        state = board.copy()
        valid_moves = get_valid_moves(state, player)
        if not valid_moves:
            return None  # No valid moves, pass the turn
        self.simulation_count = 0
        best_move = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        depth = 1

        while time.time() - self.start_time < self.time_limit:
            for move in valid_moves:
                new_board = deepcopy(board)
                execute_move(new_board, move, player)
                score = self.min_value(new_board, depth - 1, alpha, beta, player, opponent)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)

            depth += 1  # Increment depth for iterative deepening
        print(f"Simulations performed: {self.simulation_count}")
        return best_move
    
    def max_value(self, board, depth, alpha, beta, player, opponent):
        """
        Maximize the score for the current player.
        """
        if depth == 0 or check_endgame(board, player, opponent)[0] or time.time() - self.start_time >= self.time_limit:
            return self.evaluate(board, player, opponent)

        max_score = -float('inf')
        valid_moves = get_valid_moves(board, player)

        for move in valid_moves:
            self.simulation_count += 1  # Increment the simulation counter
            new_board = deepcopy(board)
            execute_move(new_board, move, player)
            score = self.min_value(new_board, depth - 1, alpha, beta, player, opponent)

            max_score = max(max_score, score)
            if max_score >= beta:
                return max_score  # Beta cut-off
            alpha = max(alpha, max_score)

        return max_score

    def min_value(self, board, depth, alpha, beta, player, opponent):
        """
        Minimize the score for the opponent.
        """
        if depth == 0 or check_endgame(board, player, opponent)[0] or time.time() - self.start_time >= self.time_limit:
            return self.evaluate(board, player, opponent)

        min_score = float('inf')
        valid_moves = get_valid_moves(board, opponent)

        for move in valid_moves:
            self.simulation_count += 1  # Increment the simulation counter
            new_board = deepcopy(board)
            execute_move(new_board, move, opponent)
            score = self.max_value(new_board, depth - 1, alpha, beta, player, opponent)

            min_score = min(min_score, score)
            if min_score <= alpha:
                return min_score  # Alpha cut-off
            beta = min(beta, min_score)

        return min_score

    def evaluate(self, board, player, opponent):
        """
        Evaluate the board state based on the disc count difference.
        """
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)
        return player_score - opponent_score

        

        
