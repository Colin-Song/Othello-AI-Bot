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
        [120, -20,  20,   5,   5,  20, -20, 120],
        [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
        [ 20,  -5,  15,   3,   3,  15,  -5,  20],
        [  5,  -5,   3,   3,   3,   3,  -5,   5],
        [  5,  -5,   3,   3,   3,   3,  -5,   5],
        [ 20,  -5,  15,   3,   3,  15,  -5,  20],
        [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
        [120, -20,  20,   5,   5,  20, -20, 120]
    ]),
    10: np.array([
    [120, -20,  20,   5,   5,   5,   5,   20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5,  -5,   -5, -40, -20],
    [ 20,  -5,  15,   3,   3,   3,   3,   15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,   3,    3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,   3,    3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,   3,    3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,   3,    3,  -5,   5],
    [ 20,  -5,  15,   3,   3,   3,   3,   15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5,  -5,   -5, -40, -20],
    [120, -20,  20,   5,   5,   5,   5,   20, -20, 120]
    ]),
    12: np.array([
    [120, -20,  20,   5,   5,   5,   5,    5,   5,   20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5,  -5,   -5,  -5,   -5, -40, -20],
    [ 20,  -5,  15,   3,   3,   3,   3,    3,   3,   15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,   3,    3,   3,    3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,   3,    3,   3,    3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,   3,    3,   3,    3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,   3,    3,   3,    3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,   3,    3,   3,    3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,   3,    3,   3,    3,  -5,   5],
    [ 20,  -5,  15,   3,   3,   3,   3,    3,   3,   15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5,  -5,   -5,  -5,   -5, -40, -20],
    [120, -20,  20,   5,   5,   5,   5,    5,   5,   20, -20, 120]
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

        def eval_moves(board, board_size, move_count, valid_moves, player, opponent):
            """
            Evaluate best moves for a player using a combination of heuristics.
            """
            # create copy of board
            board_copy = deepcopy(board)
            # make list for moves
            moves_eval = []
            for move in valid_moves:
                # get positional value of move
                value_move = int(value_boards[board_size][move[0]][move[1]])
                # value move weight
                value_move_w = 0.5

                # execute move on board_copy
                execute_move(board_copy, move, player)
                # get player's valid move given new board state
                future_player_openmoves = get_valid_moves(board_copy, player)
                # get opp's valid move given new board state
                future_opp_openmoves = get_valid_moves(board_copy, opponent)
                # calculate mobility value
                mobility = len(future_player_openmoves) - len(future_opp_openmoves)
                # mobility weight
                mobility_w = 0.7

                # calculate the board score for the player given the move
                board_value = value_move * value_move_w + mobility * mobility_w
                # append the move and the board score to moves_eval list
                moves_eval.append((move, board_value))

            # sort the list given the board score, descending order
            best_moves = sorted(moves_eval, key=lambda item: item[1], reverse=True)
            # extract only the moves (sorted by score)
            return [item[0] for item in best_moves]
        def select(node):
            """
            Select the most promising node using UCB.

            Starting from the root, traverse down to a leaf node using the UCB formula.
            """
            while node.children:
                node = node.best_child(exploration_weight=np.sqrt(2))
            return node  # returns the most promising LEAF node

        def expand(node, board_size, move_count):
            """
            Expand the node by adding a new child.

            If a node is not fully expanded, create a child for one of its unexplored valid moves.
            """
            valid_moves = get_valid_moves(node.board, player)
            best_moves = eval_moves(node.board, board_size, move_count, valid_moves, player, opponent)
            for move in best_moves:
                # check if this has already been explored
                if not any(child.move == move for child in node.children):
                    # create a new board state by applying the move
                    new_board = deepcopy(node.board)
                    execute_move(new_board, move, player)
                    # create and append new child node
                    child_node = Node(new_board, parent=node, move=move)
                    node.children.append(child_node)  # adds the child to current node
                    return child_node
            return None  # If all moves are already expanded, return none
 
        def simulate(board, board_size, move_count, current_player, max_depth):
            """
            Simulate a random playout from the given board state.

            Randomly play the game to completion and return the result.
            """
            sim_board = deepcopy(board)  # deepcopy to avoid modifying the actual game state
            sim_player = current_player
            sim_opponent = 3 - sim_player  # alternate between players 1 & 2

            depth = 0
            while depth < max_depth:
                valid_moves = get_valid_moves(sim_board, sim_player)
                if not valid_moves:
                    # if no valid moves, swap turns
                    sim_player, sim_opponent = sim_opponent, sim_player
                    valid_moves = get_valid_moves(sim_board, sim_player)
                    if not valid_moves:  # no more moves so game ends
                        break
                # select a random move and do it
                move = valid_moves[np.random.randint(len(valid_moves))]
                execute_move(sim_board, move, sim_player)
                sim_player, sim_opponent = sim_opponent, sim_player  # switch turns
                depth += 1

            # evaluate the outcome of the game
            _, player_score, opponent_score = check_endgame(sim_board, player, opponent)
            return 1 if player_score > opponent_score else 0  # win (1) or lose (0) for the player

        def backpropagate(node, result):
            """
            Backpropagate the result of a simulation up the tree.

            Update visit and win counts for all ancestors of the current node.
            """
            while node is not None:
                node.visits += 1  # increment the visit count
                node.wins += result  # add the result to the win count
                result = 1 - result  # alternate between win/lose for opponent's perspective
                node = node.parent  # move up in the tree

        def select_algorithm(board):
            """
            Decide whether to use MCTS or Minimax based on the number of remaining moves.
            """
            remaining_moves = np.sum(board == 0)
            if remaining_moves <= 10:  # Threshold for switching to Minimax
                return "minimax"
            return "mcts"

        # Choose algorithm based on the stage of the game
        algorithm = select_algorithm(board)

        if algorithm == "minimax":
            # Use Minimax for endgame
            _, best_move = minimax(board, depth=5, alpha=-float('inf'), beta=float('inf'),
                                   maximizing_player=True, player=player, opponent=opponent)
        else:
            # Use MCTS for early/midgame
            start_time = time.time()
            time_limit = 1.98
            board_size = board.shape[0]
            move_count = np.sum(board == player) - 2
            max_depth = 10
            root = Node(board)

            # MCTS loop
            while time.time() - start_time < time_limit:
                leaf = select(root)
                if not check_endgame(leaf.board, player, opponent)[0]:
                    child = expand(leaf, board_size, move_count)
                    if child:
                        result = simulate(child.board, board_size, move_count, player, max_depth)
                        backpropagate(child, result)

            best_move = root.best_child(exploration_weight=0).move

        return best_move
