#python simulator.py --player_1 second_agent --player_2 student_agent --display
#python simulator.py --player_1 second_agent --player_2 student_agent --autoplay --autoplay_runs 15
# Student agent: Add your own agent here
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


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A class for your implementation. Implements MCTS with heuristics for early game.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"

    def step(self, board, player, opponent):
        """
        Implements MCTS with heuristics to guide early-game play.
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


        def count_stable_discs(board, player):
            """
            Counts the number of stable discs for a given player on the board.
            
            Parameters:
            - board: numpy.ndarray, the current board state.
                            0: empty, 1: Player 1 (Blue), 2: Player 2 (Brown)
            - player: int, the player whose stable discs are being counted (1 or 2).
            
            Returns:
            - int, the number of stable discs for the given player.
            """
            # get size of board
            board_size = board.shape[0]
            # positions of corners
            corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
            # stability value
            stability = 0

            # if player has corner then add 1 to stability value
            for corner in corners:
                if board[corner] == player:
                    stability += 1
            
            # return stability value
            return stability
        
        def get_frontier_discs_by_player(board):
            """
            Identify frontier discs separately for each player.

            Parameters
            ----------
            board : numpy.ndarray
                The current state of the game board.

            Returns
            -------
            frontier discs count for both players
            """
            
            board_size = board.shape[0]
            frontier = np.zeros_like(board, dtype=bool)
            empty_cells = board == 0

            # Create a padded version of the board to avoid boundary checks
            padded_board = np.pad(board, pad_width=1, mode='constant', constant_values=0)
            occupied_cells = padded_board[1:-1, 1:-1] != 0

            # Check for adjacent empty cells
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),         (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]:
                shifted_empty = empty_cells.copy()
                shifted_empty = np.roll(shifted_empty, shift=(dr, dc), axis=(0, 1))
                if dr != 0:
                    shifted_empty[dr > 0 and 0 or -dr:, :] = False
                if dc != 0:
                    shifted_empty[:, dc > 0 and 0 or -dc:] = False
                frontier |= occupied_cells & shifted_empty

            # Extract player-specific frontier discs
            player_1_frontier = (frontier) & (board == 1)
            player_2_frontier = (frontier) & (board == 2)

            # Count the number of frontier discs for each player
            player_1_frontier_count = np.sum(player_1_frontier)
            player_2_frontier_count = np.sum(player_2_frontier)

            return player_1_frontier_count, player_2_frontier_count
        
        #python simulator.py --player_1 second_agent --player_2 student_agent --display
        def eval_moves(board, board_size, move_count, valid_moves, player, opponent):
            """
            Evaluate best moves for a player
            """
            # create copy of board
            board_copy = deepcopy(board)

            # make list for moves
            moves_eval = []
            # iterate through all valid moves
            for move in valid_moves:
                # get positional value of move
                value_move = int(value_boards[board_size][move[0]][move[1]])
                # value move weight
                value_move_w = 0.5

                # execute move on board_copy
                execute_move(board_copy, move, player)
                # get player's valid move given new board state
                future_player_openmoves = get_valid_moves(board_copy, player)
                # get opp's valud move given new board state
                future_opp_openmoves = get_valid_moves(board_copy, opponent)
                # calculate mobility value
                mobility = len(future_player_openmoves) - len(future_opp_openmoves)
                # mobility weight
                mobility_w = 0.7

                # get number of player's stable discs
                player_stable_discs = count_stable_discs(board_copy, player)
                # get number of opp's stable discs
                opp_stable_discs = count_stable_discs(board_copy, opponent)
                # calculate stability value
                stability = player_stable_discs - opp_stable_discs
                # stability weight
                stability_w = 2

                # get number of player's, opp's frontier discs
                player_frontiers, opp_frontiers = get_frontier_discs_by_player(board_copy)
                # calculate frontier value
                frontier = player_frontiers - opp_frontiers
                # frontier weight
                frontier_w = -0.7

                # count how many discs we flip with the move
                flip_discs = count_capture(board, move, player)
                # flipped discs weight
                flip_discs_w = -0.6

                # calculate the board score for the player given the move
                board_value = value_move*value_move_w + mobility*mobility_w + stability*stability_w + frontier*frontier_w + flip_discs*flip_discs_w
                # append the move and the board score to moves_eval list
                moves_eval.append((move, board_value))

            
            # sort the list given the board score, descending order
            best_moves = sorted(moves_eval, key=lambda item: item[1], reverse=True)
            # extract only the moves (sorted by score)
            return [item[0] for item in best_moves]

        def get_max_depth(move_count, total_moves):
            if move_count < total_moves*0.3:
                return 5
            elif move_count < total_moves*0.7:
                return 10
            else:
                return 100

        # MCTS Functions: select, expand, simulate, backpropagate
        def select(node):
            """
            Select the most promising node using UCB.

            Starting from the root, traverse down to a leaf node using the UCB formula.
            """
            while node.children:
                # select the best child based on UCB formula
                node = node.best_child(exploration_param)   
            return node # returns most promising LEAF node

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
                    node.children.append(child_node) # adds the child to current node
                    return child_node
            return None  # If all moves are already expanded, return none

        def simulate(board, board_size, move_count, current_player, max_depth):
            """
            Simulate a random playout from the given board state.

            Randomly play the game to completion and return the result.
            """
            sim_board = deepcopy(board) # deepcopy to avoid modifying actual game state
            sim_player = current_player
            sim_opponent = 3 - sim_player  # alternate between players 1 & 2

            depth = 0
            total_moves = np.sum(board == 1) + np.sum(board == 2) - 4
            max_depth = get_max_depth(move_count, total_moves)

            while depth < max_depth:
                valid_moves = get_valid_moves(sim_board, sim_player)
                best_moves = eval_moves(sim_board, board_size, move_count, valid_moves, player, opponent)
                if not best_moves:
                    # if no valid moves, swap turns
                    sim_player, sim_opponent = sim_opponent, sim_player
                    valid_moves = get_valid_moves(sim_board, sim_player)
                    best_moves = eval_moves(sim_board, board_size, move_count, valid_moves, player, opponent)
                    if not best_moves:  # no more moves so game ends
                        break
                
                # select a random move and do it
                move = best_moves[0]
                execute_move(sim_board, move, sim_player)
                sim_player, sim_opponent = sim_opponent, sim_player # switch turns
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
                node.visits += 1    # increment the visit count
                node.wins += result # add the result to the win count
                result = 1 - result # alternate between win/lose for opponent's perspective
                node = node.parent  # move up in the tree


        # MCTS Parameters
        exploration_param = np.sqrt(2)# exploration weight for the UCB formula

        # keep track of time for turn
        start_time = time.time()
        time_limit = 1.98

        # get current state of board
        board_copy = deepcopy(board)

        # get the board size
        board_size = board_copy.shape[0]

        
        # positions of corners and tiles next to corners
        corners_nexttiles = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, board_size-1), (0, board_size-2), (1, board_size-1), (1, board_size-2)],
            [(board_size-1, 0), (board_size-2, 0), (board_size-1, 1), (board_size-2, 1)],
            [(board_size-1, board_size-1), (board_size-1, board_size-2), (board_size-2, board_size-1), (board_size-2, board_size-2)]
            ]
        
        # iterate through corners_nexttiles list and update values if player/opponent has that corner
        for index, corner in enumerate(corners_nexttiles):
            # if corner is occupied by player
            if board[corner[0][0]][corner[0][1]] == player:
                # change values to postive values
                value_boards[board_size][corner[1][0]][corner[1][1]] = 20
                value_boards[board_size][corner[2][0]][corner[2][1]] = 20
                value_boards[board_size][corner[3][0]][corner[3][1]] = 20
        

        # calculate player's move count
        move_count = np.sum(board == player) - 2

        # get max depth
        max_depth = get_max_depth(move_count, np.sum(board == 1) + np.sum(board == 2) - 4)

        # initialize the root node with the current board state
        root = Node(board)

        # MCTS loop
        while time.time() - start_time < time_limit:
            # Selection
            leaf = select(root)
            # Expansion
            if not check_endgame(leaf.board, player, opponent)[0]:
                child = expand(leaf, board_size, move_count)
                if child:
                    # Simulation
                    result = simulate(child.board, board_size, move_count, player, max_depth)
                    # Backpropagation
                    backpropagate(child, result)

        # Choose the best move after search is done
        best_move = root.best_child(exploration_weight=0).move
        return best_move