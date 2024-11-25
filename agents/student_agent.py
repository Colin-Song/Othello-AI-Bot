#python simulator.py --player_1 second_agent --player_2 student_agent --display
#python simulator.py --player_1 second_agent --player_2 student_agent --autoplay_runs 15 --autoplay
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

early_value_board_6 = np.array([
    [120, -20,  20,  20, -20, 120],
    [-20, -40,  -5,  -5, -40, -20],
    [ 20,  -5,  15,  15,  -5,  20],
    [ 20,  -5,  15,  15,  -5,  20],
    [-20, -40,  -5,  -5, -40, -20],
    [120, -20,  20,  20, -20, 120]
])
#change values in board
mid_value_board_6 = np.array([
    [120, -20,  20,  20, -20, 120],
    [-20, -40,  -5,  -5, -40, -20],
    [ 20,  -5,  15,  15,  -5,  20],
    [ 20,  -5,  15,  15,  -5,  20],
    [-20, -40,  -5,  -5, -40, -20],
    [120, -20,  20,  20, -20, 120]
])

early_value_board_8 = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120]
])
#change values in board
mid_value_board_8 = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120]
])

early_value_board_10 = np.array([
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
])
#change values in board
mid_value_board_10 = np.array([
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
])

early_value_board_12 = np.array([
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
#change values in board
mid_value_board_12 = np.array([
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
    

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A class for your implementation. Implements MCTS with heuristics for early game.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"

    def step(self, chess_board, player, opponent):
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
        
        def select_board(state, size):
            return globals()[f"{state}_value_board_{size}"]
            
        def mark_stable_discs(chess_board, stable, position, player, directions):
            """
            Marks discs as stable recursively starting from a given position.

            Parameters:
            - chess_board: numpy.ndarray, the current board state.
            - stable: numpy.ndarray, a board to mark stable discs.
            - position: tuple, the starting position to check stability (row, col).
            - player: int, the player whose discs are being marked as stable.
            - directions: list, the directions to traverse the board.
            """
            board_size = chess_board.shape[0]
            queue = [position]  # Use a queue for breadth-first traversal

            while queue:
                r, c = queue.pop(0)
                if stable[r, c] == player:  # Already marked as stable
                    continue

                stable[r, c] = player  # Mark the disc as stable

                # Check all directions
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                while 0 <= nr < board_size and 0 <= nc < board_size:
                    if chess_board[nr, nc] != player or stable[nr, nc] == player:
                        break
                    # Add to queue if the disc is of the player's color and not yet stable
                    if chess_board[nr, nc] == player:
                        queue.append((nr, nc))
                    nr += dr
                    nc += dc

        def count_stable_discs(chess_board, player):
            """
            Counts the number of stable discs for a given player on the board.
            
            Parameters:
            - chess_board: numpy.ndarray, the current board state.
                            0: empty, 1: Player 1 (Blue), 2: Player 2 (Brown)
            - player: int, the player whose stable discs are being counted (1 or 2).
            
            Returns:
            - int, the number of stable discs for the given player.
            """
            board_size = chess_board.shape[0]
            stable = np.zeros_like(chess_board)  # A board to mark stable discs
            
            # Directions for traversing
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

            # Check stability starting from corners
            corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]
            for corner in corners:
                if chess_board[corner[0], corner[1]] == player:
                    mark_stable_discs(chess_board, stable, corner, player, directions)

            # Count the number of stable discs
            return np.sum(stable == player)
        
        def get_frontier_discs_by_player(chess_board):
            """
            Identify frontier discs separately for each player.

            Parameters
            ----------
            chess_board : numpy.ndarray
                The current state of the game board.

            Returns
            -------
            tuple
                Two sets: (player_1_frontier, player_2_frontier)
            """
            # All possible directions for adjacency
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
            board_size = chess_board.shape[0]

            # Sets to store frontier discs for each player
            player_1_frontier = set()
            player_2_frontier = set()

            # Iterate through the board
            for r in range(board_size):
                for c in range(board_size):
                    if chess_board[r, c] != 0:  # If the square is occupied by a disc
                        for dr, dc in directions:
                            nr, nc = r + dr, c + dc  # Neighboring row and column
                            if 0 <= nr < board_size and 0 <= nc < board_size:  # Check bounds
                                if chess_board[nr, nc] == 0:  # If adjacent square is empty
                                    if chess_board[r, c] == 1:  # Player 1's disc
                                        player_1_frontier.add((r, c))
                                    elif chess_board[r, c] == 2:  # Player 2's disc
                                        player_2_frontier.add((r, c))
                                    break  # Stop checking further directions for this disc

            return player_1_frontier, player_2_frontier
        #python simulator.py --player_1 second_agent --player_2 student_agent --display
        def eval_moves(board, value_board, valid_moves, player, opponent):
            """
            Evaluate best moves for a player
            """
            board_copy = deepcopy(board)
            # make list for moves
            moves_eval = []
            # iterate through all valid moves
            for move in valid_moves:
                # get positional value of move
                value_move = int(value_board[move])
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
                stability_w = 1

                # get number of player's, opp's frontier discs
                player_frontiers, opp_frontiers = get_frontier_discs_by_player(board_copy)
                # calculate frontier value
                frontier = len(player_frontiers) - len(opp_frontiers)
                # frontier weight
                frontier_w = 0.7

                # calculate the board score for the player given the move
                board_value = value_move*value_move_w + mobility*mobility_w + stability*stability_w + frontier*frontier_w
                # append the move and the board score to moves_eval list
                moves_eval.append((move, board_value))

            
            # sort the list given the board score
            best_moves = sorted(moves_eval, key=lambda item: item[1], reverse=True)
            # extract only the moves for return
            return [(int(item[1]), int(item[4])) for item in best_moves]
        

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

        def expand(node):
            """
            Expand the node by adding a new child.

            If a node is not fully expanded, create a child for one of its unexplored valid moves.
            """
            valid_moves = get_valid_moves(node.board, player)
            best_moves = eval_moves(node.board, value_board, valid_moves, player, opponent)
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

        def simulate(board, current_player):
            """
            Simulate a random playout from the given board state.

            Randomly play the game to completion and return the result.
            """
            sim_board = deepcopy(board) # deepcopy to avoid modifying actual game state
            sim_player = current_player
            sim_opponent = 3 - sim_player  # alternate between players 1 & 2

            while True:
                valid_moves = get_valid_moves(sim_board, sim_player)
                best_moves = eval_moves(sim_board, value_board, valid_moves, player, opponent)
                if not best_moves:
                    # if no valid moves, swap turns
                    sim_player, sim_opponent = sim_opponent, sim_player
                    valid_moves = get_valid_moves(sim_board, sim_player)
                    best_moves = eval_moves(sim_board, value_board, valid_moves, player, opponent)
                    if not best_moves:  # no more moves so game ends
                        break
                
                # select a random move and do it
                move = best_moves[0]
                execute_move(sim_board, move, sim_player)
                sim_player, sim_opponent = sim_opponent, sim_player # switch turns
                
                
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
        time_limit = 1.95  

        # get current state of board
        board_copy = deepcopy(chess_board)

        # get the board size
        board_size = board_copy.shape[0]
        # get number of player moves (number of Player 1's discs and Player 2's discs subtract 4 [starting discs])
        move_count = np.sum(chess_board == 1) + np.sum(chess_board == 2) - 4
        state = "mid" if move_count > 5 else "early"
        # get value board
        value_board = select_board(state, board_size)

        # initialize the root node with the current board state
        root = Node(chess_board)

        # MCTS loop
        while time.time() - start_time < time_limit:
            # Selection
            leaf = select(root)
            # Expansion
            if not check_endgame(leaf.board, player, opponent)[0]:
                child = expand(leaf)
                if child:
                    # Simulation
                    result = simulate(child.board, player)
                    # Backpropagation
                    backpropagate(child, result)

        # Choose the best move after search is done
        best_move = root.best_child(exploration_weight=0).move
        return best_move