from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves



@register_agent("second_agent")
class SecondAgent(Agent):
    """
    A class for implementing Monte Carlo Tree Search (MCTS).
    """

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "SecondAgent"

    def step(self, chess_board, player, opponent):
        """
        Implement the step function using Monte Carlo Tree Search (MCTS).
        """

        # MCTS Parameters
        exploration_param = np.sqrt(2)# exploration weight for the UCB formula
        time_limit = 1.95               # max 2 seconds for computation so we limit to 1.95sec
        start_time = time.time()

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
                        exploration = exploration_weight * np.sqrt(
                            np.log(self.visits) / child.visits
                        )
                        score = exploitation + exploration
                    else:
                        score = float('inf')  # prioritize and force unvisited nodes to be explored

                    # update best_child if the score is higher 
                    if score > best_score:
                        best_score = score
                        best_child = child
                return best_child

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
            for move in valid_moves:
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
                if not valid_moves:
                    # if no valid moves, swap turns
                    sim_player, sim_opponent = sim_opponent, sim_player
                    valid_moves = get_valid_moves(sim_board, sim_player)
                    if not valid_moves:  # no more moves so game ends
                        break
                
                # select a random move and do it
                random_move = valid_moves[np.random.randint(len(valid_moves))]
                execute_move(sim_board, random_move, sim_player)
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

