#python simulator.py --player_1 random_agent --player_2 human_agent --display
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

# Directions for traversing the board
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# Corner and risky positions
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
BAD_POSITIONS = [(0, 1), (1, 0), (6, 0), (7, 1), (7, 6), (6, 7), (0, 6), (1, 7)]
VERY_BAD_POSITIONS = [(1, 1), (1, 6), (6, 1), (6, 6)]

class Node:
  """
  Node class for the MCTS tree.
  """
def __init__(self, chess_board, move=None, parent=None, player=1):
    self.chess_board = chess_board
    self.move = move
    self.parent = parent
    self.children = []
    self.visits = 0
    self.wins = 0
    self.player = player

def is_fully_expanded(self):
    """
    Check if all valid moves have been expanded.
    """
    return len(self.children) == len(get_valid_moves(self.chess_board, self.player))

def select_child(self):
    """
    Select the child node using the UCT formula.
    """
    exploration_weight = 1.41  # Exploration constant
    return max(
        self.children,
        key=lambda child: (child.wins / child.visits if child.visits > 0 else 0)
        + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1)),
    )


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
    # keept track of time for turn
    start_time = time.time()
    time_limit = 1.95  

    # get current state of board
    curr_state = deepcopy(chess_board)

    # get valid moves for player
    valid_moves = get_valid_moves(curr_state, player)

    best_moves = eval_moves(curr_state, valid_moves, player)

    def eval_moves(curr_state, valid_moves, player):
        for move in best_moves:

    # get all valid moves
    # trim valid moves
    # run MCTS on trimmed moves
    # decide on best move from MCTS

    # Root node of the MCTS tree
    root = Node(deepcopy(chess_board), player=player)

    # Perform MCTS iterations
    while (time.time() - start_time < time_limit):
        leaf = self.select(root)
        simulation_result = self.simulate(leaf, player, opponent)
        self.backpropagate(leaf, simulation_result, player)

    # Select the best move based on heuristics and visits
    best_child = max(root.children, key=lambda child: self.heuristic(child), default=None)
    return best_child.move if best_child else random_move(chess_board, player)

def select(self, node):
    """
    Traverse the tree to select a leaf node for expansion.
    """
    while node.is_fully_expanded() and node.children:
        node = node.select_child()
    return self._expand(node)

def expand(self, node):
    """
    Expand a node by adding a new child node for an unvisited move.
    """
    valid_moves = get_valid_moves(node.chess_board, node.player)
    untried_moves = [move for move in valid_moves if move not in [child.move for child in node.children]]

    if untried_moves:
        move = untried_moves[0]  # Deterministic for simplicity
        new_board = deepcopy(node.chess_board)
        execute_move(new_board, move, node.player)
        child_node = Node(chess_board=new_board, move=move, parent=node, player=3 - node.player)
        node.children.append(child_node)
        return child_node
    return node

def simulate(self, node, player, opponent):
    """
    Simulate a random playout from the current node.
    """
    board_copy = deepcopy(node.chess_board)
    current_player = node.player

    while True:
        valid_moves = get_valid_moves(board_copy, current_player)
        if not valid_moves:
            current_player = 3 - current_player
            if not get_valid_moves(board_copy, current_player):
                # Game over
                _, player_score, opponent_score = check_endgame(board_copy, player, opponent)
                if player_score > opponent_score:
                    return 1  # Win
                elif player_score < opponent_score:
                    return -1  # Loss
                else:
                    return 0  # Draw
        else:
            move = self.heuristic_random(valid_moves)  # Select moves with heuristic bias
            execute_move(board_copy, move, current_player)
            current_player = 3 - current_player

def backpropagate(self, node, result, player):
    """
    Backpropagate the simulation result up the tree.
    """
    while node:
        node.visits += 1
        if node.player == player:
            node.wins += result
        else:
            node.wins -= result
        node = node.parent

def heuristic(self, node):
    """
    Evaluate a node based on position heuristics.
    """
    score = 0
    move = node.move
    if move in CORNERS:
        score += 50  # Strong preference for corners
    elif move in VERY_BAD_POSITIONS:
        score -= 100  # Avoid very bad positions
    elif move in BAD_POSITIONS:
        score -= 20  # Penalize bad positions
    return score + node.visits  # Combine with exploration factor

def heuristic_random(self, valid_moves):
    """
    Use heuristics to bias random move selection during simulation.
    """
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
        
        

        


