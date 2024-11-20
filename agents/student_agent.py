# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

class MCTSNode:
  """
  A node in the MCTS tree representing a state in the game.
  """
  def __init__(self, chess_board, move=None, parent=None, player=1):
      self.chess_board = chess_board  # Current board state
      self.move = move  # Move that led to this state
      self.parent = parent  # Parent node in the tree
      self.children = []  # Child nodes
      self.visits = 0  # Number of visits to this node
      self.wins = 0  # Wins from simulations
      self.player = player  # The player making this move

  def is_fully_expanded(self):
      """Check if all valid moves have been expanded."""
      return len(self.children) == len(get_valid_moves(self.chess_board, self.player))

  def best_child(self, exploration_weight=1.41):
      """
      Select the best child node using the UCT formula.
      UCT = (wins / visits) + exploration_weight * sqrt(log(parent_visits) / visits)
      """
      if not self.children:
          return None

      best_child = max(
          self.children,
          key=lambda child: (child.wins / child.visits if child.visits > 0 else 0)
          + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1))
      )
      return best_child

value_board = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120]
])

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    """
    Use Monte Carlo Tree Search to determine the next move.
    """
    start_time = time.time()
    time_limit = 15.0  # Limit MCTS to 2 seconds
    root = MCTSNode(chess_board=deepcopy(chess_board), player=player)

    while time.time() - start_time < time_limit:
        leaf = self._select(root)
        simulation_result = self._simulate(leaf, player, opponent)
        self._backpropagate(leaf, simulation_result, player)

    # Choose the best move based on visits
    best_child = max(root.children, key=lambda child: child.visits, default=None)
    return best_child.move if best_child else random_move(chess_board, player)

  def _select(self, node):
      """
      Traverse the tree to select a leaf node for expansion.
      """
      while node.is_fully_expanded() and node.children:
          node = node.best_child()
      return self._expand(node)

  def _expand(self, node):
      """
      Expand the node by adding a new child node for an unvisited move.
      """
      def random_move_list(moves):
        return moves[np.random.randint(len(moves))]
      
      valid_moves = get_valid_moves(node.chess_board, node.player)
      untried_moves = [
          move for move in valid_moves if move not in [child.move for child in node.children]
      ]

      if untried_moves:
          move = random_move_list(untried_moves)
          new_board = deepcopy(node.chess_board)
          execute_move(new_board, move, node.player)
          child_node = MCTSNode(
              chess_board=new_board,
              move=move,
              parent=node,
              player=3 - node.player,  # Switch player
          )
          node.children.append(child_node)
          return child_node
      return node
  
  

  def _simulate(self, node, player, opponent):
      """
      Simulate a random playout from the current node.
      """
      def random_move_list(moves):
        return moves[np.random.randint(len(moves))]
      board_copy = deepcopy(node.chess_board)
      current_player = node.player

      while True:
          valid_moves = get_valid_moves(board_copy, current_player)
          if not valid_moves:
              current_player = 3 - current_player
              if not get_valid_moves(board_copy, current_player):
                  # Game over
                  _, player_score, opponent_score = check_endgame(
                      board_copy, player, opponent
                  )
                  if player_score > opponent_score:
                      return 1  # Win
                  elif player_score < opponent_score:
                      return -1  # Loss
                  else:
                      return 0  # Draw
          else:
              random_move = random_move_list(valid_moves)
              execute_move(board_copy, random_move, current_player)
              current_player = 3 - current_player

  def _backpropagate(self, node, result, player):
      """
      Backpropagate the simulation result up the tree.
      """
      while node:
          node.visits += 1
          if node.player == player:
              node.wins += result  # Add 1 for a win, 0 for draw, -1 for loss
          else:
              node.wins -= result  # Subtract the result for the opponent
          node = node.parent
  
  #python simulator.py --player_1 random_agent --player_2 student_agent --display

