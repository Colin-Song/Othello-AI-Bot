#python simulator.py --player_1 temp_agent --player_2 random_agent --display
#python simulator.py --player_1 student_agent --player_2 random_agent --display
#python simulator.py --player_1 temp_agent --player_2 random_agent --autoplay --autoplay_runs 100
from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("temp_agent")
class TempAgent(Agent):
    def __init__(self):
        super(TempAgent, self).__init__()
        self.name = "TempAgent"
        self.time_limit = 1.98  # Limit for MCTS to make a decision

    def max_value(self, board, depth, alpha, beta, player, opponent):
        """
        Maximize the score for the current player.
        """
        if depth == 0 or check_endgame(board, player, opponent)[0] or time.time() - self.start_time >= self.time_limit:
            return self.evaluate(board, player, opponent)

        max_score = -float('inf')
        valid_moves = get_valid_moves(board, player)

        for move in valid_moves:
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

    def step(self, chess_board, player, opponent):
        def select_algo(board):
            # add in change number of moves depending on size of board
            """
            Decide whether to use MCTS or Minimax based on the number of remaining moves.
            """
            # if there are 10 moves or less then use alphabeta pruning
            remaining_moves = np.sum(board == 0)
            if remaining_moves <= 20:
                return "minimax"
            # more than 10 moves use monte carlo tree search
            return "mcts"

        class Node:
            def __init__(self, parent=None, move=None):
                self.parent = parent
                self.children = []
                self.visit_count = 0
                self.value = 0.0
                self.move = move
                self.untried_moves = None  # Will be set when needed

            def is_fully_expanded(self):
                return self.untried_moves is not None and len(self.untried_moves) == 0

            def best_child(self, c_param=1.4):
                choices_weights = [
                    (child.value / child.visit_count) + c_param * np.sqrt(
                        (2 * np.log(self.visit_count) / child.visit_count))
                    for child in self.children
                ]
                return self.children[np.argmax(choices_weights)]

        def tree_policy(node, state, current_player, opponent):
            # Selection and Expansion
            while True:
                if time.time() - start_time > time_limit:
                    # Return the current node and state to prevent errors
                    return node, state, current_player, opponent
                if node.untried_moves is None:
                    node.untried_moves = get_valid_moves(state, current_player)
                if node.untried_moves:
                    # Expand
                    move = node.untried_moves.pop()
                    execute_move(state, move, current_player)
                    child_node = Node(parent=node, move=move)
                    node.children.append(child_node)
                    return child_node, state, current_player, opponent
                else:
                    if not node.children:
                        # No children to select from, return the node
                        return node, state, current_player, opponent
                    else:
                        # Select
                        node = node.best_child()
                        execute_move(state, node.move, current_player)
                    # Switch players
                    current_player, opponent = opponent, current_player


        def default_policy(state, current_player, opponent):
            # Rollout with depth limit
            max_depth = 10
            depth = 0
            while depth < max_depth:
                if time.time() - start_time > time_limit:
                    break  # Exit if time limit is reached
                valid_moves = get_valid_moves(state, current_player)
                if valid_moves:
                    move = random_move(state, current_player)
                    execute_move(state, move, current_player)
                else:
                    # Pass turn if no valid moves
                    current_player, opponent = opponent, current_player
                    valid_moves = get_valid_moves(state, current_player)
                    if not valid_moves:
                        # Game over
                        break
                current_player, opponent = opponent, current_player
                depth += 1
            # Heuristic evaluation
            player_score = np.sum(state == player)
            opponent_score = np.sum(state == opponent)
            return player_score - opponent_score

        def backup(node, reward):
            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value += reward
                node = node.parent

        start_time = time.time()
        time_limit = self.time_limit  # Time limit in seconds

        root_node = Node()
        board = chess_board.copy()
        print(board)

        algo = select_algo(board)

        if algo == "mcts":


            while time.time() - start_time < time_limit:
                # Copy the state for this simulation
                sim_state = board.copy()
                node = root_node
                current_player = player
                sim_opponent = opponent

                # Selection and Expansion
                if time.time() - start_time > time_limit:
                    break  # Exit the loop if time limit is reached
                node, sim_state, current_player, sim_opponent = tree_policy(
                    node, sim_state, current_player, sim_opponent)

                # Simulation
                if time.time() - start_time > time_limit:
                    break  # Exit the loop if time limit is reached
                reward = default_policy(sim_state, current_player, sim_opponent)

                # Backpropagation
                backup(node, reward)

            # Choose the move with the highest visit count
            if root_node.children:
                best_move = max(root_node.children, key=lambda c: c.visit_count).move
            else:
                # If no moves were simulated, fall back to a random valid move
                best_move = random_move(chess_board, player)

            return best_move
        
        else:
    
            valid_moves = get_valid_moves(board, player)
            if not valid_moves:
                return None  # No valid moves, pass the turn

            self.start_time = time.time()
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

            return best_move

        