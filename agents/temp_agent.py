#python simulator.py --player_1 temp_agent --player_2 random_agent --display
#python simulator.py --player_1 student_agent --player_2 random_agent --display
#python simulator.py --player_1 temp_agent --player_2 random_agent --autoplay --autoplay_runs 35
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

# Define the positional weights for different board sizes
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

def get_position_weights(n):
    return value_boards.get(n)

def select_algo(board):
    remaining_moves = np.sum(board == 0)
    if remaining_moves <= 15:
        return "minimax"
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

@register_agent("temp_agent")
class TempAgent(Agent):
    def __init__(self):
        super(TempAgent, self).__init__()
        self.name = "TempAgent"
        self.time_limit = 1.96  # Time limit in seconds
        self.start_time = None  # Will be set at the beginning of step()
        self.simulation_count = 0
        self.max_depth = 10

    def step(self, chess_board, player, opponent):
        self.simulation_count = 0
        self.start_time = time.time()
        board = chess_board.copy()
        valid_moves = get_valid_moves(board, player)
        best_move = None
        if not valid_moves:
            return best_move  # No valid moves, pass the turn

        algo = select_algo(board)

        if algo == 'mcts':
            root_node = Node()
            while time.time() - self.start_time < self.time_limit:
                # Copy the state for this simulation
                sim_board = board.copy()
                node = root_node
                current_player = player
                sim_opponent = opponent

                # Selection and Expansion
                node, sim_board, current_player, sim_opponent = self.tree_policy(node, sim_board, current_player, sim_opponent)

                # Simulation
                reward = self.default_policy(sim_board, current_player, sim_opponent)

                # Backpropagation
                self.backup(node, reward)

                self.simulation_count += 1  # Increment simulation count for MCTS

            # Choose the move with the highest visit count
            if root_node.children:
                best_move = max(root_node.children, key=lambda c: c.visit_count).move
            else:
                # If no moves were simulated, fall back to a random valid move
                best_move = random_move(chess_board, player)

            print(f"MCTS simulations performed: {self.simulation_count}")
            return best_move

        else:
            best_score = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            depth = 1

            while time.time() - self.start_time < self.time_limit:
                for move in valid_moves:
                    new_board = board.copy()
                    execute_move(new_board, move, player)
                    score = self.min_value(new_board, depth - 1, alpha, beta, player, opponent)

                    if score > best_score:
                        best_score = score
                        best_move = move

                    alpha = max(alpha, best_score)

                depth += 1  # Increment depth for iterative deepening

            print(f"Minimax simulations performed: {self.simulation_count}")
            return best_move

    # Minimax functions
    def max_value(self, board, depth, alpha, beta, player, opponent):
        if depth == 0 or check_endgame(board, player, opponent)[0] or time.time() - self.start_time >= self.time_limit:
            return self.evaluate(board, player, opponent)

        max_score = -float('inf')
        valid_moves = get_valid_moves(board, player)

        if not valid_moves:
            # Pass turn to opponent
            return self.min_value(board, depth - 1, alpha, beta, player, opponent)

        for move in valid_moves:
            self.simulation_count += 1  # Increment the simulation counter
            new_board = board.copy()
            execute_move(new_board, move, player)
            score = self.min_value(new_board, depth - 1, alpha, beta, player, opponent)

            max_score = max(max_score, score)
            if max_score >= beta:
                return max_score  # Beta cut-off
            alpha = max(alpha, max_score)

        return max_score

    def min_value(self, board, depth, alpha, beta, player, opponent):
        if depth == 0 or check_endgame(board, player, opponent)[0] or time.time() - self.start_time >= self.time_limit:
            return self.evaluate(board, player, opponent)

        min_score = float('inf')
        valid_moves = get_valid_moves(board, opponent)

        if not valid_moves:
            # Pass turn to player
            return self.max_value(board, depth - 1, alpha, beta, player, opponent)

        for move in valid_moves:
            self.simulation_count += 1  # Increment the simulation counter
            new_board = board.copy()
            execute_move(new_board, move, opponent)
            score = self.max_value(new_board, depth - 1, alpha, beta, player, opponent)

            min_score = min(min_score, score)
            if min_score <= alpha:
                return min_score  # Alpha cut-off
            beta = min(beta, min_score)

        return min_score

    def evaluate(self, board, player, opponent):
        empty_squares = np.sum(board == 0)
        if empty_squares > 40:
            # Early game: Mobility is more important
            weights = {'positional': 1, 'mobility': 5}
        elif empty_squares > 15:
            # Midgame: Balance between positional and mobility
            weights = {'positional': 3, 'mobility': 3}
        else:
            # Endgame: Positional values become more important
            weights = {'positional': 5, 'mobility': 1}

        # Calculate heuristic values
        positional = self.positional_score(board, player) - self.positional_score(board, opponent)
        mobility = self.mobility(board, player, opponent)

        # Combine heuristics
        score = (weights['positional'] * positional +
                weights['mobility'] * mobility)

        return score

    def positional_score(self, board, player):
        position_weights = get_position_weights(board.shape[0])
        player_score = np.sum(position_weights[board == player])
        return player_score

    def mobility(self, board, player, opponent):
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))
        if player_moves + opponent_moves != 0:
            mobility_score = 100 * (player_moves - opponent_moves) / (player_moves + opponent_moves)
        else:
            mobility_score = 0
        return mobility_score

    # MCTS functions
    def tree_policy(self, node, state, current_player, opponent):
        # Selection and Expansion
        while True:
            if time.time() - self.start_time >= self.time_limit:
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

    def default_policy(self, state, current_player, opponent):
        # Rollout with depth limit
        depth = 0
        while depth < self.max_depth:
            if time.time() - self.start_time >= self.time_limit:
                break  # Exit if time limit is reached
            valid_moves = get_valid_moves(state, current_player)
            if valid_moves:
                # Use heuristic-based move selection
                move = self.select_move_with_heuristic(state, valid_moves, current_player)
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
        value = self.evaluate(state, current_player, opponent)
        return value

    def backup(self, node, reward):
        # Backpropagation
        while node is not None:
            node.visit_count += 1
            node.value += reward
            node = node.parent

    def select_move_with_heuristic(self, state, valid_moves, player):
        position_weights = get_position_weights(state.shape[0])
        best_value = -float('inf')
        best_move = None
        for move in valid_moves:
            value = position_weights[move[0], move[1]]
            if value > best_value:
                best_value = value
                best_move = move
        return best_move
