from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import MCTSTree, MCTSNode, MCTSOutput
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS

# Configuration class for MCTS
class MCTSConfig:
    def __init__(
        self,
        num_simulations: int = 200,
        dirichlet_alpha: float = 0.3,
        exploration_constant: float = 1.0,
        max_nodes: int = 10000,
        discount: float = -1.0,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True,
    ):
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_constant = exploration_constant
        self.max_nodes = max_nodes
        self.discount = discount
        self.temperature = temperature
        self.tiebreak_noise = tiebreak_noise
        self.persist_tree = persist_tree
