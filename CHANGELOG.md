# Changelog

## v0.1.0 — Phase 1: Core Learning Loop (2026-03-06)

First working release of the Prompt Selection RL Agent. Implements the core Q-learning training loop with human-in-the-loop reward collection.

### What's included

- **RLAgent**: Q-learning agent with configurable learning rate, discount factor, exploration rate, and decay rate. Supports ε-greedy action selection.
- **QTable**: Nested dictionary storing state-action Q-values. Defaults to 0.0 for unseen pairs. Supports serialization round-trips via `to_dict()`/`from_dict()`.
- **Environment**: Manual (HITL) environment that displays prompt+task and collects human reward scores via console input. Validates rewards to [-1.0, 1.0].
- **Error handling**: `ConfigurationError`, `ValidationError`, `ModeError`, `PersistenceError` with informative messages.
- **demo.py**: Interactive 5-episode training demo. Run with `python3 demo.py` or pipe rewards: `echo -e "0.8\n-0.2\n0.5\n1.0\n0.3" | python3 demo.py`
- **66 unit tests** covering agent init, parameter validation, ε-greedy selection, Q-learning formula, reward validation, inference mode, environment, and Q-table operations.

### Example session

```
=== Prompt Selection RL Agent Demo ===

Prompts: ['Be concise', 'Be detailed', 'Be creative']
Initial Q-table: {}

--- Episode 1 ---
Prompt: Be concise
Task: Summarize this article
Enter reward [-1.0, 1.0]: 0.8

Q-table after episode 1:
  State: 'Summarize this article'
    'Be concise': 0.0800
Exploration rate: 1.0
```

### Not yet implemented

- Experience buffer and batch training (Phase 2)
- Training/inference mode switching (Phase 3)
- Exploration decay (Phase 3)
- Metrics tracking (Phase 4)
- Policy persistence — save/load Q-tables (Phase 5)
- State encoding and extensibility interfaces (Phase 6)
