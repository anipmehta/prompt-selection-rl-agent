# Prompt Selection RL Agent

A Reinforcement Learning agent that learns to select optimal prompts for given tasks using Q-learning with offline batch training.

## Overview

This project implements an RL-based prompt selection system following industry-standard practices (similar to OpenAI's RLHF pipeline). The agent learns which prompts work best for different task contexts through trial and feedback.

## Key Features

- **Q-Learning Algorithm**: Tabular Q-learning with ε-greedy exploration
- **Offline Batch Training**: Collect experiences, train in stable batches
- **Training/Inference Modes**: Separate learning and production deployment
- **Policy Persistence**: Save and load learned Q-tables
- **Experience Management**: Buffer experiences for reproducible training
- **Extensible Architecture**: Designed for future tool integration

## Documentation

The project follows a spec-driven development approach with comprehensive documentation:

- **[Requirements](specs/requirements.md)**: Functional requirements with acceptance criteria (12 requirements)
- **[Design](specs/design.md)**: Architecture diagrams, component interfaces, and 27 correctness properties
- **[Design Rationale](specs/design-rationale.md)**: Detailed explanation of design decisions, alternatives considered, and tradeoffs

## Design Principles

1. **Offline Learning**: Industry-standard batch training approach for stability and reproducibility
2. **Simplicity First**: Tabular Q-learning for interpretability and ease of implementation
3. **Separation of Concerns**: Independent Agent, Environment, and Experience Buffer components
4. **Extensibility**: Clear extension points for future enhancements without premature complexity

## Quick Start

```python
from src import RLAgent, Environment

# Initialize agent with prompt options
agent = RLAgent(prompts=["Be concise", "Be detailed", "Be creative"])
env = Environment()  # Manual mode — you provide reward scores

# Training loop
state = "summarize an article"
for episode in range(5):
    action = agent.select_action(state)   # Agent picks a prompt
    reward = env.execute(action, state)    # You score it [-1.0, 1.0]
    agent.update(state, action, reward)    # Agent learns from your feedback

# Inspect what the agent learned
print(agent.q_table.to_dict())
```

Or run the included demo:

```bash
python3 demo.py
```

## Project Status

Phase 1 (Core Learning Loop) is complete:
- Q-table data structure with state-action value storage
- RL agent with configurable hyperparameters and parameter validation
- ε-greedy action selection (explore vs exploit)
- Q-learning update rule
- Manual HITL environment for reward collection
- 66 unit tests passing

Remaining phases: experience buffer, batch training, mode switching, exploration decay, metrics, policy persistence, state encoding, and extensibility interfaces.

## License

See [LICENSE](LICENSE) file for details.
