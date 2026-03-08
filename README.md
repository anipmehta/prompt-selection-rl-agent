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

### Online Learning (learn after each interaction)

```python
from src import RLAgent, Environment

agent = RLAgent(prompts=["Be concise", "Be detailed", "Be creative"])
env = Environment()  # Manual mode — you provide reward scores

state = "summarize an article"
for episode in range(5):
    action = agent.select_action(state)   # Agent picks a prompt
    reward = env.execute(action, state)    # You score it [-1.0, 1.0]
    agent.update(state, action, reward)    # Agent learns immediately

print(agent.q_table.to_dict())
```

### Offline Batch Learning (collect first, train later)

```python
from src import RLAgent, Environment

agent = RLAgent(prompts=["Be concise", "Be detailed", "Be creative"])
env = Environment()

state = "summarize an article"

# 1. Collect experiences without learning
for episode in range(5):
    action = agent.select_action(state)
    reward = env.execute(action, state)
    agent.store_experience(state, action, reward)  # Buffer only, no Q-update

# 2. Train on the full batch at once
agent.train_batch()
print(agent.q_table.to_dict())

# 3. Clear buffer when done (Q-values persist)
agent.clear_buffer()
```

Or run the included demo (covers both modes):

```bash
python3 demo.py
```

## Online vs Offline Learning

The agent supports two learning modes, each with different tradeoffs:

| | Online | Offline (Batch) |
|---|---|---|
| When it learns | After each interaction | After collecting a batch |
| Adapts | Immediately | After `train_batch()` |
| Data efficiency | Each experience used once | Can replay batches multiple times |
| Noise sensitivity | High — one bad reward can skew Q-values | Low — bad data can be filtered before training |
| Cost in production | Requires real-time judge (human or LLM) | Can batch judge calls for bulk pricing |
| Storage | None | Must store experiences in buffer |
| Best for | Rapid prototyping, interactive sessions | Production pipelines, reproducible training |

In practice, a hybrid approach works well: collect a batch of experiences, train, deploy the improved agent, collect another batch, repeat.

## Project Status

### Phase 1 — Core Learning Loop ✅
- Q-table data structure with state-action value storage
- RL agent with configurable hyperparameters and parameter validation
- ε-greedy action selection (explore vs exploit)
- Q-learning update rule
- Manual HITL environment for reward collection

### Phase 2 — Experience Buffer and Batch Training ✅
- Experience buffer for storing (state, action, reward) episodes
- `store_experience()` to collect interactions without learning
- `train_batch()` to replay all buffered experiences at once
- `clear_buffer()` to reset the buffer while preserving Q-values
- 83 unit tests passing

### Remaining
- Phase 3: Mode switching (training/inference) and exploration decay
- Phase 4: Metrics and monitoring
- Phase 5: Policy persistence (save/load Q-tables)
- Phase 6: State representation and extensibility
- Phase 7: Integration examples and documentation

## License

See [LICENSE](LICENSE) file for details.
