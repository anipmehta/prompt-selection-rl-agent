#!/usr/bin/env python3
"""
Demo script: Prompt Selection RL Agent — Core Training Loop

Demonstrates the end-to-end training loop:
  1. Initialize an RLAgent with sample prompts
  2. Create a manual Environment
  3. Run 5 episodes (select action → get reward → update Q-values)
  4. Print the Q-table and exploration rate after each episode

The Environment is in manual mode, so rewards are entered via stdin.
To run non-interactively, pipe rewards in:
    echo -e "0.8\n-0.2\n0.5\n1.0\n0.3" | python3 demo.py

Requirements validated: 1.1-1.5, 2.1-2.5, 3.1-3.5, 6.1, 10.1-10.5
"""

from src import RLAgent, Environment


def print_q_table(agent: RLAgent) -> None:
    """Pretty-print the current Q-table contents."""
    table = agent.q_table.to_dict()
    if not table:
        print("  (empty)")
        return
    for state, actions in table.items():
        print(f"  State: {state!r}")
        for action, q_value in actions.items():
            print(f"    {action!r}: {q_value:.4f}")


def main() -> None:
    """Run an interactive 5-episode training demo."""
    # --- 1. Initialize agent with 3 sample prompts ---
    prompts = ["Be concise", "Be detailed", "Be creative"]
    agent = RLAgent(
        prompts=prompts,
        learning_rate=0.1,       # Req 1.1
        discount_factor=0.0,     # Req 1.2
        exploration_rate=1.0,    # Req 1.3 — starts fully exploratory
    )
    print("=== Prompt Selection RL Agent Demo ===\n")
    print(f"Prompts: {agent.prompts}")                  # Req 1.5
    print(f"Initial Q-table: {agent.q_table.to_dict()}") # Req 1.4 — empty
    print(f"Mode: {agent.mode}\n")

    # --- 2. Create manual environment ---
    env = Environment()          # Defaults to manual mode (Req 10.5)
    env.set_manual_mode(True)    # Explicit for clarity

    # --- 3. Run 5 training episodes ---
    task = "Summarize this article"
    num_episodes = 5

    for episode in range(1, num_episodes + 1):
        print(f"--- Episode {episode} ---")

        # Select action via ε-greedy (Req 2.1-2.3, 2.5)
        action = agent.select_action(task)
        print(f"Selected prompt: {action!r}")

        # Get reward from manual environment (Req 10.1-10.4)
        reward = env.execute(action, task)

        # Update Q-values with the reward (Req 3.1-3.4)
        agent.update(task, action, reward)

        # Print Q-table after update (Req 3.3)
        print(f"\nQ-table after episode {episode}:")
        print_q_table(agent)

        # Print exploration rate (Req 6.1 — decay not yet implemented,
        # so this stays at 1.0; that's expected at this stage)
        print(f"Exploration rate: {agent.exploration_rate}\n")

    print("=== Phase 1 Demo complete ===\n")

    # --- 4. Batch training demo (Phase 2) ---
    print("=== Batch Training Demo ===\n")

    batch_agent = RLAgent(prompts=prompts, learning_rate=0.1)
    print("Collecting 5 episodes into experience buffer (no learning yet)...\n")

    for episode in range(1, num_episodes + 1):
        print(f"--- Episode {episode} (buffered) ---")
        action = batch_agent.select_action(task)
        print(f"Selected prompt: {action!r}")
        reward = env.execute(action, task)
        batch_agent.store_experience(task, action, reward)
        print(f"Buffer size: {batch_agent._buffer.size()}")  # pylint: disable=protected-access

        print("\nQ-table (should still be empty):")
        print_q_table(batch_agent)
        print()

    # Now train on all buffered experiences at once
    print("--- Training on all buffered experiences ---")
    batch_agent.train_batch()
    print("\nQ-table after batch training:")
    print_q_table(batch_agent)

    # Clear buffer, Q-values persist
    batch_agent.clear_buffer()
    print(f"\nBuffer cleared. Buffer size: {batch_agent._buffer.size()}")  # pylint: disable=protected-access
    print("Q-table still intact:")
    print_q_table(batch_agent)

    print("\n=== Batch Training Demo complete ===")


if __name__ == "__main__":
    main()
