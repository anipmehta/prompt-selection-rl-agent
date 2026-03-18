# Implementation Plan: Prompt Selection RL Agent

## Overview

This implementation plan breaks down the RL agent into discrete, testable phases. We start with Phase 1 (Core Learning Loop) to establish the fundamental Q-learning mechanics with a minimal working system. This allows early validation before building additional features.

The implementation uses Python and follows the offline batch training paradigm described in the design document.

## Phase 1: Core Learning Loop (MVP)

This phase implements the absolute minimum to demonstrate Q-learning in action.

- [x] 1. Set up project structure and Q-table data structure
  - Create project directory structure (src/, tests/)
  - Implement Q-table as nested dictionary: `Dict[str, Dict[str, float]]`
  - Add helper methods: get Q-value (default 0.0 for unseen pairs), set Q-value
  - _Requirements: 1.4, 4.4_

- [x] 2. Implement agent initialization with parameter validation
  - [x] 2.1 Create RLAgent class with __init__ method
    - Accept parameters: prompts, learning_rate, discount_factor, exploration_rate, decay_rate, min_exploration
    - Store prompts list and all hyperparameters as instance variables
    - Initialize empty Q-table dictionary
    - Set default mode to "training"
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 11.2_
  
  - [x] 2.2 Add parameter validation logic
    - Validate learning_rate in [0.0, 1.0], raise ConfigurationError if invalid
    - Validate discount_factor in [0.0, 1.0], raise ConfigurationError if invalid
    - Validate exploration_rate in [0.0, 1.0], raise ConfigurationError if invalid
    - Validate decay_rate in [0.0, 1.0], raise ConfigurationError if invalid
    - Validate prompts list is not empty, raise ConfigurationError if empty
    - _Requirements: 1.1, 1.2, 1.3, 9.1, 9.2, 9.5_
  
  - [ ]* 2.3 Write property test for parameter validation
    - **Property 1: Parameter Validation**
    - **Validates: Requirements 1.1, 1.2, 1.3, 6.2, 9.1, 9.2**
    - Test that valid parameters [0.0, 1.0] are accepted
    - Test that invalid parameters raise ConfigurationError
  
  - [ ]* 2.4 Write property test for initialization state
    - **Property 2: Empty Q-Table Initialization**
    - **Validates: Requirements 1.4**
    - **Property 3: Prompt List Storage**
    - **Validates: Requirements 1.5**

- [x] 3. Implement ε-greedy action selection
  - [x] 3.1 Create select_action method
    - Accept state parameter (string)
    - In training mode: generate random number, compare with exploration_rate
    - If exploring: return random prompt from prompts list
    - If exploiting: look up Q-values for state, return prompt with max Q-value
    - In inference mode: always exploit (return prompt with max Q-value)
    - Handle unseen states by treating all Q-values as 0.0
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 3.2 Write property test for action selection validity
    - **Property 4: Action Selection Returns Valid Prompt**
    - **Validates: Requirements 2.1, 2.5**
    - Test that select_action always returns a prompt from the prompts list
  
  - [ ]* 3.3 Write unit tests for ε-greedy behavior
    - Test exploration: with ε=1.0, verify random selection
    - Test exploitation: with ε=0.0 and known Q-values, verify max Q-value selection
    - Test unseen state: verify random selection when state not in Q-table
    - _Requirements: 2.2, 2.3, 2.4_

- [x] 4. Implement Q-learning update logic
  - [x] 4.1 Create update method
    - Accept parameters: state (string), action (string), reward (float)
    - Validate reward in [-1.0, 1.0], raise ValidationError if invalid
    - Check if agent is in training mode; if inference mode, return without updating
    - Get current Q-value for (state, action) pair, default to 0.0 if unseen
    - Apply Q-learning formula: Q(s,a) ← Q(s,a) + α[r - Q(s,a)] (with γ=0)
    - Store updated Q-value in Q-table
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ]* 4.2 Write property test for Q-learning formula correctness
    - **Property 7: Q-Learning Formula Correctness**
    - **Validates: Requirements 3.2**
    - Test that new Q-value equals Q₀ + α(r - Q₀) for any Q₀, α, r
  
  - [ ]* 4.3 Write property test for reward validation
    - **Property 8: Reward Value Validation**
    - **Validates: Requirements 3.4, 9.3**
    - Test that rewards in [-1.0, 1.0] are accepted
    - Test that rewards outside range raise ValidationError
  
  - [ ]* 4.4 Write unit tests for Q-value updates
    - Test that Q-value changes after update in training mode
    - Test that Q-value does NOT change in inference mode
    - Test unseen state-action pair initializes to 0.0
    - _Requirements: 3.1, 3.5, 4.4_

- [x] 5. Implement manual environment for testing
  - [x] 5.1 Create Environment class with execute method
    - Accept parameters: prompt (string), task (string)
    - In manual mode: print prompt and task, prompt user for reward input
    - Validate user input is float in [-1.0, 1.0]
    - Return reward value
    - _Requirements: 10.1, 10.2, 10.4, 10.5_
  
  - [x] 5.2 Add set_manual_mode method
    - Accept enabled parameter (boolean)
    - Store manual mode flag
    - _Requirements: 10.5_
  
  - [ ]* 5.3 Write unit tests for environment
    - Test that execute returns numeric reward
    - Test manual mode prompts for user input
    - _Requirements: 10.1, 10.4_

- [x] 6. Create simple test script to validate core loop
  - Create test script that demonstrates:
    - Initialize agent with 3 sample prompts
    - Create manual environment
    - Run 5 episodes: select action, get manual reward, update Q-values
    - Print Q-table after each episode to show learning
    - Verify exploration rate decays after each episode
  - This validates the core learning loop works end-to-end
  - _Requirements: 1.1-1.5, 2.1-2.5, 3.1-3.5, 6.1, 10.1-10.5_

- [x] 7. Checkpoint - Core learning loop functional
  - Ensure all tests pass, ask the user if questions arise.

## Phase 2: Experience Buffer and Batch Training

This phase adds offline batch training capabilities.

- [x] 8. Implement experience buffer
  - [x] 8.1 Create ExperienceBuffer class
    - Initialize with empty list to store episodes
    - Implement add method: append (state, action, reward) tuple
    - Implement get_all method: return list of all episodes
    - Implement clear method: empty the episodes list
    - Implement size method: return count of episodes
    - _Requirements: 12.1, 12.2, 12.5_
  
  - [ ]* 8.2 Write property test for experience storage
    - **Property 24: Experience Buffer Storage**
    - **Validates: Requirements 12.1, 12.2**
    - Test that added episodes are retrievable
  
  - [ ]* 8.3 Write property test for buffer clearing
    - **Property 26: Experience Buffer Clearing**
    - **Validates: Requirements 12.5**

- [x] 9. Integrate experience buffer with agent
  - [x] 9.1 Add experience buffer to agent initialization
    - Create ExperienceBuffer instance in __init__
    - _Requirements: 12.1_
  
  - [x] 9.2 Implement store_experience method
    - Accept parameters: state, action, reward
    - Call buffer.add(state, action, reward)
    - _Requirements: 12.2_
  
  - [x] 9.3 Implement train_batch method
    - Get all episodes from buffer using get_all()
    - For each episode, call update(state, action, reward)
    - _Requirements: 12.3, 12.4_
  
  - [x] 9.4 Implement clear_buffer method
    - Call buffer.clear()
    - _Requirements: 12.5_
  
  - [x]* 9.5 Write integration test for batch training workflow
    - Test: collect experiences → train_batch → verify Q-values updated
    - _Requirements: 12.1-12.5_

- [x] 10. Checkpoint - Batch training functional
  - Ensure all tests pass, ask the user if questions arise.

## Phase 2.5: Strategy Pattern Refactor

This phase extracts the learning algorithm into a pluggable strategy, enabling comparison of different RL techniques.

- [x] 10.5 Extract learning strategy pattern
  - [x] 10.5.1 Create BaseLearningStrategy ABC
    - Define interface: select_action(), update(), get_q_values(), get_table()
    - _Requirements: 8.4_
  
  - [x] 10.5.2 Create QLearningStrategy implementation
    - Extract Q-learning logic from RLAgent into QLearningStrategy
    - Owns QTable, learning_rate, discount_factor
    - _Requirements: 3.2_
  
  - [x] 10.5.3 Refactor RLAgent to accept strategy parameter
    - Add optional strategy param to __init__ (defaults to QLearningStrategy)
    - Agent delegates select_action and update to strategy
    - Backward compatible — no API changes for existing code
    - _Requirements: 8.4_
  
  - [x] 10.5.4 Write tests for strategy swapping
    - Test QLearningStrategy in isolation
    - Test agent uses custom strategy when provided
    - Test default strategy is QLearningStrategy
    - 7 new tests, 90 total passing

## Phase 3: Mode Switching and Exploration Decay

This phase adds training/inference mode separation and exploration decay.

- [x] 11. Implement mode switching
  - [x] 11.1 Add mode attribute to agent
    - Initialize mode to "training" in __init__
    - _Requirements: 11.2_
  
  - [x] 11.2 Implement set_mode method
    - Accept mode parameter ("training" or "inference")
    - Validate mode is valid, raise ModeError if invalid
    - Set mode attribute
    - If switching to inference, set exploration_rate to 0.0
    - _Requirements: 11.1, 11.3, 11.4_
  
  - [x] 11.3 Update select_action to respect mode
    - In inference mode, always exploit (skip ε-greedy check)
    - _Requirements: 2.4, 6.5_
  
  - [x] 11.4 Update update method to respect mode
    - Check mode at start; if inference, return immediately without updating
    - _Requirements: 3.5, 11.4_
  
  - [ ]* 11.5 Write property test for frozen Q-table in inference mode
    - **Property 9: Frozen Q-Table in Inference Mode**
    - **Validates: Requirements 3.5**
  
  - [ ]* 11.6 Write property test for inference mode exploitation
    - **Property 5: Inference Mode Exploitation**
    - **Validates: Requirements 2.4**

- [x] 12. Implement exploration decay
  - [x] 12.1 Add decay logic to agent
    - Create decay_exploration method
    - Apply multiplicative decay: exploration_rate *= decay_rate
    - Enforce minimum threshold: max(exploration_rate, min_exploration)
    - Only decay in training mode
    - _Requirements: 6.1, 6.3, 6.4, 6.5_
  
  - [x] 12.2 Call decay after each episode
    - Update store_experience to call decay_exploration after storing
    - _Requirements: 6.1_
  
  - [ ]* 12.3 Write property test for exploration decay
    - **Property 13: Exploration Rate Decay**
    - **Validates: Requirements 6.1, 6.3, 6.4**
  
  - [ ]* 12.4 Write property test for zero exploration in inference
    - **Property 14: Zero Exploration in Inference Mode**
    - **Validates: Requirements 6.5**

- [x] 13. Checkpoint - Mode switching and decay functional
  - Ensure all tests pass, ask the user if questions arise.

## Phase 4: Metrics and Monitoring

This phase adds performance tracking capabilities.

- [x] 14. Implement metrics tracking
  - [x] 14.1 Add metrics attributes to agent
    - Initialize episode_count to 0
    - Initialize cumulative_reward to 0.0
    - Initialize prompt_selection_counts as empty dict
    - _Requirements: 7.1, 7.2, 7.5_
  
  - [x] 14.2 Update metrics during operation
    - In store_experience: increment episode_count, add to cumulative_reward
    - In select_action: increment count for selected prompt
    - _Requirements: 7.1, 7.2, 7.5_
  
  - [x] 14.3 Implement get_metrics method
    - Return dictionary with episode_count, cumulative_reward, average_reward
    - Calculate average_reward as cumulative_reward / episode_count (handle division by zero)
    - Include exploration_rate and prompt_selection_counts
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [ ]* 14.4 Write property tests for metrics
    - **Property 15: Episode Counting**
    - **Validates: Requirements 7.1**
    - **Property 16: Cumulative Reward Tracking**
    - **Validates: Requirements 7.2**
    - **Property 17: Average Reward Calculation**
    - **Validates: Requirements 7.3**
    - **Property 18: Prompt Selection Counting**
    - **Validates: Requirements 7.5**

- [x] 15. Checkpoint - Metrics tracking functional
  - Ensure all tests pass, ask the user if questions arise.

## Phase 5: Policy Persistence

This phase adds save/load functionality for Q-tables and agent state.

- [ ] 16. Implement policy serialization
  - [ ] 16.1 Implement save_policy method
    - Create dictionary with q_table, config (hyperparameters), metrics, mode, prompts
    - Serialize to JSON file at specified filepath
    - Handle file system errors, raise PersistenceError with details
    - _Requirements: 5.1, 5.3, 9.4, 11.6_
  
  - [ ] 16.2 Implement load_policy method
    - Read JSON file from specified filepath
    - Deserialize and restore q_table, config, metrics, mode, prompts
    - Handle file not found and invalid JSON, raise PersistenceError
    - _Requirements: 5.2, 5.4, 9.4, 11.6_
  
  - [ ]* 16.3 Write property test for policy round-trip
    - **Property 12: Policy Serialization Round-Trip**
    - **Validates: Requirements 5.5**
    - Test that save → load produces equivalent agent state
  
  - [ ]* 16.4 Write unit tests for persistence errors
    - Test file not found during load
    - Test invalid JSON format
    - Test permission errors during save
    - _Requirements: 9.4_

- [ ] 17. Implement experience buffer persistence
  - [ ] 17.1 Add save method to ExperienceBuffer
    - Serialize episodes list to JSON file
    - Handle file system errors
    - _Requirements: 12.6_
  
  - [ ] 17.2 Add load method to ExperienceBuffer
    - Deserialize episodes list from JSON file
    - Handle file not found and invalid JSON
    - _Requirements: 12.6_
  
  - [ ]* 17.3 Write property test for buffer round-trip
    - **Property 27: Experience Buffer Serialization Round-Trip**
    - **Validates: Requirements 12.7**

- [ ] 18. Checkpoint - Persistence functional
  - Ensure all tests pass, ask the user if questions arise.

## Phase 6: State Representation and Extensibility

This phase adds state encoding and extensibility hooks.

- [ ] 19. Implement state representation
  - [ ] 19.1 Add state encoder to agent
    - Add optional state_encoder parameter to __init__ (defaults to identity function)
    - Apply state_encoder in select_action and update before Q-table lookup
    - _Requirements: 4.1, 4.3, 8.4_
  
  - [ ] 19.2 Create default state encoder implementations
    - Identity encoder: return string as-is
    - Hash encoder: return hash of string for consistent keys
    - Truncate encoder: return first N characters
    - _Requirements: 4.1, 4.2_
  
  - [ ]* 19.3 Write property test for state consistency
    - **Property 10: State Representation Consistency**
    - **Validates: Requirements 4.1, 4.3**
    - Test that encoding same task multiple times produces identical states

- [ ] 20. Add extensibility interfaces
  - [ ] 20.1 Document action executor interface
    - Create abstract base class or protocol for ActionExecutor
    - Define execute method signature
    - _Requirements: 8.1, 8.2_
  
  - [ ] 20.2 Document reward function interface
    - Create abstract base class or protocol for RewardFunction
    - Define compute method signature
    - _Requirements: 8.3_
  
  - [ ] 20.3 Add placeholder for tool integration
    - Add comments indicating where tool execution would be integrated
    - Document expected interface for tool actions
    - _Requirements: 8.5_

- [ ] 21. Checkpoint - State representation and extensibility complete
  - Ensure all tests pass, ask the user if questions arise.

## Phase 7: Integration and Documentation

This phase creates end-to-end examples and documentation.

- [ ] 22. Create comprehensive example scripts
  - [ ] 22.1 Create training workflow example
    - Script demonstrating: initialize → collect episodes → batch train → save policy
    - Include comments explaining each step
    - _Requirements: All training-related requirements_
  
  - [ ] 22.2 Create inference workflow example
    - Script demonstrating: load policy → set inference mode → select actions
    - Show that Q-values don't change during inference
    - _Requirements: All inference-related requirements_
  
  - [ ] 22.3 Create metrics visualization example
    - Script that tracks and plots metrics over training episodes
    - Show exploration decay, cumulative reward, prompt selection distribution
    - _Requirements: 7.1-7.5_

- [ ] 23. Write user documentation
  - Create README with:
    - Quick start guide
    - API reference for RLAgent, Environment, ExperienceBuffer
    - Configuration parameters explanation
    - Example usage patterns
    - Troubleshooting common issues

- [ ] 24. Final checkpoint - Complete system validation
  - Run full test suite (unit tests + property tests)
  - Verify all requirements are covered
  - Run example scripts to validate end-to-end workflows
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at phase boundaries
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples, edge cases, and integration points
- Phase 1 is the absolute minimum to see Q-learning working with manual testing
- Phases 2-7 add production-ready features incrementally
