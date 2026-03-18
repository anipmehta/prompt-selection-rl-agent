# AI Assistant Development Rules

This document contains rules and guidelines for AI assistants (Claude, GPT, etc.) working on this project.

## Git Workflow Rules

### NEVER Push Directly to Main

**CRITICAL:** Do NOT push commits directly to the `main` branch. Always use the PR workflow.

**Correct workflow:**
1. Create a feature branch: `git checkout -b feature/task-name`
2. Make commits to the feature branch
3. Push the feature branch: `git push -u origin feature/task-name`
4. Create a Pull Request for review
5. Wait for approval before merging

**Example:**
```bash
# ❌ WRONG - Never do this
git push origin main

# ✅ CORRECT - Always do this
git checkout -b feature/implement-agent
git add .
git commit -m "Implement agent initialization"
git push -u origin feature/implement-agent
# Then create PR via GitHub UI or gh CLI
```

### Branch Naming Convention

Use descriptive branch names following this pattern:
- `feature/task-X-description` - For new features (e.g., `feature/task-2-agent-init`)
- `fix/issue-description` - For bug fixes
- `docs/description` - For documentation updates
- `test/description` - For test additions

### Commit Strategy

**CRITICAL:** When working on the same task/branch, use `git commit --amend` to update the existing commit instead of creating new ones. Only create a new commit if:
- The user explicitly asks for a new commit
- You're starting a different task

### PR Description

**CRITICAL:** After pushing a feature branch, ALWAYS provide a PR description in markdown format. The description should include:
- Summary of what changed and why
- What was added/modified (files, methods, classes)
- Test coverage (number of tests, what's covered)
- Requirements/tasks addressed
- How to test the changes

Never skip this step — the user expects a copy-pasteable PR description after every push.

### Build Verification

**CRITICAL:** After EVERY code change, run `make check` (lint + tests). If the build fails:
1. Attempt to fix the issue immediately
2. Explain what failed and why
3. Run `make check` again to verify the fix
4. Only commit when the build is green

**Build command:** `make check` (runs ruff + pylint + tests)

Follow conventional commit format:
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

**Example:**
```
feat: implement Q-table data structure

- Create QTable class with get/set methods
- Add serialization support (to_dict/from_dict)
- Write 13 unit tests covering all operations

Implements Task 1
```

## Code Quality Rules

### Always Run Tests Before Committing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_q_table.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Write docstrings for all classes and public methods
- Keep functions focused and small (< 50 lines)

### DRY Principle (Don't Repeat Yourself)

**CRITICAL:** Avoid magic numbers and magic strings. Define them as constants IN THE SAME FILE where they're used.

**Rules:**
1. Define constants at module level (top of file) or class level
2. Use UPPER_CASE naming for constants
3. Keep constants in the SAME file where they're used (avoid code distance)
4. Only extract to shared constants file if used across multiple files

**Example:**

```python
# ❌ WRONG - Magic numbers scattered everywhere
class Agent:
    def __init__(self):
        self.learning_rate = 0.1
        
    def validate(self, rate):
        if rate < 0.0 or rate > 1.0:
            raise ValueError("Rate must be between 0.0 and 1.0")

# ✅ CORRECT - Constants at class level
class Agent:
    # Configuration constants
    DEFAULT_LEARNING_RATE = 0.1
    MIN_RATE = 0.0
    MAX_RATE = 1.0
    
    def __init__(self):
        self.learning_rate = self.DEFAULT_LEARNING_RATE
        
    def validate(self, rate):
        if rate < self.MIN_RATE or rate > self.MAX_RATE:
            raise ValueError(
                f"Rate must be between {self.MIN_RATE} and {self.MAX_RATE}"
            )

# For test files - constants at module level
# test_agent.py
import pytest
from src.agent import Agent

# Test constants - defined at module level
VALID_RATE = 0.5
INVALID_RATE_LOW = -0.1
INVALID_RATE_HIGH = 1.5

def test_validation():
    agent = Agent()
    assert agent.validate(VALID_RATE) is True
    
    with pytest.raises(ValueError):
        agent.validate(INVALID_RATE_LOW)
```

**Common constants to define:**
- Default parameter values (learning rates, thresholds, etc.)
- Valid ranges (min/max values)
- Error messages
- Mode strings ("training", "inference")
- File extensions (".json", ".pkl")
- Test data values

### Testing Requirements

- Write tests for all new functionality
- Aim for >90% code coverage
- Include unit tests and property-based tests where applicable
- Test edge cases and error conditions

## Task Execution Rules

### Task Status Updates

Always update task status in `.kiro/specs/prompt-selection-rl-agent/tasks.md`:
- Set to `in_progress` when starting a task
- Set to `completed` when task is done and tests pass
- Reference task number in commit messages

### Implementation Order

Follow the task order defined in TASKS.md:
1. Complete Phase 1 (Core Learning Loop) first
2. Get user approval before moving to next phase
3. Don't skip tasks unless explicitly approved

## Communication Rules

### Ask Before Major Decisions

Always ask the user before:
- Changing the project structure
- Adding new dependencies
- Modifying the spec documents
- Skipping tasks or tests
- Making architectural changes

### Provide Context

When presenting work:
- Show what was implemented
- Show test results
- Explain any deviations from the spec
- Highlight any issues or blockers

## Repository Structure

Maintain this structure:
```
prompt-selection-rl-agent/
├── .kiro/                    # Kiro workflow files (internal)
│   └── specs/
├── specs/                    # Public spec documentation
│   ├── requirements.md
│   ├── design.md
│   └── design-rationale.md
├── src/                      # Source code
│   ├── __init__.py
│   └── *.py
├── tests/                    # Test files
│   └── test_*.py
├── TASKS.md                  # Implementation tracking
├── CLAUDE.md                 # This file
└── README.md                 # Project overview
```

## Summary

**Golden Rules:**
1. ✅ Always create feature branches
2. ✅ Always create PRs (never push to main)
3. ✅ Always run tests before committing
4. ✅ Always update task status
5. ✅ Always ask before major changes
6. ✅ Always provide a markdown PR description after pushing

**Remember:** The user has branch protection enabled on `main`. Direct pushes will be rejected.
