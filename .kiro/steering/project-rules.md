---
inclusion: always
---

# Project Rules

## Git Workflow

- Never push directly to main. Always use feature branches and PRs.
- Branch naming: `feature/task-X-description`, `fix/issue-description`, `refactor/description`
- Use `git commit --amend` when working on the same task/branch.
- Run `make check` after every code change, before committing.
- After pushing a feature branch, always provide a markdown PR description covering: summary, what changed, tests, requirements addressed, and how to test.

## Code Style

- PEP 8, type hints, docstrings on all classes and public methods.
- No magic numbers or strings — define constants at class or module level using UPPER_CASE.
- Keep constants in the same file where they're used.
- Functions under 50 lines.

## Testing

- Write tests for all new functionality.
- Aim for >90% coverage.
- Test edge cases and error conditions.

## Task Tracking

- Update task status in `.kiro/specs/prompt-selection-rl-agent/tasks.md` when starting and completing tasks.
- Reference task numbers in commit messages.
