# LLM Systems Project

Develop, verify, and test a modern LLM systems implementation, with emphasis on new, novel, and modern design.

## Source of truth

- Current code is authoritative for implemented behavior. Inspect it; never assume an earlier suggestion was implemented.
- `docs/LLM_SYSTEM_HANDOFF.md` records architectural decisions and project state. Read it before any architecture-sensitive work.
- Use git history when it helps explain why something is the way it is.

## Keeping the handoff doc current (Claude's job, not the user's)

- Claude owns `docs/LLM_SYSTEM_HANDOFF.md`. The user should never have to update it manually.
- When a decision is agreed in conversation, add or update its entry with status **Agreed**.
- When code implementing a decision is observed in the repo, change its status to **Implemented**, noting where it lives.
- When a decision supersedes an older one, mark the old one **Superseded**, link the new one, and state what changed.
- Keep entries short. Edit this doc freely; it is not production code.

## Architecture and collaboration

- Explain the purpose of an abstraction before changing or implementing it.
- Establish ownership, data flow, tensor shapes, invariants, and lifecycle responsibilities.
- Always distinguish proposed, agreed, and implemented behavior.
- Prefer the durable target architecture when it is reasonably clear. Do not simplify an architecture merely because the simpler version is easier to implement.
- When comparing designs, explain what each preserves, loses, or postpones, and what later refactor would be required.
- Do not silently change architecture.

## Implementation workflow

- The user is normally the implementer. Only write production code when explicitly asked.
- Give contracts, responsibilities, shapes, invariants, and essential algorithm steps before any full code.
- Review correctness and architectural quality separately.
- Surface realistic failure cases and boundary conditions.
- Stay focused on the current question; don't introduce unrelated abstractions.

## Testing

- Inspect the current implementation before writing tests.
- Test contracts, tensor shapes, invariants, state transitions, boundary cases, and realistic failure modes.
- Do not modify production code to make tests pass unless explicitly asked.
- If tests reveal a mismatch between implementation and intended architecture, report it clearly.
- Run the focused tests after changes, then the relevant broader suite.
