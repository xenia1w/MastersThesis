Guidance for Dev Agent when working on `MastersThesis`.

> **Stack:** Pydantic AI + uv 
> **Principle:** Keep it simple.

---


## References

**Pydantic AI Documentation:**
- https://ai.pydantic.dev/llms.txt
- https://ai.pydantic.dev/llms-full.txt
Prioritize the txt files, only visit the other documentation pages if info is not available in them.
- https://ai.pydantic.dev/agents/
- https://ai.pydantic.dev/multi-agent-applications/

**Internal Documentation:**
- [TOOLSETS_AND_SKILLS.md](./documentation/TOOLSETS_AND_SKILLS.md) - Tools vs Skills architecture
- [DI_pattern.md](./documentation/DI_pattern.md) - Dependency injection pattern
- [TOOL_EVENTS.md](./documentation/TOOL_EVENTS.md) - Tool events debug logs when using `.iter()`
- [STREAMING_DISPATCHER.md](./documentation/STREAMING_DISPATCHER.md) - Streaming events and dispatcher architecture
- [LANGFUSE_PROMPTING.md](./documentation/LANGFUSE_PROMPTING.md) - Prompt management

---

## Golden Rules

1. **Use Serena MCP** for all development
2. **Never** stop/delete databases, directories, files or containers without backups AND explicit permission
3. **Always type everything** - No `list[dict]`, `Any`, or untyped code. Use Pydantic models, TypedDicts, or classes
4. **Use uv** for all Python commands (or uvx if dependency not installed and irrelevant to the project itself)
5. **Consult official docs or SDK code** before implementing - never guess or add unknown fallbacks
6**Keep it minimal** - only implement what's requested, avoid over-engineering

---

## Coding Standards

### Typing

- **Always use Pydantic models** for schemas and validation
- **Use `Field()` for documentation** instead of comments
- **No `list[dict]`** - create proper models or TypedDicts
- **No `# type: ignore`** without understanding the issue deeply. Before adding a ignore, 
check why the issue is happening and try to fix it properly, while also avoiding unnecessary castings.

### Error Handling

- Graceful, structured error handling
- If raising exceptions, don't log separately (auto-logged)
- Handle errors at appropriate boundaries

### Functions & Comments

- Keep functions small and focused
- Comments explain **WHY**, not how (unless code is unclear)
- Typed params and return types always
- No ARGS in docstrings for agent tools

### Logging

- Use **Loguru** with levels: ENDPOINT, DB, CLASS
- Large payloads: `logger.bind(obj=payload).debug()`
- Production logs redact sensitive data via `LOG_LEVEL`

### Testing

- Quality over quantity
- Implement only what's asked
- Ask for clarification if in doubt

## Others
- KEEP IT SIMPLE, only handle cases that are mentioned. IF YOU SEE something is extemely important but no yet handled, 
ASK FOR CLARIFICATION BEFORE IMPLEMENTING.
- Container is running with auto reload, don't need to restart for code changes, only for dependency changes. 
- Application ALWAYS RUN FROM CONTAINER. 

---

## Design Principles

**Follow:** DRY, KISS, YAGNI, SRP, OCP, LSP, DIP, SoC, POLA

- Don't change logic unnecessarily
- Preserve existing comments (unless fixing TODOs)
- Research best practices before planning
- Keep plans/analysis **concise** - no fluff, no time estimates
- When asked to create a plan file on this project, use `.claude/tmp`. This is not meant for internal planning from
agent tools, only when specifically asked by the user.
- Add `# todo: ` comments for any unhandled edge cases you notice, that are not part of the plan but 
are necessary and/or should be conceptualized later.

---

## Versioning

- Version is defined in `pyproject.toml` under `[project] version`
- Releases are versioned using **semver** (e.g., `1.2.0`) via **git tags** (e.g., `git tag v1.2.0`)
- The CI pipeline reads `$CI_COMMIT_TAG` to inject the version at build time
- **When finishing work that constitutes a release**, ask the user if the version should be bumped and a new tag created
- When bumping, update the version in `pyproject.toml` **and** create a matching git tag (e.g., `v1.2.0`)

---

