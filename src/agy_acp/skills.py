import tempfile
from pathlib import Path

from acp.schema import AvailableCommand


def _parse_skill_description(skill_md: Path) -> str | None:
    """Extract description from a SKILL.md frontmatter block, or None."""
    try:
        text = skill_md.read_text()
        if text.startswith("---"):
            end = text.index("---", 3)
            for line in text[3:end].split("\n"):
                if line.strip().startswith("description:"):
                    return line.split(":", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    return None


def _discover_skills(
    cwd: str,
    extra_skills: list[Path] | None = None,
) -> list[AvailableCommand]:
    """Scan for TOML custom commands and SKILL.md agent skills."""
    import tomllib

    commands: list[AvailableCommand] = []
    search_dirs = [
        (Path(cwd) / ".gemini" / "commands", "toml"),
        (Path(cwd) / ".gemini" / "skills", "skill"),
        (Path.home() / ".gemini" / "commands", "toml"),
        (Path.home() / ".gemini" / "skills", "skill"),
        (Path(cwd) / ".agents" / "skills", "skill"),
        (Path.home() / ".agents" / "skills", "skill"),
    ]

    seen: set[str] = set()
    for base_dir, fmt in search_dirs:
        if not base_dir.is_dir():
            continue
        if fmt == "toml":
            for toml_file in base_dir.rglob("*.toml"):
                name = toml_file.relative_to(base_dir).with_suffix("").as_posix().replace("/", ":")
                if name in seen:
                    continue
                seen.add(name)
                desc = f"Custom command: {name}"
                try:
                    data = tomllib.loads(toml_file.read_text())
                    if "description" in data:
                        desc = data["description"]
                except Exception:
                    pass
                commands.append(AvailableCommand(name=name, description=desc))
        elif fmt == "skill":
            for skill_dir in base_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                name = skill_dir.name
                if name in seen:
                    continue
                seen.add(name)
                desc = _parse_skill_description(skill_md) or f"Skill: {name}"
                commands.append(AvailableCommand(name=name, description=desc))

    for skill_dir in extra_skills or []:
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        name = skill_dir.name
        if name in seen:
            continue
        seen.add(name)
        desc = _parse_skill_description(skill_md) or f"Skill: {name}"
        commands.append(AvailableCommand(name=name, description=desc))

    return commands


def _setup_external_skills(skills: list[Path]) -> str | None:
    """Create a temp dir with symlinks to external skills. Returns the dir path, or None."""
    present = [s for s in skills if (s / "SKILL.md").exists()]
    if not present:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="agy_skills_"))
    for skill_dir in present:
        link = tmp / skill_dir.name
        if not link.exists():
            link.symlink_to(skill_dir)
    return str(tmp)


def _skills_paths(cwd: str) -> list[str]:
    """Return absolute skill directory paths to pass to the SDK."""
    base = Path(cwd).resolve()
    return [
        str(base / ".gemini" / "commands"),
        str(base / ".gemini" / "skills"),
        str(Path.home() / ".gemini" / "commands"),
        str(Path.home() / ".gemini" / "skills"),
        str(base / ".agents" / "skills"),
        str(Path.home() / ".agents" / "skills"),
    ]
