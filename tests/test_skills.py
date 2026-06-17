from agy_acp.skills import _discover_skills


def test_discover_skills_toml(tmp_path):
    """_discover_skills finds TOML custom commands."""
    cmds_dir = tmp_path / ".gemini" / "commands"
    cmds_dir.mkdir(parents=True)
    (cmds_dir / "greet.toml").write_text('prompt = "Say hello"\ndescription = "Greet the user"')
    (cmds_dir / "git" / "commit.toml").parent.mkdir()
    (cmds_dir / "git" / "commit.toml").write_text('prompt = "Commit changes"')

    skills = _discover_skills(str(tmp_path))
    names = {s.name for s in skills}
    assert "greet" in names
    assert "git:commit" in names
    greet = next(s for s in skills if s.name == "greet")
    assert greet.description == "Greet the user"


def test_discover_skills_md(tmp_path):
    """_discover_skills finds SKILL.md agent skills."""
    skills_dir = tmp_path / ".gemini" / "skills" / "review"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text('---\nname: review\ndescription: "Review code changes"\n---\nReview instructions here.')

    skills = _discover_skills(str(tmp_path))
    names = {s.name for s in skills}
    assert "review" in names
    review = next(s for s in skills if s.name == "review")
    assert review.description == "Review code changes"


def test_discover_skills_empty(tmp_path):
    """_discover_skills returns empty list when no skill dirs exist."""
    assert _discover_skills(str(tmp_path)) == []
