#!/usr/bin/env python3
"""Week 1, Day 1 environment report. Run with `make check`.

Prints what you have, what is missing, and the exact next command. Intentionally
has no third-party imports at module level so it runs before any install.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, BAD, WARN = f"{GREEN}OK{RESET}", f"{RED}MISSING{RESET}", f"{YELLOW}CHECK{RESET}"


def row(label: str, status: str, detail: str = "") -> None:
    print(f"  {label:<26} {status:<22} {DIM}{detail}{RESET}")


def have(cmd: str) -> str | None:
    return shutil.which(cmd)


def version_of(cmd: str, *args: str) -> str:
    try:
        out = subprocess.run([cmd, *args], capture_output=True, text=True, timeout=10)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        return ""


def main() -> int:
    print(f"\n{'=' * 72}\n  AI Engineer — environment check\n{'=' * 72}\n")
    missing: list[str] = []

    print("Machine")
    row("platform", OK, f"{platform.system()} {platform.release()} / {platform.machine()}")
    row("python (running this)", OK, sys.version.split()[0])
    if platform.machine() != "arm64":
        row("apple silicon", WARN, "not arm64 — MPS unavailable, CPU fallback is fine")

    print("\nToolchain")
    for cmd, args, hint in [
        ("uv", ("--version",), "brew install uv"),
        ("git", ("--version",), "brew install git"),
        ("ollama", ("--version",), "brew install ollama && brew services start ollama"),
        ("pdftotext", ("-v",), "brew install poppler   # needed in Week 10"),
        ("docker", ("--version",), "brew install --cask orbstack   # needed in Week 10"),
    ]:
        if have(cmd):
            row(cmd, OK, version_of(cmd, *args))
        else:
            row(cmd, BAD, hint)
            missing.append(hint)

    print("\nPython packages (inside the venv)")
    for mod, extra in [("torch", True), ("numpy", False), ("pytest", False)]:
        try:
            m = __import__(mod)
            detail = getattr(m, "__version__", "")
            row(mod, OK, detail)
            if extra and mod == "torch":
                mps = m.backends.mps.is_available()
                row(
                    "  torch MPS backend",
                    OK if mps else WARN,
                    "GPU acceleration active" if mps else "CPU fallback — slower but fine",
                )
        except ImportError:
            row(mod, BAD, "make setup")
            missing.append("make setup")

    print("\nOllama models")
    if have("ollama"):
        out = version_of("ollama", "list")
        tags = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
        for want, tag in (("qwen3", "qwen3:4b"), ("llama3.2", "llama3.2:3b"),
                          ("nomic-embed-text", "nomic-embed-text")):
            if want in tags:
                row(tag, OK)
            else:
                row(tag, BAD, f"ollama pull {tag}")
                missing.append(f"ollama pull {tag}")
        del out
    else:
        row("(skipped)", WARN, "ollama not installed")

    print(f"\n{'=' * 72}")
    if missing:
        print(f"  {YELLOW}Next steps:{RESET}")
        for m in dict.fromkeys(missing):
            print(f"    {m}")
    else:
        print(f"  {GREEN}Everything is ready. Start Week 1, Day 1.{RESET}")
    print(f"{'=' * 72}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
