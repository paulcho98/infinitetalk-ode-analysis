"""Minimal stand-in for fastgen.utils.logging_utils (print-based, rank-agnostic)."""
import sys

def _emit(level, msg):
    print(f"[{level}] {msg}", file=sys.stderr if level in ("WARNING", "ERROR", "CRITICAL") else sys.stdout)

def trace(msg): _emit("TRACE", msg)
def debug(msg): _emit("DEBUG", msg)
def info(msg): _emit("INFO", msg)
def success(msg): _emit("SUCCESS", msg)
def warning(msg): _emit("WARNING", msg)
def error(msg): _emit("ERROR", msg)
def critical(msg): _emit("CRITICAL", msg)
def set_log_level(level): pass
