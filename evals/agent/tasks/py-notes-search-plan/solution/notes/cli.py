"""``python -m notes <command>``."""

from __future__ import annotations

import argparse
import sys
from typing import TextIO

from notes.formatting import fmt_note, normalise_tags
from notes.store import NoteStore
from notes.textstats import top_words, word_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="notes", description="A tiny command-line notebook.")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="add a note")
    add.add_argument("text")
    add.add_argument("--tag", action="append", default=[], help="a tag (repeatable)")
    listing = commands.add_parser("list", help="list notes")
    listing.add_argument("--tag", help="only notes with this tag")
    commands.add_parser("stats", help="word statistics")
    search = commands.add_parser("search", help="notes containing a text")
    search.add_argument("text")
    return parser


def main(argv: list[str] | None = None, store: NoteStore | None = None, out: TextIO | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = store or NoteStore()
    out = out or sys.stdout
    if args.command == "add":
        if not args.text.strip():
            print("error: the note is empty", file=sys.stderr)
            return 2
        note = store.add(args.text.strip(), normalise_tags(args.tag))
        print(f"Added {fmt_note(note)}", file=out)
        return 0
    notes = store.load()
    if args.command == "list":
        tag = args.tag.lower() if args.tag else None
        for note in notes:
            if tag is None or tag in note["tags"]:
                print(fmt_note(note), file=out)
        return 0
    if args.command == "search":
        needle = args.text.lower()
        found = [n for n in notes if needle in n["text"].lower()]
        if not found:
            print("no notes found", file=out)
            return 1
        for note in found:
            print(fmt_note(note), file=out)
        return 0
    if args.command == "stats":
        words = sum(word_count(n["text"]) for n in notes)
        print(f"notes: {len(notes)}", file=out)
        print(f"words: {words}", file=out)
        for word, count in top_words([n["text"] for n in notes]):
            print(f"  {word}: {count}", file=out)
        return 0
    return 1
