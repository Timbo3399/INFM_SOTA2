"""Auswertung erzeugter Texte."""

import re

# Wörter inklusive Akzenten und innerem Apostroph (don't, Lórien), ohne "_"
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def longest_copied_span(text: str, reference: str) -> str:
    """Längste Wortfolge aus ``text``, die genau so in ``reference`` vorkommt.

    Verglichen wird Wort für Wort, ohne Groß-/Kleinschreibung und Satzzeichen;
    typografische Apostrophe zählen wie ``'``.
    Ergebnis in Kleinbuchstaben, ``""`` wenn kein einziges Wort vorkommt.
    """
    words = _words(text)
    haystack = " " + " ".join(_words(reference)) + " "
    longest: list[str] = []
    for start in range(len(words)):
        end = start + len(longest) + 1
        # Nur Folgen prüfen, die länger wären als der bisherige Fund
        while end <= len(words) and f" {' '.join(words[start:end])} " in haystack:
            longest = words[start:end]
            end += 1
    return " ".join(longest)
