"""Pipe calcite --textBuffer output through a virtual ANSI terminal and render the screen."""
import sys
import pyte

screen = pyte.Screen(80, 25)
stream = pyte.Stream(screen)

raw = sys.stdin.read()

# Extract textBuffer content
in_buffer = False
text = []
for line in raw.split('\n'):
    if '--textBuffer:' in line:
        in_buffer = True
        continue
    if in_buffer:
        if line.startswith('--') and ':' in line:
            break
        text.append(line)

content = '\n'.join(text)

# CSS content-list uses \a for newline
content = content.replace('\\a ', '\n')

# The CSS charset maps ESC (0x1B) to X, so ANSI escapes show as X[...
# Restore them: X[ followed by ANSI params → ESC[
import re
content = re.sub(r'X\[', '\x1b[', content)

# Also map remaining X (non-printable placeholder) to nothing
# But only isolated X not part of words — actually just leave them,
# some might be real 'X' characters from the game text

# Feed through virtual terminal
stream.feed(content)

# Render
for i, row in enumerate(screen.display):
    stripped = row.rstrip()
    if stripped:
        print(stripped)
