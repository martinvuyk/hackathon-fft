The .cursor/references/modular directory is where the modular
codebase is contained. It includes the Mojo and MAX
source code. The references are to be used as a guide
for how the source code works and how some complex
Mojo is written and implemented.

The .cursor/skills directory is where the skills are contained.
Some are used for gpu code generation, etc.

The agent is not allowed to commit or change git state
in any way.

Avoid adding journaling comments to the code. Only add
comments when the code is complicated enough to
warrant it. Comments should be succinct and use
judiciously when the code is very hard to understand
at a glance.