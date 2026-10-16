"""Deprecated compatibility entrypoint for the ControlSci Agent.

Use ``python -m mineru_agent.cli`` instead.
"""

import sys
from mineru_agent.cli import main as cli_main


def main():
    print(
        "[DEPRECATED] Use ``python -m mineru_agent.cli`` instead.",
        file=sys.stderr,
    )
    sys.exit(cli_main())


if __name__ == "__main__":
    main()
