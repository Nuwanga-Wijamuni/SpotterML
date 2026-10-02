#!/bin/zsh
# Double-click this file in Finder to use the project's own Python environment.
cd "${0:A:h}" || exit 1
if [[ ! -x ".venv/bin/python" ]]; then
  print "The project Python environment is missing. Follow the setup steps in README.md."
  read "?Press Return to close."
  exit 2
fi
.venv/bin/python run_all.py
spotter_exit_code=$?
if (( spotter_exit_code != 0 )); then
  print "The run stopped. Read the message above and README.md."
fi
read "?Press Return to close this window. Your browser results page stays open."
exit $spotter_exit_code
