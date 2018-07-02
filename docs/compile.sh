# Compile API documentation
sophon build

# Create LaTeX files
pandoc -o tex/tut_getting_started.tex tutorials/tut_getting_started.md
pandoc -o tex/tut_two_armed_bandit.tex tutorials/tut_two_armed_bandit.md

pandoc -o tex/doc_environments.tex api/doc_environments.md
pandoc -o tex/doc_agents.tex api/doc_agents.md
pandoc -o tex/doc_data.tex api/doc_data.md
pandoc -o tex/doc_inference.tex api/doc_inference.md
pandoc -o tex/doc_criticism.tex api/doc_criticism.md
pandoc -o tex/doc_metrics.tex api/doc_metrics.md
pandoc -o tex/doc_utils.tex api/doc_utils.md

# Compile LaTeX file
cd tex
sh makefile
cd ..
