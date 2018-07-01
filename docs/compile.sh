# Compile API documentation
sophon build

# Create LaTeX files
pandoc -o tex/doc_agents.tex api/doc_agents.md
pandoc -o tex/doc_data.tex api/doc_data.md
pandoc -o tex/doc_utils.tex api/doc_utils.md
pandoc -o tex/doc_environments.tex api/doc_environments.md


# Compile LaTeX file
cd tex
sh makefile
cd ..
