# THIS SCRIPT INSTALLS FITR (ON UNIX-BASED SYSTEMS, AT LEAST)
#	      THEN RUNS UNIT TESTS
#	      THEN COMPILES THE DOCUMENTATION
pip3 install --ignore-installed .
python -m pytest tests/
cd docs
sh compile.sh
cd ..
