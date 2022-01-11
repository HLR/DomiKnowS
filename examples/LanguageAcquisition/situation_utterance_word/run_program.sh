VENV=./
PYSCRIPT=$1
#test.py

# This will print all the arguments sent to the python script
echo "Python file and all its arguments: $@"
 
source $VENV/bin/activate
python3 $@
#$PYSCRIPT $2 $3
 
# when finished, deactivate and exit
deactivate
