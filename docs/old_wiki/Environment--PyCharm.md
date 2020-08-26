To work with PyCharm, you just need to open the root of this repository in PyCharm, with some simple configurations.

## Open

In the "Welcome" page or "File" menu, choose "Open...". Select the root directory of this repository. Click "OK" and PyCharm will try to load the project.

## Virtual Environment

It would be better to use a virtual environment (for python) with this project to avoid any side-effect when you install some specific version of dependencies.

If there is any popup notification, use them. Or go to the menu "File -> Settings...", select "Project: RelationalGraph -> Project Interpreter".
In the "Project Interpreter" select, if there is no "Python 3.6 (RelationalGraph)" (or 3.7), click the gear on the right and choose "Add...".

You have two options, to go with Virtualenv or Anaconda.

### Virtualenv

In the new window, select "Virtualenv Environment", then "New environment" option. Leave the "Location" to be something looks like `/path/to/RelationalGraph/venv`, and choose for "Base interpreter" a python 3.6 or a 3.7 instance. Leave the "Inherit global site-packages" unchecked. "Make available to all projects" depends on your own needs. Then click "OK" to activate this virtual environment.

### Anaconda

In the new window, select "Conda Environment", then "New environment" option. Leave the default "Location". Choose python version to be 3.6 or 3.7. Conda executable should be found automatically. Otherwise please specify, or check your anaconda installation. "Make available to all projects" depends on your own needs. Then click "OK" to activate this virtual environment.

## Path

To allow PyCharm to interpret the source correctly, you need to set up for source folders.
The project root is the source root by default. You need to add the specific example folder as a source root.
The easiest way is to right-click on the particular example folder (**NOT** `examples/`), in the context menu go to "Mark Directory as", and choose "Sources Root".
Otherwise, you can do this by managing the project structure. Go to the menu "File -> Settings...", select "Project: RelationalGraph -> Project Structure".
Manage the source in the tree structure. Then click "OK" to apply the change.

## Dependencies

Install the dependencies **in the terminal of this project** according to the instruction of "README.md" of the repository and each example (if you want to work with them).
In case you are using a virtual environment as advised above, you should see the promote start with `(venv) ...` (Virtualenv) or `(RelationalGraph)...` (Conda). Then your installations go all into the virtual environment.

You may need to install `gurobipy` in this shell again.
With Virtualenv, `cd` to the path of your Gurobi installation, and use `setup.py install` again.
Otherwise, consider using Anaconda virtual environment and install Gurobi with anaconda
```
conda install gurobi -c http://conda.anaconda.org/gurobi
```
which is easier.

Remember to get a license and activate it for Gurobi. For research, we can apply for an academic license here https://www.gurobi.com/downloads/free-academic-license/

## Run

To run an example, right-click on the main script file, `sprlApp.py` for example, and choose "run ..."
Check out the `README.md` in the example directory to identify the main script, which may not be necessarily named "`main.py`".
