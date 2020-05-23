## Get Started

Python 3 is required currently while we did considered (very limited) support for Python 2.

### Solver

[Gurobi](http://www.gurobi.com) is the leading commercial optimization solver.
We implement our inference interface base on Gurobi.
However, a valid licence will be required.
We also implement base on open source optimization solver [Gekko](https://gekko.readthedocs.io).
Install one of them and setup [`regr/solver/ilpConfig.py`](regr/solver/ilpConfig.py) accordingly.
You can also switch the solver by exporting a macro `REGR_SOLVER=Gurobi` or `REGR_SOLVER=GEKKO`.

#### Gurobi

[Gurobi](http://www.gurobi.com) is the leading commercial optimization solver.
Follow installation instruction from official website.
A valid licence will be required.
Make sure you have install the Python interface of Gurobi.
```bash
cd $GUROBI_HOME
sudo python3 setup.py install
```

#### Gekko

[Gekko](https://gekko.readthedocs.io) is an open source optimization solver.
You can install directly from `pip` source.
```bash
sudo python3 -m pip install gekko
```
You may also setup [`regr/solver/ilpConfig.py`](regr/solver/ilpConfig.py#L6) to make it default.

### Other dependency

Install all python dependency specified in `requirements.txt` by
```bash
sudo python3 -m pip install -r requirements.txt
```
Make sure to check out if there is any **additional prerequirements** or setup steps in specific `README`, if you want to run an example.

Last but not least, add `regr` to you `PYTHONPATH` environment variables,
```bash
export PYTHONPATH=$PYTHONPATH:$(cd regr && pwd)
```
and you are ready to go!

## Examples

We started some [examples](examples) to see if the design really work.

* [Entity Mentioned Relation](examples/emr)