# Entity Mention Relation

This is an example of the Entity Mention Relation (EMR) problem.

## Problem description

...

[//]: # (description of the problem to be added here)


## Run the example

### Prepare

#### Basic tools

The following tools are needed to set up the EMR example.
Most of them are pre-installed.
If you don't have in some case, install them with default package management tools.
For example, on Debian Linux (Ubuntu) you use `sudo apt install <package_name>`.
On MacOS you need [`homebrew`](https://brew.sh) and use `sudo brew install <package_name>`.

* `wget`: to download the dataset.
* `unzip`: to extract data from a zipped file.

#### Tools

Other tools that are unlikely to be pre-installed. May need various commands to have them on your system.

* Python 3: Python 3 is required by `allennlp`. Install by package management. Remember to mention the version in installation.
* `pip`: to install other required python packages. Follow the [installation instruction](https://pip.pypa.io/en/stable/installing/) and make sure you install it with your Python 3.
* `pytorch`: model calculation behind `allennlp`. There is a bunch of selection other than the standard pip package.
Follow the [installation instruction](https://pytorch.org/get-started/locally/) and select the correct CUDA version if any available to you.
* Anything else will be installed properly with `pip` (including `allennlp`). No worry here.

#### Setup

Follow instructions of the project to have `regr` ready.
Install all python dependency specified in `requirements.txt` by
```bash
sudo python3 -m pip install -r requirements.txt
```
Use `setup-data.sh` to prepare the data needed by the example.

### The example

The implement of the example is in package `emr`. The example can be run by using the following command.

```bash
python3.7 -m emr
```
Or run the `ner` example, which is a subset of the original problem, where only one non-composed concept is considered, by the following command.
```bash
python3.7 -m emr.ner
```