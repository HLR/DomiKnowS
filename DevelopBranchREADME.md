# Develop Branch Git README

**Develop** Branch integrates three branches: **develop**, **Doc** and **Tasks**. The **develop** branch is the main branch for development. Two other are _git submodules_ of it. The **Doc** branch is the branch for documentation of the project. The **Tasks** branch is the branch for with examples of project library usage.

In order to correctly clone or update the repository, you must use the following command.
First either clone the repository with the following command:

    git clone --branch develop https://github.com/HLR/DomiKnowS

or update the repository with the following command:

    git pull --rebase origin develop

## Initial setup of git submodules

It is **necessary to recreate locally the submodules of the repository om Windows**. 

First remove exiting submodules with the following commands:

    git submodule deinit -f Docs
    git rm -r --cached Docs
    rd /S /Q Docs

    git submodule deinit -f Tasks
    git rm -r --cached Tasks
    rd /S /Q Tasks

Then, clone the submodules with the following commands:

    git submodule  add -b Doc https://github.com/HLR/DomiKnowS Docs
    git submodule  add -b Tasks https://github.com/HLR/DomiKnowS Tasks

On **Linux you can use the following commands**:

    git submodule update --init

You can check the status of the repository with the following command:

    git status

You can also check branch of the repository with the following command:

    git branch
It should be **develop**.

You can also check branch of the Doc subfolder with the following command:

    cd Docs
    git branch 
It should be **Doc**.

You can also check branch of the Tasks subfolder with the following command:

    cd Tasks
    git branch
It should be **Tasks**.

## Updating the git repository with your changes

When you makes change to the code in the Tasks or Docs folder. You have to first commit changes to them. Then, you have to commit changes to the develop branch. To do this, you have to use the following commands:

If you want to commit changes to the Doc branch:

    cd Docs
    git add .
    git commit -m "Commit message"
    git push

or  if you want to commit changes to the Tasks branch:

    cd Tasks
    git add .
    git commit -m "Commit message"
    git push

then always commit changes to the develop branch:

    cd ..       # go to the root of the repository
    git add .
    git commit -m "Commit message"
    git push

If you only change the develop branch, you only need to commit changes to this branch:

    git add .
    git commit -m "Commit message"
    git push

## Releasing pip package (develop and stable version)

The github repository is linked to the pip package. When you want to release a new version of the pip package, you have to follow the following steps.

    To release develop version of pip library use tag name starting with d followed by version number. For example:

        git tag d0.501
        git push origin d0.501

    the library version will be in this case 0.501.dev0

    To release stable version of pip library use tag name starting with v followed by version number. For example:
        
            git tag v0.501
            git push origin v0.501

    the library version will be in this case 0.501

To check already used tags, you can execute the following command:

    git tag
