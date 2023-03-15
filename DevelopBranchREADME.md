# Develop Branch Git README

**Develop** Branch integrates three branches: **develop**, **Doc** and **Examples**. The **develop** branch is the main branch for development. Two other are _git submodules_ of it. The **Doc** branch is the branch for documentation of the project. The **Examples** branch is the branch for with examples of project library usage.

In order to correctly clone or update the repository, you must use the following command.
First either clone the repository with the following command:

    git clone --branch develop https://github.com/HLR/DomiKnowS

or update the repository with the following command:

    git pull --rebase origin develop

## Initial setup of git submodules

It is **necessary to recreate locally the submodules of the repository**. 

First remove exiting submodules with the following commands:

    git submodule deinit -f Docs
    git rm -r --cached Docs
    rd /S /Q Docs

    git submodule deinit -f Examples
    git rm -r --cached Example
    rd /S /Q Example

Then, clone the submodules with the following commands:

    git submodule  add -b Doc https://github.com/HLR/DomiKnowS Docs
    git submodule  add -b Examples https://github.com/HLR/DomiKnowS Example

You can check the status of the repository with the following command:

    git status

You can also check branch of develop branch with the following command:

    git branch
It should be **develop**.

You can also check branch of Doc folder with the following command:

    cd Docs
    git branch 
It should be **Doc**.

You can also check branch of Example folder with the following command:

    cd Example
    git branch
It should be **Examples**.

## Updating the git repository with your changes

When you makes change to the code in the Example or Docs folder. You have to first commit changes to them. Then, you have to commit changes to the develop branch. To do this, you have to use the following commands:

If you want to commit changes to the Doc branch:

    cd Docs
    git add .
    git commit -m "Commit message"
    git push

or  if you want to commit changes to the Examples branch:

    cd Example
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
