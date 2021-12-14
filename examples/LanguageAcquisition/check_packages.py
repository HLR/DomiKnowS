# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 08:58:44 2021

@author: Juan
"""
import subprocess
import sys

if "../../" not in sys.path:
    sys.path.append("../../")


# get the contents of the package

# Read the list of requirements
fp = open("../../requirements.txt")
#myoutput = open("result.txt","w")
L = []
for package in fp:
    
    package = package.strip()
    og_package = package
    # remove the version numbers
    
    
    if ">" in package:
        index = package.find(">")
        package = package[:index]
    elif "<" in package:
        index = package.find("<")
        package = package[:index]
    elif "=" in package:
        index = package.find("=")
        package = package[:index]
    elif "[" in package:
        index = package.find("[")
        package = package[:index]
    
    
    # find the package in conda
    output = subprocess.run(['conda','list',package],shell=True, stdout=subprocess.PIPE)
    print("Package name:",  package)
    
    result = str(output.stdout).strip().split("\\r\\n")[-2].split()[:2]
    
    L.append((og_package, package," ".join(result)))

fp = open("result_pkg.txt","w")

print("{:25s}{:25s}".format("Original Package", "Installed"), file=fp)
for x in L:
    print("{:25s}{:25s}".format(x[0],x[2]), file=fp)

fp.close()

#myoutput.close() 