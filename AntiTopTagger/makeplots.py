import os 
import sys 
import argparse

usage = " generalised plotting macro"
parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-n","--numberoffiles", dest="inputfile", type=int) ## to double check how many files are there, and also to decide weather to read histo from same file or different files
parser.add_argument("-i", "--inputfile",  dest="inputfile",default="") ## provide list in "1.root 2.root" else provide the .txt file with all the files
parser.add_argument("-merge","--mergefiles",dest="mergefiles",default="") ## if some files need to be merged you can provide same id for those files, should provide like, 1 2 3 4 4 5, it will merge two files with process id 4 


parser.add_argument("-var", "--variable", dest="variable", default="") ## variable which you want to see save as pdf 

parser.add_argument("-leg", "--legend", dest="legend", default="") ## legend, would be in the same order as the file names or histogram names 

parser.add_argument("-axisT", "--axistitle", dest="axistitle", default="") ## "xAxis yAxis"

parser.add_argument("-exp", "--experiment", dest="experiment", default="CMS") ## add experiment name or some other text
''' add one more argument later to set the position of the legend ''' 

parser.add_argument("-plotType", "--plotType", dest="plotType", default="Preliminary")

parser.add_argument("-makeratio", "--makeratio", action="store_true", dest="makeratio")

parser.add_argument("-logY", "--saveAlsoLogY", action="store_true", dest="saveAlsoLogY")
parser.add_argument("-logX", "--saveAlsoLogX", action="store_true", dest="saveAlsoLogX")
parser.add_argument("-logXY", "--saveAlsoLogXY", action="store_true", dest="saveAlsoLogXY")

parser.add_argument("-saveType", "--saveType", dest="saveType", default=".pdf") ## you can save in more than one type, just provide them in double quotes seprated by single space 

#parser.add_argument("", "--", dest="", default="")



from plotutils import plotutils
pu_ = plotutils("input","output")






