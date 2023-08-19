from arepl_dump import dump
import re
import numpy as np

e = eval("7>4")


pattern = re.compile(r"^\(.*-.*\).*/.*$", re.IGNORECASE)
m = pattern.match("(123 - 456)    /    456")

pattern = re.compile(r"[,%\$]", re.IGNORECASE)
s = "$1,2456-$1,235"
sub = re.sub(r"[,%\$]", "", s)

a = np.array([20,10,3,5,6,12,7,8,9,10])

b = (a < 4).nonzero()[0]




