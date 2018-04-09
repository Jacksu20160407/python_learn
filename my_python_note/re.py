import re

keys = [r"afiouwehrfuichuxiuhong@hit.edu.cnaskdjhfiosueh"]
p1 = r"chuxiuhong@hit\.edu\.cn"
pattern1 = re.compile(p1)
print [pattern1.findall(key) for key in keys]
