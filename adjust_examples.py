#!/usr/bin/env python
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
GLPK_DIR= str(dir_path) + "/glpk"
examples = ["combi_example", "combi_example_faults", "gene_distributed", "gene_distributed_linear"]
for example in examples:
    pfilein = open(str(dir_path)+ "/distributedcombigrid/examples/" + example + "/Makefile.template" ,'r')
    temp = pfilein.read()
    pfilein.close()
    temp = temp.replace('$(SGPP)', str(dir_path))
    temp = temp.replace('$(GLPK)', GLPK_DIR)
    pfileout = open(str(dir_path)+ "/distributedcombigrid/examples/" + example + "/Makefile" ,'w')
    pfileout.write(temp)
    pfileout.close()

pfilein = open(str(dir_path)+ "/distributedcombigrid/examples/gene_distributed/preproc.py" ,'r')
temp = pfilein.read()
pfilein.close()
temp = temp.replace('$(SGPP)', str(dir_path))
pfileout = open(str(dir_path)+ "/distributedcombigrid/examples/gene_distributed/preproc.py" ,'w')
pfileout.write(temp)
pfileout.close()

pfilein = open(str(dir_path)+ "/distributedcombigrid/examples/gene_distributed_linear/preproc.py" ,'r')
temp = pfilein.read()
pfilein.close()
temp = temp.replace('$(SGPP)', str(dir_path))
pfileout = open(str(dir_path)+ "/distributedcombigrid/examples/gene_distributed_linear/preproc.py" ,'w')
pfileout.write(temp)
pfileout.close()
