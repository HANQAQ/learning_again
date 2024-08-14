import os
os.add_dll_directory(os.environ['minGwPath'])


from build import pybindDemo

print(pybindDemo.add(1, 2))