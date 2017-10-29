from os import listdir

from yapf.yapflib.yapf_api import FormatFile

for f in listdir("."):
    if f[-3:] == ".py":
        print(FormatFile(f, in_place=True))
