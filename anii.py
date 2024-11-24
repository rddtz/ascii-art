import time
import sys
from os import system

def read_file(nome: str) -> str:

    try:
        file = open(nome, "r")
    except:
        print("\033[1;91mErro:\033[0m Nao foi possivel abrir o arquivo.")
        exit(-2)
    x = file.read()
    file.close()

    return x

#-------

file = read_file(sys.argv[1]).split("#$\n")

while True:
	for frame in file:
		system("clear")
		print()
		print(frame)
		time.sleep(0.3)
		print()
