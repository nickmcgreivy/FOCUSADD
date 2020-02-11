

def readAxis(filename):
	with open(filename, "r") as file:
		# NF, NP header
		file.readline()
		NF, NP = file.readline().split(" ")
		NF = int(NF)
		NP = int(NP)
		print(NF)
		print(NP)

			

def main():
	filename = "../initFiles/axes/defaultAxis.txt"
	readAxis(filename)

if __name__ == "__main__":
	main()
