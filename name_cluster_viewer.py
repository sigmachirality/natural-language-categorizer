import json


file = open("ML Output Data/id_cluster.json", "r")
clusters = json.load(file)
for c in clusters:
	[print(i) for i in c[:10]] 
	print("**********************")

file.close()

#print(a)
