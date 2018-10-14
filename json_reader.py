from json import JSONDecoder, JSONDecodeError
import json
import re, os, sys

NOT_WHITESPACE = re.compile(r'[^\s]')

def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj

all_data = []


def read_json(name):
	file = open(name)
	s = file.read()
	file.close()

	# decoded = list(decode_stacked(s))
	# print(decoded[0]["id"])
	lis = []

	for obj in decode_stacked(s):
		lis.append(json.loads(obj))

	all_data.extend(lis)


def get_all_data():


	directory = os.fsencode(sys.path[0])

	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.endswith(".json"): 
			read_json(file)

	return all_data

#print(get_all_data()[0])