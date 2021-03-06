import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy
import sys
from model import model
from data import chars, dataX, n_vocab


print("Enter path to saved weights:")
filename = input()

# load the network weights
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# create reverse mapping of int to char
int_to_char = dict((i, c) for i, c in enumerate(chars))

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# create the report
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
