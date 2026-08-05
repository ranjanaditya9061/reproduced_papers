import numpy
# shots_v2\cfae9a1d\d2ad667e\counts.npz
b = numpy.load("shots_v2\\f9776f6f\\d2ad667e\\counts.npz")
print(b.files)

indptr = b['indptr']
counts = b['counts']
keys = b['keys']

print(indptr)
print(counts)
print(keys.shape)
print(keys)