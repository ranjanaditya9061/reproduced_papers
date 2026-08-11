import numpy
# shots_v2\cfae9a1d\d2ad667e\counts.npz
b = numpy.load("datasets\\b69147a7\\exact\\dist.npz")
print(b.files)

probs = b['probs']
probs_0 = b['probs_at_zero']
keys = b['keys']

print(probs.shape)
print(probs[0])
print(probs_0.shape)
print(probs_0)
print(keys.shape)
print(keys)