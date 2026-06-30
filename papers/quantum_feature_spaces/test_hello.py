def primes(n):
    """Return the first n prime numbers."""
    seq = []
    candidate = 2
    while len(seq) < n:
        if all(candidate % p for p in seq if p * p <= candidate):
            seq.append(candidate)
        candidate += 1
    return seq


if __name__ == "__main__":
    print(primes(10))