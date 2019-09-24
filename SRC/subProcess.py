# fungsi merubah string menjadi bit
def wordToBit(words):
    result = []
    for c in words:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

# fungsi merubah bit menjadi string
def bitToWord(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    flag = ''.join(chars)
    return flag

# fungsi merubah int menjadi bit
def intToBit(val):
    result = list(format(val, "b"))
    result = list(map(int, result))
    return result

# fungsi merubah bit menjadi int
def bitToInt(bit):
    val = ''.join(str(e) for e in bit)
    result = int(val, 2)
    return result