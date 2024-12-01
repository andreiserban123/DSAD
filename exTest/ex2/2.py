# ( def create_arr(1,20)). Folosind un lambda function extrageti doar nr pare.

def create_arr(start, end):
    return [i for i in range(start, end + 1)]


arr = create_arr(1, 20)

listNrPare = list(filter(lambda x: x % 2 == 0, arr))

print(listNrPare)