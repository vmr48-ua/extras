def kMinMax(lista,k):
    j = 0
    v = []
    a = lista.copy()
    for _ in range(k):
        for j in range (len(a)):
            mina = min(a) # minimo(a) si min() no se puede usar
            if a[j] == mina:   
                l = j
        v.append(a[l])
        a.pop(l)

    a = lista.copy()
    for _ in range(k):
        for j in range (len(a)):
            maxa = max(a) # maximo(a) si max() no se puede usar
            if a[j] == maxa:
                l = j
        v.append(a[l])
        a.pop(l)
    return v

w = [5,20,3,7,6,8]
a = kMinMax(w,2)
print(a)
""" 
def minimo(v):
    m = 0
    if len(v) != 0:
        m = v[0]
        for i in range(len(v)):
            if v[i] < m:
                m = v[i]
    return m

def maximo(v):
    m = 0
    if len(v) != 0:
        m = v[0]
        for i in range(len(v)):
            if v[i] > m:
                m = v[i]
    return m """