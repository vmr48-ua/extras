def divisores(n):
    v = []
    i = 2
    while n != 1:
        if esprimo(i) and n%i == 0:
            v.append(i)
            n /= i
        else:
            i += 1
    return v

def esprimo(n):
    ans = False
    for i in range (n-1,1,-1):
        if n%i == 0:
            ans = False
        else:
            ans = True
    if n == 2:
        ans = True
    return ans

def esta(v,i):
    ans = False
    for j in range(len(v)):
        if v[j] == i:
            ans = True
    return ans

def repeticiones(a):
    v = []
    w = []
    for i in range(len(a)):
        if not(esta(v,a[i])):
            v.append(a[i])
    for i in range(len(v)):
        c = 0
        for j in range(len(a)):
            if a[j] == v[i]:
                c += 1
        w.append(c)
    return v,w

def imprimir(v,w):
    for i in range (len(v)-1):
        print('{}^{}'.format(v[i],w[i]), end=' ')
    print('{}^{}'.format(v[len(v)-1],w[len(v)-1]))
    
num = int(input('Introduce el número a descomponer: '))
a = repeticiones(divisores(num))
imprimir(a[0],a[1])