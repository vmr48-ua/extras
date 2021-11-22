def recorta(A,i):
    B = A.copy()
    B.pop(i)
    v = []
    for j in range(len(A[0])-1):
        v = B[j][::-1] 
        v.pop()
        B[j] = v[::-1]
    return B

def determinante(A):   
    if(len(A)==1):
        return A[0][0]     
    return (sum(A[i][0]*((-1)**i*determinante(recorta(A,i))) for i in range(len(A))))

A = [[1,0,0],[0,1,0],[0,0,1]]
print(determinante(A))