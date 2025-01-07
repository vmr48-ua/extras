import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate

def ajustar_cifras_significativas(num, err):
    """
    Ajusta un número y su error a cifras significativas según las reglas especificadas:
    1. El error se ajusta a 1 cifra significativa (2 si empieza por 1)
    2. Redondeo especial: 1-4 hacia abajo, 5-9 hacia arriba
    3. El número se ajusta a la precisión del error

    Args:
        num (float): El número a ajustar
        err (float): El error asociado

    Returns:
        str: String en formato LaTeX con el número y error ajustados
    """
    import math

    def primera_cifra(x):
        if x == 0:
            return 0
        x = abs(x)
        exp = math.floor(math.log10(x))
        x = x / (10 ** exp)
        return int(x)

    def redondeo_especial(x, decimales):
        if x == 0:
            return 0
        factor = 10 ** decimales
        return math.floor(x * factor + 0.5) / factor

    # Paso 1: Ajustar el error
    orden_err = math.floor(math.log10(abs(err)))
    primera_cifra_err = primera_cifra(err)
    
    # El error se ajusta a una cifra significativa (dos si empieza por 1)
    cifras_err = 2 if primera_cifra_err == 1 else 1
    precision_err = -orden_err + (cifras_err - 1)
    err_ajustado = redondeo_especial(err, precision_err)
    
    # Paso 2: Determinar la precisión para el número basándose en el error ajustado
    orden_err_ajustado = math.floor(math.log10(abs(err_ajustado)))
    orden_err_ajustado -= 1 if orden_err_ajustado < 0 and cifras_err == 2 else 0
    precision_num = -orden_err_ajustado
    num_ajustado = redondeo_especial(num, precision_num)
    
    # Formatear resultado
    decimales = max(-orden_err_ajustado, 0)
    formato = f'{{:.{decimales}f}}'
    
    return f'{formato.format(num_ajustado)} ± {formato.format(err_ajustado)}'

def calculate_errors(n1, n2, n3, h):
    """
    Calculate measurement errors according to the specified formulas:
    Δy = sqrt(Δy_media² + Δy_altura² + Δy_contador²)
    """
    # Calculate mean
    n_mean = (n1 + n2 + n3) / 3
    
    # Calculate statistical error (Δy_media)
    N = len([n1, n2, n3])  # number of measurements
    diff_squared = (n1 - n_mean)**2 + (n2 - n_mean)**2 + (n3 - n_mean)**2
    sigma = np.sqrt(diff_squared / (N - 1))
    delta_y_media = sigma / np.sqrt(N)
    
    # Calculate height error (Δy_altura)
    # Using finite difference approximation for dn/dh
    delta_h = 1  # given in the example
    delta_y_altura = np.zeros_like(h)
    for i in range(len(h)-1):
        delta_y_altura[i] = abs(n_mean[i+1] - n_mean[i]) * delta_h
    delta_y_altura[-1] = delta_y_altura[-2]  # Use last valid value for the endpoint
    
    # Counter error (Δy_contador)
    delta_y_contador = 5  # given constant
    
    # Calculate total error
    total_error = np.sqrt(delta_y_media**2 + delta_y_altura**2 + delta_y_contador**2)
    
    return total_error, delta_y_media, delta_y_altura, delta_y_contador


def derivada(y, x=None, ventana=5):
    """
    Calcula la derivada de un array y respecto a x usando ajuste por mínimos cuadrados.
    
    Parametros:
    - y: array de valores (e.g., n)
    - x: array de valores independientes (e.g., altura). Si es None, se asume x = np.arange(len(y)).
    - ventana: número de puntos en la ventana deslizante (debe ser impar).
    
    Retorna:
    - derivada: array con la derivada calculada para cada punto de y.
    """
    if x is None:
        x = np.arange(len(y))  # Asume x equidistante si no se proporciona
    
    n = len(y)
    mitad = ventana // 2
    derivada = np.zeros(n)
    
    # Recorre cada punto del array
    for i in range(mitad, n - mitad):
        x_ventana = x[i - mitad : i + mitad + 1]  # Ventana en x
        y_ventana = y[i - mitad : i + mitad + 1]  # Ventana en y
        
        # Ajuste por mínimos cuadrados para una recta: y = ax + b
        A = np.vstack([x_ventana, np.ones(len(x_ventana))]).T
        a, b = np.linalg.lstsq(A, y_ventana, rcond=None)[0]  # a es la pendiente
        
        derivada[i] = a  # La pendiente es la derivada en el centro de la ventana
    
    # Opcional: manejar bordes donde no se puede calcular la ventana completa
    derivada[:mitad] = derivada[mitad]  # Rellena los bordes iniciales
    derivada[-mitad:] = derivada[-mitad-1]  # Rellena los bordes finales
    
    return derivada

def simpson38(y, x=None, ventana=7):
    """
    Calcula la integral de un array y respecto a x usando el método de Simpson 3/8 con ventana móvil.
    
    Parámetros:
    - y: array de valores a integrar
    - x: array de valores independientes. Si es None, se asume x = np.arange(len(y))
    - ventana: número de puntos en la ventana deslizante (debe ser múltiplo de 3 más 1)
    
    Retorna:
    - integral: array con la integral acumulada calculada para cada punto de y
    """
    if x is None:
        x = np.arange(len(y))
    
    if (ventana - 1) % 3 != 0:
        raise ValueError("La ventana debe ser múltiplo de 3 más 1 (e.g., 4, 7, 10, ...)")
    
    n = len(y)
    mitad = ventana // 2
    integral = np.zeros(n)
    
    # Primer valor es 0
    integral[0] = 0
    
    # Recorre cada punto del array
    for i in range(3, n):
        # Si hay suficientes puntos para la ventana completa
        if i >= ventana - 1:
            start_idx = i - (ventana - 1)
            x_ventana = x[start_idx:i+1]
            y_ventana = y[start_idx:i+1]
        else:
            # Para los primeros puntos donde no hay suficientes datos previos
            x_ventana = x[:i+1]
            y_ventana = y[:i+1]
        
        # Aplicar regla de Simpson 3/8 en la ventana
        h = (x_ventana[-1] - x_ventana[0]) / (len(x_ventana) - 1)
        
        # Coeficientes para Simpson 3/8
        coef = np.ones(len(y_ventana))
        coef[1:-1:3] = 3  # Multiplica por 3 cada tercer punto
        coef[2:-1:3] = 3  # Multiplica por 3 cada tercer punto
        coef[3:-1:3] = 2  # Multiplica por 2 cada tercer punto
        
        # Calcula la integral en este segmento
        segmento_integral = (3 * h / 8) * np.sum(coef * y_ventana)
        
        # Almacena el resultado
        integral[i] = segmento_integral
    
    # Interpola los primeros puntos donde no se pudo calcular
    integral[1:3] = np.linspace(integral[0], integral[3], 4)[1:3]
    
    return integral

def encontrar_cruce(altura, derivada, objetivo=0.5):
    # Crear función de interpolación
    f = interpolate.interp1d(derivada, altura, kind='linear')
    
    # Encontrar el punto donde cruza 0.5
    try:
        return float(f(objetivo))
    except ValueError:
        return None

def normalizar(arr):
    """
    Normaliza un array para que su mínimo sea 0 y su máximo sea 1.
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


n1 = np.array([    54, 69, 128, 160, 184, 211, 263, 305, 357, 412, 521, 523, 592, 614, 652, 704, 736, 754, 754, 795, 809, 778, 709, 714, 717, 690])[::-1]
n2 = np.array([    35, 40,  81, 114, 139, 165, 230, 318, 297, 334, 418, 501, 525, 602, 624, 650, 720, 728, 722, 754, 804, 790, 772, 746, 749, 710])[::-1]
n3 = np.array([    28, 50,  70,  85,  91, 176, 208, 235, 304, 350, 391, 465, 488, 561, 568, 646, 709, 746, 742, 758, 774, 782, 737, 695, 709, 721])[::-1]
N7 = ((n1+n2+n3)/3)
h7 = np.array([0,  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,   25])

total_error, delta_y_media, delta_y_altura, delta_y_contador = calculate_errors(n1, n2, n3, h7)

# Sinusoidal, 10Hz, 20s, 0.8A
h8 = np.array([0,  1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,   17,  18,  19,  20,  21,  22, 23, 24, 25]) #mm
n8 = np.array([42, 70, 106, 122, 135, 178, 187, 242, 279, 312, 327, 414, 439, 446, 472, 427, 425, 423, 344, 288, 288, 247, 153, 97, 15, 0])[::-1]
# Sinusoidal, 40Hz, 20s, 0.8A
h9 = np.array([ 0,  1,  2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25]) #mm
n9 = np.array([54, 92, 115, 175, 194, 226, 254, 274, 322, 331, 354, 391, 400, 407, 430, 455, 460, 488, 503, 502, 470, 446, 434, 450, 426, 458])[::-1]
# Sinusoidal, 40Hz, 20s, 0.6A
h10 = np.array([ 0, 1,  2,  3,  4,   5,  6,  7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,   25]) #mm
n10 = np.array([      5, 24, 32, 35, 54, 74, 96, 144, 144, 207, 235, 279, 293, 328, 349, 352, 401, 408, 401, 417, 396, 398, 388, 405, 403, 409])[::-1]


def f(x,A,mu,kT):
    return A*(np.e**((x-mu)/kT))/(kT*(np.e**((x-mu)/kT)+1)**2)


# Cuadrada, 10Hz, 20s, 0.8A
cuadrada = np.array([    34, 59, 75, 104, 125, 163, 185, 209, 249, 279, 323, 349, 418, 433, 465, 483, 498, 482, 479, 449, 381, 352, 336, 281])[::-1]
ht = np.array([5,  6,  7,  8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, 19,])
hc = h10[2:]
triangular = np.array([     2, 10, 36, 60, 117, 157, 186, 231, 256, 262, 223, 193, 178, 148, 89])[::-1]
sinusoidal = np.array([    42, 70, 106, 122, 135, 178, 187, 242, 279, 312, 327, 414, 439, 446, 472, 467, 445, 423, 344, 288, 288, 247, 153, 97, 15, 0])[::-1]

###########################################################################################################
plt.figure(figsize=(12,8), tight_layout=True)
plt.title('Sinusoidal, 10Hz, 30s, 1.0A')
h = np.linspace(0,h8[-1],100)

popt, pcov = curve_fit(f, h7, N7)
perr = np.sqrt(np.diag(pcov))
plt.plot(h,f(h,*curve_fit(f,h7,N7)[0]), label=f'$c={ajustar_cifras_significativas(popt[0],perr[0])}$\n' + 
                                              f'$\\mu={ajustar_cifras_significativas(popt[1],perr[1])}$\n' +
                                              f'$kT={ajustar_cifras_significativas(popt[2],perr[2])}$', c='palevioletred')

# Sinusoidal, 10Hz, 30s, 1.0A
plt.errorbar(h7, N7, yerr=total_error, fmt='x', label='Sinusoidal, 10Hz, 30s, 1.0A', capsize=5, c='palevioletred')
plt.scatter(h7,n1, label='Mediciones', alpha=0.8, marker='x', c='mediumturquoise')
plt.scatter(h7,n2, alpha=0.8, marker='x', c='mediumturquoise')
plt.scatter(h7,n3, alpha=0.8, marker='x', c='mediumturquoise')
plt.xlabel('Altura (mm)')
plt.ylabel('Número de esferas')
plt.legend()
plt.grid(True, alpha=0.5)
##############################################################################################################

##############################################################################################################
plt.figure(figsize=(12,8), tight_layout=True)
plt.title('Variación de Frecuencia')

total_error, delta_y_media, delta_y_altura, delta_y_contador = calculate_errors(n8, n8, n8, h8)
popt, pcov = curve_fit(f, h8, n8)
perr = np.sqrt(np.diag(pcov))
plt.errorbar(h8, n8, yerr=total_error, fmt='x', label='Sinusoidal, 10Hz, 20s, 0.8A', capsize=5, c='slateblue')
plt.plot(h,f(h,*curve_fit(f,h8,n8)[0]), label=f'$c={ajustar_cifras_significativas(popt[0],perr[0])}$\n' + 
                                              f'$\\mu={ajustar_cifras_significativas(popt[1],perr[1])}$\n' +
                                              f'$kT={ajustar_cifras_significativas(popt[2],perr[2])}$',c='slateblue')

total_error, delta_y_media, delta_y_altura, delta_y_contador = calculate_errors(n9, n9, n9, h9)
popt, pcov = curve_fit(f, h9, n9)
perr = np.sqrt(np.diag(pcov))
plt.errorbar(h9, n9, yerr=total_error, fmt='x', label='Sinusoidal, 40Hz, 20s, 0.8A', capsize=5, c='mediumseagreen')
plt.plot(h,f(h,*curve_fit(f,h9,n9)[0]), label=f'$c={ajustar_cifras_significativas(popt[0],perr[0])}$\n' + 
                                              f'$\\mu={ajustar_cifras_significativas(popt[1],perr[1])}$\n' +
                                              f'$kT={ajustar_cifras_significativas(popt[2],perr[2])}$', c='mediumseagreen')
plt.xlabel('Altura (mm)')
plt.ylabel('Número de esferas')
plt.legend()
plt.grid(True, alpha=0.5)
##############################################################################################################

##############################################################################################################
plt.figure(figsize=(12,8), tight_layout=True)
plt.title('Variación de Amplitud')

total_error, delta_y_media, delta_y_altura, delta_y_contador = calculate_errors(n10, n10, n10, h10)
popt, pcov = curve_fit(f, h10, n10)
perr = np.sqrt(np.diag(pcov))
plt.errorbar(h10, n10, yerr=total_error, fmt='x', label='Sinusoidal, 40Hz, 20s, 0.6A', capsize=5, c='orchid')
plt.plot(h,f(h,*curve_fit(f,h10,n10)[0]), label=f'$c={ajustar_cifras_significativas(popt[0],perr[0])}$\n' + 
                                              f'$\\mu={ajustar_cifras_significativas(popt[1],perr[1])}$\n' +
                                              f'$kT={ajustar_cifras_significativas(popt[2],perr[2])}$', c='orchid')

total_error, delta_y_media, delta_y_altura, delta_y_contador = calculate_errors(n9, n9, n9, h9)
popt, pcov = curve_fit(f, h9, n9)
perr = np.sqrt(np.diag(pcov))
plt.errorbar(h9, n9, yerr=total_error, fmt='x', label='Sinusoidal, 40Hz, 20s, 0.8A', capsize=5, c='mediumseagreen')
plt.plot(h,f(h,*curve_fit(f,h9,n9)[0]), label=f'$c={ajustar_cifras_significativas(popt[0],perr[0])}$\n' + 
                                              f'$\\mu={ajustar_cifras_significativas(popt[1],perr[1])}$\n' +
                                              f'$kT={ajustar_cifras_significativas(popt[2],perr[2])}$', c='mediumseagreen')
plt.xlabel('Altura (mm)')
plt.ylabel('Número de esferas')
plt.legend()
plt.grid(True, alpha=0.5)
##############################################################################################################

##############################################################################################################
plt.figure(figsize=(12,8), tight_layout=True)
plt.title('Extra: Variación de la función de vibración')

total_error, delta_y_media, delta_y_altura, delta_y_contador = calculate_errors(sinusoidal, sinusoidal, sinusoidal, h10)
popt, pcov = curve_fit(f, h10, sinusoidal)
perr = np.sqrt(np.diag(pcov))
plt.errorbar(h10, sinusoidal, yerr=total_error, fmt='x', label='Sinusoidal, 10Hz, 20s, 0.8A', capsize=5, c='slateblue')
plt.plot(h,f(h,*curve_fit(f,h10,sinusoidal)[0]), label=f'$c={ajustar_cifras_significativas(popt[0],perr[0])}$\n' + 
                                              f'$\\mu={ajustar_cifras_significativas(popt[1],perr[1])}$\n' +
                                              f'$kT={ajustar_cifras_significativas(popt[2],perr[2])}$', c='slateblue')

total_error, delta_y_media, delta_y_altura, delta_y_contador = calculate_errors(cuadrada, cuadrada, cuadrada, hc)
popt, pcov = curve_fit(f, hc, cuadrada)
perr = np.sqrt(np.diag(pcov))
plt.errorbar(hc, cuadrada, yerr=total_error, fmt='x', label='Cuadrada, 10Hz, 20s, 0.8A', capsize=5, c='#827A04')
plt.plot(h,f(h,*curve_fit(f,hc,cuadrada)[0]), label=f'$c={ajustar_cifras_significativas(popt[0],perr[0])}$\n' + 
                                              f'$\\mu={ajustar_cifras_significativas(popt[1],perr[1])}$\n' +
                                              f'$kT={ajustar_cifras_significativas(popt[2],perr[2])}$', c='#827A04')

total_error, delta_y_media, delta_y_altura, delta_y_contador = calculate_errors(triangular, triangular, triangular, ht)
popt, pcov = curve_fit(f, ht, triangular)
perr = np.sqrt(np.diag(pcov))
plt.errorbar(ht, triangular, yerr=total_error, fmt='x', label='Triangular, 10Hz, 20s, 0.8A', capsize=5, c='#802f2d')
plt.plot(h,f(h,*curve_fit(f,ht,triangular)[0]), label=f'$c={ajustar_cifras_significativas(popt[0],perr[0])}$\n' + 
                                              f'$\\mu={ajustar_cifras_significativas(popt[1],perr[1])}$\n' +
                                              f'$kT={ajustar_cifras_significativas(popt[2],perr[2])}$', c='#802f2d')

plt.xlabel('Altura (mm)')
plt.ylabel('Número de esferas')
plt.legend()
plt.grid(True, alpha=0.5)
##############################################################################################################

fig, ax1 = plt.subplots(figsize=(12,8), tight_layout=True)
# Cuadrada, 10Hz, 20s, 0.8A
altura = np.array([0,  1,  2,  3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24, 25]) #mm
n = np.array([    34, 59, 75, 104, 125, 163, 185, 209, 249, 279, 323, 349, 418, 433, 465, 483, 498, 482, 479, 449, 381, 352, 336, 281, 173, 10])[::-1]

n = n[2:]
der = derivada(n, ventana=7)
#ax2.plot(altura[2:],normalizar(der), c='#827A04')
ax1.scatter(altura[2:],n,marker='x', label='cuadrada', c='#827A04')
der_cuadrada = normalizar(der)
altura_cuadrada = altura[2:]

# Triangular, 10Hz, 20s, 0.8A
# altura = np.array([0, 1, 2, 3, 4, 5,  6,  7,  8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, 19,  20,  21,  22,  23, 24, 25])*1e-3 #mm
# n = np.array([     0, 0, 0, 0, 0, 2, 10, 36, 60, 117, 157, 186, 231, 256, 262, 223, 193, 238, 168, 89, 105, 142, 129, 104, 81, 121])[::-1]
altura = np.array([5,  6,  7,  8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, 19,]) #mm
n = np.array([     2, 10, 36, 60, 117, 157, 186, 231, 256, 262, 223, 193, 178, 148, 89,])[::-1]
ax1.scatter(altura[len(altura)-len(n):],n,marker='x', c='#802f2d', label='triangular')
der = derivada(n, ventana=7)
#ax2.plot(altura[len(altura)-len(n):],normalizar(der), c='#802f2d')
der_triangular = normalizar(der)
altura_triangular = altura[len(altura)-len(n):]

# Sinusoidal, 10Hz, 20s, 0.8A
altura = np.array([0,  1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,   17,  18,  19,  20,  21,  22, 23, 24, 25]) #mm
n4 = np.array([    42, 70, 106, 122, 135, 178, 187, 242, 279, 312, 327, 414, 439, 446, 472, 467, 445, 423, 344, 288, 288, 247, 153, 97, 15, 0])[::-1]
ax1.scatter(altura[0:len(n4)],n4,marker='x', label='sinusoidal', c='slateblue')
der = derivada(n4, ventana=7)
#ax2.plot(altura[0:len(n4)],normalizar(der), c='slateblue')
der_sinusoidal = normalizar(der)
altura_sinusoidal = altura[:len(n4)]

cruce_cuadrada = encontrar_cruce(altura_cuadrada, der_cuadrada)
cruce_triangular = encontrar_cruce(altura_triangular, der_triangular)
cruce_sinusoidal = encontrar_cruce(altura_sinusoidal, der_sinusoidal)

ax1.axvline(cruce_sinusoidal, color='slateblue', linestyle='--', label=f'Sinusoidal $\mu$ = {cruce_sinusoidal:.2f}')
ax1.axvline(cruce_cuadrada,   color='#827A04',   linestyle='--', label=f'Cuadrada $\mu$ = {cruce_cuadrada:.2f}')
ax1.axvline(cruce_triangular, color='#802f2d',   linestyle='--', label=f'Triangular $\mu$ = {cruce_triangular:.2f}')

# Etiquetas y leyendas
ax1.set_xlabel('Altura (mm)')
ax1.set_ylabel('Amplitud')
ax1.legend(loc='upper right')
plt.title('Potencial químico asociado a la Distribución de Fermi-Dirac\n para varias formas de vibración')
plt.show()

plt.show()