import numpy as np
from numpy import sqrt, exp, pi
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

'''
MECÁNICA ESTADÍSTICA, PRÁCTICA II
MODELO DE ISING

Víctor Mira Ramírez
Mireia Serrano Beltrà

74528754Z
54379937W

vmr48@alu.ua.es
fvjm1@alu.ua.es
'''

def energia(S,E,J):
    E[:] = -J * 0.5 * S * (
            np.roll(S, shift=1, axis=0) +  # Vecino arriba
            np.roll(S, shift=-1, axis=0) +  # Vecino abajo
            np.roll(S, shift=1, axis=1) +  # Vecino izquierda
            np.roll(S, shift=-1, axis=1)  # Vecino derecha
        )
    return E

def ising(N: int, J: float = 1) -> None:
    S = np.random.choice([-1,1], p=[0.5,0.5], size=(N,N))
    E = np.zeros((N,N))
    E = energia(S,E,J)
    return S, np.sum(E), np.sum(S)

def comparar_N_T(N_values, T_values, steps=int(4e4), J=1.0, K=1.0):
    # Primera figura: evolución de energía y magnetización
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig1.suptitle('Evolución del modelo de Ising para diferentes N y T')

    colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFA500', 
             '#800080', '#008080', '#FFD700', '#4B0082']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']
    cmap = ListedColormap(['rebeccapurple', 'lemonchiffon'])
    plot_interval = steps // 100

    # Crear una figura separada para cada N
    figures = []

    for n_idx, N in enumerate(N_values):
        fig2, axes = plt.subplots(1, len(T_values), figsize=(4 * len(T_values), 4))
        if len(T_values) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        fig2.suptitle(f'Estados finales para N = {N}')
        
        for t_idx, T in enumerate(T_values):
            beta = 1 / (K * T)
            S_old, E_old, M_old = ising(N, J)
            energias = []
            magnetizaciones = []

            for _ in range(steps):
                i_r, j_r = np.random.randint(0, N, size=(2, 1))
                S_new = S_old.copy()
                S_new[i_r, j_r] *= -1
                E_new = np.sum(energia(S_new, np.zeros((N, N)), J))

                p_old = 1 / (1 + np.exp(-beta * (E_new - E_old)))
                p_new = 1 - p_old

                if np.random.choice([0, 1], p=[p_old, p_new]):
                    S_old = S_new
                    E_old = E_new
                    M_old = np.sum(S_new)

                energias.append(E_old)
                magnetizaciones.append(M_old)

            energias = np.array(energias) / (N * N)
            magnetizaciones = np.array(magnetizaciones) / (N * N)

            x = np.arange(0, steps, plot_interval)
            energias_plot = energias[::plot_interval]
            magnetizaciones_plot = magnetizaciones[::plot_interval]

            label = f'N={N}, T={T:.2f}'
            plot_kwargs = {
                'color': colors[n_idx % len(colors)],
                'marker': markers[t_idx % len(markers)],
                'label': label,
                'markevery': max(len(x) // 20, 1),
                'markersize': 6,
                'alpha': 0.7
            }

            ax1.plot(x, energias_plot, **plot_kwargs)
            ax2.plot(x, magnetizaciones_plot, **plot_kwargs)

            # Estado final
            axes[t_idx].imshow(S_old, cmap=cmap)
            axes[t_idx].set_title(f'T = {T:.2f}')
            axes[t_idx].axis('off')

        # Ocultar ejes extras si los hay
        for idx in range(len(T_values), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        figures.append(fig2)

    # Ajustes finales figura 1
    ax1.set_xlabel('Pasos de Monte Carlo')
    ax1.set_ylabel('Energía')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.set_xlabel('Pasos de Monte Carlo')
    ax2.set_ylabel('Magnetización')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.figure(fig1.number)
    plt.tight_layout()

    return fig1, figures

def M_E_equilibrio(N, J, beta, tol):
    tol = tol / 100
    S_old, E_old, M_old = ising(N, J)
    energias, magnetizaciones, espines = [E_old], [M_old], [S_old]
    converged = False
    pasos = 0
    ventana = 1500  # Tamaño de la ventana para calcular el promedio de cambios en energía
    
    while not converged:
        i_r, j_r = np.random.randint(0, N, size=(2, 1))
        S_new = S_old.copy()
        S_new[i_r, j_r] *= -1
        E_new = np.sum(energia(S_new, np.zeros((N, N)), J))

        p_old = 1 / (1 + np.exp(-beta * (E_new - E_old)))
        p_new = 1 - p_old

        if np.random.choice([0, 1], p=[p_old, p_new]):
            S_old = S_new
            E_old = E_new
            M_old = np.sum(S_new)

        energias.append(E_old)
        magnetizaciones.append(M_old)
        espines.append(S_old)

        # Verificar convergencia después de un número suficiente de pasos
        if pasos >= 10*ventana:
            # Calcular el promedio absoluto de las diferencias de energía en la ventana
            diff_prom = np.mean([abs(energias[-k] - energias[-k-1]) for k in range(1, ventana)])
            if diff_prom < tol * abs(E_old):
                print(f"Promedio de cambio en energía: {diff_prom}")
                print(f"Tolerancia ajustada: {tol * abs(E_old)}")
                print(f"Energía final: {np.mean(energias[-ventana:])}")
                print(f"Magnetización final: {np.mean(magnetizaciones[-ventana:])}")
                print(f"Pasos: {pasos}")
                print()
                converged = True
        pasos += 1

    return energias, magnetizaciones, pasos

def comparar_M_E_equilibrio(N, T_values, J=1.0, K=1.0, tol=5, ventana=1500):
    figuras, datos = [], []
    for T in T_values:
        beta = 1 / (K * T)
        energias, magnetizaciones, pasos = M_E_equilibrio(N, J, beta, tol)
        datos.append([np.mean(energias[-ventana:]), np.mean(magnetizaciones[-ventana:]), pasos])

        # Graficar resultados
        fig, ax = plt.subplots()
        ax.set_title(f'N = {N}, T = {T}, Pasos = {pasos}')
        xo = np.arange(pasos + 1)
        ax.plot(xo, energias, label='Energía', c='teal')
        ax.plot(xo, magnetizaciones, label='Magnetización', c='firebrick')

        ax.legend(
            prop={'size': 10},
            facecolor='lemonchiffon',
            edgecolor='black',
            framealpha=0.5,
            fancybox=True,
        )
        ax.xlabel = 'Pasos de Monte Carlo'
        ax.ylabel = 'Valor'
        ax.grid(alpha=0.3)
        figuras.append(fig)
    return figuras, datos

def animacion(espines, T, steps, N):   
    cmap = ListedColormap(['rebeccapurple', 'lemonchiffon'])
    fig, ax = plt.subplots(figsize=(5,5))
    plt.tight_layout()
    im = ax.imshow(espines[0], cmap=cmap, animated=True)
    title = ax.set_title(f'Modelo de Ising T={T} - Iteración 0/{steps}')
    def update(frame):
        title.set_text(f'Modelo de Ising T={T} - Iteración {frame}/{steps}')
        im.set_array(espines[frame])
        return [im, title]
    anim = animation.FuncAnimation(fig, update, frames=range(0,len(espines),750), interval=1)
    plt.xlim(0,N-1), plt.ylim(0,N-1)
    plt.xticks(np.arange(0, N+1, 10)), plt.yticks(np.arange(0, N+1, 10))
    plt.show()
    
    writervideo = animation.FFMpegWriter(fps=120) 
    anim.save('ising.mp4', writer=writervideo) 
    print('video guardado')
    anim.save('ising.gif')
    print('gif guardado')

def calor_especifico(I,k,S,M):
    T = np.linspace(0.5, 4, 40)
    dispersion = []
    Cv = []

    tau = I//2

    for temp in T:
        beta = 1 / (k * temp)
        
        #Monte Carlo
        E, _, _ = Montecarlo(beta, S, M, I)
        
        var = np.var(E[tau: ])
        dispersion.append(var)
        
        #Calor especifico
        Cv.append(var / (k * temp ** 2))

    #Dispersion en funcion de T
    plt.figure()
    plt.title('Dispersión de la energía en función de T')
    plt.plot(T, dispersion, 'o', label='$<\\Delta E^2>$')
    plt.grid()
    plt.xlabel('$T$')
    plt.ylabel('$<\\Delta E^2>$')
    plt.legend()
    plt.show()

    #Cv en funcion de T
    plt.figure()
    plt.title('Calor específico en función de T')
    plt.plot(T, Cv, 'o', label='$Cv$')
    plt.grid()
    plt.xlabel('$T$')
    plt.ylabel('$Cv$')
    plt.legend()
    plt.show()

def main() -> None:
    J = 1.
    N = 100
    K = 1.
    T = 0.1
    beta = 1/(K*T)
    # Kb * T / J
    
    espines, energias, magnetizaciones = [], [], []
    steps = int(1e6)
    S_old, E_old, M_old = ising(N)
    for _ in range(steps):
        i_r, j_r = np.random.randint(0,N, size=(2,1))
        
        S_new = S_old.copy()
        S_new[i_r,j_r] *= -1
        E_new = np.sum(energia(S_new, np.zeros((N,N)), J))
        
        p_old = 1/(1+np.exp(-beta*(E_new-E_old)))
        p_new = 1-p_old #1/1+exp(2*J*E)
        
        if np.random.choice([0,1], p=[p_old, p_new]):
            S_old = S_new
            E_old = E_new
            M_old = np.sum(S_new)
        energias.append(E_old)
        magnetizaciones.append(M_old)
        espines.append(S_old)

    # calor_especifico(steps, k, S_old, M_old)
    animacion(espines, T, steps, N)
    
    # N_values, T_values = [7, 20, 50], [0.1, 2., 15.]
    # fig1 = comparar_N_T(N_values, T_values, steps=int(1e4), J=J, K=K)
    
    # tol = 1e-4 # % de tolerancia
    # T_values = np.arange(0.1,1.3,0.1)
    # N = 30
    # fig2, datos = comparar_M_E_equilibrio(N, T_values, J=J, K=K, tol=tol)
    # plt.figure()
    # plt.scatter(T_values, [dato[0] for dato in datos], label='Energía',       marker='x', c='teal')
    # plt.scatter(T_values, [abs(dato[1]) for dato in datos], label='Magnetización absoluta', marker='x', c='firebrick')
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.title(f'Energía y magnetización en función de la temperatura para N={N}')
    # plt.xlabel('Temperatura')
    # plt.show()

if __name__ == '__main__':
    main()
    plt.show()