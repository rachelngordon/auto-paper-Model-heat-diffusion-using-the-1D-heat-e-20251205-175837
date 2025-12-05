# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate_heat(D, t_end, dx, dt, u0, save_times):
    u = u0.copy()
    r = D * dt / dx**2
    if r > 0.5:
        raise ValueError(f'Unstable scheme: r={r}')
    saved = {}
    current_time = 0.0
    saved[0.0] = u.copy()
    save_set = set(save_times)
    steps = int(np.ceil(t_end / dt))
    for _ in range(steps):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u = u_new
        current_time += dt
        for t in list(save_set):
            if current_time >= t - 1e-12:
                saved[t] = u.copy()
                save_set.remove(t)
        if not save_set:
            # all required times saved; continue to maintain stability until t_end
            pass
    if t_end not in saved:
        saved[t_end] = u.copy()
    return saved

def main():
    L = 1.0
    Nx = 201
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    sigma = 0.05
    u0 = np.exp(-0.5 * ((x - L / 2) / sigma) ** 2)
    # Experiment 1: baseline diffusion profile
    D1 = 1.0
    r = 0.4
    dt = r * dx ** 2 / D1
    times_exp1 = [0.0, 0.1, 0.5, 1.0]
    sol1 = simulate_heat(D1, max(times_exp1), dx, dt, u0, times_exp1)
    plt.figure()
    for t in times_exp1:
        plt.plot(x, sol1[t], label=f't={t}')
    plt.xlabel('Position x')
    plt.ylabel('Temperature')
    plt.title('Temperature profiles over time (D=1.0)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temperature_profiles_over_time.png')
    plt.close()
    # Experiment 2: diffusivity parameter sweep
    D_vals = [0.1, 0.5, 1.0, 2.0]
    t_fixed = 0.5
    plt.figure()
    for D in D_vals:
        dt = r * dx ** 2 / D
        sol = simulate_heat(D, t_fixed, dx, dt, u0, [t_fixed])
        plt.plot(x, sol[t_fixed], label=f'D={D}')
    plt.xlabel('Position x')
    plt.ylabel('Temperature')
    plt.title(f'Temperature at t={t_fixed} for various D')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temperature_vs_position_various_D.png')
    plt.close()
    # Primary numeric answer: maximum temperature at t=1.0 for D=1.0
    answer = np.max(sol1[1.0])
    print('Answer:', answer)

if __name__ == '__main__':
    main()

