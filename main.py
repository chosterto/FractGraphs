import plotly.graph_objects as ply
from plotly.subplots import make_subplots
import numpy as np
from math import pi, sqrt, exp, sin

def f(x):
    return x


def f_prime(x):
    h = 0.0001
    return (f(x + h) - f(x)) / h


def I(a, b):
    n = 10000
    delta_x = (b - a) / n
    sum_ = 0
    for i in range(1, n + 1):
        sum_ += f(a + delta_x * i) + f(a + delta_x * (i - 1))
    return sum_ * delta_x / 2


def gamma(z):
    # Lanczos approximation
    # https://en.wikipedia.org/wiki/Lanczos_approximation
    g = 7
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]

    if z < 0.5:
        if z == 0.0:
            z = 0.00000001
        y = pi / (sin(pi * z) * gamma(1 - z))  # Reflection formula
    else:
        z -= 1
        x = p[0]
        for i in range(1, len(p)):
            x += p[i] / (z + i)
        t = z + g + 0.5
        y = sqrt(2 * pi) * t ** (z + 0.5) * exp(-t) * x
    return y


def D(a, x):
    def binomial_coeffs(p, k):
        if p > 0:
            return (-1)**k * gamma(p + 1) / (gamma(k + 1) * gamma(p - k + 1))
        return gamma(-p + k) / (gamma(k + 1) * gamma(-p))

    # Grunwald Letnikov method
    # bounds of integral are 0 and x
    if a == 0.0:
        return f(x)
    h = x / 142
    sum_GL = 0
    for m in range(142):
        sum_GL += f(x - m*h) * binomial_coeffs(a, m)
    if a > 0:
        return sum_GL / (h ** a)
    elif a < 0:
        return sum_GL * (h ** -a)


def main():
    num_of_points = 50
    num_of_steps = 30
    x_vals = np.arange(0.0001, 10.0, 10.0 / num_of_points)
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}]])

    fig.add_trace(
        ply.Scatter(
            name='f(x)',
            visible=True,
            line=dict(color='#00FF00'),
            x=x_vals,
            y=f(x_vals)
        ),
        row=1, col=1
    )

    fig.add_trace(
        ply.Scatter(
            name='f\'(x)',
            visible=True,
            line=dict(color='#FF0000'),
            x=x_vals,
            y=f_prime(x_vals)
        ),
        row=1, col=1
    )

    fig.add_trace(
        ply.Scatter(
            name='∫f(x)dx',
            visible=True,
            line=dict(color='#0000FF'),
            x=x_vals,
            y=I(0.0, x_vals)
        ),
        row=1, col=1
    )

    steps = []
    c = 0
    color_step = 255 // (num_of_steps // 2)
    for i in np.linspace(-1.0, 1.0, num_of_steps):
        if i < 0:
            blue = 255 - color_step * c
            rgb = 'rgb(0,{0},{1})'.format(255 - blue, blue)
        else:
            green = 255 - color_step * (c - num_of_steps // 2)
            rgb = 'rgb({0},{1},0)'.format(255 - green, green)
        
        DI = D(i, x_vals)
        fig.add_trace(
            ply.Scatter(
                name='alpha = ' + str(round(i, 2)),
                visible=False,
                line=dict(color=rgb, dash='dot'),
                x=x_vals,
                y=DI
            ),
            row=1, col=1
        )

        fig.add_trace(
            ply.Scatter3d(
                line=dict(color=rgb, width=20),
                x=[c] * num_of_points,
                y=x_vals,
                z=DI
            ),
            row=1, col=2
        )

        # Extremely scuffed but whatever
        step = dict(
            method='update',
            args=[dict(visible=[True] * 3 + [False, True] * num_of_steps)]
        )
        step['args'][0]['visible'][2 * c + 3] = True
        steps.append(step)
        c += 1


    fig.data[0].visible = True
    sliders = [dict(
        active=0,
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders, 
        showlegend=False,
        scene=dict(
            xaxis_title='step',
            yaxis_title='x',
            zaxis_title='f(x)',
            xaxis=dict(range=[-1, num_of_steps + 1]),
            yaxis=dict(range=[0.0, 10.0]),
            zaxis=dict(range=[-40, 40])
        )
    )

    fig.update_yaxes(range=[-20, 20], secondary_y=False)

    fig.show()


if __name__ == '__main__':
    main()