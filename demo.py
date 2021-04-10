#  Copyright (c) 2021 Robert Lieck

import numpy as np
import torch
from torch.nn import Module, Parameter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import EllipseCollection, PatchCollection

from asadam import ASAdam


class M(Module):

    def __init__(self, loc, sigma=((1, 0), (0, 1)), center=(0, 0)):
        super().__init__()
        self.loc = Parameter(torch.Tensor(loc))
        self.center = torch.Tensor(center)
        self.prec = torch.inverse(torch.Tensor(sigma))

    def loss(self, loc):
        diff = loc - self.center
        return torch.einsum('...a,ab,...b->...', diff, self.prec, diff) / 10

    def forward(self):
        return self.loss(self.loc)


def plot_pins(locs, heads, ax, head_style='o', pin_style='-', label=None, **kwargs):
    ax.plot(*(heads).t(), head_style, label=label, **kwargs)
    ax.plot(*np.array([[list(x), list(y), [np.nan, np.nan]] for x, y in zip(locs, heads)]).reshape(-1, 2).transpose(),
        pin_style, **kwargs)


def plot_halos(widths, heights, offsets, ax, **kwargs):
    ax.add_collection(EllipseCollection(widths=widths, heights=heights,
                                        angles=np.zeros_like(widths),
                                        units='xy',
                                        transOffset=ax.transData,
                                        offsets=offsets,
                                        **kwargs))


def plot_boxes(sizes, offsets, ax, label=None, **kwargs):
    label_ = label
    for loc, s in zip(offsets, sizes):
        ax.add_patch(Rectangle(loc - s / 2, *s, label=label_, **kwargs))
        label_ = None


if __name__ == "__main__":

    plot_range = 10
    n_grid = 50
    D = 2
    n_iterations = 100
    lr = 1e-0

    l1 = None
    glue = 0.21
    safety = 1.5
    active = True

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for asadam, ax in [(False, axes[0]), (True, axes[1])]:

        if asadam:
            ax.set_title("Active Set Adam")
        else:
            ax.set_title("Normal Adam")

        module = M(loc=[-6, 9], sigma=[[1, 0], [0, 4]], center=(5, -7))
        stats = {"exp_avg": [],
                 "exp_avg_corr": [],
                 "exp_avg_std": [],
                 "exp_avg_std_corr": [],
                 "true_grad": [],
                 "eff_grad": [],}
        if asadam:
            optimizer = ASAdam(module.parameters(), lr=lr, l1=l1, glue=glue, safety=safety, active=active, debug=True)
            stats = {**stats,
                     "exp_avg_var": [],
                     "exp_avg_var_corr": [],
                     "uncertainty": []}
        else:
            optimizer = ASAdam(module.parameters(), lr=lr, l1=None, glue=None, safety=safety, active=active, debug=True)
            # optimizer = Adam(module.parameters(), lr=lr)

        trace = [module.loc.detach().clone()]
        for it in range(n_iterations):
            optimizer.zero_grad()
            loss = module()
            loss.backward()
            optimizer.step()
            # collect stats
            trace += [module.loc.detach().clone()]
            for key, val in stats.items():
                val.append(getattr(optimizer, key).detach().clone())
            print(f"{module.loc.detach().numpy()}: {loss}")

        # post process stats
        trace = torch.cat(tuple(t[None, :] for t in trace))
        for key, val in stats.items():
            stats[key] = torch.cat(tuple(t[None, :] for t in val))

        # define grid for heatmap
        # 1D space used for each dimension
        grid = torch.linspace(-plot_range, plot_range, n_grid)
        # meshgrid with D dimensions (tuple of coordinates)
        xyz = torch.meshgrid(*(grid,) * D)
        # flatten and concatenate along new dimension to get grid of coordinates
        locs = torch.cat(tuple(l.flatten()[..., None] for l in xyz), dim=-1)
        # evaluate function on grid
        f = module.loss(locs).reshape(*(n_grid,) * D).t()

        # objective
        ax.contourf(grid, grid, f, 100, zorder=-10, cmap='Reds_r')
        # optimum
        ax.scatter(*module.center, marker='+', c='k', zorder=10, label='optimum')
        # zero lines
        ax.plot([-plot_range, plot_range, np.nan, 0, 0], [0, 0, np.nan, -plot_range, plot_range], c=(0, 0, 0, 0.1))
        # trace
        c = (0.1, 0.1, 0.1)
        ax.plot(*trace.t(), '-o', linewidth=2, markersize=4, color=c, label='trace')
        ax.scatter(*trace[0], s=5, color=c)
        # step
        plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["eff_grad"] * lr,
                  ax=ax, markersize=2, linewidth=1, color=(1, 0.1, 1), label='step')
        # true grad
        c = (1, 0.1, 0.1)
        plot_pins(locs=trace, heads=trace[:-1] - stats["true_grad"],
                  ax=ax, linewidth=1, markersize=2, label='gradient', color=c, zorder=2)
        # average
        c = (0.1, 0.1, 1)
        plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["exp_avg"],
                  ax=ax, linewidth=1, markersize=2, label='average', color=c, zorder=3)
        plot_halos(*(stats["exp_avg_std"].t()), offsets=trace[:-1],
                   ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='-', zorder=3)
        # corrected average
        c = (0.1, 0.9, 0.1)
        plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["exp_avg_corr"],
                  ax=ax, pin_style='--', linewidth=1, markersize=2, label='corrected average', color=c, zorder=1)
        plot_halos(*(stats["exp_avg_std_corr"].t()), offsets=trace[:-1],
                   ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='--', zorder=1)
        if asadam:
            # variance
            c = (0.1, 0.9, 0.9)
            plot_halos(*(stats["exp_avg_var"].sqrt().t()), offsets=trace[:-1] - stats["exp_avg"],
                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='-', label="variance")
            plot_halos(*(stats["exp_avg_var_corr"].sqrt().t()), offsets=trace[:-1] - stats["exp_avg"],
                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='--')
            # L1 boxes
            c = (1, 0.1, 0.1)
            if l1 is not None:
                plot_boxes(sizes=np.ones((n_iterations, 2)) * 2 * l1, offsets=trace,
                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='--', label="L1")
            if glue is not None:
                plot_boxes(sizes=np.ones((n_iterations, 2)) * 2 * ((l1 if l1 is not None else 0.) + glue), offsets=trace,
                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='-',
                           label=f"{('L1 + ' if l1 is not None else '')}glue")
            # confidence boxes
            c = (0, 0, 1)
            plot_boxes(sizes=stats["uncertainty"] * 2 * safety, offsets=trace[:-1] - stats["exp_avg"],
                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='--', label="uncertainty")

        # adjust stuff
        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)

    # show in tight layout
    fig.tight_layout()
    plt.show()
