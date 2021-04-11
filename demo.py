#  Copyright (c) 2021 Robert Lieck

import numpy as np
import torch
from torch.nn import Module, Parameter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import EllipseCollection, PolyCollection

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
    ax.plot(*(heads).transpose(), head_style, label=label, **kwargs)
    ax.plot(*np.array([[list(x), list(y), [np.nan, np.nan]] for x, y in zip(locs, heads)]).reshape(-1, 2).transpose(),
        pin_style, **kwargs)


def plot_halos(widths, heights, offsets, ax, label=None, **kwargs):
    if label is not None:
        ax.add_patch(Ellipse(xy=offsets[0], width=widths[0], height=heights[0], label=label, **kwargs))
        widths = widths[1:]
        heights = heights[1:]
        offsets = offsets[1:]
    ax.add_collection(EllipseCollection(widths=widths, heights=heights,
                                        angles=np.zeros_like(widths),
                                        units='xy',
                                        transOffset=ax.transData,
                                        offsets=offsets,
                                        **kwargs))


def plot_boxes(sizes, offsets, ax, label=None, **kwargs):
    bottom_left = offsets - sizes / 2
    bottom_right = bottom_left.copy()
    bottom_right[:, 0] += sizes[:, 0]
    top_left = bottom_left.copy()
    top_left[:, 1] += sizes[:, 1]
    top_right = offsets + sizes / 2
    verts = np.concatenate((bottom_left[:, None, :],
                            bottom_right[:, None, :],
                            top_right[:, None, :],
                            top_left[:, None, :]), axis=1)
    if label is not None:
        ax.add_patch(Rectangle(offsets[0] - sizes[0] / 2, *sizes[0], label=label, **kwargs))
        verts = verts[1:]
    ax.add_collection(PolyCollection(verts=verts, **kwargs))


if __name__ == "__main__":

    plot_range = 10
    n_grid = 50
    D = 2
    n_iterations = 100
    lr = 2

    l1 = None
    glue = 0.21
    safety = 1
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
        trace = torch.cat(tuple(t[None, :] for t in trace)).detach().clone().numpy()
        for key, val in stats.items():
            stats[key] = torch.cat(tuple(t[None, :] for t in val)).detach().clone().numpy()

        # define grid for heatmap
        # 1D space used for each dimension
        grid = torch.linspace(-plot_range, plot_range, n_grid)
        # meshgrid with D dimensions (tuple of coordinates)
        xyz = torch.meshgrid(*(grid,) * D)
        # flatten and concatenate along new dimension to get grid of coordinates
        locs = torch.cat(tuple(l.flatten()[..., None] for l in xyz), dim=-1)
        # evaluate function on grid
        f = module.loss(locs).reshape(*(n_grid,) * D).transpose(-2, -1)

        # objective
        ax.contourf(grid, grid, f, 100, zorder=-10, cmap='Reds_r')
        # optimum
        ax.scatter(*module.center, marker='+', c='k', zorder=10, label='optimum')
        # zero lines
        ax.plot([-plot_range, plot_range, np.nan, 0, 0], [0, 0, np.nan, -plot_range, plot_range], c=(0, 0, 0, 0.1))
        # trace
        c = (0.1, 0.1, 0.1)
        ax.plot(*trace.transpose(), '-o', linewidth=2, markersize=4, color=c, label='trace')
        ax.scatter(*trace[0], s=5, color=c)
        # step
        plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["eff_grad"] * lr,
                  ax=ax, markersize=2, linewidth=1, color=(1, 0.1, 1), label='step')
        # true grad
        c = (1, 0.1, 0.1)
        plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["true_grad"],
                  ax=ax, linewidth=1, markersize=2, label='gradient', color=c, zorder=2)
        # average
        c = (0.1, 0.1, 1)
        plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["exp_avg"],
                  ax=ax, linewidth=1, markersize=2, label='average', color=c, zorder=3)
        plot_halos(*(2 * stats["exp_avg_std"].transpose()), offsets=trace[:-1],
                   ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='-', zorder=3)
        # corrected average
        c = (0.1, 0.9, 0.1)
        plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["exp_avg_corr"],
                  ax=ax, pin_style='--', linewidth=1, markersize=2, label='corrected average', color=c, zorder=1)
        plot_halos(*(2 * stats["exp_avg_std_corr"].transpose()), offsets=trace[:-1],
                   ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='--', zorder=1)
        if asadam:

            # L1 boxes
            c = (1, 0.1, 0.1)
            if l1 is not None:
                plot_boxes(sizes=np.ones((n_iterations, 2)) * 2 * l1, offsets=trace[:-1],
                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='--', label="L1")
            # L1 + glue boxes
            if glue is not None:
                plot_boxes(sizes=np.ones((n_iterations, 2)) * 2 * ((l1 if l1 is not None else 0.) + glue),
                           offsets=trace[:-1],
                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='-',
                           label=f"{('L1 + ' if l1 is not None else '')}glue")
            # variance
            c = (0.1, 0.9, 0.9)
            plot_halos(*(2 * np.sqrt(stats["exp_avg_var"]).transpose()), offsets=trace[:-1] - stats["exp_avg"],
                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='-')
            plot_halos(*(2 * np.sqrt(stats["exp_avg_var_corr"]).transpose()), offsets=trace[:-1] - stats["exp_avg"],
                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='--', label="STD around average")
            # bounds
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
