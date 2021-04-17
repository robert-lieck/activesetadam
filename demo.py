#  Copyright (c) 2021 Robert Lieck

import numpy as np
import torch
from torch.nn import Module, Parameter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import EllipseCollection, PolyCollection

from asadam import ASAdam


class M(Module):

    def __init__(self, loc, sigma=((1, 0), (0, 1)), center=(0, 0), div=1.):
        super().__init__()
        self.loc = Parameter(torch.Tensor(loc))
        self.center = torch.Tensor(center)
        self.prec = torch.inverse(torch.Tensor(sigma))
        self.div = div

    def loss(self, loc):
        diff = loc - self.center
        return torch.einsum('...a,ab,...b->...', diff, self.prec, diff) / self.div

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


def demo(module_kwargs,
         asadam_kwargs=None,
         module_class=M,
         plot_range=((-10, 10), (-10, 10)),
         n_grid=50,
         n_iterations=100,
         lr=1,
         real_adam=False
):
    if asadam_kwargs is None:
        asadam_kwargs = {}
    asadam_kwargs = {**dict(lr=lr, debug=True), **asadam_kwargs}

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    # fig, axes = plt.subplots(1, 1, figsize=(13, 13))

    for asadam, ax in [
        # (True, axes),
        (False, axes[0]),
        (True, axes[1])
    ]:

        module = module_class(**module_kwargs)

        if asadam:
            ax.set_title("Active Set Adam")
        else:
            ax.set_title("Normal Adam")
        stats = {
            "grad": [],
            "avg_grad": [],
            "avg_sq_grad": [],
            "avg_grad_corr": [],
            "avg_sq_grad_corr": [],
            "direction": [],
        }
        if asadam:
            optimizer = ASAdam(module.parameters(), **asadam_kwargs)
            stats = {**stats,
                     "avg_var_corr": [],
                     "eff_var": [],
                     "uncertainty": [],
                     "safe_grad": [],
                     "avg_var": [],
                     }
        else:
            if real_adam:
                stats = {}
                optimizer = torch.optim.Adam(module.parameters(), lr=lr)
            else:
                optimizer = ASAdam(module.parameters(), **{**asadam_kwargs,
                                                           **dict(l1=None,
                                                                  glue=None,
                                                                  sub_uncertain=False)})

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
        x_grid = torch.linspace(*plot_range[0], n_grid)
        y_grid = torch.linspace(*plot_range[1], n_grid)
        # meshgrid with D dimensions (tuple of coordinates)
        xy = torch.meshgrid([x_grid, y_grid])
        # flatten and concatenate along new dimension to get grid of coordinates
        locs = torch.cat(tuple(l.flatten()[..., None] for l in xy), dim=-1)
        # evaluate function on grid
        f = module.loss(locs).reshape(n_grid, n_grid).transpose(-2, -1)

        # objective
        ax.contourf(x_grid, y_grid, f, 100, zorder=-10, cmap='Reds_r')
        # optimum
        ax.scatter(*module.center, marker='+', c='k', zorder=100, label='optimum')
        # zero lines
        ax.plot([plot_range[0][0], plot_range[0][1], np.nan, 0, 0],
                [0, 0, np.nan, plot_range[1][0], plot_range[1][1]],
                c=(0, 0, 0, 0.1))
        # trace
        c = (0.1, 0.1, 0.1)
        ax.plot(*trace.transpose(), '-o', linewidth=2, markersize=4, color=c, label='trace')
        ax.scatter(*trace[0], s=5, color=c, zorder=0)
        if asadam or not real_adam:
            # true grad
            c = (1, 0.1, 0.1)
            plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["grad"],
                      ax=ax, linewidth=2, markersize=3, label='grad', color=c, zorder=10)
            # # average
            # c = (0.1, 0.1, 1)
            # plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["avg_grad"],
            #           ax=ax, pin_style='-', linewidth=2, markersize=3, label='average', color=c, zorder=20)
            # plot_halos(*(2 * np.sqrt(stats["avg_sq_grad"]).transpose()), offsets=trace[:-1],
            #            ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='-', zorder=20)
            # corrected average
            c = (0.1, 0.9, 0.1)
            plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["avg_grad_corr"],
                      ax=ax, pin_style='-', linewidth=2, markersize=3, label='corrected average', color=c, zorder=30)
            plot_halos(*(2 * np.sqrt(stats["avg_sq_grad_corr"]).transpose()), offsets=trace[:-1],
                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='-', zorder=30)
            # step
            c = (1, 0.1, 1)
            plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["direction"],
                      ax=ax, markersize=2, linewidth=1, color=c, label='step', zorder=40)
            plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["direction"] * lr,
                      ax=ax, pin_style='--', markersize=2, linewidth=1, color=c, zorder=40)
            if asadam:
                # L1 boxes
                c = (1, 0.1, 0.1)
                l1, glue, safety = asadam_kwargs['l1'], asadam_kwargs['glue'], asadam_kwargs['safety']
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
                plot_halos(*(2 * np.sqrt(stats["avg_var_corr"]).transpose()), offsets=trace[:-1] - stats["avg_grad_corr"],
                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='-', label="STD")
                plot_halos(*(2 * np.sqrt(stats["eff_var"]).transpose()),
                           offsets=trace[:-1] - stats["avg_grad_corr"],
                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='--', label="effective STD")
                # bounds
                c = (0, 0, 1)
                plot_boxes(sizes=stats["uncertainty"] * 2 * safety, offsets=trace[:-1] - stats["avg_grad_corr"],
                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='--', label="uncertainty")
                # effective gradient
                plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["safe_grad"],
                          ax=ax, pin_style='--', linewidth=1, markersize=2, label='effective average', color=c, zorder=30)

        # adjust stuff
        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlim(*plot_range[0])
        ax.set_ylim(*plot_range[1])

    # show in tight layout
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo(module_kwargs=dict(loc=[-6, 9], sigma=[[1, 0], [0, 4]], center=(5, -7), div=10.),
         asadam_kwargs=dict(l1=None, glue=0.21, safety=2),
         lr=2)
    demo(module_kwargs=dict(loc=[0, 0], sigma=[[1, -0.999], [-0.999, 1]], center=(5, 5), div=10000),
         asadam_kwargs=dict(active=False, l1=0.1, glue=None, log_scale=1, safety=1, max_activation=1, min_steps=5),
         plot_range=((-1, 10), (-1, 10)))
