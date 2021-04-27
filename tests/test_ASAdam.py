#  Copyright (c) 2021 Robert Lieck

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import torch
from torch.nn import Module, Parameter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import EllipseCollection, PolyCollection

from asadam import ASAdam

class TestASAdam(TestCase):

    do_plot = False

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

    @classmethod
    def plot_pins(cls, locs, heads, ax, head_style='o', pin_style='-', label=None, **kwargs):
        ax.plot(*(heads).transpose(), head_style, label=label, **kwargs)
        ax.plot(*np.array([[list(x), list(y), [np.nan, np.nan]] for x, y in zip(locs, heads)]).reshape(-1, 2).transpose(),
                pin_style, **kwargs)

    @classmethod
    def plot_halos(cls, widths, heights, offsets, ax, label=None, **kwargs):
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

    @classmethod
    def plot_boxes(cls, sizes, offsets, ax, label=None, **kwargs):
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

    @classmethod
    def demo(cls,
             module_kwargs,
             asadam_kwargs=None,
             module_class=M,
             plot_range=((-10, 10), (-10, 10)),
             n_grid=50,
             n_iterations=100,
             lr=1.,
             real_adam=False,
             asadam_as_realadam=False,
             asadam=True,
             do_plot=None
             ):
        if do_plot is None:
            do_plot = cls.do_plot
        if asadam_kwargs is None:
            asadam_kwargs = {}
        asadam_kwargs = {**dict(lr=lr, debug=True, l1=None, glue=None, l1_log=None, log_scale=None, safety=1),
                         **asadam_kwargs}

        mode_list = []
        if real_adam:
            mode_list += ["Adam"]
        if asadam_as_realadam:
            mode_list += ['ASAdam as Adam']
        if asadam:
            mode_list += ['Active Set Adam']

        fig, axes = plt.subplots(1, len(mode_list), figsize=(10 * len(mode_list), 10))
        axes = np.atleast_1d(axes)
        # fig, axes = plt.subplots(1, 1, figsize=(13, 13))

        module_list = []
        trace_list = []
        for mode, ax in zip(mode_list, axes):

            module = module_class(**module_kwargs)

            ax.set_title(mode)
            stats = {
                "grad": [],
                "avg_grad": [],
                "avg_sq_grad": [],
                "avg_grad_corr": [],
                "avg_sq_grad_corr": [],
                "direction": [],
            }
            if mode == 'Active Set Adam':
                optimizer = ASAdam(module.parameters(), **asadam_kwargs)
                stats = {**stats,
                         "avg_var_corr": [],
                         "eff_var": [],
                         "uncertainty": [],
                         "safe_grad": [],
                         "avg_var": [],
                         }
            elif mode == 'ASAdam as Adam':
                optimizer = ASAdam(module.parameters(), **{**asadam_kwargs,
                                                           **dict(l1=None,
                                                                  glue=None,
                                                                  l1_log=None,
                                                                  log_scale=None,
                                                                  sub_uncertain=False)})
            elif mode == 'Adam':
                stats = {}
                optimizer = torch.optim.Adam(module.parameters(), lr=lr)
            else:
                raise ValueError(f"Unsupported mode '{mode}'")

            trace = [module.loc.clone().detach()]
            for it in range(n_iterations):
                optimizer.zero_grad()
                loss = module()
                loss.backward()
                optimizer.step()
                # collect stats
                trace += [module.loc.clone().detach()]
                for key, val in stats.items():
                    val.append(getattr(optimizer, key).clone().detach())
                # print(f"{module.loc.detach().numpy()}: {loss}")

            # post process stats
            trace = torch.cat(tuple(t[None, :] for t in trace)).clone().detach().numpy()
            for key, val in stats.items():
                stats[key] = torch.cat(tuple(t[None, :] for t in val)).clone().detach().numpy()

            if do_plot:
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
                if mode in ['Active Set Adam', 'ASAdam as Adam']:
                    # true grad
                    c = (1, 0.1, 0.1)
                    cls.plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["grad"],
                                  ax=ax, linewidth=2, markersize=3, label='grad', color=c, zorder=10)
                    # # average
                    # c = (0.1, 0.1, 1)
                    # plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["avg_grad"],
                    #           ax=ax, pin_style='-', linewidth=2, markersize=3, label='average', color=c, zorder=20)
                    # plot_halos(*(2 * np.sqrt(stats["avg_sq_grad"]).transpose()), offsets=trace[:-1],
                    #            ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='-', zorder=20)
                    # corrected average
                    c = (0.1, 0.9, 0.1)
                    cls.plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["avg_grad_corr"],
                                  ax=ax, pin_style='-', linewidth=2, markersize=3, label='corrected average', color=c,
                                  zorder=30)
                    cls.plot_halos(*(2 * np.sqrt(stats["avg_sq_grad_corr"]).transpose()), offsets=trace[:-1],
                                   ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='-', zorder=30)
                    # step
                    c = (1, 0.1, 1)
                    cls.plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["direction"],
                                  ax=ax, markersize=2, linewidth=1, color=c, label='step', zorder=40)
                    cls.plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["direction"] * lr,
                                  ax=ax, pin_style='--', markersize=2, linewidth=1, color=c, zorder=40)
                    if mode == 'Active Set Adam':
                        # L1 boxes
                        c = (1, 0.1, 0.1)
                        l1, glue, l1_log, safety = asadam_kwargs['l1'], asadam_kwargs['glue'], asadam_kwargs['l1_log'], asadam_kwargs['safety']
                        if l1 is not None:
                            l1 = torch.tensor(l1).numpy()
                        if glue is not None:
                            glue = torch.tensor(glue).numpy()
                        if l1_log is not None:
                            l1_log = torch.tensor(l1_log).numpy()
                        if l1 is None:
                            if l1_log is None:
                                eff_l1 = None
                            else:
                                eff_l1 = l1_log
                        else:
                            if l1_log is None:
                                eff_l1 = l1
                            else:
                                eff_l1 = l1_log + l1
                        if eff_l1 is not None:
                            cls.plot_boxes(sizes=np.ones((n_iterations, 2)) * 2 * eff_l1, offsets=trace[:-1],
                                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='--', label="L1")
                        # L1 + glue boxes
                        if glue is not None:
                            cls.plot_boxes(sizes=np.ones((n_iterations, 2)) * 2 * ((eff_l1 if eff_l1 is not None else 0.) + glue),
                                           offsets=trace[:-1],
                                           ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='-',
                                           label=f"{('L1 + ' if eff_l1 is not None else '')}glue")
                        # variance
                        c = (0.1, 0.9, 0.9)
                        cls.plot_halos(*(2 * np.sqrt(stats["avg_var_corr"]).transpose()),
                                       offsets=trace[:-1] - stats["avg_grad_corr"],
                                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='-', label="STD")
                        cls.plot_halos(*(2 * np.sqrt(stats["eff_var"]).transpose()),
                                       offsets=trace[:-1] - stats["avg_grad_corr"],
                                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=0.5, linestyle='--', label="effective STD")
                        # bounds
                        c = (0, 0, 1)
                        cls.plot_boxes(sizes=stats["uncertainty"] * 2 * safety, offsets=trace[:-1] - stats["avg_grad_corr"],
                                       ax=ax, ec=c, fc=(0, 0, 0, 0), linewidth=1, linestyle='--', label="uncertainty")
                        # effective gradient
                        cls.plot_pins(locs=trace[:-1], heads=trace[:-1] - stats["safe_grad"],
                                      ax=ax, pin_style='--', linewidth=1, markersize=2, label='effective average', color=c,
                                      zorder=30)

                # adjust stuff
                ax.legend()
                ax.set_aspect('equal')
                ax.set_xlim(*plot_range[0])
                ax.set_ylim(*plot_range[1])

            module_list.append(module)
            trace_list.append(trace)

        if do_plot:
            # show in tight layout
            fig.tight_layout()
            plt.show()

        return module_list, trace_list

    def test_Adam_equivalence(self):
        for active, safety in [(True, 1), (False, 0.9), (False, 1.1)]:
            _, (t1, t2, t3) = self.demo(
                module_kwargs=dict(loc=[0, 0], sigma=[[1, 0], [0, 4]], center=(5, 7), div=10.),
                asadam_kwargs=dict(l1=0, glue=0, l1_log=0, log_scale=1, safety=safety, active=active),
                lr=2,
                real_adam=True,
                asadam_as_realadam=True,
                asadam=True)
            if active or safety < 1:
                # if all dimensions are active, all three should be equivalent
                assert_array_almost_equal(t1, t2)
                assert_array_almost_equal(t2, t3)
                assert_array_almost_equal(t3, t1)
            else:
                # without activation, ASAdam should start late if safety is > 1 because of uncertainty,
                # even with zero l1 and glue
                assert_array_almost_equal(t1, t2)
                self.assertFalse(np.all(t1 == t3))
                self.assertFalse(np.all(t2 == t3))
                assert_array_equal(t3[0:2], 0)

    def test_delayed_activation_and_log_scale(self):
        min_steps = 5
        _, (trace,) = self.demo(
            module_kwargs=dict(loc=[0, 0], sigma=[[1, -0.999], [-0.999, 1]], center=(5, 5), div=10000),
            asadam_kwargs=dict(glue=None, l1_log=0.1, log_scale=1, safety=1,
                               max_activation=1, min_steps=min_steps, betas=(0.5, 0.99, 0.5)),
            asadam_as_realadam=False,
            plot_range=((-1, 11), (-1, 11)))
        assert_array_almost_equal(trace[0], [0, 0])
        if trace[1, 0] == 0:
            # 1st dimension has delay
            self.assertTrue(np.all(trace[0:min_steps + 1, 0] == 0))  # starts at zero
            self.assertFalse(np.any(trace[min_steps + 1:min_steps + 10, 0] == 0))  # is non-zero afterwards
            self.assertTrue(np.all(trace[-10:, 0] == 0)) # ends at zero
            # 2nd dimension is non-zero except on first step
            self.assertFalse(np.any(trace[1:, 1] == 0))
        else:
            # 2nd dimension has delay
            self.assertTrue(np.all(trace[0:min_steps + 1, 1] == 0))  # starts at zero
            self.assertFalse(np.any(trace[min_steps + 1:min_steps + 10, 1] == 0))  # is non-zero afterwards
            self.assertTrue(np.all(trace[-10:, 1] == 0))  # ends at zero
            # 1st dimension is non-zero except on first step
            self.assertFalse(np.any(trace[1:, 0] == 0))

    def test_l1_strength(self):
        # quadratic loss L(x0, x1) = c0 * (x0 - d0) ** 2 + c1 * (x1 - d1) ** 2 + l0 * |x0| + l1 * |x1|
        # gradient dL/dxi = 2 * ci * (xi - di) + li * sign(xi)
        # gradient at xi = 0 is -2 * ci * di
        # optimum (assuming xi > 0) at xi = di - 1/2 * li / ci
        d = torch.Tensor([2, 1])
        c = torch.Tensor([0.1, 0.3])
        grad_at_zero = -2 * c * d  # [-0.4, -0.6]
        for l1, glue, l1_log, held_at_zero in [
            (0, 0, 0, [False, False]),
            (0.1, 0, 0, [False, False]),
            (0, 0.1, 0, [False, False]),
            (0, 0, 0.1, [False, False]),
            (0.1, 0.1, 0, [False, False]),
            (0, 0.1, 0.1, [False, False]),
            (0.1, 0, 0.1, [False, False]),
            (0.1, 0.1, 0.1, [False, False]),
            (0.5, 0, 0, [True, False]),
            (0.7, 0, 0, [True, True]),
            (torch.Tensor([0.3, 0.7]), 0, 0, [False, True]),
        ]:
            # print(f"l1: {l1}\n"
            #       f"glue: {glue}\n"
            #       f"l1_log: {l1_log}\n"
            #       f"optimum: {d - (l1 + l1_log) / c / 2}")
            _, (trace,) = self.demo(module_kwargs=dict(loc=[0, 0],
                                                       sigma=[[1 / c[0], 0], [0, 1 / c[1]]],
                                                       center=d,
                                                       div=1),
                                    asadam_kwargs=dict(l1=l1, glue=glue, l1_log=l1_log, log_scale=1e10, safety=0,
                                                       betas=(0.5, 0.99, 0.5)),
                                    lr=0.05,
                                    n_iterations=200,
                                    plot_range=((-0.1, 3), (-0.1, 3)))
            # get stickiness in vector form
            sticky = l1 + glue + l1_log
            if not isinstance(sticky, torch.Tensor):
                sticky = torch.Tensor([sticky, sticky])
            for dim in [0, 1]:
                # check for initial stickiness
                if sticky[dim] < abs(grad_at_zero[dim]):
                    # not holding params at zero
                    self.assertFalse(held_at_zero[dim])
                    # optimum shifted (just add l1_log because we are using extreme scale for that)
                    assert_array_almost_equal(trace[-1, dim], (d - (l1 + l1_log) / c / 2)[dim])
                else:
                    # params held at zero
                    self.assertTrue(held_at_zero[dim])
                    assert_array_equal(trace[:, dim], 0)

    def test_demo(self):
        # simple example
        self.demo(module_kwargs=dict(loc=[-6, 9], sigma=[[1, 0], [0, 4]], center=(5, -7), div=10.),
                  asadam_kwargs=dict(glue=0.21, safety=2),
                  asadam_as_realadam=True,
                  asadam=True,
                  lr=2)
        # delayed activation and log_scale
        self.demo(module_kwargs=dict(loc=[0, 0], sigma=[[1, -0.999], [-0.999, 1]], center=(5, 5), div=10000),
                  asadam_kwargs=dict(active=False, l1_log=0.1, log_scale=1, max_activation=1,
                                     min_steps=5,
                                     # betas=(0.5, 0.99, 0.5),
                                     ),
                  asadam_as_realadam=True,
                  asadam=True,
                  plot_range=((-1, 11), (-1, 11)))
        # vectorised l1/glue/log_scale
        self.demo(module_kwargs=dict(loc=[0, 0], sigma=[[1, 0], [0, 1]], center=(5, 5), div=10),
                  asadam_kwargs=dict(active=False,
                                     glue=torch.Tensor([0.2, 0.4]),
                                     l1_log=torch.Tensor([0.2, 0.4]),
                                     log_scale=torch.Tensor([0.1, 2]),
                                     safety=1, lr=1e-1, ),
                  asadam_as_realadam=True,
                  asadam=True,
                  plot_range=((-1, 10), (-1, 10)))
        # prioritised activation, no regularisation
        self.demo(module_kwargs=dict(loc=[0, 0], sigma=[[1, 0], [0, 1]], center=(5, 5), div=10),
                  asadam_kwargs=dict(active=False, glue=0, safety=0, lr=1,
                                     # betas=(0.1, 0.9, 0.1),
                                     # betas=(0.5, 0.99, 0.5),
                                     max_activation=1, min_steps=10),
                  asadam_as_realadam=True,
                  asadam=True,
                  plot_range=((-1, 10), (-1, 10)))
