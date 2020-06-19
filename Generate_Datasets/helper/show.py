import matplotlib
matplotlib.use('Agg')
from contextlib import contextmanager
import skvideo.io

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
except ImportError:
    plt = None
    mpl_animation = None

# ================ Description ================
# Utility functions for plots and animations
# =============================================

@contextmanager
def canvas(image_file=None, viz=None, **kwargs):
    """
    Generic matplotlib context
    """
    mode = image_file.split(".")[-2].split("/")[-1]
    fig, ax = plt.subplots(**kwargs)
    ax.grid(linestyle='dotted')
    ax.set_aspect(1.2, 'datalim')
    ax.set_axisbelow(True)
    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=300)
    if viz:
        viz.matplot(fig, win=mode)
    plt.close(fig)


@contextmanager
def animation(n, scene, movie_file=None, writer=None, viz=None, **kwargs):
    """
    Context for animations
    """
    fig, ax = plt.subplots(**kwargs)
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set_axisbelow(True)
    ax.set_title(scene)

    context = {'ax': ax, 'update_function': None}
    yield context

    ani = mpl_animation.FuncAnimation(fig, context['update_function'], range(n))
    if movie_file:
        ani.save(movie_file, writer=writer)
        print("video saved under %s" % (movie_file))
    if viz:
        video = skvideo.io.vread(movie_file)
        viz.video(tensor=video)
    plt.close(fig)
