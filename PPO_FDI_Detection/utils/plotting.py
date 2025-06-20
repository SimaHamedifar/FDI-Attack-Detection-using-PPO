# utils/plotting.py
import matplotlib.pyplot as plt
import os

def plot_rewards(reward_history, save_path=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_ieee_style(x, y_dict, xlabel, ylabel, title='', save_path=None, dpi=300):
    """
    Parameters
    ----------
    x : list or array
        x_axis values.
    y_dict : Dictinary
        keys are labels and values are y_axis arrays.
    xlabel : str
        x_axis label.
    ylabel : str
        y_axis label.
    title : str, optional
        Title of the figure. The default is ''.
    save_path : str, optional
        If provided, saves the figure to this path. The default is None.
    dpi : int, optional
        Dot per inch for the saved figure. The default is 300.

    Returns
    -------
    A plot in IEEE style.

    """
    plt.figure(figsize=(3.5,2.5))
    plt.rcParams.update({'font.size': 9,
                         'font.family': 'sans-serif',
                         'font.sans-serif': 'Arial', 
                         'axes.linewidth': 0.8,
                         'lines.linewidth': 1.2})
    
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd', 'x']
    
    for idx, (label,y) in enumerate(y_dict.items()):
        plt.plot(x, y, label=label, linestyle=styles[idx % len(styles)], 
                 marker=markers[idx % len(markers)], markersize=3)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend(loc='best', fontsize=8, frameon=True)
    plt.tight_layout()
    
    if save_path:
        plt.save(save_path, dpi=dpi, bbox_inches='tight')
    
    plt.show()