import socket
import visdom

def connect2visdom(viz_server, viz_port, viz_env):
    """
    Connect to Visdom server
    """
    servername = socket.gethostname()
    if "node" in servername:
        server = viz_server
    else:
        server = 'http://localhost'

    return visdom.Visdom(server=server, port=viz_port, env=viz_env)