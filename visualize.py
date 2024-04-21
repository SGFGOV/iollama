from pyvis.network import Network
from llama_index.core import KnowledgeGraphIndex
def visualize(index:KnowledgeGraphIndex,file_name:str = "Knowledge_graph.html"):
    g = index.get_networkx_graph()
    net = Network(notebook=False,cdn_resources="in_line",directed=True)
    net.from_nx(g)
    #net.show("graph.html")
    net.save_graph(file_name)
#
# import IPython
# IPython.display.HTML(filename="/content/Knowledge_graph.html")