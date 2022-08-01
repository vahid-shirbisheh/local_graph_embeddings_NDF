# Local Graph Embeddings Based on Neighbors Degree Frequency of Nodes
## By Vahid Shirbisheh

This is the repository of python soure codes for the implementations of algorithms and examples of the article mentioned above. 

For routine computations on graphs (such as various centrality measures, PageRank, etc), we use the library NetworkX. 
For data science and deep learning algorithms, we use pandas, numpy, matplotlib and pytorch. 
We use datasets that are freely available in the internet and appropriate references are given inside the article.
 
### Undirected Graphs:
We have implemented a costume class for undirected graphs called "MyGraph" (implemented in mygraph.py).
We have also written several helper functions (implemented in helpers.py) to make flow of our implementations more straightforward.
To use the methods and functions in these files one has to simply import everything from helpers.py as follows:
from helpers import * 
### Directed Graphs:
Similarly, we have a costume class for directed graphs (named MyDiGraph and implemented in mydigraph.py), 
and a set of helper functions needed to work with directed graphs (implemented in dihelpers.py).
To use the methods and functions in these files one has to simply import everything from dihelpers.py as follows:
from dihelpers import * 

### Source codes of examples in Sections 2, 3, 4, 5 can be found in the following folder:

../sections_2_3_4_5

### Source codes of examples in Section 6 can be found in the following folder:

../pagerank_section_6

### Source codes of examples in Section 7 can be found in the following folder:

../closeness_section_7

### Source codes of examples in Section 8 can be found in the following folder:
../directed_graphs/pagerank_section_8


Important Notice:
1. Not every method and function in classes MyGraph and MyDiGraph has been used inside the article!
2. These codes meant to explain the content of the article and we do not claim they are the most efficient implementations.  
3. To contact us, please send your emails to shirbisheh@gmail.com. 