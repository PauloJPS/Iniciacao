import igraph as ig
import community
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.decomposition import PCA
import scipy as sp
import numpy as np
import os


def graph_my(n, m):
	ba = ig.Graph.Barabasi(n, m=5).get_edgelist()
	ws = np.array(ig.Graph.Watts_Strogatz(1, n, 5, 0.05).get_edgelist()) + 100
	er = np.array(ig.Graph.Erdos_Renyi(n, 0.05).get_edgelist()) + 2*n 
	ed = ba + ws.tolist() + er.tolist()
	g = ig.Graph(edges=ed)
	adj = g.get_adjlist() 
	comun_list =  [list(range(1,n)),list(range(n,2*n)),list(range(2*n, 3*n))]
	for i in comun_list:
		for j in i:
			c1 = np.random.sample()
			c2 = np.random.sample()
			me = comun_list.index(i)
			c1t = np.mod(me + 1, 3)
			c2t = np.mod(me + 2, 3)
			if c1 < m[me][c1t]:
				new_e = np.random.choice(comun_list[c1t])
				g.add_edge(j, new_e)
			if c2 < m[me][c2t]:
				new_e = np.random.choice(comun_list[c2t])
				g.add_edge(j, new_e)
	
	g = g.clusters().giant()
	dendrogram = g.community_edge_betweenness()
	clusters = dendrogram.as_clustering()            
	membership = clusters.membership                                              
	visual_style = {}                                                     
	visual_style["vertex_color"] = membership
	return g


def sbm(n, m, l):
	g = ig.Graph.SBM(n, m, l, directed=True)
	dendrogram = g.community_edge_betweenness()
	clusters = dendrogram.as_clustering()            
	membership = clusters.membership                                              
	visual_style = {}                                                     
	visual_style["vertex_color"] = membership
	ig.plot(g, "SBM1.pdf", **visual_style, palette = ig.ClusterColoringPalette(len(set(membership)) + 3)).save('oi')

	
def Comu_Wax(a, nc, size, plot=0):
	c_list = []
	""" x = [-2, -1, 1, 2] 
	y =	[1, 1, 1, 1] 
	"""
	x = np.random.normal(size=nc, scale=0.5)
	y = np.random.normal(size=nc, scale=0.5)

	for i, n, j, k  in zip(range(nc), size, x, y):
		x1 = np.random.normal(loc=j, size=n, scale=0.25)
		y1 = np.random.normal(loc=k, size=n, scale=0.25)
		c_list = c_list + np.column_stack((x1, y1)).tolist()
	layout = [ (i[0], i[1]) for i in c_list]
	c_list = np.exp(-a * distance.cdist(np.array(c_list), np.array(c_list)))
	adjcency = 1*(np.random.sample((len(c_list),len(c_list))) < c_list + c_list.T)
	g = ig.Graph.Adjacency(adjcency.tolist())
	g.vs['pos'] = layout
	g = g.clusters().giant()
	p = np.array(layout).T
	
	if plot==1:
		G = nx.DiGraph(g.get_edgelist())
		pos = {i:layout[i] for i in range(len(layout))}

		dendrogram = g.community_edge_betweenness()
		clusters = dendrogram.as_clustering()
		m = g.modularity(clusters)
		membership = clusters.membership
		partition = {i:membership[i] for i in range(len(membership))}

		nx.draw_networkx_nodes(G, pos,  node_size = 20, p=plt.cm.RdYlBu, node_color=list(partition.values()), alpha=0.8)
		nx.draw_networkx_edges(G, pos, alpha=0.1)
		plt.title("Modularity=%f"%m)
		plt.scatter(p[0], p[1], alpha=0.3)
		plt.savefig('c_gaussianas.pdf', fmt='pdf')
		     
	return g, layout


def Poisson_point_process(r, n, plot=False):
	n = np.random.poisson(n)
	area = np.pi*r*r
	lamb = n/area
	ratio = r
	u1 = np.random.uniform(0,1, n)
	u2 = np.random.uniform(0,1, n)
	radi = r*np.sqrt(u1)
	theta = 2*np.pi*u2
	x = [radi[i] * np.cos(theta[i]) for i in range(n)]
	y = [radi[i] * np.sin(theta[i]) for i in range(n)]
	pos = {i:(x[i],y[i]) for i in np.arange(n)}

	points = np.column_stack((x,y))
	a = 1/np.sqrt(area)
	waxman = np.triu(np.exp(-a*30 *distance.cdist(points, points)), 1)
	adjcency = 1*(np.random.sample((len(waxman),len(waxman))) < waxman )
	adjcency = adjcency + adjcency.T
	G = ig.Graph.Adjacency(adjcency.tolist())
	for i in range(n):
		G.vs[i]['pos'] = pos[i]
	G = G.clusters().giant()
	if plot==True:
		g = nx.DiGraph(ig.Graph.Adjacency(adjcency.tolist()).get_edgelist())
		fig, ax = plt.subplots()
		nx.draw_networkx_nodes(g, pos,  node_size = 10, ax=ax)
		nx.draw_networkx_edges(g, pos, alpha=0.5, ax=ax)
		ax.scatter(x,y, alpha=0.4)
		circ = plt.Circle((0, 0), radius=ratio, color='r', linewidth=2, fill=False)
		ax.add_artist(circ)
		return G, lamb
	else:
		return G, lamb
	
def community_space(n, area):
	"""	
	xcp = np.random.uniform(-area, area, n)
	ycp = np.random.uniform(-area, area, n)
	"""
	xcp = np.array([-3.5, -3.5, 3.5, 3.5]) 
	ycp = np.array([-3.5, 3.5, -3.5, 3.5])
	
	
	
	radis = np.random.normal(1.5, size=n)

	graphs = [Poisson_point_process(radis[i], 1000) for i in range(n)]
	graphs.append(Poisson_point_process(area, 1000))
	pos1 = [np.array(i[0].vs['pos']).T for i in graphs]
	for i in range(n):
		pos1[i][0] += xcp[i]
		pos1[i][1] += ycp[i]
	pos_all = [[], []]
	for i in pos1:
		pos_all[0] += list(i[0])
		pos_all[1] += list(i[1])
	lambs = np.array(graphs).T[1]
	interval = [0] + [i[0].vcount() for i in graphs]
	pos_all = np.column_stack((pos_all[0], pos_all[1]))
	dist = distance.cdist(pos_all, pos_all)

	adjacency = []
	acum_interval = np.trim_zeros([np.sum(interval[0:1+i]) for i in range(len(interval))])
	idx = 0
	for i, j in zip(dist, range(acum_interval[-1])) :
		if j < acum_interval[idx]:
			alpha = lambs[idx]
		else:
			idx += 1
		aux = np.exp(-np.sqrt(alpha)*1.4*i)
		aux = (np.random.sample((1,acum_interval[-1])) < aux) * 1
		adjacency.append(aux[0].tolist())
	adjacency = np.triu(adjacency, 1)
	adjacency = adjacency + adjacency.T
	g = ig.Graph.Adjacency(adjacency.tolist())
	
	pos = {i:(pos_all[i][0], pos_all[i][1]) for i in range(len(pos_all))}
	for i in range(len(pos)):
		g.vs[i]['pos'] = pos[i]
	g = g.clusters().giant()
	g['centers'] = list(zip(xcp, ycp))
	g['radius'] = radis
	centers = find_centers(g)
	g['central_points']=centers

	return g
"""
def find_communities(g):
	G = nx.DiGraph(g.get_edgelist())
	pos = {i:g.vs['pos'][i] for i in range(g.vcount())}

	dendrogram = g.community_edge_betweenness()
	clusters = dendrogram.as_clustering()
	m = g.modularity(clusters)
	membership = clusters.membership
	partition = {i:membership[i] for i in range(len(membership))}

	nx.draw_networkx_nodes(G, pos,  node_size = 20, p=plt.cm.RdYlBu, node_color=list(partition.values()), alpha=0.8)
	nx.draw_networkx_edges(G, pos, alpha=0.1)
	plt.savefig('Comunidades_espaciais.pdf', fmt='pdf')
"""
def find_communities(g):
	dendrogram = g.community_edge_betweenness()
	clusters = dendrogram.as_clustering()
	m = g.modularity(clusters)
	membership = clusters.membership
	return membership

	ig.plot(graph, 'test.png', layout =g.vs['pos'] , vertex_color=membership, vertex_size = 10)

	 

def write_xnet(g):
	lv = '#vertices %d nonweighted\n'%g.vcount()
	le = '#edges nonweighted\n'
	ve = ''.join(["'%d'\n"%i for i in range(g.vcount())])
	ed = g.get_edgelist()
	ed = ''.join(['{0:1d}{1:2d}\n'.format(i[0],i[1]) for i in ed])
	f = open('temporary.xnet','w')
	f.write(lv + ve + le + ed)
	f.close()
	return 'temporary.xnet'

def plot_points(g, colors='b', alpha=0.2):
	x = np.array(g.vs['pos'])
	y = x.T[1]
	x = x.T[0]
	plt.scatter(x, y, alpha=0.2, color=colors)
	c = find_centers(g)
	c = np.array([list(i.values()) for i in c])
	y = c.T[1]
	x = c.T[0]
	plt.scatter(x,y, alpha=1, color='k')
	

def find_centers(g):
	n = g.vcount()
	x = np.array(g.vs['pos'])
	y = x.T[1]
	x = x.T[0]
	centers = np.array(g['centers']).T
	radius = np.array(g['radius'])
	dic_list = []	
	for i in range(len(radius)):
		x_p = np.abs(x - centers[0][i]) < radius[i]/10
		y_p = np.abs(y - centers[1][i]) < radius[i]/10
		cen_pos = {j:(x[j], y[j])  for j in range(n)  if x_p[j] and y_p[j]}
		i = 1
		while len(cen_pos.keys()) == 0:
			x_p = np.abs(x - centers[0][i]) < radius[i]/10*i
			y_p = np.abs(y - centers[1][i]) < radius[i]/10*i
			cen_pos = {j:(x[j], y[j])  for j in range(n)  if x_p[j] and y_p[j]}
			print('trying to find centers')
			i += 0.5
		c = np.random.choice(list(cen_pos.keys()))
		dic_list.append({c:cen_pos[c]})

	return dic_list

def coloring_centers(g, dic_list):
	c_v = []
	for i in dic_list:
		c_v += i.values()
	colors = ['k' if i in c_v else 'gray' for i in g.vs['pos']]
	alpha = [1 if i=='k' else 0.1 for i in colors]
	return colors, alpha

def accessibility(g):
	xnet_file = write_xnet(g)
	out_str = 'out.txt'
	out = open(out_str, 'w')
	os.system('./CVAccessibility temporary.xnet out.txt')
	out.close()
	acc_list = np.loadtxt('%s'%out_str)
	os.remove('temporary.xnet')
	os.remove('out.txt')
	return acc_list


def coloring_hierarchical_degree(g):
	keys = [list(i.keys())[0] for i in g['central_points']]
	neighbors = [np.array(g.shortest_paths_dijkstra(i)) for i in keys]
	for i in range(1, max(neighbors[0][0])):
		aux = (neighbors[0] <= i)*1
		aux += (neighbors[1] <= i)*1
		aux += (neighbors[2] <= i)*1
		aux += (neighbors[3] <= i)*1
		col = aux
		col = ['b' if i==0 else 'k' for i in col[0]]
		plot_points(g, col)
		plt.savefig('/home/bilu/Dropbox/IC/IC2/Community_plots/Color_hierarchical_Degree/%d.png'%i)

def distance_to_centers(g):
	d = []
	centers = [list(i.keys())[0] for i in g['central_points']]
	for i in range(g.vcount()):
		d.append([])
		for j in centers:
			d[i].append(g.shortest_paths_dijkstra(i,j)[0][0])
	return d

def coloring_by_proximiy(g, d=0, plot=True):
	if d==0:
		d = distance_to_centers(g)
	else:pass
	col = []
	color_list = ['red', 'blue', 'green', 'yellow', 'gray', 'purple']
	for i in d:
		col.append(color_list[np.argmin(i)])
	if plot==True:
		plot_points(g, col)
		return col
	else:
		return col
	
	
def min_distance_distribution(g):
	centers = [list(i.keys())[0] for i in g['central_points']]
	color_list = ['r', 'b', 'green', 'y']
	fig, ax = plt.subplots(4)
	for j, k, w in zip(centers, color_list, range(4)):
		l = g.shortest_paths_dijkstra(target=j)
		hist, bins = np.histogram(l, bins=20, density=True)
		ax[w].plot(bins[:-1], hist, color=k, marker='o')

	plt.savefig('min_distante_dist%s.pdf'%k, fmt='pdf')


def plot_networkx(g, col):
	layout = g.vs['pos']
	ig.plot(g, 'fig1.png', layout=layout, vertex_color=col, bbox=(1000,1000))
	 

def PCA_distance_vector(g, plot_network=False):
	d = distance_to_centers(g)
	pca = PCA(n_components=2)
	col = coloring_by_proximiy(g, d, plot=False)
	reduced_data = pca.fit(d).transform(d)
	plt.scatter(reduced_data.T[0], reduced_data.T[1], color=col)
	variance_rate = pca.explained_variance_ratio_
	plt.xlabel('%.2f'%variance_rate[0], fontsize=15)
	plt.ylabel('%.2f'%variance_rate[1], fontsize=15)
	plt.title('PCA Vetores de distância')
	if plot_network==True:
		plot_networkx(g, col)
	else:pass


	
	


















