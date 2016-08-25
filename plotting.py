import logging
import sys
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os.path
import seaborn as sns

def plot_ordered_heatmap_from_clustering(g, column_order, outfig='/tmp/Johannes_clustering.png', dpi=100):
    table = g.data2d[column_order]
    fig = plt.figure(figsize=(11,11))
    ax = sns.heatmap(table, annot=True, fmt='.2g', linewidths=0.5, annot_kws={'size': 18})
    fig.add_axes(ax)

    logging.debug('Writing ordered heatmap to %s ...', outfig)
    fig.savefig(outfig, dpi=dpi)
    plt.close(fig)

def plotCorrelationGraph(G, title='Correlation plot', outputPath='/tmp/CorrelationGraph.png', pos=None, dpi=100):
    eW = []
    for src, dest in G.edges_iter():
        eW.append(G.get_edge_data(src, dest)['weight'])

    f, ax = plt.subplots(figsize=(8, 8))
    if not pos:
        pos = nx.spring_layout(G)
    node_axes = nx.draw_networkx_nodes(G, pos, ax=ax, with_labels=True, node_color='white', cmap='Greens', vmin=-1.0, vmax=1.0, node_size=1200, alpha=0.8)
    label_axes = nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='black')
    edge_axes = nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, edge_color=eW, edge_cmap=plt.cm.RdYlGn, edge_vmin=min(eW), edge_vmax=max(eW), width=5, alpha=0.8)
    plt.tight_layout()
    edgeStrengh = f.colorbar(edge_axes, orientation='vertical', shrink=0.5, pad=0.0, ticks=[min(eW), max(eW)])
    edgeStrengh.set_label('Pearson Correlation')

    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_bgcolor('white')
    path = os.path.abspath(outputPath)
    logging.info('Writing correlation graph to %s' % (path))
    f.suptitle(title)
    f.savefig(path, dpi=dpi)
    plt.close(f)

def plotGraph(G, title='Correlation plot', outputDir='/tmp', pos=None, node_color='white', node_cmap=None, edge_color='black', edge_cmap=None, edge_colorbar_title='Edge Strength', node_colorbar_title='Node Strength', arrows=True, edge_weight_attr='weight', node_attr=None, nv_limits=None, edge_width=1, dpi=100):
    f, ax = plt.subplots(figsize=(8, 8))
    if not pos:
        pos = nx.spring_layout(G)

    if edge_color is None:
        eW = []
        for src, dest in G.edges_iter():
            eW.append(G.get_edge_data(src, dest)[edge_weight_attr])
    else:
        eW = edge_color

    nvmin = -1
    nvmax = 1
    if node_attr is not None:
        nW = []
        for node, ndata in G.nodes(data=True):
            nW.append(ndata[node_attr])
        node_color=nW
        nvmin = np.round(min(nW), decimals=2)
        nvmax = np.round(max(nW), decimals=2)
    if nv_limits is not None:
        try:
            nvmin, nvmax = nv_limits
        except Exception as e:
            logging.error('Invalid nv_limits setting, falling back to defaults!')

    node_axes = nx.draw_networkx_nodes(G, pos, ax=ax, with_labels=True, node_color=node_color, cmap=node_cmap, vmin=nvmin, vmax=nvmax, node_size=1200, alpha=0.8)
    label_axes = nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='black')
    edge_axes = nx.draw_networkx_edges(G, pos, ax=ax, arrows=arrows, edge_color=eW, edge_cmap=edge_cmap, width=edge_width, alpha=0.8)
    plt.tight_layout()

    if edge_color is None:
        edgeStrength = f.colorbar(edge_axes, orientation='vertical', shrink=0.5, pad=0.0, ticks=[min(eW), max(eW)])
        edgeStrength.set_label(edge_colorbar_title)

    if node_cmap is not None:
        nodeStrength = f.colorbar(node_axes, orientation='horizontal', shrink=0.5, pad=0.0, ticks=[nvmin, nvmax])
        nodeStrength.set_label(node_colorbar_title)

    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_bgcolor('white')
    sns.despine(top=True, bottom=True, left=True, right=True)

    path = os.path.join(os.path.abspath(outputDir), title.replace(' ', '') +'.png')
    logging.info('Writing %s to %s' % (title, path))
    f.suptitle(title)
    f.savefig(path, dpi=dpi)
    plt.close(f)
    sns.despine()

    return pos

def plotNodeBars(values, G, xlabel='Regions', ylabel=None, title=None, outPath=None, dpi=100):
    yData = list(values.values())
    regions = list(values.keys())
    assert len(yData) > 0, 'no Data for Regions to plot!'

    #If yData is a tuple, or list, generate multiple barplots on top of each other
    if isinstance(yData[0], (tuple, list)):
        nPlots = len(yData[0])
        f, ax = plt.subplots(nPlots, 1, figsize=(8, 8), sharex=True)
    else:
        nPlots = 1
        #Pack the data into tuples so they can be accessed by indexing later
        yData = zip(yData, yData)
        f, ax = plt.subplots(nPlots, 1, figsize=(8, 8), sharex=True)
        ax = [ax]

    if ylabel is not None:
        if isinstance(ylabel, (tuple, list)):
            ylabels = ylabel
        else:
            ylabels = [ylabel] * nPlots
    else:
        ylabels = [''] * nPlots
    #if xlabel is not None:
    #    ax.set_xlabel(xlabel)
    if title is not None:
        f.suptitle(title)

    for pn in range(nPlots):
        #Rank order yData
        labels = []
        yT = [y[pn] for y in yData]
        y = []
        for i in np.argsort(yT):
            labels.append(regions[i])
            y.append(yT[i])

        sns.set_style('whitegrid')
        sns.barplot(labels, y, ax=ax[pn], palette='RdYlGn')
        ax[pn].set_ylabel(ylabels[pn])

    if outPath is not None:
        logging.info('writing model %s plot to %s ...', title, outPath)
        f.savefig(outPath, dpi=dpi)
        plt.close(f)
        sns.set_style()
        return None
    else:
        plt.show()
        return f

def generateCFOSPlots(fos, G, outputDir, pos=None, dpi=100):
    eW = []
    for src, dest in G.edges_iter():
        eW.append(G.get_edge_data(src, dest)['weight'])

    emin = min(eW)
    emax = max(eW)
    #Extract the cFos expression rates
    try:
        fosRates = {}
        for exp in fos.index.levels[1].unique().tolist():
            rates = []
            fosRegions = fos.index.levels[0]
            for region in G.nodes():
                if (region in fosRegions) and (exp in fos.loc[region].index.tolist()):
                    rates.append(fos.loc[(region, exp)].values[0])
                else:
                    #rates.append(np.nan)
                    rates.append(0.0)
            fosRates[exp] = rates
    except Exception as e:
        logging.error('Generating cFos expression table failed! %s', e)

    if not pos:
        pos = nx.spring_layout(G)
    #Generate Plots for each Experimental group
    for expGrp in fosRates.keys():
        pth = os.path.join(os.path.abspath(outputDir), '%s.png' % expGrp)
        logging.info('Generating graph %s', pth)
        f, ax = plt.subplots(figsize=(8, 8))
        nmax = max(fosRates[expGrp])
        nmin = min(fosRates[expGrp])
        node_axes = nx.draw_networkx_nodes(G, pos, ax=ax, with_labels=True, node_color=fosRates[expGrp], cmap='Greens', vmin=nmin, vmax=nmax, node_size=1200, alpha=1.0)
        label_axes = nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='black')
        edge_axes = nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, edge_color=eW, edge_cmap=plt.cm.OrRd, edge_vmin=emin, edge_vmax=emax, width=5, alpha=0.8)
        f.suptitle('%s' % (expGrp))
        plt.tight_layout()
        cfosEx = f.colorbar(node_axes, orientation='horizontal', shrink=0.5, pad=0.0, ticks=[nmin, 0.0, nmax])
        edgeStrengh = f.colorbar(edge_axes, orientation='vertical', shrink=0.5, pad=0.0, ticks=[min(eW), max(eW)])
        cfosEx.set_label('z-Scored CFOS expression')
        edgeStrengh.set_label('Projection strengh')

        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_axis_bgcolor('white')
        f.savefig(pth, dpi=dpi)
        plt.close(f)

    return pos
