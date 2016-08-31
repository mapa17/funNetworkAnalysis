"""
funNetworkAnalysis - peform a functional network analysis of CFOS expression data

@author:     Pasieka Manuel , manuel.pasieka@vbcf.ac.at

@copyright:  2016 BioComp/VBCF. All rights reserved.

@license:    MIT

@contact:    manuel.pasieka@vbcf.ac.at
"""
import logging
import time
import sys
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os.path
from scipy.stats.mstats import zscore
import scipy as sp
import seaborn as sns
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import configparser
from collections import OrderedDict
from pudb import set_trace as st
import tools
import plotting

PROJECT_NAME = 'funNetworkAnalysis'
__version__ = 1.0
__date__ = '2016-August-22'
__updated__ = 'XXXX-XX-XX'

__logfile__ = 'funNetworkAnalysis.log'
__logfilelevel__ = logging.DEBUG

def buildCorrelationMatrix(T, testG, treatmentG, generatePlots=False, outputDir='/tmp', dpi=100):
    """
    Calculate the pearson correlation between each injection region.
    Two groups are defined testG, and treatmentG
    If generatePlots=True a correlation plot will be generate for the two groups
    """
    N = len(T.index.levels[0])
    testCorr = np.empty(shape=(N, N))
    treatCorr = np.empty(shape=(N, N))
    for i1, r1 in enumerate(T.index.levels[0]):
        for i2, r2 in enumerate(T.index.levels[0]):
            testCorr[i1, i2] = sp.stats.pearsonr(T.loc[([r1], testG), :], T.loc[([r2], testG), :])[0][0]
            treatCorr[i1, i2] = sp.stats.pearsonr(T.loc[([r1], treatmentG), :], T.loc[([r2], treatmentG), :])[0][0]

    if generatePlots:
        mask = np.zeros(shape=testCorr.shape)
        mask[np.triu_indices_from(testCorr)] = True
        regions = T.index.levels[0].tolist()

        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(treatCorr, ax=ax, annot=True, xticklabels=regions, yticklabels=regions, mask=mask, cmap=plt.cm.RdYlGn)
        f.suptitle('Drug Correlation Matrix')
        path = os.path.join(os.path.abspath(outputDir), 'DrugCM.png')
        logging.info('Writing Correlation Matrix Plot to %s' % (path))
        f.savefig(path, dpi=dpi)
        plt.close(f)

        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(testCorr, ax=ax, annot=True, xticklabels=regions, yticklabels=regions, mask=mask, cmap=plt.cm.RdYlGn)
        f.suptitle('Saline Correlation Matrix')
        path = os.path.join(os.path.abspath(outputDir), 'NoDrugCM.png')
        logging.info('Writing Correlation Matrix Plot to %s' % (path))
        f.savefig(path, dpi=dpi)
        plt.close(f)

        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(treatCorr - testCorr, ax=ax, annot=True, xticklabels=regions, yticklabels=regions, mask=mask, cmap=plt.cm.RdYlGn)
        f.suptitle('Treatment - Test (CM)')
        path = os.path.join(os.path.abspath(outputDir), 'DiffCorrelationMatrix.png')
        logging.info('Writing Difference correlation Matrix Plot to %s' % (path))
        f.savefig(path, dpi=dpi)
        plt.close(f)

    return pd.DataFrame(data=testCorr, index=T.index.levels[0], columns=T.index.levels[0]), pd.DataFrame(data=treatCorr, index=T.index.levels[0], columns=T.index.levels[0])

def normalizeMatrix(T, axis=0):
    return T.divide(T.sum(axis=1-axis), axis=0+axis)

def getSubNetwork(completeNetwork, nodeList, includeNeighbours=False, normalizeAxis=0):
    srcNodes = []
    destNodes = []
    injRegions = completeNetwork.index.tolist()
    targetRegions = completeNetwork.columns.tolist()

    #Build a list of injection and target nodes to keep
    for node in nodeList:
        if node in injRegions:
            srcNodes.append(node)
        if node in targetRegions:
            destNodes.append(node)
            if includeNeighbours:
                srcNodes += injRegions

    srcNodes = np.unique(srcNodes)
    destNodes = np.unique(destNodes)
    subN = completeNetwork.loc[srcNodes, destNodes]
    subN.index.name = 'Nodes'
    subN = subN.groupby(level=0).median()

    if normalizeAxis is not None:
        #Normalize the new subnetwork
        normalizeAxis = int(normalizeAxis)
        subN = normalizeMatrix(subN, axis=normalizeAxis)

    return subN

def genGraph(M, edgeWeigthThreshold=0.0, directedGraph=True):
    if directedGraph:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(set(M.index.tolist()) | set(M.columns.tolist()))
    M.apply(lambda x: tools.addEdges(G, x, threshold=edgeWeigthThreshold))
    return G

def readConnectum(aBAPath, mappingPath, normalizeAxis=0, aggregation_function='median'):
    """
    Read the connectum from the allen brain atlas
    Calculate the median for each injection Region
    and apply a row wise normalization (outgoing edge normalized)
    """
    try:
        nM = pd.read_csv(mappingPath)
        fC = pd.read_csv(aBAPath)
        #Map Injection sites and target sites
        fC = tools.replaceInColumn(fC, nM, 0)
        fC = tools.renameColumns(fC, nM)

        #Set primary injection site to index, and remove secondary injection site
        C = fC.copy()
        C.drop(C.columns[1], axis=1, inplace=True)
        C.set_index(C.columns[0], inplace=True)

        #Filter all target sites that are either primary inj. or seconday inj sites
        for row in fC.iterrows():
            idx = row[0]
            pInj = row[1][0]
            try:
                sInj = row[1][1].split('|')
            except AttributeError:
                sInj = []
            sInj.append(pInj)
            sInj = [X for X in sInj if X in C.columns]
            C.ix[idx, sInj] = np.nan
            #logging.debug('Setting NaN %d:%s', idx, sInj)

        if aggregation_function == 'median':
            conn = C.groupby(level=0).median()
        else:
            conn = C.groupby(level=0).mean()

        if normalizeAxis is not None:
            return normalizeMatrix(conn, axis=normalizeAxis)
        else:
            return conn

    except Exception as e:
        logging.error('Reading allen brain atlas failed! %s', e)

def prepareCFOSTable(T, filterHomeCage=True, aggregation_function='mean'):
    """
    Load cFos data, first generating different experiment groups based on columns ['shock', 'EPM', 'CTB']
    Calculate the median cfos expression value for all regions of the same mouse
    (there are multiple samples for the same region and mouse) and than calculate the median
    for each region.
    At last calculate a zscore for the expression rates inside the groups testGroup and treatmentGroup
    """
    #Drop all EPM x2 rows
    T = T.query('EPM != "EPM x2"').copy()

    #Generate experiment column in oder to differenciate between samples
    T['shock'].replace(['no', 'yes'], ['', 'shock'], inplace=True)
    T['EPM'].replace(['no', 'yes'], ['', 'EPM'], inplace=True)
    T['CTB'].replace(['no', 'yes'], ['', 'CTB'], inplace=True)
    T['Injection'].replace(['no'], ['noInjection'], inplace=True)
    T.loc[:, 'experiment'] = T[['Injection', 'shock', 'EPM']].apply(lambda x: '_'.join([ v for v in x.values if v != '']), axis=1)

    #Filter all samples that have less than two regions
    T = T[T.nRegions > 1]

    #Calculate median inside mouse an than for each region
    reducedTable = T[['Mouse.ID', 'experiment', 'Region', 'cfos/DAPI.Percent']]
    if aggregation_function == 'median':
        reducedTable = reducedTable.groupby(['Region', 'experiment', 'Mouse.ID']).median().groupby(level=[0, 1]).median()
    else:
        reducedTable = reducedTable.groupby(['Region', 'experiment', 'Mouse.ID']).mean().groupby(level=[0, 1]).mean()

    #Apply zscore to each Region, and for treatment and test group
    #treatment = reducedTable.loc[(reducedTable.index.levels[0], treatmentGroup), :].groupby(level=[0]).apply(zscore)
    #test = reducedTable.loc[(reducedTable.index.levels[0], testGroup), :].groupby(level=[0]).apply(zscore)
    #reducedTable.loc[(reducedTable.index.levels[0], treatmentGroup), :] = np.concatenate(np.concatenate(treatment))
    #reducedTable.loc[(reducedTable.index.levels[0], testGroup), :] = np.concatenate(np.concatenate(test))

    if filterHomeCage:
        #Filter noInjection Experiments
        reducedTable = reducedTable.loc[pd.IndexSlice[:, reducedTable.index.levels[1].difference(['noInjection'])], :]

    #Remove garbage row XXX
    reducedTable.drop('XXX', inplace=True)
    reducedTable = reducedTable.reset_index().set_index(['Region', 'experiment']) #Bug in pandas, XXX is still an index

    #zscore all groups together
    reducedTable.loc[:, reducedTable.columns[0]] = np.concatenate(np.concatenate(reducedTable.groupby(level=[0]).apply(zscore)))

    return reducedTable

def readCFOS(cfosTablePath, mappingPath, aggregation_function, filterHomeCage=True):
    cT = pd.read_csv(cfosTablePath)
    nM = pd.read_csv(mappingPath)
    cFos = tools.replaceInColumn(cT, nM, 'Region')
    cFos = prepareCFOSTable(cFos, filterHomeCage=filterHomeCage, aggregation_function=aggregation_function)
    return cFos

def anlayseProjectionSource(connectome, regions, outputPath='/tmp/ProjectionAnalysis.png', dpi=100):
    """
    Analyse for each region of the reduced connectome how many projections
    are comming from other regions in the reduced connectome or from outside.
    """
    inNormalized = normalizeMatrix(connectome, axis=1)
    outNormalized = normalizeMatrix(connectome, axis=0)

    localInput = inNormalized.loc[regions, regions].sum()
    localOutput = outNormalized.loc[regions, regions].sum(axis=1) * (-1.0)

    order = np.argsort(localInput.values)
    regs = []
    inS = []
    outS = []
    for idx in reversed(order):
        r = regions[idx]
        regs.append(r)
        inS.append(localInput[r] * 100.0)
        outS.append(localOutput[r] * 100.0)

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    sns.barplot(regs, inS, palette='Set3', ax=ax1)
    ax1.set_ylabel('Local Input [%]')
    sns.barplot(regs, outS, palette='Set3', ax=ax2)
    ax2.set_ylabel('Local Output [%]')

    #Make the xaxis visible between the plots
    plt.setp(ax1.get_xticklabels(), visible=True)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.tight_layout(h_pad=0)
    f.suptitle('Projection Analysis')
    logging.info('Writing plot to %s', outputPath)
    f.savefig(outputPath, dpi=dpi)
    plt.close(f)

def anlayseCorrelationNeighborhood(node, G, weight=False, attrName='weight', average=True):
    """
    Perform an analysis of the Graph G, summing all attributes of name attrName for each node and its K=1 neighbourhood.
    If weight=True the attributes are scaled by the edge strengh between the nodes
    If average=True take the average vale of the sum
    If the graph is directed, perform the analysis separate for the inbound and outbound edges

    weight ... takes care of giving more importance to strongly connected neighbours
    average ... makes nodes compareable even if their degree varies
    """
    if G.is_directed():
        inSum = 0
        outSum = 0

        nElements = 0
        for u, v, data in G.out_edges(node, data=True):
            try:
                if weight:
                    outSum = outSum + data[attrName] * data['weight']
                else:
                    outSum = outSum + data[attrName]
                nElements = nElements + 1
            except KeyError:
                logging.warning('Edge %s-%s has no weight attribute!', u, v)
        try:
            if average:
                outSum = outSum/nElements
        except ZeroDivisionError:
            outSum = np.nan
        noutEdges = nElements

        nElements = 0
        for u, v, data in G.in_edges(node, data=True):
            try:
                if weight:
                    inSum = inSum + data[attrName] * data['weight']
                else:
                    inSum = inSum + data['weight']
                nElements = nElements + 1
            except KeyError:
                logging.warning('Edge %s-%s has no weight attribute!', u, v)

        try:
            if average:
                inSum = inSum/nElements
        except ZeroDivisionError:
            outSum = np.nan
        ninEdges = nElements

        score = (inSum, outSum)
    else:
        score = 0
        nElements = 0
        for u, v, data in G.edges(node, data=True):
            try:
                if weight:
                    score = score + data[attrName] * data['weight']
                else:
                    score = score + data['weight']
                nElements = nElements + 1
            except KeyError:
                logging.warning('Edge %s-%s has no weight attribute!', u, v)
        try:
            if average:
                score = score / nElements
        except ZeroDivisionError:
            score = np.nan

    return score

def nodeAnalysis(G, nodeFunction, resultFunction, nodeArguments={}, resultArguments={}):
    node_results = {}
    for node in G.nodes():
        node_results[node] = nodeFunction(node, G, **nodeArguments)

    post_results = resultFunction(node_results, G, **resultArguments)
    return node_results, post_results

def export_cfos_data(cfos, outDir, csv_file='cfosExpression.csv', clustering_figure='Clustering.png', cluster_metric='cityblock', dpi=100):
    #Export cfos data as csv table
    T = cfos.reset_index()
    T = T.pivot(index='experiment', columns='Region', values='cfos/DAPI.Percent')
    logging.info('Writing cfos expression values to cfosExpression.csv')
    T.to_csv(outDir + csv_file)

    #In addition peform hierahical clustering of the data and export the results as a heatmap
    g = sns.clustermap(T.T, linewidths=.5, figsize=(11,11), z_score=0, col_cluster=True, row_cluster=True, annot=True, metric=cluster_metric)
    logging.info('Writing cluster plot to Clustering.png ...')
    g.savefig(outDir + clustering_figure, dpi=dpi)

    #Manualy sort the heatmap columns (only valid if the previous hierahical tree is conserved, valid for OUR data and cityblock metric!)
    column_order = ['noInjection', 'DZP', 'DZP_shock', 'DZP_EPM', 'saline', 'DZP_shock_EPM', 'saline_shock', 'saline_EPM', 'saline_shock_EPM']
    plotting.plot_ordered_heatmap_from_clustering(g, column_order, outfig=outDir + 'sorted_' + clustering_figure, dpi=dpi)


def neighbourhood_correlation_analysis(G, testCM, treatCM, pos, directional=False, weight=True, average=True, sum_absolute=True, outDir='.', dpi=100):
    """
    Perform a neighbourhood correlation analysis on the graph G
    """
    # Add to the edge of the graph the correlation data
    G = tools.updateEdges(G.to_directed(), treatCM, 'treat_correlation')
    G = tools.updateEdges(G, testCM, 'test_correlation')
    G = tools.updateEdges(G, treatCM - testCM, 'effect_correlation')

    # Perform a node correlation analysis for each node (for each node sum up the correleation values for each edge; for ingoing and outgoing edges separatly)
    cS, _ = nodeAnalysis(G, anlayseCorrelationNeighborhood, plotting.plotNodeBars, nodeArguments={'weight':weight, 'attrName':'treat_correlation', 'average':average}, resultArguments={'ylabel':['Avg. Sum Input Correlation', 'Avg. Sum Out Correlation'], 'title':'Test Neighborhood Correlation', 'outPath':outDir + 'TestCorrelationSum.png', 'dpi':dpi})
    cS, _ = nodeAnalysis(G, anlayseCorrelationNeighborhood, plotting.plotNodeBars, nodeArguments={'weight':weight, 'attrName':'test_correlation', 'average':average},resultArguments={'ylabel':['Avg. Sum Input Correlation', 'Avg. Sum Out Correlation'], 'title':'Treatment Neighborhood Correlation', 'outPath':outDir + 'TreatmentCorrelationSum.png', 'dpi':dpi})
    cS, _ = nodeAnalysis(G, anlayseCorrelationNeighborhood, plotting.plotNodeBars, nodeArguments={'weight':weight, 'attrName':'effect_correlation', 'average':average}, resultArguments={'ylabel':['Avg. Sum Input Correlation', 'Avg. Sum Out Correlation'], 'title':'[Treat - Test] Neighborhood Correlation', 'outPath':outDir + 'DrugCorrelationSum.png', 'dpi':dpi})

    logging.info('Performing a network correlation analysis for inbound and outgoing edges ...')
    G = tools.updateNodes(G, dict(zip(cS.keys(), [x[0] for x in cS.values()])), 'effect_correlation_sum_input')
    G = tools.updateNodes(G, dict(zip(cS.keys(), [x[1] for x in cS.values()])), 'effect_correlation_sum_output')

    # Plot the graph, color coding the nodes depending on the effective correlation sum
    # Make sure the node color scale is the same for both plots
    scI = [nAttr['effect_correlation_sum_input'] for nName, nAttr in G.nodes(data=True)]
    scO = [nAttr['effect_correlation_sum_output'] for nName, nAttr in G.nodes(data=True)]
    nmin = np.round(min(min(scI), min(scO)), decimals=2)
    nmax = np.round(max(max(scI), max(scO)), decimals=2)
    pos = plotting.plotGraph(G, title='Neighborhood Correlation Sum Analysis (Input Edges)', outputDir=outDir, edge_color=None, edge_cmap=plt.cm.Blues, arrows=False, node_cmap=plt.cm.RdYlGn, pos=pos, node_attr='effect_correlation_sum_input', node_colorbar_title='Phi', nv_limits=(nmin, nmax), dpi=dpi)
    pos = plotting.plotGraph(G, title='Neighborhood Correlation Sum Analysis (Output Edges)', outputDir=outDir, edge_color=None, edge_cmap=plt.cm.Blues, arrows=False, node_cmap=plt.cm.RdYlGn, pos=pos, node_attr='effect_correlation_sum_output', node_colorbar_title='Phi', nv_limits=(nmin, nmax), dpi=dpi)

    # Optional calculate the sum of the correlation of ingoing and outgoing edges
    table = tools.get_nodes_as_table(G)
    if not directional:
        if sum_absolute:
            logging.info('Using absolute values to sum up correlation values!')
            table['total_effect_correlation_sum'] = table['effect_correlation_sum_input'].abs() + table['effect_correlation_sum_output'].abs()
        else:
            logging.info('Using relative values to sum up correlation values!')
            table['total_effect_correlation_sum'] = table['effect_correlation_sum_input'] + table['effect_correlation_sum_output']

            logging.debug('Generating a bar plot containing the total of the effective correlation changes to %s ...', outDir + 'Total_Correlation_Effect')
            plotting.plotNodeBars(table.to_dict()['total_effect_correlation_sum'], None, xlabel='Regions', ylabel=None, title='Total Correlation Effect', outPath=outDir + 'Total_Correlation_Effect.png', dpi=dpi)

    logging.debug('Writing neighborhood_correlation_analysis results to %s ...', outDir + 'neighborhood_correlation_analysis.csv')
    table.to_csv(outDir + 'neighborhood_correlation_analysis.csv')


def helper_args_and_cfg():
    try:
        parser = createArgumentParser()
        try:
            args = parser.parse_args()
        except SystemExit:
            logging.error('Parsing arguments failed!')
            raise

        if args.verbosity >= 2:
            consoleloglevel = logging.DEBUG
        elif args.verbosity >= 1:
            consoleloglevel = logging.INFO
        else:
            consoleloglevel = logging.WARN

        setupLogging(args.logfile, __logfilelevel__, consoleloglevel)

        logging.info(format('Starting %s v%s on the %s...'
                            % (PROJECT_NAME, __version__, time.ctime())))

        logging.info(format("Reading config file [%s]" % (args.config)))
        cfg = readConfigFile(args.config, PROJECT_NAME)

        return args, cfg
    except Exception as e:
        sys.exit(1)


####################################  MAIN #####################################

def main(argv=None):
    # Get Arguments and config
    args, cfg = helper_args_and_cfg()

    # Helper. Store all output data inside the output directory
    outDir = args.outputDirectory + os.path.sep

    # Write csv containing the cfos data that will be used for the analysis (containing the HomeCage data)
    fos = readCFOS(args.CFOSExpression, args.CFOSExpressionMapping, cfg['cfos_aggregation'],  filterHomeCage=False)
    export_cfos_data(fos, outDir, csv_file='cfosExpression.csv', clustering_figure='Clustering.png', cluster_metric=cfg['cluster_metric'], dpi=cfg['dpi'])

    # Read cfos data without homecage
    fos = readCFOS(args.CFOSExpression, args.CFOSExpressionMapping, cfg['cfos_aggregation'], filterHomeCage=True)
    regionsOI = fos.index.levels[0].tolist()

    # Read complete connectome
    connectome = readConnectum(args.connectome, args.connectomeMapping, normalizeAxis=None, aggregation_function=cfg['connectome_aggregation'])

    # Generate the structural graph out of the allen brain atlas data, apply normalization, write the subgraph as csv table
    M = getSubNetwork(connectome, regionsOI, includeNeighbours=False, normalizeAxis=cfg['connectome_normalization'])
    logging.info('Exporting reduced connectome to ReducedConnectom.csv ...')
    M.to_csv(outDir + 'ReducedConnectum.csv')

    # Analyse incoming projection strength of ROIs
    anlayseProjectionSource(connectome, regionsOI, outputPath = outDir + 'ProjectionAnalysis.png')

    # Graph representation of the reduced connectome, and plot it, showing projection strength
    G = genGraph(M, 0.0)
    pos = plotting.plotGraph(G, title='Regions of Interest', outputDir=outDir, edge_color=None, edge_cmap=plt.cm.Blues, arrows=False)
    plotting.generateCFOSPlots(fos, G, outDir, pos=pos, dpi=cfg['dpi'])

    # Calculate correlation values for each region depending on the cfos activity between the treatment and test state
    testCM, treatCM = buildCorrelationMatrix(fos, ['saline', 'saline_EPM', 'saline_shock', 'saline_shock_EPM'], ['DZP', 'DZP_EPM', 'DZP_shock', 'DZP_shock_EPM'], generatePlots=True, dpi=cfg['dpi'], outputDir=outDir)

    # Add correlation values to the network graph (combinig structural and functional data)
    testG = tools.updateEdges(G.to_undirected(), testCM, 'weight')
    treatG = tools.updateEdges(G.to_undirected(), treatCM, 'weight')
    plotting.plotCorrelationGraph(testG, title='Correlation under Test', pos=pos, outputPath=outDir + 'TestCorrelationGraph.png', dpi=cfg['dpi'])
    plotting.plotCorrelationGraph(treatG, title='Correlation under Treatment', pos=pos, outputPath=outDir + 'TreatmentCorrelationGraph.png', dpi=cfg['dpi'])

    # Analyse the correlation neighbourhood for each region
    neighbourhood_correlation_analysis(G, testCM, treatCM, pos,
        directional=cfg['directional_correlation_analysis'],
        weight=cfg['weight_edges'],
        average=cfg['average_node'],
        sum_absolute=cfg['correlation_sum_absolute'],
        outDir=outDir, dpi=cfg['dpi'])

    logging.info('Finished funtional network analysis!')
    sys.exit(0)

##### Helper Code #####

def helperIsFile(parser, filename):
    fName = os.path.abspath(filename)
    if os.path.isfile(fName):
        return fName
    else:
        parser.error('%s is no file' % (fName))

def helperIsDir(parser, path):
    fName = os.path.abspath(path)
    if os.path.isdir(fName):
        return fName
    else:
        parser.error('Cannot read from folder %s. Error %s' % (fName, e))

def checkOutputDir(parser, path, create=True):
    fName = os.path.abspath(path)
    if os.path.isdir(fName):
        return fName
    else:
        if create:
            try:
                os.mkdir(fName)
                return fName
            except Exception as e:
                parser.error('Cannot create %s. Error %s' % (fName, e))

def createArgumentParser():
    """
    Create an Argument Parser
    """
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

      Created by Pasieka Manuel on %s.
      Copyright 2016 BioComp. All rights reserved.

      Distributed on an "AS IS" basis without warranties
      or conditions of any kind, either express or implied.

    ''' % (program_shortdesc, str(__date__))

    # Setup argument parser
    parser = ArgumentParser(prog=PROJECT_NAME,
                            description=program_license,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('connectome',
                        help="CSV file containing the connectome",
                        type=lambda x: helperIsFile(parser, x))

    parser.add_argument('CFOSExpression',
                        help="CSV file containing the CFOS Expression data",
                        type=lambda x: helperIsFile(parser, x))

    parser.add_argument('connectomeMapping',
                        help="CSV file containing a mapping of region names from connectome to internal naming",
                        type=lambda x: helperIsFile(parser, x))

    parser.add_argument('CFOSExpressionMapping',
                        help="CSV file containing a mapping of region names from CFOS expression data to internal naming",
                        type=lambda x: helperIsFile(parser, x))

    parser.add_argument('outputDirectory',
                        help="path to output folder",
                        type=lambda x: checkOutputDir(parser, x))

    parser.add_argument('-c', '--config', default='config.ini',
                        help="Path to config file",
                        type=lambda x: helperIsFile(parser, x))

    parser.add_argument('-L', '--logfile', default=__logfile__,
                        help="Log file")
    parser.add_argument("-v", "--verbosity", action='count', default=0,
                        help="increase output verbosity. Use -vv, -vvv for more")
    parser.add_argument('--version', action='version', version=program_version_message)

    return parser


def setupLogging(logfile, logfilelevel, logconsolelevel):
    """
    Define python.logging to file and console
    """
    #Normal settings
    #logformat = '[%(levelname)s] %(module)s.%(funcName)s : %(message)s'

    #Debugging settings
    logformat = '[%(levelname)s] \'%(filename)s +%(lineno)d\' %(funcName)s : %(message)s'

    #Setup logging to file
    logging.basicConfig(level=logfilelevel, format=logformat,
                        filename=logfile, filemode="w")

    #Add logging to the console
    console = logging.StreamHandler()
    console.setLevel(logconsolelevel)
    console.setFormatter(logging.Formatter(logformat))
    logging.getLogger('').addHandler(console)


def readConfigFile(fileName, tagName):
    """
    Read python config file using ConfigParser
    Will read all options under the region "tagName" and if possible convert a
    value into float.
    """
    logging.debug(format('Reading section %s from config file [%s] ...' % (tagName, fileName)))
    Config = configparser.ConfigParser(allow_no_value=True)
    Config.optionxform = str  # Keeps options to be all set to lowercases
    Config.read_file(open(fileName))
    options = Config.options(tagName)

    parameters = dict()
    for option in options:
        try:
            parameters[option] = Config.get(tagName, option)
        except KeyError:
            logging.error(format("Exception on config file option [%s]!" % option))
            parameters[option] = None

    oDict = OrderedDict()
    for k in sorted(parameters.keys()):
        v = parameters[k]
        if v == '':
            v = None
        elif v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        elif v.isnumeric():
            try:
                v = float(v)
            except ValueError:
                pass
        oDict[k] = v
    return oDict
    #return OrderedDict(sorted(parameters.items(), key=lambda t: t[0]))

if __name__ == "__main__":
    main(sys.argv)
