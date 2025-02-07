#!/home/reynoa12/localbin/bin/python3.9


import pandas as pd
import numpy as np
import sklearn
import sklearn.impute
#from ppca import PPCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import scipy
from scipy.stats import pearsonr
from scipy.stats import linregress
import re
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import umap.umap_ as umap
import csv

MISSINGNESS_THRESH=1
SAMPLE_MISSINGNESS_THRESH=1
IMPUTE_METHOD=None

def trim_string(input_string):
    # Define the pattern to match special characters
    pattern = r'[^_a-zA-Z0-9\s]'
    # Use regex to substitute special characters with an empty string
    output_string=re.sub(pattern, '', input_string)
    return output_string

####################################################################################################
class Preprocess:
    """
    This contains functions for preprocessing of data:
    """
    def _drop_samples(self, sample_list):
        """ Exclude sample info for analysis """
        for sample in sample_list:
            for col in self.rawinput.columns:
                if sample in col:
                    self.rawinput= self.rawinput.drop(col,axis=1)

        for sample in sample_list:
            for index,row in self.mat.iterrows():
                if sample in row.name:
                    self.mat = self.mat.drop(index,axis=0)
                    self.sample=self.mat.index
            for index,row in self.metadata.iterrows():
                samplekey=row[row.index=='samplekey'].values
                if sample in samplekey[0]:
                    print(samplekey[0])
                    self.metadata= self.metadata.drop(index,axis=0)
        self.sample=self.mat.index

    def _obtain_numeric_matrix(self, sample_list):
        """ Exclude sample info for analysis """
        self.mat = self.mat(drop(sample_list))
        #self.metadata = self.metadata[~self.metadata[self.sample].isin(sample_list)]

    def _normalization(self, method="zscore"):
        print("Normalizing data with "+method)
        if method=='zscore':
            scaler=sklearn.preprocessing.StandardScaler()
            normalized_array=scaler.fit_transform(self.mat.values)
        elif method=='quantile':
            qt=sklearn.preprocessing.QuantileTransformer(random_state=0)
            normalized_array=qt.fit_transform(self.mat.values)
        self.mat = pd.DataFrame(
            normalized_array, index=self.mat.index, columns=self.mat.columns
        )

    def _imputation(self, method="constant"):
        print("Imputing data with "+method)
        if method=='constant':
            print('fill_value for imputation = ',0)

        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy=method, fill_value=0.0)
        imputation_array = imp.fit_transform(self.mat.values)
        self.mat = pd.DataFrame(
             imputation_array, index=self.mat.index, columns=self.mat.columns
        )

    def preprocess_print_info(self):
        """
        Print summary of preprocessing steps
        """
        print(pd.DataFrame(self.preprocessing_info.items()))

    def _log2_transform(self):
        self.mat = np.log2(self.mat.astype(float) + 0.1)
        self.preprocessing_info.update({"Log2-transformed":True})
        print("Data has been log2-transformed")

    def _take_cit_total_ratio(self):
        """
        This is specific to citrullination as we need to do cit/total ratio instead of cit level only
            - First loop over cit peptides
            - Find the protein cit-peptide is assigned to
            - Loop over samples
            - Look up the PG table for the level of that protein in that sample

        At the end of this the matrix values in self.mat is cit/total 
        """
        ###First read PG matrix###
        pg_mat = pd.read_csv('../raw/report.pg_matrix.tsv', delimiter='\t', dtype=str)
        pg_df = pd.DataFrame(pg_mat)

        #print(self.rawinput[['Protein.Group','Precursor.Id']])
        cit_total_debug=[]
        for index, sample in self.mat.iterrows():
            filename=index
            iloc=0
            for col in self.mat.columns:
                peptide=col
                #assuming order is the same in self.mat and self.rawinput
                protein=self.rawinput['Protein.Group'].iloc[iloc]
                #protein=self.rawinput['Protein.Group'][self.rawinput['Precursor.Id']==peptide][0]
                row_index = pg_df[pg_df['Protein.Group']==protein].index.values
                #print(protein,row_index,filename)
                protein_intensity=float(pg_df.iloc[row_index][filename].values)
                cit_total=0
                #print(protein,protein_intensity)
                if not np.isnan(protein_intensity):
                    """take log2 of ratio (peptide/protein) instead of log2(peptide)/log2(protein)"""
                    """intensity was NOT log2 transformed"""
                    #print(index,col,protein,protein_intensity,self.mat.loc[index, col],protein_intensity)
                    cit_total=float(self.mat.loc[index, col])/protein_intensity*100

                    ####Uncomment when we want cit peptide intensity
                    #cit_total=float(self.mat.loc[index, col])
                    ####Uncomment when we want log2 of cit peptide intensity
                    #cit_total=np.log2(float(self.mat.loc[index, col])+0.00001)

                    #cit_total=np.log2(float(self.mat.loc[index, col])+0.00001)-np.log2(protein_intensity+0.00001)
                #if protein=='E9PV24':
                    #print('PROTEIN FGA ',protein_intensity,self.mat.loc[index, col],cit_total)
                #if np.isnan(cit_total):
                    #print(iloc, protein, peptide, protein_intensity)
                self.mat.loc[index, col] = cit_total
                cit_total_debug.append([index,peptide,protein,self.mat.loc[index, col],protein_intensity,cit_total])
                iloc=iloc+1

        cit_df=pd.DataFrame(cit_total_debug)
        cit_df.columns=['samplekey','Peptide','Protein','Peptide.Intensity','Protein.Intensity','Cit.Total.Ratio']
        cit_df.to_csv('cit_total_debug_matrix.csv',sep='\t')

    def preprocess_data(
        self,
        log2_transform: bool=True,
        normalization: str=None,
        imputation: str=None,
        remove_samples: list=None
        ):

        #self._obtain_numeric_matrix(sample_list=remove_samples)
        print("No normalization is done")
        #if normalization is not None:
        #    print("Normalizing...")
        #    self._normalization(method=normalization)
        if imputation is not None:
            print("Imputing...")
            self._imputation(method=imputation)

        if log2_transform:
            print("Log2 transforming...")
            self._log2_transform()

        print("Take ratio cit to total protein...")
        self._take_cit_total_ratio()
        self.mat=self.mat.loc[:, (self.mat != 0).any(axis=0)]
        self.mat.to_csv("cit_total_ratio_matrix_cit_for_analysis.tsv",sep='\t')

        self.preprocessed = True


class Loader():
    """
    Load DIANN output
    """
    def __init__(self, input_file, intensity_column, index_column):
        """
        Args:
            input_file = (report.pg_matrix.tsv or report.pr_matrix.tsv)
            intensity_column = samples
            index_column = Protein.Group (or Precursor.Id)
        """
        self.input_file = input_file
        #### OK this is for precursor search pr matrix assuming looking at cit 
        rawinput = pd.read_csv(input_file, sep='\t', dtype=str)
        cit = []
        pg_pr=[]
        rawinput_filter = []
        cit_ratio=[]

        ###Print number of unique peptides from rawinput
        n_peptides=rawinput['Modified.Sequence'].unique().size
        print('Number of unique peptides from raw data before filtering missingness =',n_peptides)


        print("Filtering out non-cit peptides and cit peptides with more than ",str(MISSINGNESS_THRESH)," when loading pr_matrix...")
        """Number of samples = number of columns - 10"""
        with open ('../cit_peptides_list_'+str(100*MISSINGNESS_THRESH)+'missingness_.txt', 'w') as f:
            for index,peptide in rawinput.iterrows():
                percent_missing=peptide.isnull().values.sum()/(peptide.size-10)
                if percent_missing<MISSINGNESS_THRESH:
                 
                    rawinput_filter.append(peptide)
                    if pd.isnull(peptide['Precursor.Id']) == False and 'R(UniMod:7)' in peptide['Precursor.Id']:
                        if peptide['Modified.Sequence'].endswith('R(UniMod:7)') == False:
                            #print(peptide)
                            pg_pr_info=str(peptide['Genes'])+"___"+str(peptide['Precursor.Id'])
                            pg_pr.append(pg_pr_info)
                            cit.append(peptide)
                            f.write(pg_pr_info+"\n")
        f.close()

        self.rawinput=pd.DataFrame(cit)
        self.rawinput['Precursor.Id']=pg_pr
        self.rawinput.reset_index(inplace=True)
        #print(self.rawinput.index)
        self.rawinput.to_csv("report.pr_matrix_cit_for_analysis.tsv",sep='\t')
        print("Done loading cit only pr matrix. Number of precursors with R(UniMod:7) that are not at C-terminal = ", self.rawinput.shape[0])

        n_cit_peptides=len(self.rawinput['Modified.Sequence'].unique())
        print('Number of unique cit peptides after filtering missingness and C-terminus =',n_cit_peptides)
        self.intensity_column=intensity_column
        self.index_column=index_column


        print("Number of precursors before filtering missingness = ",len(rawinput))
        print("Number of precursors after filtering missingness = ",len(rawinput_filter))

        self.allpeptides=pd.DataFrame(rawinput_filter)
        self.allpeptides.reset_index(inplace=True)

    def print_load_parameters(self):
        print("Input File = "+self.input_file)
        print("Intensity Column = "+self.intensity_column)
        print("Index Column = "+self.index_column)


class Statistics:

    def __init__(self, dataset,
                 group1: Union[str, list],
                 group2: Union[str, list],
                 column: str=None,
                 method: str="ttest"):
        """
        Args: column(str) = column name in metadata file with the 2 groups to compare
        group1, group2 (str/list) = name of group to compare
        method (str, optional) = statistical method to calculate differential expression

        Returns: pd.DataFrame with foldchange, foldchange_log2, and pvalue for each protein/precursor
        between group2 and group2
        """


        self.dataset=dataset
        self.group1=group1
        self.group2=group2
        self.column=column
        self.method=method


        print("Finished initializing Statistics")

    def compute_fdr(self, p_vals):
        ranked_p_vals = scipy.stats.rankdata(np.array(p_vals), nan_policy='omit')
        fdr = p_vals * len(p_vals)/ranked_p_vals
        fdr[fdr > 1] = 1

        return fdr

    def unpairedttest(self) -> pd.DataFrame:
        print("****** Starting independent t-test ***********")
        group1_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"].tolist()
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"])
        group2_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"].tolist()
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"])
        #group1_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group1][
        #                                       self.dataset.sample].tolist() 
        #group2_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group2][
        #                                       self.dataset.sample].tolist() 



        #transpose mat, now [PG x samples] (row=proteins, column=samples)
        self.dataset.mat = self.dataset.mat.astype(float)
        mat_transpose=self.dataset.mat.transpose()

        p_values=mat_transpose.apply(lambda row: scipy.stats.ttest_ind(
                                    row[group1_samples].values.flatten(),
                                    row[group2_samples].values.flatten(),
                                   )[1],axis=1)

        """
        Adjusted p values
        """
        p_to_fdr = self.compute_fdr(p_values)

        fc = self._calculate_foldchange(
                              mat_transpose=mat_transpose,
                              group1_samples=group1_samples,
                              group2_samples=group2_samples)


        df = pd.DataFrame()
        df[self.dataset.index_column], df["pval"], df["adj.pval"]=p_values.index.tolist(), p_values.values, p_to_fdr.values
        #print(df)
        #df[self.dataset.index_column], df["pval"]=p_values.index.tolist(), p_values.values
        df["log2fc"] = fc

        return df

    def pairedttest(self) -> pd.DataFrame:
        print("****** Starting t-test ***********")
        group1_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"].tolist()
        #print("group1_samples")
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"])
        group2_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"].tolist()
        #print("group2_samples")
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"])
        #group1_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group1][
        #                                       self.dataset.sample].tolist() 
        #group2_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group2][
        #                                       self.dataset.sample].tolist() 
        mat_transpose=self.dataset.mat.transpose()
        ###print(mat_transpose)
        p_values=mat_transpose.apply(lambda row: scipy.stats.ttest_rel(
                                    row[group1_samples].values.flatten(),
                                    row[group2_samples].values.flatten(),
                                   )[1],axis=1)

        """
        Adjusted p values
        """
        p_to_fdr = self.compute_fdr(p_values)

        fc = self._calculate_foldchange(
                              mat_transpose=mat_transpose,
                              group1_samples=group1_samples,
                              group2_samples=group2_samples)


        df = pd.DataFrame()
        df[self.dataset.index_column], df["pval"], df["adj.pval"]=p_values.index.tolist(), p_values.values, p_to_fdr.values
        #print(df)
        #df[self.dataset.index_column], df["pval"]=p_values.index.tolist(), p_values.values
        df["log2fc"] = fc

        return df

    def _calculate_foldchange(self, mat_transpose:pd.DataFrame, group1_samples:list, group2_samples:list):
        mat_transpose += 0.00001

        if self.dataset.preprocessing_info["Log2-transformed"]:
            print("Data have already been log2 transformed for log2 fold change calculation. No Log2 done.")
            fc = (mat_transpose[group2_samples].T.mean().values-mat_transpose[group1_samples].T.mean().values)
        else:
            fc = (mat_transpose[group2_samples].T.mean().values/mat_transpose[group1_samples].T.mean().values)
            fc=np.log2(fc+0.00001)

        print("Finished calculating fold change: group2/group1")
        return fc

    def diff_expression_analysis(self) -> pd.DataFrame:

        print("Starting differential expression analysis with paired t-test")
        df = self.unpairedttest()
        return df



class DataSet(Preprocess, Statistics):
    """
    """
    def __init__(self, loader, metadata_path=None, sample_column=None):
        """
        Create DataSet
        Args: metadata_path(str) = path to metadata file (default=None)
              sample_column(str) = column in metadata file indicating the sample IDs (default=None)
        """

        self.metadata=None
        self.loader=loader
        self.rawinput=loader.rawinput
        self.allpeptides=loader.allpeptides #### This is to do PCA
        self.index_column=loader.index_column
        self.intensity_column=loader.intensity_column

        if metadata_path is not None:
            self.sample = sample_column
            self.load_metadata(file_path=metadata_path)
        else:
            print("Need to provide path to metadata")

        self.create_matrix()
        print("DataSet has been created")
        #self._check_matrix_values()
        self._check_missingness()

        self.overview()
        self.preprocessing_info = self._save_dataset_info()
        self.preprocessed=False

        pg_mat = pd.read_csv('../raw/report.pg_matrix.tsv', delimiter='\t', dtype=str)
        self.pg_rawinput= pd.DataFrame(pg_mat)

    def create_matrix(self):
        """
        output file matrix with columns displaying Features (PGs/Precursors) and rows the samples.
        """
        df = self.rawinput
        df = df.set_index(self.index_column)
        """ Removing all columns but samples"""
        df = df.filter(regex="\.d$")
        #Transpose dataframe
        mat= df.transpose()
        self.mat=mat
        self.sample=mat.index
        self.preprocessing_info = self._save_dataset_info()

        df_all = self.allpeptides
        df_all.set_index(self.index_column)
        df_all = df_all.filter(regex="\.d$")
        mat_all = df_all.transpose()
        self.mat_allpeptides=mat_all


    ### Get intensities for one particular cit peptide from self.rawdata
    def get_cit_intensity(self,precursor,is_present=False):

        row_index=0
        if not is_present:
            for index, pep in self.rawinput.iterrows():
                if precursor in pep['Precursor.Id']:
                    is_present=True
                    row_index=index
            if not is_present:
                print(f"Cit peptide {precursor} not found.")
                return pd.DataFrame() 

        imputedata=self.mat.loc[:,precursor]

        rowdata = imputedata.T
        cit_df=pd.DataFrame(rowdata)
        cit_df.columns=[precursor]

        cit_df.to_csv('_debug_corr_intensity_'+precursor+'.csv',sep='\t')
        return(cit_df)


    ### Get intensities for one particular protein from self.pg_rawdata
    def get_protein_intensity(self,protein_group):

        row=self.pg_rawinput.index[self.pg_rawinput['Genes']==protein_group].tolist()

        if not row:
             print(f"Protein {protein_group} not found in pg_rawinput.")
             return

        ## self.pg_rawinput has rows as PGs and columns samplekey
        rowdata = self.pg_rawinput.loc[row]
        protein_df = pd.DataFrame(rowdata).reset_index()
        protein_df=protein_df.T

        pg = protein_df.loc['Protein.Group', 0]
        genes = protein_df.loc['Genes', 0]
        new_column_name = f"{pg}_{genes}"

        protein_df=protein_df.drop(['index', 'Protein.Group', 'Protein.Ids', 'Protein.Names','Genes','First.Protein.Description'])
        protein_df.columns = [new_column_name]
        protein_df.to_csv('_debug_corr_intensity_'+protein_group+'.csv',sep='\t')
        return(protein_df)


    def analyze_cit_IDs(self, feature, groups=[], output=""):
        """
        This function creates box plots with missingness values for different sample groups
        based on feature in DOE specified from input
        """
        
        plt.rcParams["figure.figsize"] = (16, 5)
        outputfile="PR_cit_"+str(feature)+"_"+output
        cit_IDs = []
        cit_intensity=[]
        color='goldenrod' 
        for index, sample in self.mat.iterrows():
            sample=sample.astype(float)
            counts = sample.size-sample.isnull().values.sum()
            cit_IDs.append(counts)
            intensity=sample.sum(skipna=True)
            cit_intensity.append(np.log2(intensity+0.00000001))

        cit_df = pd.DataFrame(cit_IDs,columns=['CitIDs'])
        cit_df['Sample'] = self.sample
        cit_df['Total.Cit.Intensity']=cit_intensity
        cit_df[feature] = self.metadata[feature]
        cit_df.set_index('Sample')

        ##cit_df is the matrix containing feature key mapped to sample ID
        #print(cit_df)
        cit_df.to_csv(outputfile+".csv",sep=',')

        plotdf=pd.DataFrame()
        plot_intensity_df=pd.DataFrame()
        if not len(groups):
            groups = self.metadata[column].unique()
        for group in groups:
            group_samples = self.metadata[self.metadata[feature]==group]["samplekey"].tolist()
            newcol = []
            newcol2 = []
            for sample in group_samples:
                subset = cit_df[cit_df['Sample']==sample]
                newcol.append(subset['CitIDs'].values)
                newcol2.append(subset['Total.Cit.Intensity'].values)
            plotdf[group]=pd.DataFrame(newcol)
            plot_intensity_df[group]=pd.DataFrame(newcol2)

        print("Box plot for Cit IDs in different sample groups")
        fig,ax=plt.subplots()
        plotdf.plot(kind='box', patch_artist='True',boxprops=dict(facecolor=color,alpha=0.7, linewidth=2),
                                medianprops=dict(color='black', linewidth=2),
                                capprops=dict(color='black', linewidth=2),
                                flierprops=dict(color='black', markerfacecolor='white'),
                                whiskerprops=dict(color='black', linewidth=2),ax=ax)
        for i, group in enumerate(groups):
            y = plotdf[group]
            x = np.random.normal(i + 1, 0.02, size=len(y))  # i+1 positions the points; adjust 0.04 for less/more spread
            #x = [i + 1] * len(y)  # Generate a list of x positions for each data point
            ax.scatter(x, y, color='black', alpha=0.9, s=25,zorder=4)  # Adjust alpha and s (size) as needed
            ax.scatter(x, y,
                           s=25,
                           c='grey',
                           edgecolor='black',
                           linewidth=1,
                           zorder=4)

        plt.gca().set_facecolor('whitesmoke')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)


        plt.ylabel("Cit IDs",fontsize=24)
        plt.xlabel(feature,fontsize=24)
        plt.xticks(rotation="vertical",fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(linestyle=':')
        plt.grid(linestyle=':')
        plt.title('Cit IDs for Different Sample Groups',fontsize=25)
        figure_file=outputfile+"_IDs.svg"
        plt.savefig(figure_file, bbox_inches="tight")
        plt.clf() 


        fig,ax=plt.subplots()
        plot_intensity_df.plot(kind='box', patch_artist='True',boxprops=dict(facecolor=color,alpha=0.8,linewidth=2),
                                medianprops=dict(color='black',linewidth=2),
                                capprops=dict(color='black',linewidth=2),
                                flierprops=dict(color='black', markerfacecolor='white'),
                                whiskerprops=dict(color='black',linewidth=2),ax=ax)
        for i, group in enumerate(groups):
            y = plot_intensity_df[group]
            x = np.random.normal(i + 1, 0.02, size=len(y))  # i+1 positions the points; adjust 0.04 for less/more spread
            #x = [i + 1] * len(y)  # Generate a list of x positions for each data point
            ax.scatter(x, y, color='black', alpha=1, s=25)  # Adjust alpha and s (size) as needed
            ax.scatter(x, y,
                           s=25,
                           c='grey',
                           edgecolor='black',
                           linewidth=1,
                           zorder=4)

        plt.gca().set_facecolor('whitesmoke')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)

        plt.ylabel("Log2[Average Cit Intensity]",fontsize=24)
        plt.xlabel(feature,fontsize=24)
        plt.xticks(rotation="vertical",fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(linestyle=':')
        plt.grid(linestyle=':')
        plt.title('Total Cit Intensity for Different Sample Groups',fontsize=16)
        figure_file=outputfile+"_intensity.svg"
        plt.savefig(figure_file, bbox_inches="tight")
        plt.clf() 

        plt.close()


    def analyze_missingness(self, feature, cit_only=False, output=""):
        """
        This function creates box plots with missingness values for different sample groups
        based on feature in DOE specified from input
        """
        
        percent_missingness = []
        color='linen'
        if cit_only:
            color='goldenrod' 
            for index, sample in self.mat.iterrows():
                percent = sample.isnull().values.sum()/sample.size
                if percent > 0.8:
                    print('High missingness in sample index ',index, sample.isnull().values.sum(), sample.size)
                percent_missingness.append(percent)
        else:
            color='thistle'
            for index, sample in self.mat_allpeptides.iterrows():
                percent = sample.isnull().values.sum()/sample.size
                if percent > 0.8:
                    print('High missingness in sample index ',index, sample.isnull().values.sum(), sample.size)
                percent_missingness.append(percent)

        missingness_df = pd.DataFrame(percent_missingness,columns=['Missingness'])
        missingness_df['Sample'] = self.sample
        missingness_df[feature] = self.metadata[feature]
        missingness_df.set_index('Sample')
        missingness_df['Missingness'] = 100*missingness_df['Missingness'].astype(float)

        groups = self.metadata[feature].unique()

        plotdf=pd.DataFrame()
        for group in groups:
            group_samples = self.metadata[self.metadata[feature]==group]["samplekey"].tolist()
            newcol = []
            for sample in group_samples:
                subset = missingness_df[missingness_df['Sample']==sample]
                newcol.append(subset['Missingness'].values)
           
            plotdf[group]=pd.DataFrame(newcol)

        print("Box plot for missingness different sample groups")
        fig, ax = plt.subplots()
        plotdf.plot(kind='box', patch_artist='True',boxprops=dict(facecolor=color,alpha=0.5),
                    medianprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(color='black', markerfacecolor=color, markersize=2),
                    whiskerprops=dict(color='black'),
                    ax=ax)

        for i, group in enumerate(groups):
            y = plotdf[group]
            x = np.random.normal(i + 1, 0.02, size=len(y))  # i+1 positions the points; adjust 0.04 for less/more spread
            #x = [i + 1] * len(y)  # Generate a list of x positions for each data point
            ax.scatter(x, y, color='black', alpha=0.9, s=10)  # Adjust alpha and s (size) as needed

        #data_df = plotdf[tg_group].melt(var_name=column,value_name="Log2[Cit/Total]")
        #print(data_df)
        plt.ylabel("Percent Missingness")
        plt.xlabel(feature)
        plt.xticks(rotation="vertical")
        plt.grid(linestyle=':')
        plt.ylim(0,100)
        plt.title('Data missingness with sample groups: cit_only='+str(cit_only))
        figure_file="PR_missingness_cit_"+str(feature)+"_"+str(cit_only)+output+".svg"
        plt.savefig(figure_file, bbox_inches="tight")
        plt.clf() 
        plt.close()

    def _check_matrix_values(self):
        if np.isinf(self.mat).values.sum() > 0:
            print("Data contains infinite values.")

    def _check_missingness(self):
        pmissing = self.mat.isnull().values.sum()/self.mat.size*100
        print("Percent missingness = ",pmissing)

    def load_metadata(self, file_path):
        """
        Load Metadata
        Args: file_path(str)
        """
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            df = None
            print("ERROR: Metadata must be be an xlsx file")
        df.columns = df.columns.astype(str)
        self.metadata=df

    def _save_dataset_info(self):
        n_proteingroups = self.mat.shape[1]
        preprocessing_dict = {
            "Number of protein groups ": n_proteingroups,
            "Intensity used for analysis ": self.intensity_column,
            "Log2-transformed": False,
            "Normalization": None,
            "Imputation": None
        }
        return preprocessing_dict

    def overview(self):
        """Print overview of the DataSet"""
        dataset_overview = (
            "Attributes of the DataSet can be accessed using: \n"
            + "DataSet.rawinput:\t Raw Protein data.\n"
            + "DataSet.mat:\tProcessed data matrix with ProteinIDs/ProteinGroups as columns and samples as rows. All computations are performed on this matrix.\n"
            + "DataSet.metadata:\tMetadata for the samples in the matrix. Metadata will be matched with DataSet.mat when needed (for instance Volcano Plot).\n"
            )
        print(dataset_overview)

############################################################################ PCA #################################

    def plot_heatmap(self, feature='Well',output=''):

        plotdata=self.rawinput.drop(['Protein.Group','Protein.Ids','Protein.Names','First.Protein.Description','Proteotypic',
                                 'Stripped.Sequence','Modified.Sequence','Precursor.Charge','Precursor.Id'],axis=1)
        plotdata.drop('index',axis=1,inplace=True)
        ### sort proteins by intensity of the last column (last sample)
        sample=plotdata.columns[-1]
        plotdata.sort_values(by=sample, axis=0, ascending=True, inplace=True)

        sample_list= ['Genes']
        sample_list.extend(self.metadata[feature].values)
        column_dict = {old_column : new_column for old_column,new_column in zip(plotdata.columns, sample_list)}
        plotdata=plotdata.rename(columns=column_dict)
        plotdata.set_index('Genes',inplace=True)
        plotdata=plotdata.astype(float)
        maskna=1e-8
        plotdata=plotdata.fillna(maskna)
        plotdata=np.log2(plotdata+maskna)
        print('Plotting heatmap with missing values imputed with zero')

        #just heatmap no correlation
        heatmap=sns.clustermap(plotdata.T, cmap='viridis',vmin=-25,vmax=25,col_cluster=False, row_cluster=False,figsize=(70, 20))
        # Add horizontal lines to separate the blocks of similar y-values
        feature_values = self.metadata[feature].values
        unique_features = np.unique(feature_values)
        y_labels = heatmap.ax_heatmap.get_yticklabels()
        y_pos = []
                    
        for label in unique_features:
            pos = np.where(feature_values == label)[0]
            if len(pos) > 0:
                y_pos.append(pos[-1])

        for y in y_pos:
            heatmap.ax_heatmap.hlines(y + 1, *heatmap.ax_heatmap.get_xlim(), color='white', linewidth=2)
        plt.savefig('sns_heatmap_'+feature)


        #just seaborn correlation
        #metric='correlation'
        #heatmap=sns.clustermap(plotdata.T, metric=metric,cmap='vlag',col_cluster=False,vmin=-1,vmax=1,z_score=0,figsize=(70, 20))
        #heatmap.ax_row_dendrogram.set_visible(True)
        #heatmap.ax_col_dendrogram.set_visible(True)
        #plt.savefig('sns_clustermap_'+metric+"_"+feature)

        figure=px.imshow(plotdata.T, width=6000,height=1000)
        plotdata.reset_index(inplace=True)
        plotdata.drop('Genes',axis=1,inplace=True)
        figure=px.imshow(plotdata.T)

        outputfile='heatmap_T_cit_sorted_'+feature+str(output)
        pio.write_html(figure,outputfile+'.html')

        plt.clf()
        plt.close()


 ########################################################################### PCA #################################
    def plot_pca(self, column=None,output="", marker="o",circle=False):

        ################## Impute within group in column using min#######################

        imputed_mat = self.mat.copy(deep=True)
        imputed_mat = imputed_mat.astype(float)
        #imputed_mat has the dimension of number_samples x number_cit_peptides
        #imputed_mat.to_csv('_pca_debug_imputed_mat.csv',sep='\t')

        #imputed_mat = imputed_mat.dropna(axis=0, how='any').dropna(axis=1, how='any')


        #####################################################################################
        ######Fill missing values with the minimum value in each row axis=0 per sample
        ######This is assuming that the missing values = detection limit = lowest intensity in that sample
        ###uncomment the following lines
        #imputed_mat= imputed_mat.apply(lambda x: x.fillna(x.min()), axis=1)
        # Identify missing values and prepare optimization
        is_null = pd.isnull(imputed_mat)
        missing_indices = list(zip(*np.where(is_null)))
        #imputed_mat = [imputed_mat.iloc[:, col].median() for _, col in missing_indices]
        #impute='_median'


        """
        ###### MIN DET #####
        # Optimized initial guess using column median to handle outliers better
        initial_guess = [imputed_mat.iloc[:, col].median() for _, col in missing_indices]

        # Optimization to minimize the determinant of the covariance matrix
        res = minimize(det_cov, initial_guess, args=(imputed_mat, missing_indices), method='SLSQP', options={'maxiter': 100, 'ftol': 1e-06})

        if res.success:
            # Batch update of imputed values
                for (index, col), value in zip(missing_indices, res.x):
                        imputed_mat.iat[index, col] = value
        """

        #####impute data per column of proteins for each sample group with min or constant
        #fill_value=imputed_mat.min(skipna=True).min() #filling all missing values with min of matrix
        #imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=fill_value, keep_empty_features=True)
        #####################################################################################


        ######impute data within group
        impute='_within_groups'
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, keep_empty_features=True)

        for group in self.metadata[column].unique():
            group_samples = self.metadata[self.metadata[column]==group]["samplekey"].tolist()
            imputation_array = imp.fit_transform(imputed_mat.loc[group_samples].values)
            imputed_mat.loc[group_samples]= pd.DataFrame(imputation_array,
                                            index=imputed_mat.loc[group_samples].index, columns=imputed_mat.loc[group_samples].columns)
        ################################# END IMPUTE ###################################
        #standardize
        #scaler = StandardScaler()
        #scaled_mat = scaler.fit_transform(imputed_mat.astype(float))

        scaled_mat=imputed_mat #no scaling

        ### Now run PCA on imputed matrix
        pca=sklearn.decomposition.PCA(n_components=2)
        components=pca.fit_transform(scaled_mat) #these are the n_components  egenvectors

        """Transform components into a DataFrame for seaborn plot"""
        data_to_plot = pd.DataFrame(components, columns=['PCA1', 'PCA2'])
        # Reset the index of plotdf to ensure it's using a simple integer index
        data_to_plot=data_to_plot.reset_index(drop=True)
        #data_to_plot.to_csv('_pca_cit_data_to_plot_'+column+'.csv',sep='\t')

        label_column = self.metadata[column].astype(str)
        #label_column.to_csv('_pca_cit_label_column_debug_'+column+'.csv',sep='\t')
        label_column_df = label_column.to_frame(name=column).reset_index(drop=True)

        label_file=self.metadata['samplekey'].astype(str)
        #label_file.to_csv('_pca_cit_label_file_debug_'+column+'.csv',sep='\t')
        label_file_df = label_file.to_frame(name='samplekey').reset_index(drop=True)


        plotdf=pd.concat([data_to_plot, label_column_df, label_file_df], axis=1)
        plotdf.to_csv('_pca_cit_plotdf_debug_'+column+impute+'.csv',sep='\t')

        figure=px.scatter(plotdf,x='PCA1',y='PCA2',color=column,hover_name="samplekey")
        figure.update_traces(marker_size=10)
        figure.update_layout(
            title='PCA Citrullinome Level : '+impute,
            xaxis_title="PCA 1(%.2f%%)" % (pca.explained_variance_ratio_[0]*100),
            yaxis_title="PCA 2(%.2f%%)" % (pca.explained_variance_ratio_[1]*100),
            legend_title=column,
            font=dict(
                family='Tahoma',
                size=18,
                color='RebeccaPurple'
            )
        )
        outputfile='pca_cit_'+column+"_"+output+"_"+str(MISSINGNESS_THRESH)+impute
        outputfile=trim_string(outputfile)
        #outputfile='pca_cit_'+column+"_"+output+'_standardized_impute_within_group'
        pio.write_html(figure,outputfile+'.html')

        ### Get the feature importance for the top PCA components
        loadings = pca.components_.T
        feature_names = imputed_mat.columns
        loading_matrix = pd.DataFrame(loadings, columns=['PCA1', 'PCA2'], index=feature_names)

        ### Save the loading matrix for later analysis
        loading_matrix.to_csv('pca_loadings_'+outputfile+'.csv')


        """
        snsplot=sns.scatterplot(data=plotdf, x="PCA1", y="PCA2", hue=column, edgecolor="black", marker=marker, s=80, alpha=0.9)
        plt.xlabel("PC 1(%.2f%%)" % (pca.explained_variance_ratio_[0]*100))
        plt.ylabel("PC 2(%.2f%%)" % (pca.explained_variance_ratio_[1]*100))
        plt.legend(loc="best", bbox_to_anchor=(1,1))
        plt.savefig(output, bbox_inches="tight")
        plt.clf() 
        """

        ################# Create the same plot with seaborn for presentation #############################
        # Set the color palette to use with seaborn
        unique_labels = plotdf[column].unique()

        #color blind palette
        sns.set_palette("colorblind")
        palette = sns.color_palette("bright", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='PCA1', y='PCA2', data=plotdf, hue=column, palette=color_map, s=100, edgecolor='black', alpha=0.7)

        plt.title(f'PCA Plot Citrullinome Level', fontsize=18)
        plt.xlabel("PCA 1(%.2f%%)" % (pca.explained_variance_ratio_[0]*100),fontsize=18)
        plt.ylabel("PCA 2(%.2f%%)" % (pca.explained_variance_ratio_[1]*100),fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
    
        # Show the plot
        plt.legend(title=column,fontsize=16)
        plt.tight_layout()
    
        # Optionally, save the plot if an output file path is provided
        plt.savefig(outputfile+'_seaborn_plot.svg')  # Saves the plot

        plt.clf() 
        plt.close()
   
##########################################################################################################
#    def plot_pca(self, group=None,output="", marker="o",circle=False):
#        pca=sklearn.decomposition.PCA(n_components=2)
#        components=pca.fit_transform(self.mat) #these are the n_components  egenvectors
#        #Transform components into a DataFrame for seaborn plot
#        plotdf = pd.DataFrame(components, columns=['PCA1', 'PCA2'])
#        label_column = self.metadata[group].astype(str)
#        plotdf = plotdf.join(label_column)
#        label_file=self.metadata['samplekey'].astype(str)
#        plotdf = plotdf.join(label_file)
#
#        figure=px.scatter(plotdf,x='PCA1',y='PCA2',color=group,hover_name="samplekey")
#        figure.update_traces(marker_size=20)
#        figure.update_layout(
#            title='PCA Citrullinome Level : '+' Impute with '+IMPUTE_METHOD,
#            xaxis_title="PCA 1(%.2f%%)" % (pca.explained_variance_ratio_[0]*100),
#            yaxis_title="PCA 2(%.2f%%)" % (pca.explained_variance_ratio_[1]*100),
#            legend_title=group,
#            font=dict(
#                family='Tahoma',
#                size=18,
#                color='RebeccaPurple'
#            )
#        )
#        #plotly.offline.plot(figure,output+'.html')
#        pio.write_html(figure,output+'.html')
#
#        plt.clf() 
#        plt.close()

##########################################################################################################

    def plot_umap(self, group='Row', output='umap.svg'):

        """Need to impute data"""

        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        imputation_array = imp.fit_transform(self.mat_allpeptides.values)
        self.mat_allpeptides = pd.DataFrame(
             imputation_array, index=self.mat_allpeptides.index, columns=self.mat_allpeptides.columns
        )
        
        nneigh=5
        mindist=0.2
        um = umap.UMAP(n_neighbors=nneigh,min_dist=mindist,metric='correlation')
        x_fit=um.fit(self.mat_allpeptides)
        x_umap=um.transform(self.mat_allpeptides)


        """Transform components into a DataFrame for seaborn plot"""
        data_to_plot = pd.DataFrame(x_umap, columns=['col1', 'col2'])
        # Reset the index of plotdf to ensure it's using a simple integer index
        data_to_plot=data_to_plot.reset_index(drop=True)
        #data_to_plot.to_csv('_pca_cit_data_to_plot_'+column+'.csv',sep='\t')

        label_column = self.metadata[group].astype(str)
        #label_column.to_csv('_pca_cit_label_column_debug_'+column+'.csv',sep='\t')
        label_column_df = label_column.to_frame(name=column).reset_index(drop=True)

        label_file=self.metadata['samplekey'].astype(str)
        #label_file.to_csv('_pca_cit_label_file_debug_'+column+'.csv',sep='\t')
        label_file_df = label_file.to_frame(name='samplekey').reset_index(drop=True)

        plotdf=pd.concat([data_to_plot, label_column_df, label_file_df], axis=1)
        plotdf.to_csv('_debug/_umap_cit_plotdf_debug_'+column+'.csv',sep='\t')





        figure=px.scatter(plotdf,x='col1',y='col2',color=group,hover_name="samplekey")
        figure.update_traces(marker_size=7)
#        snsplot=sns.scatterplot(data=plotdf, x="col1", y="col2", hue=group, edgecolor="black", s=40, alpha=0.8)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(loc="best", bbox_to_anchor=(1,1))
 #       print(umap_df)
        output_file='umap_'+group+'_nneigh'+str(nneigh)+'_mindist'+str(mindist)+'_'+output
#        plt.savefig('./'+output_file+'.svg', bbox_inches="tight")
        #plotly.offline.plot(figure,output_file+'.html')
        pio.write_html(figure,output_file+'.html')




        ################# Create the same plot with seaborn for presentation #############################
        # Set the color palette to use with seaborn
        unique_labels = plotdf[column].unique()

        #color blind palette
        sns.set_palette("colorblind")
        palette = sns.color_palette("bright", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='col1', y='col2', data=plotdf, hue=column, palette=color_map, s=100, edgecolor='black', alpha=0.7)

        plt.title(f'UMap Plot Citrullinome Level', fontsize=18)
        plt.xlabel('UMAP 1',fontsize=18)
        plt.ylabel('UMAP 2',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
    
        # Show the plot
        plt.legend(title=column,fontsize=16)
        plt.tight_layout()
    
        # Optionally, save the plot if an output file path is provided
        plt.savefig(output_file+'_seaborn_plot.svg')  # Saves the plot

        plt.clf() 
        plt.close()
        

#######################################################################################################################3

    def volcano_plot(self, group1, group2, column, output="", method="ttest"):

        outputfile='volcano_'+group1+'_'+group2+output

        #compute missingness for features in group2 samples
        group2_samples = self.metadata[self.metadata[column]==group2]["samplekey"].tolist()
        NA_rate=self.mat.loc[group2_samples].isnull().sum()/self.mat.loc[group2_samples].shape[0]*100

        #impute data per column of proteins for each sample group
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, keep_empty_features=True)

        for group in self.metadata[column]:
            group_samples = self.metadata[self.metadata[column]==group]["samplekey"].tolist()
            #print(self.mat.loc[group_samples])
            imputation_array = imp.fit_transform(self.mat.loc[group_samples].values)
            self.mat.loc[group_samples]= pd.DataFrame(
                imputation_array, index=self.mat.loc[group_samples].index, columns=self.mat.loc[group_samples].columns
            )
            #print(self.mat.loc[group_samples])


        stat = Statistics(dataset=self, group1=group1, group2=group2, column=column, method=method)
        plotdf=stat.diff_expression_analysis()
        plotdf['negLogP']=-np.log10(plotdf['adj.pval'])
        plotdf['NA_rate']=NA_rate.values
        #print("PRINT PLOTDF")
        #print(plotdf)
        plotdf.to_csv('table_'+outputfile+'.csv',sep=',')
        #snsplot=sns.scatterplot(data=plotdf, x="log2fc", y="negLogP", edgecolor="black", color="gray",s=4, alpha=0.4)
        figure=px.scatter(plotdf,x='log2fc',y='negLogP',color='NA_rate',hover_name="Precursor.Id",
                          color_continuous_scale='viridis')
        figure.update_traces(marker_size=10)
        figure.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title='Volcano Plot :  '+str(group1)+' vs '+str(group2),
            #title='Volcano Plot :  Impute with '+str(IMPUTE_METHOD)+'  '+str(group1)+' vs '+str(group2),
            xaxis_title="Log2FC",
            yaxis_title="-Log(Adj.P)",
            legend_title="%NA rate",
            xaxis_range=[-2,2],
            yaxis_range=[0,6],
            font=dict(
                family='Tahoma',
                size=18,
                color='RebeccaPurple'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='#F0F8FF',
                gridwidth=1,
                showticklabels=True
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#F0F8FF',
                gridwidth=1,
                showticklabels=True
            )
        )
        figure.update_xaxes(showline=True,linewidth=2,linecolor='black',zeroline=True)
        figure.update_yaxes(showline=True,linewidth=2,linecolor='black',zeroline=True)
        #plotly.offline.plot(figure,output+'.html')
        pio.write_html(figure,'volcano_plots_pr_'+outputfile+'.html')

        """
        Print significant values
        """
        sigval=[]
        fig, ax = plt.subplots()
        ax.scatter(data=plotdf, x="log2fc", y="negLogP", edgecolor="gray", color="gray",s=3, alpha=0.7)
        #plt.xlim(-2, 2)
        #plt.ylim(0, 6)
        """ADJ.P VALUE AND LOG2FC threshhold"""
        p_thresh=1.3
        fc_thresh=1

        outputfile=output+"_p"+str(p_thresh)+"_fc"+str(fc_thresh)
        plt.axhline(p_thresh, linewidth=1, color="gray", linestyle=':')
        plt.axvline(fc_thresh, linewidth=1, color="gray", linestyle=':')
        plt.axvline(-fc_thresh, linewidth=1, color="gray", linestyle=':')
        for index, line in plotdf.iterrows():
            stripped_sequence=line[0].replace("(UniMod:21)","_")
            stripped_sequence=stripped_sequence.replace("(UniMod:35)","_")
            stripped_sequence=stripped_sequence.replace("(UniMod:4)","_")
            stripped_sequence=stripped_sequence.replace("(UniMod:28)","_")
            stripped_sequence=stripped_sequence.replace("(UniMod:27)","_")
            stripped_sequence=stripped_sequence.replace("(UniMod:26)","_")
            stripped_sequence=stripped_sequence.replace("(UniMod:1)","_")
            if line['negLogP'] > p_thresh and line['log2fc']>fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="red", color="red",s=5, alpha=0.9)
                #plt.text(line["log2fc"], line["negLogP"], line[0])
                #plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                #if self.rawinput['Genes'].loc[index] is not np.nan and (line['negLogP']>2 or line['log2fc']>4):
                    #plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                    #plt.annotate(line[0].replace("(UniMod:21)","_"), (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                sigval.append(line) 
            elif line['negLogP'] > p_thresh and line['log2fc']<-fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="blue", color="blue",s=5, alpha=0.9)
            if line['negLogP'] > p_thresh and line['log2fc']>fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="red", color="red",s=5, alpha=0.9)
                #plt.text(line["log2fc"], line["negLogP"], line[0])
                #plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                #if self.rawinput['Genes'].loc[index] is not np.nan and (line['negLogP']>2 or line['log2fc']>4):
                    #plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                    #plt.annotate(line[0].replace("(UniMod:21)","_"), (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                sigval.append(line) 
            elif line['negLogP'] > p_thresh and line['log2fc']<-fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="blue", color="blue",s=5, alpha=0.9)
            if line['negLogP'] > p_thresh and line['log2fc']>fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="red", color="red",s=5, alpha=0.9)
                #plt.text(line["log2fc"], line["negLogP"], line[0])
                #plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                #if self.rawinput['Genes'].loc[index] is not np.nan and (line['negLogP']>2 or line['log2fc']>4):
                    #plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                    #plt.annotate(line[0].replace("(UniMod:21)","_"), (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                sigval.append(line) 
            elif line['negLogP'] > p_thresh and line['log2fc']<-fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="blue", color="blue",s=5, alpha=0.9)
                #plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="blue")
                #if self.rawinput['Genes'].loc[index] is not np.nan and (line['negLogP']>2 or line['log2fc']<-4):
                    #plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="blue")
                sigval.append(line) 
            if line['log2fc']>4:
                plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
            elif line['log2fc']<-4:
                plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="blue")
                
            if self.rawinput['Genes'].loc[index]=="PADI4":
                print(stripped_sequence, line["log2fc"], line["negLogP"])
                if line['negLogP'] > p_thresh and line['log2fc']>fc_thresh:
                    plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
                elif line['negLogP'] > p_thresh and line['log2fc']<-fc_thresh:
                    plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="blue")
                else:
                    plt.annotate(stripped_sequence, (line["log2fc"], line["negLogP"]), fontsize=4, color="gray")
        sigval_df=pd.DataFrame(sigval)
        sigval_df['Protein.Names']=self.rawinput['Protein.Names'].loc[sigval_df.index]
        sigval_df['Protein.Description']=self.rawinput['First.Protein.Description'].loc[sigval_df.index]
        sigval_df.to_csv("sigval_pr_"+outputfile+".csv",sep=',')
        #print("Max log2fc = ", max(plotdf['log2fc']), "   Min log2fc = ",min(plotdf['log2fc']))
        #plt.axline((0, p_thresh), (1, p_thresh), linewidth=4, color="gray")
        plt.xlabel("log2 FC")
        plt.ylabel("-log10(adj.p value)")
        #plt.grid()
        #plt.legend(loc="best", bbox_to_anchor=(1,1))

        plt.savefig("volcano_pr_"+outputfile+".svg", bbox_inches="tight")
        plt.clf() 
        plt.close()
        
    def plot_cit_ratio(self, feature='Row', output=""): ###Imputed with 0 

        print("Using imputed matrix for cit to total ratio plot")
        groups = self.metadata[feature].unique()
        plotdf = pd.DataFrame() ###plotdf will have #rows=n_peptides and #columns=#groups
        #plotdf.columns=groups

        icol=0
        error_df = pd.DataFrame()
        self.mat=self.mat.astype(float)
        for precursor in self.mat.columns: #Over ALL CIT PEPTIDES
            #Note that precursor = PG_PR here which is "gene___precursor.Id"
            mat_col = self.mat[precursor]
            plotdf.at[icol, 'Peptide']=precursor
            error_df.at[icol, 'Peptide']=precursor

            for group in groups:
                group_samples = self.metadata[self.metadata[feature]==group]["samplekey"].tolist()
                newcol=[]
                for sample in group_samples:
                    sample_cit_ratio=(mat_col[mat_col.index==sample])[0].astype(float)
                    newcol.append(sample_cit_ratio)
                plotdf.at[icol, group]=np.nanmean(newcol)
                error_df.at[icol, group]=np.nanstd(newcol)
            icol=icol+1

        logfile='cit_missing_thresh_'+str(MISSINGNESS_THRESH*100)+'_feature_'+feature+"_"+output
        plotdf.to_csv(logfile+'.csv')
        #figure = pd.melt(plotdf, id_vars='Peptide', value_vars= plotdf.columns[:-1])
        figure=go.Figure()
        peptides_list = plotdf['Peptide']
        plotdf_T = plotdf.set_index('Peptide').transpose()
        error_T=error_df.set_index('Peptide').transpose()
        for col in peptides_list:
            #print(plotdf_T[col])
            figure.add_trace(go.Scatter(x=plotdf_T.index, y=plotdf_T[col],
                                        error_y=dict(type='data', array=error_T[col],visible=True),name=col))

        figure.update_layout(title='Citrullinated Peptide / Total Protein Level',
                             xaxis_title=feature,
                             yaxis_title='Cit/Total',
                             legend_title='cit-Peptides less than '+str(MISSINGNESS_THRESH*100)+'%  Missingness',
                             font=dict(family="Tahoma",
                                       size=18,
                                       color="RebeccaPurple"))
        output=output+"NoImpute"
        #plt.savefig("./cit_pep_plot_"+feature+"_"+output+".svg", bbox_inches="tight")
        #plotly.offline.plot(figure,'cit_pr_.html')
        pio.write_html(figure,'cit_ratio_pr.html')
        plt.clf() 
        plt.close()

        
    def plot_cit_peptides(self, feature='Row', output=""): #### NO IMPUTATION ####

        print("Using rawinput (no impute) for cit peptides plot")
        #groups = self.metadata[feature].unique()
        plotdf = pd.DataFrame()
        #plotdf.columns=groups
        for index, rowdata in self.rawinput.iterrows(): #Over ALL CIT PEPTIDES
            precursor=rowdata['Precursor.Id']
            plotdf.at[index, 'Peptide']=precursor

            for group in groups:
                group_samples = self.metadata[self.metadata[feature]==group]["samplekey"].tolist()
                newcol=(rowdata[group_samples].T).astype(float)
                plotdf.at[index, group]=np.nanmean(np.log2(newcol.values))

        plotdf.to_csv('citpeptides_feature.csv')
        #figure = pd.melt(plotdf, id_vars='Peptide', value_vars= plotdf.columns[:-1])
        figure=go.Figure()
        peptides_list = plotdf['Peptide']
        plotdf_T = plotdf.set_index('Peptide').transpose()
        for col in peptides_list:
            print(plotdf_T.index)
            print(plotdf_T[col])
            figure.add_trace(go.Scatter(x=plotdf_T.index, y=plotdf_T[col],
                                        name=col))

        figure.update_layout(title='Citrullinated Peptide Level',
                             xaxis_title=feature,
                             yaxis_title='Log2[Intensity]',
                             legend_title='cit-Peptides more than '+str(MISSINGNESS_THRESH*100)+'%  Missingness',
                             font=dict(family="Tahoma",
                                       size=18,
                                       color="RebeccaPurple"))
        output=output+"NoImpute"
        #plt.savefig("./cit_pep_plot_"+feature+"_"+output+".svg", bbox_inches="tight")
        plotly.offline.plot(figure,'cit_pr_.html')
        plt.clf() 
        plt.close()

    def bar_plot(self, precursor, column, groups=[], output="", color='goldenrod',impute=None):

        stripped_precursor=precursor.replace("(UniMod:7)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:4)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:1)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:35)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:26)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:27)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:28)","_")
        rowdata=pd.DataFrame()
        outputfile="bar_cit_"+stripped_precursor+"_"+column+"_"+output

        #is_present = [i for i in  self.rawinput['Precursor.Id'] if precursor in i] 
        is_present = False
        row_index=0
        for index, pep in self.rawinput.iterrows():
            if precursor in pep['Precursor.Id']:
                is_present=True
                row_index=index
        if not is_present:
            print(precursor+' not present')
            return
        else:
            print('Bar plot for '+precursor+' ...(index=',str(row_index),') in rawdata')


        imputedata=self.mat.loc[:,precursor]
        rowdata = imputedata.T
            
        rowdata.T.to_csv("../bar_plots/csv/"+outputfile+".csv",sep=',')
        if not len(groups):
            groups = self.metadata[column].unique()

        plt.rcParams["figure.figsize"] = (30, 5)
        plotdf=[]
        plot_labels=[]
        errordf=[]
        for group in groups:
            group_samples = self.metadata[self.metadata[column]==group]["samplekey"].tolist()
            newcol=(rowdata[group_samples].T).astype(float)
            newcol=newcol.fillna(0)
            plot_labels.append(group)
            plotdf.append(np.mean(newcol.values))
            errordf.append(np.std(newcol.values))
 
        print("Bar plot for "+precursor)
        fig,ax=plt.subplots()
        ax.bar(groups,plotdf,yerr=errordf,capsize=5,
        #ax.bar(groups,plotdf,capsize=10,
               label=plot_labels,color=color,edgecolor='black',linewidth=1,width=0.9)
        plt.ylabel("%Cit/Total")
        plt.xlabel(column)
#        plt.ylim(-0.5,1.5)
        plt.xticks(rotation="vertical")
        plt.title(precursor)
        if impute is not None:
            output=output+'_impute_'+impute
        else:
            output=output+"NoImpute"
        #plt.figure(figsize=(60,10))
        #plt.figure().set_figwidth(10)
        plt.savefig("../bar_plots/"+outputfile+".svg", bbox_inches="tight")
        plt.clf() 
        plt.close()
        
    def box_plot(self, precursor, column, groups=[], color='skyblue',output="", figuresize1=12, figuresize2=6,impute=None):

        plt.rcParams["figure.figsize"] = (figuresize1, figuresize2)
        stripped_precursor=precursor.replace("(UniMod:7)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:4)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:1)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:35)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:26)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:27)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:28)","_")
        rowdata=pd.DataFrame()
        outputfile="box_cit_"+stripped_precursor+"_"+column+"_"+output

        #is_present = [i for i in  self.rawinput['Precursor.Id'] if precursor in i] 
        is_present = False
        row_index=0
        for index, pep in self.rawinput.iterrows():
            if precursor in pep['Precursor.Id']:
                is_present=True
                row_index=index
        if not is_present:
            print(precursor+' not present')
            return
        else:
            print('Box plot for '+precursor+' ...(index=',str(row_index),') in rawdata')


        imputedata=self.mat.loc[:,precursor]
        rowdata = imputedata.T
            
        rowdata.T.to_csv("../box_plots_pr/"+outputfile+".csv",sep=',')
        if not len(groups):
            groups = self.metadata[column].unique()

        plotdf=pd.DataFrame()
        print(groups)
        for group in groups:
            group_samples = self.metadata[self.metadata[column]==group]["samplekey"].tolist()
            newcol=(rowdata[group_samples].T).astype(float)
            plotdf[group]=pd.DataFrame(newcol.values)
            if impute is None:
                missingness = plotdf[group].isnull().values.sum()/plotdf[group].size*100
                print(group, "   ", missingness)

        fig, ax = plt.subplots()
        plotdf[groups].plot(kind='box', patch_artist='True',boxprops=dict(facecolor=color,alpha=1,linewidth=2),
                            medianprops=dict(color='black',linewidth=2),
                            capprops=dict(color='black',linewidth=2),
                            #flierprops=dict(marker='o', color='black', alpha=0.8, markersize=8),
                            whiskerprops=dict(color='black',linewidth=2),
                            ax=ax)

        for i, group in enumerate(groups):
            y = plotdf[group]
            #x = [i + 1] * len(y)  # Generate a list of x positions for each data point
            x = np.random.normal(i + 1, 0.04, size=len(y))  # i+1 positions the points; adjust 0.04 for less/more spread
            ax.scatter(x, y,
                       s=60,
                       c='lightgrey',
                       edgecolor='black',
                       linewidth=2,
                       zorder=4)

        plt.gca().set_facecolor('whitesmoke')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)

        #data_df = plotdf[tg_group].melt(var_name=column,value_name="Log2[Cit/Total]")
        #print(data_df)
        #snsplot=sns.boxplot(x=column,y="Log2[Cit/Total]",data=data_df)
        #snsplot=sns.swarmplot(x=column,y="Log2[Cit/Total]",data=data_df,color="black",zorder=0)
        plt.ylabel("%Cit/Total",fontsize=28)
        plt.xlabel(column,fontsize=28)
#        plt.ylim(-0.5,1.5)
        plt.xticks(rotation="vertical",fontsize=18)
        plt.yticks(fontsize=24)
        plt.grid(linestyle=':')
        plt.title(precursor,fontsize=18)
        if impute is not None:
            output=output+'_impute_'+impute
        else:
            output=output+"NoImpute"
        plt.savefig('box_cit/'+outputfile+".svg", bbox_inches="tight")
        plt.clf() 
        plt.close()
        
#####################################################################################################################

    # plot cit peptide 1 vs protein 2 and color by sample groups
    def scatter_corr_plot(self, protein_1, precursor_2, column, groups=[], output=''):

        row_index=0
        is_present = False
        for index, pep in self.rawinput.iterrows():
            if precursor_2 in pep['Precursor.Id']:
                is_present=True
                row_index=index
        if not is_present:
            print(f"Cit peptide {precursor_2} not found.")
            return np.nan
        else:
            print('---- Plotting scatter correlation plot for '+precursor_2+' and '+protein_1+' ----')

        output=protein_1+'_'+precursor_2+'_'+column+'_'+output
        plt.rcParams["figure.figsize"] = (20, 7)
        stripped_precursor=precursor_2.replace("(UniMod:7)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:4)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:1)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:35)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:26)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:27)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:28)","_")


        protein_1_df=self.get_protein_intensity(protein_1)
        precursor_2_df=self.get_cit_intensity(precursor_2,is_present)


        if not len(groups):
            groups = self.metadata[column].unique()

        # Filter the data for the two proteins
        protein_1_name=protein_1_df.columns[0]
        protein_data = protein_1_df.merge(precursor_2_df,left_index=True,right_index=True)
        #protein_data.to_csv('_corr_pr_protein_data_debug_'+column+'.csv',sep='\t')

        protein_data=protein_data.astype(float)
        protein_data=np.log2(protein_data)
    
#        protein_data.to_csv('_scatter_'+output+".csv",sep=',')
        if not len(groups): #plot all unique features in column
            groups = self.metadata[column].unique()

        # Reset the index of plotdf to ensure it's using a simple integer index
        data_to_plot=protein_data.reset_index(drop=True)
        #data_to_plot.to_csv('_corr_pr_data_to_plot_debug_'+column+'.csv',sep='\t')

        label_column = self.metadata[column].astype(str)
        #label_column.to_csv('_corr_pr_label_column_debug_'+column+'.csv',sep='\t')
        label_column_df = label_column.to_frame(name=column).reset_index(drop=True)

        label_file=self.metadata['samplekey'].astype(str)
        #label_file.to_csv('_corr_pr_label_file_debug_'+column+'.csv',sep='\t')
        label_file_df = label_file.to_frame(name='samplekey').reset_index(drop=True)


        plotdf=pd.concat([data_to_plot, label_column_df, label_file_df], axis=1)
        #plotdf.to_csv('_corr_pr_plotdf_debug_'+column+'.csv',sep='\t')

        # Assuming the necessary data preparation is done before this line as per your original code
        # plotdf is ready with 'precursor_2', 'protein_1', 'column' for color, and 'samplekey' for hover information

        plotdf_nona = plotdf.dropna(subset=[precursor_2, protein_1_name])
        plotdf=plotdf.fillna(0.0)
        # Calculate Pearson correlation coefficient
        corr_coeff, _ = pearsonr(plotdf_nona[protein_1_name], plotdf_nona[precursor_2])
        # Create an interactive scatter plot with Plotly
        #fig = px.scatter(plotdf, x=protein_1, y=precursor_2
        #                 color=column,  # Defines colors based on the feature column
        #                 hover_data=['samplekey'],  # Defines what appears on hover
        #                 title=f'Scatter plot for {protein_1} vs. {precursor_2} - Pearson Correlation: {corr_coeff:.2f}')


        # Calculate linear regression
        regression_result = linregress(plotdf_nona[protein_1_name], plotdf_nona[precursor_2])
        slope = regression_result.slope
        intercept = regression_result.intercept
        r_value = regression_result.rvalue
    
        # Prepare a line for plotting
        plotdf['regression_line'] = plotdf_nona[protein_1_name] * slope + intercept
    
        # Create the scatter plot
        fig = px.scatter(plotdf, x=protein_1_name, y=precursor_2, color=column,
                         hover_data=['samplekey'])
    
        # Add the regression line
        fig.add_traces(go.Scatter(x=plotdf[protein_1_name], y=plotdf['regression_line'],
                                  mode='lines', name='Regression Line',
                                  line=dict(dash='dot', width=1),
                                  showlegend=True))
    
        # Add annotations for the equation and R2 value
        equation_text = f'y = {slope:.2f}x + {intercept:.2f}<br>R^2 = {r_value**2:.2f}'
        fig.add_annotation(x=1, y=max(plotdf_nona[protein_1_name]), text=equation_text, showarrow=False)
    
        fig.update_layout(title=f'Scatter Plot with Linear Regression for {protein_1} vs {precursor_2} - Pearson Correlation: {corr_coeff:.2f}',
                          xaxis_title=protein_1,
                          yaxis_title=precursor_2)

        fig.write_html('cit_corr_'+protein_1+'/corr_plot_'+output + '.html')
         

        ################# Create the same plot with seaborn for presentation #############################
        # Set the color palette to use with seaborn
        unique_labels = plotdf[column].unique()

        #color blind palette
        sns.set_palette("colorblind")
        palette = sns.color_palette("bright", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    
        # Create the scatter plot with regression line
        plt.figure(figsize=(10, 7))
        sns.regplot(x=protein_1_name, y=precursor_2, data=plotdf_nona, scatter_kws={'s':50, 'alpha':0.5}, line_kws={'color':'gray', 'ls':'dotted'}, ci=None)
        sns.scatterplot(x=protein_1_name, y=precursor_2, data=plotdf, hue=column, palette=color_map, s=100, edgecolor='black', alpha=0.7)
    
        # Annotate the plot with the regression equation and R^2 value
        equation_text = f'y = {slope:.2f}x + {intercept:.2f}\nR^2 = {r_value**2:.2f}'
        plt.annotate(equation_text, xy=(1, 0), xycoords='axes fraction', ha='right', va='bottom', fontsize=12,
        #plt.annotate(equation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.5))
    
        plt.title(f'{protein_1} vs {precursor_2} \n Pearson Corr = {corr_coeff:.2f}',fontsize=16)
        plt.xlabel(protein_1,fontsize=18)
        plt.ylabel(precursor_2,fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
    
        # Show the plot
        plt.legend(title=column,fontsize=16)
        plt.tight_layout()
    
        # Optionally, save the plot if an output file path is provided
        plt.savefig('cit_corr_seaborn_plot_'+protein_1+'/corr_seaborn_plot_'+output+'.svg')  # Saves the plot

        plt.clf() 
        plt.close()

        return corr_coeff
       
#######################################################################################################################################

    def bar_plot_stats(self, stats,feature, doe_file,color='dodgerblue'):

        #First read stats
        stats_mat = pd.read_excel(doe_file)
        stats_mat[stats]=stats_mat[stats].astype(float)

        #Second group by samples in feature
        groups = self.metadata[feature].unique()
        plotdf=[]
        plot_labels=[]
        errordf=[]
        #plt.rcParams["figure.figsize"] = (10, 5)
        for group in groups:
            group_samples = self.metadata[self.metadata[feature]==group]["samplekey"].tolist()
            newcol = []
            for sample in group_samples:
                subset = stats_mat[stats_mat['samplekey']==sample]
                newcol.append(subset[stats].values)

            col_df=pd.DataFrame(newcol)
            plot_labels.append(group)
            plotdf.append(np.mean(col_df.values))
            errordf.append(np.std(col_df.values))
 
        print("Bar plot for "+stats+" with "+feature)
        fig,ax=plt.subplots()
        ax.bar(groups,plotdf,yerr=errordf,capsize=5,
               label=plot_labels,color=color,alpha=0.7,edgecolor='black',linewidth=1,width=0.9)
        plt.ylabel(stats,fontsize=16)
        #plt.ylim(8500,9500)
        #plt.xlabel(feature,fontsize=16)
        plt.xticks(rotation="vertical",fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(linestyle=':')
        plt.title('Bar Plot for '+str(stats)+' vs. '+str(feature),fontsize=16)
        plt.savefig('bar_stats_'+stats+'_'+feature+'.svg', bbox_inches="tight")
        plt.clf() 
        plt.close()
        

#####################################################################################################################
def read_cit(peptides_file):
   peptides = pd.read_csv(peptides_file, delimiter='\t', dtype=str, header=None, names=['Cit.Peptides'])
   peptides_df = pd.DataFrame(peptides)
   return peptides_df


"""
#####################################################################################################################
"""

#""" First read in the list of targeted peptides of interest """
#cit=read_cit('targeted_cit_peptides_list.txt')
#print('LIST OF PRIORITY CIT PEPTIDES FOR PAD4')
#print(cit)


print("LOAD PR_MATRIX")
pr_mat=Loader(input_file="../raw/report.pr_matrix.tsv",intensity_column="[sample]", index_column="Precursor.Id")
pr_mat.print_load_parameters()

doe_file='EPJ_241126_RAKIR_GS134_LPS_Peritonitis_PD_PAD24_DKO_DOE'
doe_file_path='../DOE/'+doe_file+'.xlsx'
ds = DataSet(
    loader=pr_mat,
    metadata_path=doe_file_path,
    sample_column="sample"
)
"""Print out Metadata"""
ds.metadata.to_csv(doe_file+".csv",sep='\t')
print(ds.metadata)
"""Print out rawinput - more columns that ds.mat""" 
print(ds.rawinput)

"""Print out precursor-data matrix - transposed of rawdata"""
""" samples (row) x peptides (column) """
print(ds.mat)

############################ drop control sample ###########################
#sample_list_to_drop=['_P1_', '_P3_','_P4_']
#print('control samples to drop = ',sample_list_to_drop)
#ds._drop_samples(sample_list_to_drop)
##print(ds.rawinput)
#ds.rawinput.to_csv("report.pr_matrix_cit_for_analysis_samples_dropped.tsv",sep='\t')
#ds.metadata.to_csv("metadata_samples_dropped.csv",sep='\t')
####################### DONE drop control sample ###########################

"""needs to analyze missingness before preprocessing"""
column="Row"
print(ds.metadata[column].unique())
ds.analyze_missingness(feature=column, cit_only=True,output="")
column="Column"
print(ds.metadata[column].unique())
ds.analyze_missingness(feature=column,cit_only=True,output='')

"""UMAP for Dimensionality Reduction"""
output='all_peptides'
column="Contrast"
ds.plot_umap(group=column, output=output)

""" Preprocess data (log2 transform etc.) """
ds.preprocess_data(log2_transform=False,imputation=IMPUTE_METHOD)

groups=[
        'WT_1h',
        'WT_LPS_1h',
        'DKO_LPS_1h',
        'DKO_LPS_PBS_1h',
        'DKO_LPS_rhPAD2_1h',
        'DKO_LPS_rhPAD4_1h',
        'DKO_LPS_rhPAD2_rhPAD4_1h',
        'WT_3h',
        'WT_LPS_3h',
        'DKO_LPS_3h',
        'DKO_LPS_PBS_3h',
        'DKO_LPS_rhPAD2_3h',
        'DKO_LPS_rhPAD4_3h',
        'DKO_LPS_rhPAD2_rhPAD4_3h']

column="Contrast"
ds.analyze_cit_IDs(column,groups=groups)
exit()

"""HEATMAPS"""
output=''
ds.plot_heatmap(feature='Contrast',output=output)

"""PCA"""
#column='PAD'
#ds.plot_pca(column=column, output='')

"""


    NOTE: Before scatter plots need to comment out take_cit_ratio
          Make sure cit_ratio not taken and rawdata contains raw intensities


"""
"""SCATTER PLOTS"""
"""
citfile='../cit_peptides_list_'+str(100*MISSINGNESS_THRESH)+'missingness_.txt'
column='SampleType_PAD2_PAD4'
protein_1='Padi4'
all_cit=pd.read_csv(citfile,delimiter='\t',dtype=str, header=None, names=['Cit.Peptides'])
with open('_pearson_' + str(100 * MISSINGNESS_THRESH) + 'missingness_' + protein_1 + '.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for pep in all_cit['Cit.Peptides']:
        corr_coeff=ds.scatter_corr_plot(protein_1, pep,column)
        if (corr_coeff):
            writer.writerow([protein_1, pep, corr_coeff])
f.close()
"""


"""
******NOTE******
Using matrix after preprocessing so imputation is determined above
impute option here only dictates file name
"""
"""


    NOTE: Before box plots need to uncomment take_cit_ratio
          Make sure cit_ratio is taken


"""
print("\n\n Creating box plots for cit peptides of interest")
column='Contrast'
print(ds.metadata[column].unique())
citfile='../cit_peptides_list_'+str(100*MISSINGNESS_THRESH)+'missingness_.txt'
all_cit=pd.read_csv(citfile,delimiter='\t',dtype=str, header=None, names=['Cit.Peptides'])

output='_allcit_PAD_Inhib'
groups=[
        'WT_1h',
        'WT_LPS_1h',
        'DKO_LPS_1h',
        'DKO_LPS_PBS_1h',
        'DKO_LPS_rhPAD2_1h',
        'DKO_LPS_rhPAD4_1h',
        'DKO_LPS_rhPAD2_rhPAD4_1h',
        'WT_3h',
        'WT_LPS_3h',
        'DKO_LPS_3h',
        'DKO_LPS_PBS_3h',
        'DKO_LPS_rhPAD2_3h',
        'DKO_LPS_rhPAD4_3h',
        'DKO_LPS_rhPAD2_rhPAD4_3h']

for pep in all_cit['Cit.Peptides']:
    ds.box_plot(precursor=pep,column=column, groups=groups,color='magenta',impute=IMPUTE_METHOD,output='cit_ratio'+output,figuresize1=30,figuresize2=6)

exit()
