#!/home/reynoa12/localbin/bin/python3.9


import pandas as pd
import numpy as np
import sklearn
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#from ppca import PPCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import scipy
import re
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import umap.umap_ as umap

MISSINGNESS_THRESH=0.5
SAMPLE_MISSINGNESS_THRESH=1
IMPUTE_METHOD=None
IS_LOG2=True

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
        self.mat = np.log2(self.mat.astype(float) + 0.00001)
        self.preprocessing_info.update({"Log2-transformed":True})
        print("Data has been log2-transformed")

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

        self.mat=self.mat.loc[:, (self.mat != 0).any(axis=0)]
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
        #### OK this is for precursor search pr matrix assuming looking at DBIA
        rawinput = pd.read_csv(input_file, sep='\t', dtype=str)
        dbia= []
        pg_pr=[]
        rawinput_filter = []
        print("Filtering out non-DBIA labeled peptides and DBIA labeled peptides with more than ",str(MISSINGNESS_THRESH)," when loading pr_matrix...")
        """Number of samples = number of columns - 10"""
        with open ('../DBIA_labeled_peptides_list_'+str(100*MISSINGNESS_THRESH)+'percent_missingness.txt', 'w') as f:
            for index,peptide in rawinput.iterrows():
                percent_missing=peptide.isnull().values.sum()/(peptide.size-10)
                if percent_missing<MISSINGNESS_THRESH:
                 
                    rawinput_filter.append(peptide)
                    if pd.isnull(peptide['Precursor.Id']) == False and '(DBIA)' in peptide['Precursor.Id']:
                        #print(peptide)
                        pg_pr_info=str(peptide['Genes'])+"___"+str(peptide['Precursor.Id'])
                        pg_pr.append(pg_pr_info)
                        dbia.append(peptide)
                        f.write(pg_pr_info+"\n")
        f.close()

        self.rawinput = pd.DataFrame(dbia)
        self.rawinput['Precursor.Id']=pg_pr
        self.rawinput.reset_index(inplace=True)
        #print(self.rawinput.index)
        #exit()
        self.rawinput.to_csv("report.pr_matrix_DBIA_for_analysis.tsv",sep='\t')
        print("Done loading DBIA only pr matrix. Number of precursors with DBIA = ", self.rawinput.shape[0])
        self.intensity_column=intensity_column
        self.index_column=index_column

        print("Number of precursors before filtering missingness = ",len(rawinput))
        print("Number of precursors after filtering missingness = ",len(rawinput_filter))

        n_dbia_peptides=len(self.rawinput['Modified.Sequence'].unique())
        print('Number of unique DBIA labeled peptides after filtering missingness =',n_dbia_peptides)

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

    def unpairedttest(self, imputed_mat) -> pd.DataFrame:
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
        mat_transpose=imputed_mat.transpose()

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

    def pairedttest(self, imputed_mat) -> pd.DataFrame:
        print("****** Starting t-test ***********")
        group1_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"].tolist()
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"])
        group2_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"].tolist()
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"])
        #group1_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group1][
        #                                       self.dataset.sample].tolist() 
        #group2_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group2][
        #                                       self.dataset.sample].tolist() 

        #transpose mat, now [PG x samples] (row=proteins, column=samples)
        mat_transpose=imputed_mat.transpose()
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

    def diff_expression_analysis(self, imputed_data) -> pd.DataFrame:

        #print("Starting differential expression analysis with paired t-test")
        #df = self.pairedttest(imputed_data)
        print("Starting differential expression analysis with unpaired t-test")
        df = self.unpairedttest(imputed_data)
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


    def analyze_missingness(self, feature, dbia_only=False, output=""):
        """
        This function creates box plots with missingness values for different sample groups
        based on feature in DOE specified from input
        """
        
        percent_missingness = []
        color='linen'
        if dbia_only:
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
        figure=plotdf.plot(kind='box', patch_artist='True',boxprops=dict(facecolor=color),
                                               medianprops=dict(color='black'),
                                               capprops=dict(color='black'),
                                               flierprops=dict(color='black', markerfacecolor='white'),
                                               whiskerprops=dict(color='black'))
        plt.ylabel("Percent Missingness")
        plt.xlabel(feature)
        plt.xticks(rotation="vertical")
        plt.grid(linestyle=':')
        plt.ylim(0,100)
        plt.title('Data missingness with sample groups: dbia_only='+str(dbia_only))
        figure_file="PR_missingness_dbia_"+str(feature)+"_"+str(dbia_only)+output+".svg"
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

    def plot_heatmap(self, output='eluate_peptides_heatmap.png'):
        data=self.rawinput.drop(['Protein.Group','Protein.Ids','Protein.Names','Genes','First.Protein.Description','Proteotypic',
                                 'Stripped.Sequence','Modified.Sequence','Precursor.Charge','Precursor.Id'],axis=1)
        #data=data.set_index('Protein.Group')
        maskna=0.00001
        data=data.astype(float)
        data=np.log2(data+maskna)
        data=data.fillna(maskna)
        data=data.T.drop(index='index')
        figure=px.imshow(data, width=6000,height=1000)
        pio.write_html(figure,'dbia_pr_heatmap_T.html')

        #plt.savefig(output)
        #plt.clf()
        #plt.close()

    def plot_pca(self, group=None,output="pca.png", marker="o",circle=False):
        pca=sklearn.decomposition.PCA(n_components=2)
        components=pca.fit_transform(self.mat) #these are the n_components  egenvectors
        """Transform components into a DataFrame for seaborn plot"""
        plotdf = pd.DataFrame(components, columns=['PCA1', 'PCA2'])
        label_column = self.metadata[group].astype(str)
        plotdf = plotdf.join(label_column)
        label_file=self.metadata['samplekey'].astype(str)
        plotdf = plotdf.join(label_file)

        figure=px.scatter(plotdf,x='PCA1',y='PCA2',color=group,hover_name="samplekey")
        figure.update_traces(marker_size=20)
        figure.update_layout(
            title='PCA Protein Group Level : '+' Impute with '+IMPUTE_METHOD,
            xaxis_title="PCA 1(%.2f%%)" % (pca.explained_variance_ratio_[0]*100),
            yaxis_title="PCA 2(%.2f%%)" % (pca.explained_variance_ratio_[1]*100),
            legend_title=group,
            font=dict(
                family='Tahoma',
                size=18,
                color='RebeccaPurple'
            )
        )
        pio.write_html(figure,'pca_'+str(group)+"_"+output+'.html')

        plt.clf() 
        plt.close()

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

        plotdf = pd.DataFrame(data=x_umap,columns=['col1','col2'])
        label_column = self.metadata[group].astype(str)
        plotdf = plotdf.join(label_column)
        label_file=self.metadata['samplekey'].astype(str)
        plotdf = plotdf.join(label_file)

        figure=px.scatter(plotdf,x='col1',y='col2',color=group,hover_name="samplekey")
        figure.update_traces(marker_size=7)
#        snsplot=sns.scatterplot(data=plotdf, x="col1", y="col2", hue=group, edgecolor="black", s=40, alpha=0.8)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(loc="best", bbox_to_anchor=(1,1))
 #       print(umap_df)
        output_file='umap_'+group+'_nneigh'+str(nneigh)+'_mindist'+str(mindist)+'_'+output
#        plt.savefig('./'+output_file+'.svg', bbox_inches="tight")
        pio.write_html(figure,output_file+'.html')
        plt.clf() 
        plt.close()

    def volcano_plot(self, group1, group2, column,group1_cutoff=50,group2_cutoff=50, figuresize1=5,figuresize2=8, output="", method="ttest"):

        plt.rcParams["figure.figsize"] = (figuresize1,figuresize2)
        print("Volcano plot for contrasts: "+group1+" vs "+group2+" using column "+column)
        outputfile='volcano_'+group1+'_'+group2+output

        imputed_mat = self.mat.copy(deep=True)
        imputed_mat = imputed_mat.astype(float)
        group1_samples = self.metadata[self.metadata[column]==group1]["samplekey"].tolist()
        group2_samples = self.metadata[self.metadata[column]==group2]["samplekey"].tolist()
        concat_samples = group1_samples+group2_samples

        ####compute missingness for features in group1 and group 2 samples
        NA_rate=self.mat.loc[concat_samples].isnull().sum()/self.mat.loc[concat_samples].shape[0]*100

        #print(imputed_mat.loc[concat_samples].shape)
        #drop_index=[]
        #for index, value in NA_rate.items():
        #    if value > group1_cutoff:
        #        imputed_mat.drop(columns=[index], inplace=True)
        #        drop_index.append(index)
        #print('after',imputed_mat.loc[concat_samples].shape)
        #NA_rate.drop(labels=drop_index, inplace=True)

        ####compute missingness for features in group1 samples
        NA_rate=self.mat.loc[group1_samples].isnull().sum()/self.mat.loc[group1_samples].shape[0]*100

        print(imputed_mat.loc[group1_samples].shape, NA_rate)
        drop_index=[]
        for index, value in NA_rate.items():
            if value > group1_cutoff:
                imputed_mat.drop(columns=[index], inplace=True)
                drop_index.append(index)
        print('Group1: after',imputed_mat.loc[group1_samples].shape)

        NA_rate.drop(labels=drop_index, inplace=True)
        print('Group1', NA_rate.size)

        ####compute missingness for features in group2 samples
        NA_rate=imputed_mat.loc[group2_samples].isnull().sum()/imputed_mat.loc[group2_samples].shape[0]*100
        print(imputed_mat.loc[group2_samples].shape, NA_rate)
        drop_index=[]
        for index, value in NA_rate.items(): ###this NA_rate is after dropping peptides with high missingness
            if value > group2_cutoff:
                imputed_mat.drop(columns=[index], inplace=True)
                drop_index.append(index)
        print('Group2: after',imputed_mat.loc[group2_samples].shape)
        NA_rate.drop(labels=drop_index, inplace=True)
        print('Group2', NA_rate.size)


        #impute data per column of proteins for each sample group
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, keep_empty_features=True)
        #imp = sklearn.impute.IterativeImputer(missing_values=np.nan, sample_posterior=True,
        #                                      min_value=0,verbose=0, keep_empty_features=True)

      
        """
        for group in self.metadata[column]:
            group_samples = self.metadata[self.metadata[column]==group]["samplekey"].tolist()
            imputation_array = imp.fit_transform(imputed_mat.loc[group_samples].values)
            imputed_mat.loc[group_samples]= pd.DataFrame(
                         imputation_array, index=imputed_mat.loc[group_samples].index, columns=imputed_mat.loc[group_samples].columns
            )
        """
        ####Only need to impute group 1 and group 2 samples
        imputation_array_1 = imp.fit_transform(imputed_mat.loc[group1_samples].values)
        imputed_mat.loc[group1_samples]= pd.DataFrame(imputation_array_1,
                                            index=imputed_mat.loc[group1_samples].index, columns=imputed_mat.loc[group1_samples].columns)
        imputation_array_2 = imp.fit_transform(imputed_mat.loc[group2_samples].values)
        imputed_mat.loc[group2_samples]= pd.DataFrame(imputation_array_2,
                                            index=imputed_mat.loc[group2_samples].index, columns=imputed_mat.loc[group2_samples].columns)

        stat = Statistics(dataset=self, group1=group1, group2=group2, column=column, method=method)
        plotdf=stat.diff_expression_analysis(imputed_mat)

        plotdf['negLogP']=-np.log10(plotdf['adj.pval'])
        plotdf['NA_rate']=NA_rate.values
        #print("PRINT PLOTDF")
        #print(plotdf)
        #plotdf.to_csv('full_tables/table_'+outputfile+'.csv',sep=',')
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
            xaxis_range=[-6,6], ########################### X AXIS RANGE FOR VOLCANO PLOTS ####################
            yaxis_range=[0,10], ########################### Y AXIS RANGE FOR VOLCANO PLOTS ####################
            font=dict(
                family='Tahoma',
                size=24,
                color='RebeccaPurple'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='#F0F8FF',
                gridwidth=1,
                tickfont=dict(size=40),
                showticklabels=True
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#F0F8FF',
                gridwidth=1,
                tickfont=dict(size=40),
                showticklabels=True
            )
        )
        figure.add_hline(y=2,line_width=2, line_dash="dot",line_color='gray')
        figure.add_vline(x=1, line_width=2, line_dash="dot",line_color='gray')
        figure.add_vline(x=-1, line_width=2, line_dash="dot",line_color='gray')
        figure.update_xaxes(showline=True,linewidth=2,linecolor='black',zeroline=True)
        figure.update_yaxes(showline=True,linewidth=2,linecolor='black',zeroline=True)
        pio.write_html(figure,'interactive_freescale_'+outputfile+'.html')
        #pio.write_html(figure,'../volcano_plots/'+outputfile+'.html')

        ################################### MAKE STATIC PLOTS ##########################################
        """ADJ.P VALUE AND LOG2FC threshhold"""
        p_thresh=2
        fc_thresh=2
        sigval=[] ####Precursors with adjusted pval > p_thresh and |log2FC|>fc_thresh

        fig, ax = plt.subplots()
        ax.scatter(data=plotdf, x="log2fc", y="negLogP", edgecolor="darkgray", color="lightgray",s=5, alpha=1)
        outputfile=outputfile+"_p"+str(p_thresh)+"_fc"+str(fc_thresh)
        plt.axhline(p_thresh, linewidth=1, color="gray", linestyle=':')
        plt.axvline(fc_thresh, linewidth=1, color="gray", linestyle=':')
        plt.axvline(-fc_thresh, linewidth=1, color="gray", linestyle=':')
        for index, line in plotdf.iterrows():
            sequence=line[0]
            #red for upmodulation
            if line['negLogP'] > p_thresh and line['log2fc']>fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="red", color="red",s=5, alpha=1)
                sigval.append(line) 
            #blue for downmodulation
            elif line['negLogP'] > p_thresh and line['log2fc']<-fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="blue", color="blue",s=5, alpha=1)
                sigval.append(line) 

            #if line['log2fc']>4:
            #    plt.annotate(sequence, (line["log2fc"], line["negLogP"]), fontsize=2, color="red")
            #if line['log2fc']>1 and line['negLogP'] > 5:
            #    plt.annotate(sequence, (line["log2fc"], line["negLogP"]), fontsize=2, color="red")
            #if line['log2fc']<-1 and line['negLogP'] > 5:
            #    plt.annotate(sequence, (line["log2fc"], line["negLogP"]), fontsize=2, color="blue")
            #elif line['log2fc']<-4:
            #    plt.annotate(sequence, (line["log2fc"], line["negLogP"]), fontsize=2, color="blue")
                
        for index, line in plotdf.iterrows():
            sequence=line[0]
            if "TDP1" in str(sequence):
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="black", color="hotpink",s=15, alpha=1)
                print(sequence, line["log2fc"], line["negLogP"])
                plt.annotate(sequence, (line["log2fc"], line["negLogP"]), fontsize=6, color="orange")

        sigval_df=pd.DataFrame(sigval)
        sigval_df['Protein.Names']=self.rawinput['Protein.Names'].loc[sigval_df.index]
        sigval_df['Protein.Description']=self.rawinput['First.Protein.Description'].loc[sigval_df.index]
#        sigval_df.to_csv("volcano_plots_csv/significant_pr_list_"+outputfile+".csv",sep=',')
        plt.xlabel("log2 FC", fontsize=18)
        plt.ylabel("-log10(adj.p value)", fontsize=18)
        plt.title(group1+'_'+group2, fontsize=18)
        plt.ylim(0,15)
        plt.xlim(-6,6)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        #plt.grid()
        #plt.legend(loc="best", bbox_to_anchor=(1,1))

        plt.savefig("static_"+outputfile+"_wide_xlim.svg", bbox_inches="tight")
        plt.clf() 
        plt.close()
        
    def box_plot(self, precursor, column, groups=[], output="", impute=None,color='goldenrod',ylim1=-1, ylim2=-1, figuresize1=10,figuresize2=3):

        stripped_precursor=precursor.replace("(UniMod:7)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:4)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:21)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:1)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:35)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:26)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:27)","_")
        stripped_precursor=stripped_precursor.replace("(UniMod:28)","_")
        rowdata=pd.DataFrame()
        outputfile="box_dbia_"+stripped_precursor+"_"+column+"_"+output
        if figuresize1>0:
            plt.rcParams["figure.figsize"] = (figuresize1, figuresize2)
        else:
            plt.rcParams["figure.figsize"] = (10, 3)
        rowdata.T.to_csv("box_plots/csv/"+outputfile+".csv",sep=',')

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
            
        rowdata.T.to_csv("box_plots/csv/"+outputfile+".csv",sep=',')
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
        plotdf[groups].plot(kind='box', patch_artist='True',boxprops=dict(facecolor=color,alpha=0.7),
                            medianprops=dict(color='black'),
                            capprops=dict(color='black'),
                            flierprops=dict(color='black', markerfacecolor=color, markersize=2),
                            whiskerprops=dict(color='black'),
                            ax=ax)

        for i, group in enumerate(groups):
            y = plotdf[group]
            #x = np.random.normal(i + 1, 0.02, size=len(y))  # i+1 positions the points; adjust 0.04 for less/more spread
            x = [i + 1] * len(y)  # Generate a list of x positions for each data point
            ax.scatter(x, y, color='black', alpha=0.9, s=12, zorder=3)

        if(ylim1>0):
            plt.ylim(ylim1,ylim2)
        plt.ylabel("Log2[Intensity]",fontsize=18)
        plt.xlabel(column,fontsize=18)
        plt.xticks(rotation="vertical",fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(linestyle=':')
        plt.title(precursor, fontsize=18)
        if impute is not None:
            output=output+'_impute_'+impute
        else:
            output=output+"NoImpute"
        plt.savefig("box_plots/"+outputfile+".svg", bbox_inches="tight")
        plt.clf() 
        plt.close()

        
"""
#####################################################################################################################
"""

""" First read in the list of targeted peptides of interest """
print("LOAD PR_MATRIX")
pr_mat=Loader(input_file="../../raw/report.pr_matrix.tsv",intensity_column="[sample]", index_column="Precursor.Id")
pr_mat.print_load_parameters()

print('LOAD DATASET')
doe_file='EPJ-241211-MAPUF_RCP_DOE'
ds = DataSet(
    loader=pr_mat,
    metadata_path="../../DOE/"+doe_file+".xlsx",
    sample_column="samplekey"
)
"""Print out Metadata"""
ds.metadata.to_csv(doe_file+".csv",sep='\t')
#print(ds.metadata)
"""Print out rawinput - more columns that ds.mat""" 
#print(ds.rawinput)

"""Print out precursor-data matrix - transposed of rawdata"""
""" samples (row) x peptides (column) """
#print(ds.mat)

### drop samples with high missingness 
#sample_list_to_drop=['P02B06','P02B11','P02G04']
#print('control samples to drop = ',sample_list_to_drop)
#ds._drop_samples(sample_list_to_drop)
#ds.rawinput.to_csv("report.pr_matrix_for_analysis_samples_dropped.tsv",sep='\t')
#ds.metadata.to_csv("metadata_samples_dropped.csv",sep='\t')

""" MISSINGNESS ANALYSIS """
#column="Row"
#ds.analyze_missingness(feature=column, dbia_only=True,output="")
#ds.plot_umap(group=column)
#column="Column"
#ds.analyze_missingness(feature=column, dbia_only=True,output="")
#ds.plot_umap(group=column)
#column="Contrast"
#ds.analyze_missingness(feature=column, dbia_only=True,output="")
#ds.plot_umap(group=column)
#column="Compound"
#ds.analyze_missingness(feature=column, dbia_only=True,output="")
#ds.plot_umap(group=column)

""" Preprocess data (log2 transform etc.) """
ds.preprocess_data(log2_transform=IS_LOG2,imputation=IMPUTE_METHOD)

#""" Draw heatmap """
#ds.plot_heatmap()

column="Contrast"
groups=ds.metadata[column].unique().copy().astype(str)
print(ds.metadata[column].unique())

#######################################################################################################################
groups=['DMSO_4h',
        'EXT-004525_1uM_4h',
        'EXT-004525_10uM_4h',
        'EXT-004525_30uM_4h',
        'BMT-885960_1uM_4h',
        'BMT-885960_10uM_4h',
        'BMT-885960_30uM_4h',
        'BMT-945203_1uM_4h',
        'BMT-945203_10uM_4h',
        'BMT-945203_30uM_4h',
        'DMSO_24h',
        'EXT-004525_1uM_24h',
        'EXT-004525_10uM_24h',
        'EXT-004525_30uM_24h',
        'BMT-885960_1uM_24h',
        'BMT-885960_10uM_24h',
        'BMT-885960_30uM_24h',
        'BMT-945203_1uM_24h',
        'BMT-945203_10uM_24h',
        'BMT-945203_30uM_24h']


#######################################################################################################################

#******NOTE******
#Using matrix after preprocessing so imputation is determined above
#impute option here only dictates file name

#proteins=['TDP1']
#for protein in proteins:
#    peptides=ds.rawinput[ds.rawinput['Genes']==protein]['Precursor.Id']
#    for pep in peptides:
#        ds.box_plot(precursor=pep,column=column, groups=groups, ylim1=8, ylim2=14, figuresize1=12, figuresize2=7, impute=IMPUTE_METHOD)
#
#proteins=['BRAT1']
#for protein in proteins:
#    peptides=ds.rawinput[ds.rawinput['Genes']==protein]['Precursor.Id']
#    for pep in peptides:
#        ds.box_plot(precursor=pep,column=column, groups=groups, ylim1=-8, ylim2=14, figuresize1=12, figuresize2=7, impute=IMPUTE_METHOD)

#proteins=['FAM136A', 'THBS1','EEF1A1']
#proteins=['PUM1', 'MYH14','SNRNP200','TOR1A1P2','SETD5','OGT','FYCO1','IDH3G']
proteins=['FANCD2']
for protein in proteins:
    peptides=ds.rawinput[ds.rawinput['Genes']==protein]['Precursor.Id']
    for pep in peptides:
        ds.box_plot(precursor=pep,column=column, groups=groups, ylim1=-8, ylim2=14, figuresize1=12, figuresize2=7, impute=IMPUTE_METHOD)
exit()
"""##################### VOLCANO PLOTS ########################"""
group1_cutoff=15
group2_cutoff=75

column='Contrast'
group1='DMSO_4h'
group2='DMSO_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")

group2='EXT-004525_1uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2='EXT-004525_10uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2='EXT-004525_30uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")

group2= 'BMT-885960_1uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'BMT-885960_10uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'BMT-885960_30uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")

group2= 'BMT-945203_1uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'BMT-945203_10uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'BMT-945203_30uM_4h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")

###
group1= 'DMSO_24h'
group2= 'EXT-004525_1uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'EXT-004525_10uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'EXT-004525_30uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")

group2= 'BMT-885960_1uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'BMT-885960_10uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'BMT-885960_30uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")

group2= 'BMT-945203_1uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'BMT-945203_10uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
group2= 'BMT-945203_30uM_24h'
ds.volcano_plot(group1, group2, column, group1_cutoff=group1_cutoff,group2_cutoff=group2_cutoff,output="", method="ttest")
