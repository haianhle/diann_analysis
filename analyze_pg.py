#!/home/reynoa12/localbin/bin/python3.9


import pandas as pd
import numpy as np
import sklearn
import sklearn.impute
from sklearn.preprocessing import StandardScaler
#from ppca import PPCA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Union
import scipy
from scipy.stats import pearsonr
from scipy.stats import linregress
from scipy.optimize import minimize
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import csv
import re

MISSINGNESS_THRESH=0.3
SAMPLE_MISSINGNESS_THRESH=1
IMPUTE_METHOD=None
MAX_COL_LENGTH=20 #in box_plot
IS_LOG2=True


def trim_string(input_string):
    # Define the pattern to match special characters
    pattern = r'[^_a-zA-Z0-9\s]'
    # Use regex to substitute special characters with an empty string
    output_string=re.sub(pattern, '', input_string)
    return output_string

# Function to calculate determinant of the covariance matrix
def det_cov(x, df, missing_indices):
    # Convert the DataFrame to numpy array for faster operations
    array_copy = df.values.copy()
            
    # Update the numpy array directly
    for value, (index, col) in zip(x, missing_indices):
        array_copy[index, col] = value

    # Calculate the covariance matrix using numpy
        cov_matrix = np.cov(array_copy, rowvar=False)
                                        
    # Return the determinant of the covariance matrix
        return np.linalg.det(cov_matrix)




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
                    self.metadata= self.metadata.drop(index,axis=0)

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

    def _imputation(self, method="mean"):
        print("Imputing data with "+method)
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy=method, fill_value=0)
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

    def preprocess_data(
        self,
        log2_transform: bool=True,
        normalization: str="quantile",
        imputation: str="mean",
        remove_samples: list=None
        ):

        #self._obtain_numeric_matrix(sample_list=remove_samples)
        """NORMALIZATION HERE IS SUSPICIOUS"""
        if normalization is not None:
            print("Normalizing...")
            #self._normalization(method=normalization)
        if imputation is not None:
            print("Imputing...")
            self._imputation(method=imputation)
        if log2_transform:
            print("Log2 transforming...")
            self._log2_transform()

        #filters out columns that contain only zeros
        #self.mat=self.mat.loc[:, (self.mat != 0).any(axis=0)]
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
            index_column = Protein.Group (or Modified.Sequence)
        """
        self.input_file = input_file
        rawinput = pd.read_csv(input_file, sep='\t', dtype=str)
        #print(self.rawinput.index)
        self.intensity_column=intensity_column
        self.index_column=index_column

        pg_gene =[]
        ##### FILTERING OUT SAMPLES #####
        print("Filtering out samples with more than ",str(SAMPLE_MISSINGNESS_THRESH)," when loading pg_matrix...")
        total_protein=rawinput.shape[0]
        print('Number of Protein Groups in report.pg_matrix.tsv = ',total_protein)
        """Number of samples = number of columns - 10"""
        number_samples=rawinput.shape[1]-5
        print('Number of Samples Before Filtering Samples Missingness = ',number_samples)
        bad_samples=[]
        for column in rawinput:
            if column not in ['Protein.Group','Protein.Ids','Protein.Names','Genes','First.Protein.Description']:
                percent_missing=rawinput[column].isnull().values.sum()/total_protein
                #print(column, percent_missing)
                if percent_missing > SAMPLE_MISSINGNESS_THRESH:
                    print(column, percent_missing)
                    bad_samples.append(column)

        if bad_samples:
            print('Found ',len(bad_samples), ' Samples with Missingness more than Cutoff ',SAMPLE_MISSINGNESS_THRESH)
        rawinput=rawinput.drop(labels=bad_samples,axis=1) #axis=1 is column

        rawinput_filter = []
        number_samples=rawinput.shape[1]-5 ###should be number of samples after filtering bad samples out

        print('Filtering out protein groups with more than ',str(MISSINGNESS_THRESH)," out of ",number_samples,
              'samples  when loading pg_matrix...')
        outputfile='_protein_groups_high_missing_'+str(MISSINGNESS_THRESH)
        outputfile=trim_string(outputfile)
        with open(outputfile+'.txt','w',newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for index,protein in rawinput.iterrows():
                pg_gene_info=str(protein['Genes'])+"___"+str(protein['Protein.Ids'])
                number_missing=protein.isnull().values.sum()
                percent_missing=number_missing/number_samples
                if (number_missing > number_samples):
                    print("**** ERROR!!!! More missing values than total number of samples in raw input filtering")
                    exit()
                if percent_missing<MISSINGNESS_THRESH:
                    pg_gene.append(pg_gene_info)
                    rawinput_filter.append(protein)
                else:
                    writer.writerow([pg_gene_info, percent_missing])

        self.rawinput = pd.DataFrame(rawinput_filter)
        self.outlier_samples = bad_samples #### samples with more than cutoff missing data
        self.rawinput['Protein.Ids']=pg_gene
        self.rawinput['Protein.Group']=pg_gene
        self.rawinput.reset_index(inplace=True)
        print("Done loading pg matrix. Number of protein groups remaining = ", self.rawinput.shape[0])
        self.intensity_column=intensity_column
        self.index_column=index_column

        print("Number of protein groups before filtering missingness = ",len(rawinput))
        print("Number of proteing roups after filtering missingness = ",len(rawinput_filter))

    def print_load_parameters(self):
        print("Input File = "+self.input_file)
        print("Intensity Column = "+self.intensity_column)
        print("Index Column = "+self.index_column)
        print("Bad Samples = ",self.outlier_samples)


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

    def unpairedttest(self, imputed_mat:pd.DataFrame) -> pd.DataFrame:
        print("****** Starting independent t-test ***********")
        group1_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"].tolist()
        print("group1_samples")
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"])
        group2_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"].tolist()
        print("group2_samples")
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"])
        mat_transpose=imputed_mat.transpose()
        ###print(mat_transpose)
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

    def pairedttest(self, imputed_mat:pd.DataFrame) -> pd.DataFrame:
        print("****** Starting t-test ***********")
        group1_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"].tolist()
        print("group1_samples")
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group1]["samplekey"])
        group2_samples = self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"].tolist()
        print("group2_samples")
        print(self.dataset.metadata[self.dataset.metadata[self.column]==self.group2]["samplekey"])
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
            fc=np.log2(fc)

        print("Finished calculating fold change group2/group1")
        return fc

    def diff_expression_analysis(self, imputed_mat:pd.DataFrame) -> pd.DataFrame:

        print("Starting differential expression analysis with unpaired t-test")
        df = self.unpairedttest(imputed_mat)
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

    def analyze_missingness(self, feature, ylim0=50, output=''):
        plt.rcParams["figure.figsize"] = (10, 2)
        percent_missingness = []
        color='mediumpurple'
        for index, sample in self.mat.iterrows():
            percent = 100*sample.isnull().values.sum()/sample.size
            percent_missingness.append(percent)

        missingness_df = pd.DataFrame(percent_missingness,columns=['Missingness'])
        missingness_df['Sample'] = self.sample
        missingness_df[feature] = self.metadata[feature]
        missingness_df.set_index('Sample')
        missingness_df['Missingness'] = missingness_df['Missingness'].astype(float)

        groups = self.metadata[feature].unique()

        plotdf=pd.DataFrame()
        for group in groups:
            group_samples = self.metadata[self.metadata[feature]==group]["samplekey"].tolist()
            newcol = []
            for sample in group_samples:
                subset = missingness_df[missingness_df['Sample']==sample]
                if subset['Missingness'].values:
                    newcol.append(subset['Missingness'].values)
           
            if newcol:
                plotdf[group]=pd.DataFrame(newcol)

        print("Box plot for missingness different sample groups")
        fig, ax = plt.subplots()
        plotdf.plot(kind='box', patch_artist='True',boxprops=dict(facecolor=color,alpha=0.5),
                                               medianprops=dict(color='black'),
                                               capprops=dict(color='black'),
                                               flierprops=dict(color='black', markerfacecolor='black'),
                                               whiskerprops=dict(color='black'),ax=ax)
        for i, group in enumerate(groups):
            y = plotdf[group]
            #x = [i + 1] * len(y)  # Generate a list of x positions for each data point
            x = np.random.normal(i + 1, 0.02, size=len(y))  # i+1 positions the points; adjust 0.04 for less/more spread
            ax.scatter(x, y, color='black', alpha=0.9, s=10)  # Adjust alpha and s (size) as needed

        plt.ylabel("% Missingness",fontsize=16)
        plt.xlabel(feature)
        plt.xticks(rotation="vertical",fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(linestyle=':')
        plt.ylim(0,ylim0)
        #plt.title('Data missingness in pg with sample groups')
        figure_file="missingness_pg_"+str(feature)+output+".svg"
        plt.savefig(figure_file, bbox_inches="tight")


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

        #### drop samplekey values that are associated with bad samples
        #### these samples were already excluded in Loader rawinput
        #### need to drop rows with samplekey value == bad samples

        df = df.apply(lambda row: row[~df['samplekey'].isin(self.loader.outlier_samples)])
        #print(df['samplekey'].values)

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

    def plot_heatmap(self, feature='Well',output=''):

        plotdata=self.rawinput.drop(['Protein.Group','Protein.Ids','Protein.Names','First.Protein.Description'],axis=1)
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
        #plotdata_np = plotdata.T.to_numpy()

        metric='correlation'
        heatmap=sns.clustermap(plotdata.T, metric=metric,cmap='vlag',col_cluster=False,vmin=-1,vmax=1,z_score=0,figsize=(70, 50))
        #heatmap=sns.clustermap(plotdata.T, metric=metric, cmap='vlag',standard_scale=1,method="weighted", figsize=(70, 15))
        heatmap.ax_row_dendrogram.set_visible(True)
        heatmap.ax_col_dendrogram.set_visible(True)
        plt.savefig('sns_clustermap_'+metric+"_"+feature)

        figure=px.imshow(plotdata.T, width=6000,height=1000)
        plotdata.reset_index(inplace=True)
        plotdata.drop('Genes',axis=1,inplace=True)
        figure=px.imshow(plotdata.T)
        outputfile='heatmap_T_pg_sorted_'+feature+str(output)
        pio.write_html(figure,outputfile+'.html')



        ####Plot seaborn for this
        # Transposing the DataFrame for the heatmap
        data_for_heatmap = plotdata.T
        data_for_heatmap=data_for_heatmap.sort_index()
        # Create a figure and adjust its size as needed
        plt.figure(figsize=(20, 5))  # Width and height values can be adjusted to your preference
        # Create the heatmap
        sns.heatmap(data_for_heatmap, cmap="viridis")  # You can change the colormap by adjusting the 'cmap' parameter
        # Save the plot as an image, then embed into an HTML file
        outputfile = 'heatmap_T_pg_seaborn_sorted_' + feature + str(output)
        plt.savefig(outputfile + '.svg')




        heatmap2=sns.clustermap(plotdata.T, metric=metric,cmap='vlag',vmin=-1,vmax=1,col_cluster=False,z_score=0,figsize=(70, 50))
        #heatmap2=sns.clustermap(plotdata.T, metric=metric, cmap='vlag',standard_scale=1,method="weighted", figsize=(70, 15))
        heatmap2.ax_row_dendrogram.set_visible(True)
        heatmap2.ax_col_dendrogram.set_visible(True)
        plt.savefig('sns_clustermap_index_'+metric+"_"+feature)
        plt.clf()
        plt.close()

############################################################################ PCA #################################
    def plot_pca(self, column=None,output="", marker="o",circle=False):

        ################## Impute within group in column using min#######################

        imputed_mat = self.mat.copy(deep=True)
        imputed_mat = imputed_mat.astype(float) ###already log2

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
        #data_to_plot.to_csv('_pca_pg_data_to_plot_'+column+'.csv',sep='\t')

        label_column = self.metadata[column].astype(str)
        #label_column.to_csv('_pca_pg_label_column_debug_'+column+'.csv',sep='\t')
        label_column_df = label_column.to_frame(name=column).reset_index(drop=True)

        label_file=self.metadata['samplekey'].astype(str)
        #label_file.to_csv('_pca_pg_label_file_debug_'+column+'.csv',sep='\t')
        label_file_df = label_file.to_frame(name='samplekey').reset_index(drop=True)


        plotdf=pd.concat([data_to_plot, label_column_df, label_file_df], axis=1)
        plotdf.to_csv('_pca_pg_plotdf_debug_'+column+impute+'.csv',sep='\t')

        figure=px.scatter(plotdf,x='PCA1',y='PCA2',color=column,hover_name="samplekey")
        figure.update_traces(marker_size=10)
        figure.update_layout(
            title='PCA Protein Group Level : '+impute,
            #title='PCA Protein Group Level : '+' Impute within sample groups',
            xaxis_title="PCA 1(%.2f%%)" % (pca.explained_variance_ratio_[0]*100),
            yaxis_title="PCA 2(%.2f%%)" % (pca.explained_variance_ratio_[1]*100),
            legend_title=column,
            font=dict(
                family='Tahoma',
                size=18,
                color='RebeccaPurple'
            )
        )
        outputfile='pca_pg_'+column+"_"+output+"_"+str(MISSINGNESS_THRESH)+impute
        outputfile=trim_string(outputfile)
        #outputfile='pca_pg_'+column+"_"+output+'_standardized_impute_within_group'
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

        plt.title(f'PCA Plot Protein Group Level', fontsize=18)
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
        
############################### VOLCANO ##############################################################################################################

    def volcano_plot(self, group1, group2, column, output="", figuresize1=10, figuresize2=5, method="ttest"):

        plt.rcParams["figure.figsize"] = (figuresize1, figuresize2)
        print("Volcano plot for contrasts: "+group1+" vs "+group2+" using column "+column)
        outputfile='volcano_'+group1+'_'+group2+output

        group1_cutoff=75
        group2_cutoff=75

        imputed_mat = self.mat.copy(deep=True)
        imputed_mat = imputed_mat.astype(float)
        group1_samples = self.metadata[self.metadata[column]==group1]["samplekey"].tolist()
        group2_samples = self.metadata[self.metadata[column]==group2]["samplekey"].tolist()

        ####compute missingness for features in group2 samples
        NA_rate=self.mat.loc[group1_samples].isnull().sum()/self.mat.loc[group1_samples].shape[0]*100

        print(imputed_mat.loc[group1_samples].shape)
        drop_index=[]
        for index, value in NA_rate.items():
            if value > group1_cutoff:
                imputed_mat.drop(columns=[index], inplace=True)
                drop_index.append(index)
        print('Group1: after',imputed_mat.loc[group1_samples].shape)

        NA_rate.drop(labels=drop_index, inplace=True)
        #print('Group1', NA_rate.size)

        ####compute missingness for features in group2 samples
        NA_rate=imputed_mat.loc[group2_samples].isnull().sum()/imputed_mat.loc[group2_samples].shape[0]*100
        print(imputed_mat.loc[group2_samples].shape)
        drop_index=[]
        for index, value in NA_rate.items(): ###this NA_rate is after dropping peptides with high missingness
            if value > group2_cutoff:
                imputed_mat.drop(columns=[index], inplace=True)
                drop_index.append(index)
        print('Group2: after',imputed_mat.loc[group2_samples].shape)

        NA_rate.drop(labels=drop_index, inplace=True)
        #print('Group2', NA_rate.size)

        #impute data per column of proteins for each sample group
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, keep_empty_features=True)

      
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
                                            index=imputed_mat.loc[group1_samples].index, columns=imputed_mat.loc[group2_samples].columns)
        imputation_array_2 = imp.fit_transform(imputed_mat.loc[group2_samples].values)
        imputed_mat.loc[group2_samples]= pd.DataFrame(imputation_array_2,
                                            index=imputed_mat.loc[group2_samples].index, columns=imputed_mat.loc[group2_samples].columns)

        stat = Statistics(dataset=self, group1=group1, group2=group2, column=column, method=method)
        plotdf=stat.diff_expression_analysis(imputed_mat)

        plotdf['negLogP']=-np.log10(plotdf['pval'])
        plotdf['NA_rate']=NA_rate.values

        plotdf.to_csv('new_table_'+outputfile+'.csv',sep=',')
        #snsplot=sns.scatterplot(data=plotdf, x="log2fc", y="negLogP", edgecolor="black", color="gray",s=4, alpha=0.4)
        figure=px.scatter(plotdf,x='log2fc',y='negLogP',color='NA_rate',hover_name="Protein.Group",
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
            xaxis_range=[-5,5], ############ VOLCANO PLOTS X AXIS #######################
            yaxis_range=[0,14],  ############ VOLCANO PLOTS Y AXIS #######################
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
        figure.update_xaxes(showline=True,linewidth=2,linecolor='black',zeroline=True)
        figure.update_yaxes(showline=True,linewidth=2,linecolor='black',zeroline=True)
        figure.add_hline(y=2,line_width=2, line_dash="dot",line_color='dimgray')
        figure.add_vline(x=2, line_width=2, line_dash="dot",line_color='dimgray')
        figure.add_vline(x=-2, line_width=2, line_dash="dot",line_color='dimgray')
        pio.write_html(figure,outputfile+'.html')


        """
        Print significant values
        """
        sigval=[]
        fig, ax = plt.subplots()
        ax.scatter(data=plotdf, x="log2fc", y="negLogP", edgecolor="dimgray", color="dimgray",s=3, alpha=0.7)
        #plt.xlim(-2, 2)
        #plt.ylim(0, 6)
        """ADJ.P VALUE AND LOG2FC threshhold"""
        p_thresh=2
        fc_thresh=2
        #plt.xlim(-4, 4) ############ VOLCANO PLOTS X AXIS #######################
        #plt.ylim(0, 6)  ############ VOLCANO PLOTS Y AXIS #######################
        plt.axhline(p_thresh, linewidth=1, color="dimgray", linestyle=':')
        plt.axvline(fc_thresh, linewidth=1, color="dimgray", linestyle=':')
        plt.axvline(-fc_thresh, linewidth=1, color="dimgray", linestyle=':')
        sigval_df=pd.DataFrame(sigval)
        for index, line in plotdf.iterrows():
            prot=line[0]
            gene=prot.split('_')[0]
            #red for upmodulation
            if line['negLogP'] > p_thresh and line['log2fc']>fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="red", color="red",s=5, alpha=0.7)
                sigval.append(line) 
            #blue for downmodulation
            elif line['negLogP'] > p_thresh and line['log2fc']<-fc_thresh:
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="blue", color="blue",s=5, alpha=0.7)
                sigval.append(line) 

            #if line['log2fc']>2:
            #    plt.annotate(gene, (line["log2fc"], line["negLogP"]), fontsize=3, color="red")
            if line['log2fc']>fc_thresh and line['negLogP'] > p_thresh:
                plt.annotate(gene, (line["log2fc"], line["negLogP"]), fontsize=4, color="red")
            if line['log2fc']<-fc_thresh and line['negLogP'] > p_thresh:
                plt.annotate(gene, (line["log2fc"], line["negLogP"]), fontsize=4, color="blue")
            elif line['log2fc']<-2:
                plt.annotate(gene, (line["log2fc"], line["negLogP"]), fontsize=4, color="blue")
                
            if "TDP2" in str(prot):
                print(gene, line["log2fc"], line["negLogP"])
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="black", color="yellow",s=5, alpha=0.9)
                plt.annotate(gene, (line["log2fc"], line["negLogP"]), fontsize=10, color="yellow")
            if "TDP1" in str(prot):
                print(gene, line["log2fc"], line["negLogP"])
                ax.scatter(line["log2fc"], line["negLogP"], edgecolor="indigo", color="mediumpurple",s=5, alpha=0.9)
                plt.annotate(gene, (line["log2fc"], line["negLogP"]), fontsize=10, color="indigo")

        sigval_df=pd.DataFrame(sigval)
        sigval_df['Protein.Names']=self.rawinput['Protein.Names'].loc[sigval_df.index]
        sigval_df['Protein.Description']=self.rawinput['First.Protein.Description'].loc[sigval_df.index]
        sigval_df.to_csv("sigval_pg_"+outputfile+".csv",sep=',')
        print("Max log2fc = ", max(plotdf['log2fc']), "   Min log2fc = ",min(plotdf['log2fc']))
        #plt.axline((0, p_thresh), (1, p_thresh), linewidth=4, color="gray")
        plt.xlabel("log2 FC", fontsize=18)
        plt.ylabel("-log10(adj.p value)", fontsize=18) #### ACTUALLY USE P VAL
        #plt.ylabel("-log10(p value)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(group1+'_'+group2+output,fontsize=24)
        #plt.grid()
        #plt.legend(loc="best", bbox_to_anchor=(1,1))

        plt.savefig('_static_'+outputfile+".svg", bbox_inches="tight")
        plt.clf() 
        plt.close()
        

    def get_sub_pg_matrix(self, column):
        # get proteins identified in each unique group in column from report.pg_matrix
        groups = self.metadata[column].unique()
        rawdata=self.rawinput.copy()
        rawdata.set_index('Genes')
        print(rawdata['Genes'])
        for group in groups:
            group_samples = self.metadata[self.metadata[column] == group]["samplekey"].tolist()
            sub_pg=pd.DataFrame(rawdata[group_samples].T)
            print(sub_pg)
            exit()
            sub_pg['Genes'].to_csv('sub_pg_'+group+'.csv')

    def box_plot(self, protein, column, groups=[], output="",color='dodgerblue',ylim0=-1,ylim1=-1,figuresize1=10,figuresize2=7):

        plt.rcParams["figure.figsize"] = (figuresize1, figuresize2)
        row=self.rawinput.index[self.rawinput['Genes']==protein].tolist()
        if not row:
             print("Box Plot: No "+protein+" in PG matrix!!!")
             return
    
        rowdata = self.rawinput.loc[row]
        ###rowdata.T.to_csv("_debug_pg"+protein+"_"+column+"_intensity.csv",sep=',')
        if not len(groups):
            groups = self.metadata[column].unique()
    
        #plotdf=pd.DataFrame()
        plotdf_dict = {}  # This will hold all your DataFrames
        number_isoforms = 0  # Will be updated based on the actual number of isoforms

        # Initialize DataFrames for each isoform outside of the groups loop
        # It's important to do this once based on the maximum number of isoforms you expect across all groups

        MAX_COL_LENGTH=20 ### max number of samples per group 
        for group in groups:
            group_samples = self.metadata[self.metadata[column] == group]["samplekey"].tolist()
            newcol = (rowdata[group_samples].T).astype(float)

            # Apply log2 transformation while preserving the DataFrame structure
            if IS_LOG2:
                newcol = np.log2(newcol + 1e-6)  # Adding 1 to avoid log2(0)

            # Update number_isoforms if it's the first iteration or if there's a group with more isoforms
            number_isoforms = max(number_isoforms, newcol.shape[1])

        # Initialize plotdf_dict with empty DataFrames
        for i in range(number_isoforms):
            unique_name = f'isoform_{i}'
            plotdf_dict[unique_name] = pd.DataFrame(columns=groups)

        # Populate DataFrames
        for group in groups:
            group_samples = self.metadata[self.metadata[column] == group]["samplekey"].tolist()
            newcol = (rowdata[group_samples].T).astype(float)
            if IS_LOG2:
                newcol = np.log2(newcol + 1e-6)  # Applying log2 transformation

            for i, col in enumerate(newcol.columns):
                unique_name = f'isoform_{i}'
                plotdf_dict[unique_name][group] = np.full(MAX_COL_LENGTH, np.nan)
                values_length = len(newcol[col].values)
                plotdf_dict[unique_name][group][:values_length] = newcol[col].values
                #plotdf_dict[unique_name][group] = newcol[col].values


        print("Box plot with log2 transform of intensity="+str(IS_LOG2)+" and imputation="+str(IMPUTE_METHOD)+' for '+protein)

        for unique_name, df in plotdf_dict.items():
            fig, ax = plt.subplots()
    
            df[groups].plot(kind='box', patch_artist='True',boxprops=dict(facecolor=color,alpha=0.6),
                            medianprops=dict(color='black'),
                            capprops=dict(color='black'),
                            flierprops=dict(color='black', markerfacecolor=color, markersize=4),
                            whiskerprops=dict(color='black'),
                            ax=ax)

            for i, group in enumerate(groups):
                y = df[group]
                x = [i + 1] * len(y)  # Generate a list of x positions for each data point
                #x = np.random.normal(i + 1, 0.02, size=len(y))  # i+1 positions the points; adjust 0.04 for less/more spread
                ax.scatter(x, y, color='black', alpha=1.0, s=24, zorder=3)


            if IS_LOG2:
                plt.ylabel("Log2[Intensity]", fontsize=18)
            else:
                plt.ylabel("Intensity", fontsize=18)
            plt.xlabel(column, fontsize=18)
            plt.xticks(rotation="vertical",fontsize=18)
            plt.yticks(fontsize=18)
            if(ylim0>0):
                plt.ylim(ylim0,ylim1)
            plt.grid(linestyle=':')
            if (number_isoforms>1):
                plt.title(protein+'  '+unique_name, fontsize=22)
                plt.savefig("box_pg_"+protein+"_"+column+'_'+unique_name+output+"_NoImpute.svg", bbox_inches="tight")
            else:
                plt.title(protein, fontsize=22)
                plt.savefig("box_pg_"+protein+"_"+column+output+"_NoImpute.svg", bbox_inches="tight")

            plt.clf() 
            plt.close(fig)

        
    def violin_plot(self, protein_keys):

        plt.rcParams["figure.figsize"] = (15, 5)
        plt.rcParams.update({'figure.autolayout': True})
        plt.rc ('axes', axisbelow=True)
        ### First need to identify topN proteins based on sort_key = sample where protein intensities will be sorted by
        ### read protein_keys
        protein_df = pd.read_csv(protein_keys, sep=',', dtype=str)
        proteins = protein_df['Genes_Protein.Group'].values
        #print(proteins)

        ### Second extract subset of matrix
        topN_mat = self.rawinput.copy(deep=True)
        #topN_mat['Protein.Group'].to_csv('proteins.csv')
        topN_mat = topN_mat.apply(lambda row: row[topN_mat['Protein.Group'].isin(proteins)])
        print('Number of proteins in protein_keys = ',topN_mat.shape[0],' from ',protein_keys)
        ### Sort topN_mat such that proteins are alphabetical
        topN_mat.sort_values(by='Protein.Group', axis=0, ascending=True, inplace=True)


        ### Third compute and plot CV
        protein_list = topN_mat['Protein.Group'].tolist()  #### take the entire mat but could be an explicit list of proteins

        row=topN_mat.index[topN_mat['Protein.Group']==protein_list].tolist()
        samples = self.metadata["samplekey"].tolist()

        topN_mat[samples]=topN_mat[samples].astype(float)
        self.rawinput[samples]=self.rawinput[samples].astype(float)
        #topN_mat[samples]=np.log2(topN_mat[samples]+0.000001)
        topN_mat[samples]=topN_mat[samples]/self.rawinput[samples].sum(axis=0, skipna=True)
        #maskna=0.00001
        #topN_mat=topN_mat.fillna(maskna)
        #topN_mat[samples]=np.log2(topN_mat[samples]+0.00001) ##Log2 transform)

        topN_mat.drop(columns=['index','Protein.Ids','Protein.Names','Genes','First.Protein.Description'], inplace=True)
        fig, ax = plt.subplots()
        #plotdf=pd.DataFrame()
        plot_df=pd.DataFrame()
        title=protein_keys.split('.')[0]
        gene_list=[]
        for protein in protein_list:
            gene_list.append(protein.split('___')[0])
            #ax.violinplot(dataset=topN_mat[topN_mat['Protein.Group']==protein], positions=[i])
            column = topN_mat[topN_mat['Protein.Group']==protein].set_index('Protein.Group').T
            plot_df[protein]=pd.DataFrame(column)

        figure = sns.violinplot(data=plot_df)
        #sns.stripplot(data=plot_array, color='black',alpha=0.5)
        figure.set_xticklabels(gene_list)
        plt.xticks(rotation=45,fontsize=14)
        #plt.axes.ticklabel_format(scilimits=(0,0))

        plt.xlabel("Proteins",fontsize=16)
        plt.ylabel("signal/total signal",fontsize=16)
        plt.yscale(value='log', base=2)
        figure.yaxis.set_major_formatter('{x:.8f}')

        plt.title("Violin plot for "+protein_keys)
        plt.grid(visible=True,which='major',axis='both',color='lightgrey',linestyle=':')

        plt.savefig('violin_plot_'+title+'.svg', format='svg', transparent=True)
        plt.clf()
        plt.close()

    def cv_plot(self, title=''):

        #### Make a copy of self.rawinput
        rawinput = self.rawinput.copy(deep=True)

        #### First need to calculate CV for each protein
        #### CV calculated using scipy === np.std(x, axis=axis, ddof=ddof) / np.mean(x)
        samples = self.metadata["samplekey"].tolist()
        rawinput.drop(columns=['index','Protein.Ids','Protein.Names','Genes','First.Protein.Description'], inplace=True)

        cv_df=[]
        for index, row in rawinput.iterrows():
            array=row[samples].values.astype(float)
            array=np.log2(array+0.0001)
            cv=scipy.stats.variation(array, ddof=0,nan_policy='omit')
            #if np.isnan(cv)==True:
            #   print(row)
            #   exit()
            cv_df.append(cv*100)
        rawinput.insert(0,'CV',cv_df,allow_duplicates=False)

  


        # Calculate mean, median, and 90th percentile of the CV
        mean_cv = np.mean(rawinput['CV'])
        median_cv = np.median(rawinput['CV'])
        percentile_90_cv = np.percentile(rawinput['CV'], 90)

        # Plotting the density distribution of CV
        sns.displot(rawinput, x='CV', kde=True, fill=True, color='skyblue', alpha=0.5, bins=150, height=4, aspect=2.5)
        plt.axvline(mean_cv, color='red', linewidth=3,linestyle='--', label=f'Mean: {mean_cv:.1f}%')
        plt.axvline(median_cv, color='black', linewidth=3,linestyle='-', label=f'Median: {median_cv:.1f}%')
        plt.axvline(percentile_90_cv, color='blue', linewidth=3,linestyle=':', label=f'90th Percentile: {percentile_90_cv:.1f}%')

        # Adding legend, titles, and labels
        plt.legend(prop={'size': 16}, title='CV Distribution')
        plt.title(title, fontsize=20)
        plt.xlabel('% Coefficient of Variation', fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(0, 30)
        plt.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle=':')
        
        # Saving the plot
        plt.savefig('cv_plot_'+title+'.svg', format='svg', transparent=True, bbox_inches='tight')
        plt.clf()
        plt.close()

#######################################################################################################################################

    # calculate corr coeff protein 1 vs protein 2 and color by sample groups
    def scatter_corr(self, protein_1, protein_2, column, groups=[]):

        # Filter the data for the two proteins
        if ('___' in protein_1):
            protein_data_1 = self.rawinput[self.rawinput['Protein.Ids'].isin([protein_1])]
            protein_data_2 = self.rawinput[self.rawinput['Protein.Ids'].isin([protein_2])]
            protein_data=pd.concat([protein_data_1,protein_data_2],axis=0)
        else:
            protein_data_1 = self.rawinput[self.rawinput['Genes'].isin([protein_1])]
            protein_data_2 = self.rawinput[self.rawinput['Genes'].isin([protein_2])]
            protein_data=pd.concat([protein_data_1,protein_data_2],axis=0)

        if (protein_data.shape[0] != 2):
            print('Error: Shape of protein_data is not 2 protein_1 or protein_2 not present',protein_data.shape[0])
            exit()
        #protein_data.T.to_csv('_scatter_'+output+".csv",sep=',')
        if not len(groups): #plot all unique features in column
            groups = self.metadata[column].unique()

        protein_data.set_index('Protein.Group',inplace=True) 
        data_to_plot=(protein_data.drop(columns=['index','Protein.Ids','Protein.Names','Genes','First.Protein.Description'])).T

        #protein_data.T.to_csv('_scatter_'+output+".csv",sep=',')
        if not len(groups): #plot all unique features in column
            groups = self.metadata[column].unique()

        data_to_plot=(protein_data.drop(columns=['index','Protein.Ids','Protein.Names','Genes','First.Protein.Description'])).T

        # Reset the index of plotdf to ensure it's using a simple integer index
        data_to_plot=data_to_plot.reset_index(drop=True)
        data_to_plot.columns=[protein_1,protein_2]
        data_to_plot=data_to_plot.astype(float)
        data_to_plot=np.log2(data_to_plot)

        label_column = self.metadata[column].astype(str)
        #label_column.to_csv('_corr_pg_label_column_debug_'+column+'.csv',sep='\t')
        label_column_df = label_column.to_frame(name=column).reset_index(drop=True)

        label_file=self.metadata['samplekey'].astype(str)
        #label_file.to_csv('_corr_pg_label_file_debug_'+column+'.csv',sep='\t')
        label_file_df = label_file.to_frame(name='samplekey').reset_index(drop=True)


        plotdf=pd.concat([data_to_plot, label_column_df, label_file_df], axis=1)
        plotdf_nona = plotdf.copy().dropna(subset=[protein_1, protein_2])
        try:
            corr_coeff, _ = pearsonr(plotdf_nona[protein_1], plotdf_nona[protein_2])
            return corr_coeff
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

#######################################################################################################################################

    # plot protein 1 vs protein 2 and color by sample groups
    def scatter_corr_plot(self, protein_1, protein_2, column, groups=[], output=''):

        plt.rcParams["figure.figsize"] = (10, 7)
        output=protein_1+'_'+protein_2+'_'+column+'_'+output
        output=trim_string(output)


        print('---- Plotting scatter correlation plot for '+protein_1+' and '+protein_2+' ----')
        # Filter the data for the two proteins
        protein_data=pd.DataFrame()
        if ('___' in protein_1):
            protein_data_1 = self.rawinput[self.rawinput['Protein.Ids'].isin([protein_1])]
            protein_data_2 = self.rawinput[self.rawinput['Protein.Ids'].isin([protein_2])]
            protein_data=pd.concat([protein_data_1,protein_data_2],axis=0)
        else:
            protein_data_1 = self.rawinput[self.rawinput['Genes'].isin([protein_1])]
            protein_data_2 = self.rawinput[self.rawinput['Genes'].isin([protein_2])]
            protein_data=pd.concat([protein_data_1,protein_data_2],axis=0)

        #protein_data.T.to_csv('_scatter_'+output+".csv",sep=',')
        if not len(groups): #plot all unique features in column
            groups = self.metadata[column].unique()

        protein_data.set_index('Protein.Group',inplace=True) 
        data_to_plot=(protein_data.drop(columns=['index','Protein.Ids','Protein.Names','Genes','First.Protein.Description'])).T

        # Reset the index of plotdf to ensure it's using a simple integer index
        data_to_plot=data_to_plot.reset_index(drop=True)
        data_to_plot.columns=[protein_1,protein_2]
        data_to_plot=data_to_plot.astype(float)
        data_to_plot=np.log2(data_to_plot)

        label_column = self.metadata[column].astype(str)
        #label_column.to_csv('_corr_pg_label_column_debug_'+column+'.csv',sep='\t')
        label_column_df = label_column.to_frame(name=column).reset_index(drop=True)

        label_file=self.metadata['samplekey'].astype(str)
        #label_file.to_csv('_corr_pg_label_file_debug_'+column+'.csv',sep='\t')
        label_file_df = label_file.to_frame(name='samplekey').reset_index(drop=True)


        plotdf=pd.concat([data_to_plot, label_column_df, label_file_df], axis=1)
        plotdf.to_csv('_corr_pg_plotdf_debug_'+column+'.csv',sep='\t')

        # Assuming the necessary data preparation is done before this line as per your original code
        # plotdf is ready with 'protein_1', 'protein_2', 'column' for color, and 'samplekey' for hover information

        plotdf_nona = plotdf.copy().dropna(subset=[protein_1, protein_2])
        plotdf=plotdf.fillna(0.0)
        # Calculate Pearson correlation coefficient
        corr_coeff, _ = pearsonr(plotdf_nona[protein_1], plotdf_nona[protein_2])
        # Create an interactive scatter plot with Plotly
        #fig = px.scatter(plotdf, x=protein_1, y=protein_2,
        #                 color=column,  # Defines colors based on the feature column
        #                 hover_data=['samplekey'],  # Defines what appears on hover
        #                 title=f'Scatter plot for {protein_1} vs. {protein_2} - Pearson Correlation: {corr_coeff:.2f}')


        # Calculate linear regression
        regression_result = linregress(plotdf_nona[protein_1], plotdf_nona[protein_2])
        slope = regression_result.slope
        intercept = regression_result.intercept
        r_value = regression_result.rvalue
    
        # Prepare a line for plotting
        plotdf_nona['regression_line'] = plotdf_nona[protein_1] * slope + intercept
    
        # Create the scatter plot
        fig = px.scatter(plotdf, x=protein_1, y=protein_2, color=column,
                         hover_data=['samplekey'])
    
        # Add the regression line
        fig.add_traces(go.Scatter(x=plotdf_nona[protein_1], y=plotdf_nona['regression_line'],
                                  mode='lines', name='Regression Line',
                                  line=dict(dash='dot', width=1),
                                  showlegend=True))
    
        # Add annotations for the equation and R2 value
        equation_text = f'y = {slope:.2f}x + {intercept:.2f}<br>R^2 = {r_value**2:.2f}'
        fig.add_annotation(x=1, y=max(plotdf[protein_2]), text=equation_text, showarrow=False)
    
        fig.update_layout(title=f'Scatter Plot with Linear Regression for {protein_1} vs {protein_2} - Pearson Correlation: {corr_coeff:.2f}',
                          xaxis_title=protein_1,
                          yaxis_title=protein_2)

        fig.write_html('corr_plots/pg_corr_plot_'+output + '.html')
        #fig.write_html('pg_corr_plot_'+protein_1+'/pg_corr_plot_'+output + '.html')


        ################# Create the same plot with seaborn for presentation #############################
        # Set the color palette to use with seaborn
        unique_labels = plotdf[column].unique()

        #color blind palette
        sns.set_palette("colorblind")
        palette = sns.color_palette("bright", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    
        # Create the scatter plot with regression line
        plt.figure(figsize=(10, 7))
        sns.regplot(x=protein_1, y=protein_2, data=plotdf_nona, scatter_kws={'s':50, 'alpha':0.5}, line_kws={'color':'gray', 'ls':'dotted'}, ci=None)
        sns.scatterplot(x=protein_1, y=protein_2, data=plotdf, hue=column, palette=color_map, s=100, edgecolor='black', alpha=0.7)
        plt.xlabel(protein_1,fontsize=20)
        plt.ylabel(protein_2,fontsize=20)
    
        # Annotate the plot with the regression equation and R^2 value
        equation_text = f'y = {slope:.2f}x + {intercept:.2f}\nR^2 = {r_value**2:.2f}'
        plt.annotate(equation_text, xy=(1, 0), xycoords='axes fraction', ha='right', va='bottom', fontsize=12,
        #plt.annotate(equation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.5))
    
        plt.title(f'{protein_1} vs {protein_2} - Pearson Corr = {corr_coeff:.2f}',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    
        # Show the plot
        plt.legend(title=column,fontsize=16)
        plt.tight_layout()
    
        # Optionally, save the plot if an output file path is provided
        plt.savefig('corr_plots/pg_corr_seaborn_plot_'+output+'.svg')  # Saves the plot
        #plt.savefig('pg_corr_seaborn_plot_'+protein_1+'/pg_corr_seaborn_plot_'+output+'.svg')  # Saves the plot
        print(protein_1,protein_2,corr_coeff)

        plt.clf() 
        plt.close()

        return corr_coeff
        
#######################################################################################################################################

    def bar_plot_stats(self, stats,feature, doe_file,groups=[],color='cornflowerblue'):

        #First read stats
        stats_mat = pd.read_excel(doe_file)
        stats_mat[stats]=stats_mat[stats].astype(float)

        #Second group by samples in feature
        if groups=='':
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
 
        filename='bar_stats_'+stats+'_'+feature

        printdata={'Group' : plot_labels, stats : plotdf, 'STD' : errordf}
        printdf=pd.DataFrame(printdata)
        printdf.to_csv('_'+filename+'.csv')
        print("Bar plot for "+stats+" with "+feature)
        plt.rcParams["figure.figsize"] = (10, 7)
        fig,ax=plt.subplots()
        ax.bar(groups,plotdf,yerr=errordf,capsize=5,
               label=plot_labels,color=color,alpha=0.7,edgecolor='black',linewidth=2,width=0.8)
        plt.ylabel(stats,fontsize=18)
        #plt.ylim(8500,9500)
        #plt.xlabel(feature,fontsize=16)
        plt.xticks(rotation="vertical",fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(linestyle=':')
        plt.title('Bar Plot for '+str(stats)+' vs. '+str(feature),fontsize=18)
        plt.savefig(filename+'.png', bbox_inches="tight")
        plt.clf() 
        plt.close()
        
#######################################################################################################################################

    def waterfall_plot(self, title='Waterfall plot'):

        ### sort proteins by intensity of the last column (last sample)
        rawinput = self.rawinput.copy(deep=True)
        rawinput.drop(columns=['index','Protein.Ids','Protein.Names','Protein.Group','Genes','First.Protein.Description'], inplace=True)

        plotdata=pd.DataFrame()
        plotdata['Protein.Group']=self.rawinput['Protein.Group']
        plotdata['Intensity']=rawinput.median(axis=1)

        plotdata['Intensity']=plotdata['Intensity'].astype(float)
        maskna=1e-8
        plotdata['Intensity']=plotdata['Intensity'].fillna(maskna)
        plotdata['Log2Intensity']=np.log2(plotdata['Intensity'])
        plotdata.sort_values(by='Log2Intensity',axis=0, ascending=False, inplace=True)
        plotdata.reset_index(inplace=True)
        plotdata.to_csv('Waterfall_Plot_PGs.csv',sep=',')
        print('Plotting waterfall with missing values imputed with zero')

        figure=px.scatter(plotdata,y='Log2Intensity',color='Protein.Group',hover_name="Protein.Group")
        figure.update_traces(marker_size=20,opacity=0.7)
        figure.update_layout(
            title=title,
            xaxis_title="Index",
            yaxis_title="Log2Intensity",
            legend_title='Protein.Group',
            font=dict(
                family='Tahoma',
                size=30,
                color='RebeccaPurple'
            )
        )
        pio.write_html(figure,'waterfall_plot.html')

        """Seaborn plot"""
        # Set the theme for Seaborn
        sns.set_theme(style="whitegrid")
        sns.set_palette("colorblind")

        # Create the scatter plot
        plt.figure(figsize=(20, 10))  # Set the figure size
        sns.set_palette("colorblind")
        scatter_plot = sns.scatterplot(data=plotdata, x=plotdata.index, y='Log2Intensity', color='grey',edgecolors='lightgrey', s=100, alpha=0.7)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Other Proteins', markerfacecolor='grey', markeredgecolor='grey', markersize=10)]


        # Overlay the highlighted points and create custom legend entries
        colors = ['magenta', 'lime', 'yellow','red', 'purple', 'orange'] 
        index_labels = {2081: 'TDP1', 311: 'TOP1', 468: 'TOP2B', 986 : 'TOP2A'}
        highlighted_index=[2081,311,468,986]
        palette = sns.color_palette("bright", len(highlighted_index))
        color_map = {label: palette[i] for i, label in enumerate(highlighted_index)}
        highlighted_data = plotdata.loc[highlighted_index]
        legend_elements = []  # To store legend elements

        for i, index in enumerate(highlighted_index):
            color = colors[i % len(colors)]  # Cycle through colors list
            plt.scatter(x=[index], y=plotdata.loc[index, 'Log2Intensity'], facecolors=color, edgecolors='black', s=250, zorder=20)
            # Create a legend entry for each highlighted point
            label = index_labels.get(index, 'Unknown')  # Use the label from the dictionary, default to 'Unknown'
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markeredgecolor='black', markersize=14))

        # Add the custom legend for highlighted points
        plt.legend(handles=legend_elements,fontsize=18)

        # Customize the plot
        scatter_plot.set_title(title, fontsize=25)
        scatter_plot.set_xlabel('Index', fontsize=25)
        scatter_plot.set_ylabel('Log2Intensity', fontsize=25)
        #scatter_plot.legend(title='Protein Group', fontsize='medium')
    
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.setp(scatter_plot.get_legend().get_texts(), fontsize='13')  # for legend text
        #plt.setp(scatter_plot.get_legend().get_title(), fontsize='14')  # for legend title
    
        # Save the plot as a file
        plt.savefig('waterfall_plot_seaborn.svg')
        # Display the plot
        plt.show()
        plt.clf()
        plt.close()
    
#######################################################################################################################################

"""
#TESTING CODE ABOVE
"""

pg_mat=Loader(input_file="../raw/report.pg_matrix.tsv",intensity_column="[sample]", index_column="Protein.Group")
pg_mat.print_load_parameters()

doe_file='EPJ-241211-MAPUF_RCP_DOE'
doe_file_path='../DOE/'+doe_file+'.xlsx'
ds = DataSet(
    loader=pg_mat,
    metadata_path=doe_file_path,
    sample_column="sample"
)
print(ds.mat)
print(ds.sample)
ds.metadata.to_csv('metadata_csv_format_read.csv',sep='\t')
ds.rawinput.to_csv('_debug_rawinput.csv',sep=',')

############################ drop control sample ###########################
#sample_list_to_drop=['F09', 'C12','A12','A11']
#print('control samples to drop = ',sample_list_to_drop)
#ds._drop_samples(sample_list_to_drop)
#ds.rawinput.to_csv("report.pr_matrix_cit_for_analysis_samples_dropped.tsv",sep='\t')
#ds.metadata.to_csv("metadata_samples_dropped.csv",sep='\t')
####################### DONE drop control sample ###########################

"""needs to analyze missingness before preprocessing"""
"""
column="Row"
print(ds.metadata[column].unique())
ds.analyze_missingness(feature=column, ylim0=20)
column="Column"
print(ds.metadata[column].unique())
ds.analyze_missingness(feature=column, ylim0=20)
column="Contrast"
print(ds.metadata[column].unique())
ds.analyze_missingness(feature=column, ylim0=20)
"""

ds.preprocess_data(log2_transform=IS_LOG2,imputation=IMPUTE_METHOD)

""" ###########  WATERFALL PLOT ###########"""
#ds.waterfall_plot()

""" ###########  HEATMAP ###########"""
output=''
#ds.plot_heatmap(output=output,feature='Contrast')

""" ###########  PCA PLOTS ###########"""
#ds.plot_pca(column='Contrast')

""" ###########  CV PLOT ###########"""
#ds.cv_plot(doe_file)


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


"""Bar Plots"""
#feature='Contrast'
#stats='Precursors.Identified'
#ds.bar_plot_stats(stats,feature, doe_file_path, groups=groups,color='cornflowerblue')
#stats='Proteins.Identified'
#ds.bar_plot_stats(stats,feature, doe_file_path, groups=groups,color='magenta')


"""###################BOX PLOTS#############################"""
column='Contrast'
#groups = ds.metadata[column].unique()
#print(groups)

ds.box_plot('TDP2',column, groups=groups,output="",color='cornflowerblue',ylim0=-11,ylim1=10,figuresize1=15,figuresize2=5)
ds.box_plot('TDP1',column, groups=groups,output="",color='cornflowerblue',ylim0=-11,ylim1=10,figuresize1=15,figuresize2=5)

#############################
"""##################### VOLCANO PLOTS ########################"""
column='Contrast'
group1='DMSO_4h'

group2='EXT-004525_1uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2='EXT-004525_10uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2='EXT-004525_30uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")

group2= 'BMT-885960_1uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'BMT-885960_10uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'BMT-885960_30uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")

group2= 'BMT-945203_1uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'BMT-945203_10uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'BMT-945203_30uM_4h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")

###
group1= 'DMSO_24h'
group2= 'EXT-004525_1uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'EXT-004525_10uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'EXT-004525_30uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")

group2= 'BMT-885960_1uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'BMT-885960_10uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'BMT-885960_30uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")

group2= 'BMT-945203_1uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'BMT-945203_10uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
group2= 'BMT-945203_30uM_24h'
ds.volcano_plot(group1, group2, column, output="", method="ttest")
