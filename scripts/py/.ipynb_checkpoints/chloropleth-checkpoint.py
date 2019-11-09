
def DrawPlots(scheme, var, db, k, colorscheme, categorical=False, figsize=(16,8), saveto=None):
    '''
    Plot the distribution over value and geographical space of variable `var` using scheme `scheme`
    ...
    
    Arguments
    ---------
    scheme   : str
               Name of the classification scheme to use 
    var      : str
               Column name 
    db       : GeoDataFrame
               Table with input data
    k        : integer
               Number of classes for classification. Default is 4.
    figsize  : Tuple
               [Optional. Default = (16, 8)] Size of the figure to be created.
    saveto   : None/str
               [Optional. Default = None] Path for file to save the plot.
    '''
    
    '''
    Determine data type and set key variables
    '''
    
    

    '''
    Set up plots
    '''
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1,2,1)
      

    '''
    Set color values
    '''
    cmap = cm.get_cmap(colorscheme)
    
    if categorical == True:
        values = db[var].value_counts()
        labels = db[var].value_counts().index

#         print('Categories:Count')
#         for x in range(0,len(labels)):
#             print(labels[x],values[x])
        
        # add stacked  bar charts
#         ax1.pie(values,labels=labels,autopct='%1.1f%%')
#         ax1.set_title('Value Counts')

        # chloropleth
        db.plot(ax=ax1,column=var, categorical=True,legend=True, linewidth=0.1)
    else:
        ax2 = fig.add_subplot(1,2,2,facecolor=(.18, .31, .31))
        # Classify data     
        from mapclassify import Quantiles, Equal_Interval, Fisher_Jenks
        schemes = {'equal_interval': Equal_Interval, \
                   'quantiles': Quantiles, \
                   'fisher_jenks': Fisher_Jenks}
        classi = schemes[scheme](db[var], k=k)
#         print(classi)
        
        # KDE
        sns.kdeplot(db[var], shade=True, color=cmap(.5), ax=ax1)
        sns.rugplot(db[var], alpha=0.5, color=cmap(.5), ax=ax1)
        for cut in classi.bins:
            ax1.axvline(cut, color=cmap(1.0), linewidth=0.75)
        ax1.set_title('Value Distribution')
        
        # chloropleth
        p = db.plot(column=var, scheme=scheme, alpha=0.75, k=k, cmap=cmap, ax=ax2, legend=True,linewidth=0.1)
        ax2.axis('equal')
        ax2.set_axis_off()
        ax2.set_title('Geographical Distribution')

        fig.suptitle(scheme, size=25)
        if saveto:
            plt.savefig(saveto)
        plt.show()

        


