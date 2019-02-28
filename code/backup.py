
dim_list = [data_dim_nobs, data_dim_vars]
for dim in dim_list:
    plot_data = batch_benchmark(func_list=func_list, data_dimensions=dim)
    function_names = plot_data.columns
    plot_data.reset_index(inplace=True)

    for col in ['nobs', 'nvariables']:
        if len(plot_data[col].unique()) == 1:
            reduced_data = plot_data.drop(col, axis=1)
        else:
            x_name = col

    fig, ax = plt.subplots()

    
    for funcname in function_names:
        sns.lineplot(
            x=x_name, y=funcname, data=reduced_data,
            label=funcname, ax=ax)
    ax.set_xlabel(x_name, fontsize='xx-small')
    ax.set_ylabel('Time taken', fontsize='xx-small')
    plt.title('Ols Implementations', fontsize='x-small')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize = 'xx-small')
    plt.savefig("Perfomance_ols_{}.png".format(x_name), bbox_inches='tight')
    plt.show()
    plt.close()