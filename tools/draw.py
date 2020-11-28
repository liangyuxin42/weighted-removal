import matplotlib.pyplot as plt

def draw_overall(original_result, cn_result, abtt_result_ds, wr_result_ds, D, task_name,emb_type):
    x = D
    corr_origin = [original_result]*len(D)
    corr_cn = [cn_result]*len(D)
    corr_abtt = abtt_result_ds
    corr_wr = wr_result_ds

    # Labels to use in the legend for each line
    line_labels = ["WR", "ABTT", "CN", "Original"]

    fig, ax = plt.subplots(figsize=(8,6))

    # Create the sub-plots, assigning a different color for each line.
    # Also store the line objects created
    ax.set_title(task_name, fontsize=20)
    l1 = ax.plot(x, corr_wr,     c="orangered", linewidth=2)[0]
    l2 = ax.plot(x, corr_abtt,   c="royalblue", linewidth=2)[0]
    l3 = ax.plot(x, corr_cn,     c="gray", linewidth=2)[0]
    l4 = ax.plot(x, corr_origin, c="darkgreen", linewidth=2)[0]
    ax.set_xlabel('d', fontsize=18)
    ax.set_ylabel('correlation', fontsize=18)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Create the legend
    fig.legend([l1,l2,l3,l4],     # The line objects
            labels=line_labels,   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
            title="methods",   # Title for the legend
            fontsize = 12
            )
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=0.84)

    plt.rcParams['figure.figsize'] = (8,6) #图片像素 
    plt.rcParams['savefig.dpi'] = 300       #分辨率 

    plt.savefig('%s_%s_overall.png' %(task_name, emb_type), dpi=300)#指定分辨率

    plt.show()
