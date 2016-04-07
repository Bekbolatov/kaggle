


from pylab import rcParams
rcParams['figure.figsize'] = 20, 15

fig = plt.figure()
st = fig.suptitle("Unfiltered event counts (hourly)", fontsize="x-large")
for i, event_type in enumerate(['page_view', 'ad_request', 'ad_impression', 'ad_click', 'contact_impression', 'professional_impression']):
    ax = fig.add_subplot(3, 2, i)
    for event_date in ssl_data.loc[event_type].index.unique():
        if event_date == '2016-04-03':
            continue
        dat = ssl_data.loc[(event_type, event_date)].sort_values(by='hh')
        ax.plot(dat['hh'], dat['cnt'], label = "{0}".format(event_date))
    plt.xlabel("Hour")
    plt.ylabel("Counts")
    plt.title(event_type)
    ax.legend()
plt.show()

st.set_y(0.95)
fig.subplots_adjust(top=0.85)
#plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=5.0)
plt.tight_layout()

